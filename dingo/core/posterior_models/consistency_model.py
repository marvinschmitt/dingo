import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import Base
from dingo.core.nn.cfnets import create_cf_model

import matplotlib.pyplot as plt

class ConsistencyModel(Base):
    """
    Class for consistency model.
    """
    def __init__(self, **kwargs):
        # self.total_steps = 1000  # FIXME: hardcoded
        super().__init__(**kwargs)
        
        # Compute number of gradient steps for the consistency model schedulers
        self.train_budget = self.metadata["train_settings"]["data"]["total_budget"] * self.metadata["train_settings"]["data"]["train_fraction"]
        self.total_steps = 0
        i = 0
        while 1:
            key = f"stage_{i}"
            if key in self.metadata["train_settings"]["training"]:
                stage_epochs = self.metadata["train_settings"]["training"][key]["epochs"]
                stage_batch_size = self.metadata["train_settings"]["training"][key]["batch_size"]
                self.total_steps += stage_epochs * self.train_budget // stage_batch_size
                i += 1
            else:
                break
        
        self.theta_dim = self.metadata["train_settings"]["model"]["posterior_kwargs"]["input_dim"]
        self.s0 = torch.tensor(self.model_kwargs["posterior_kwargs"]["consistency_args"]["s0"], dtype=torch.float32)
        self.s1 = torch.tensor(self.model_kwargs["posterior_kwargs"]["consistency_args"]["s1"], dtype=torch.float32)
        self.tmax = torch.tensor(self.model_kwargs["posterior_kwargs"]["consistency_args"]["tmax"], dtype=torch.float32)
        self.epsilon = torch.tensor(self.model_kwargs["posterior_kwargs"]["consistency_args"]["epsilon"], dtype=torch.float32)
        self.sigma2 = torch.tensor(self.model_kwargs["posterior_kwargs"]["consistency_args"]["sigma2"], dtype=torch.float32)
        self.current_step = 0
        self.c_huber = torch.tensor(0.00054 * math.sqrt(self.theta_dim), dtype=torch.float32)
        self.c_huber2 = self.c_huber**2

        print("init successful!")

    def initialize_network(self):
        model_kwargs = {k: v for k, v in self.model_kwargs.items() if k != "type"}
        if self.initial_weights is not None:
            model_kwargs["initial_weights"] = self.initial_weights
        self.network = create_cf_model(**model_kwargs)

        
    def _schedule_discretization(self):
        k_ = math.floor(self.total_steps / (math.log(self.s1 / self.s0) / math.log(2.0) + 1.0))
        out = min(self.s0 * math.pow(2.0, math.floor(self.current_step / k_)), self.s1) + 1.0
        return int(out)

    def _discretize_time(self, num_steps, rho=7.0):
        N = num_steps + 1.0
        indices = torch.linspace(1, N, steps=int(N), dtype=torch.float32, device=self.s0.device)  # Use linspace and add device for efficiency
        one_over_rho = 1.0 / rho
        discretized_time = (
            self.epsilon**one_over_rho + (indices - 1.0) / (N - 1.0) * (self.tmax**one_over_rho - self.epsilon**one_over_rho)
        ) ** rho
        return discretized_time

    def forward(self, xz, conditions=None, inverse=False, **kwargs):
        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        raise NotImplementedError("Consistency Models are not invertible")

    def _forward_train(self, x, noise, t, conditions=None, **kwargs):
        inp = x + t.unsqueeze(1) * noise
        return self.consistency_function(inp, t, conditions=conditions, **kwargs)

    def _inverse(self, z, conditions=None, steps=10, **kwargs):
        x = z.clone()
        discretized_time = torch.flip(self._discretize_time(steps), dims=[-1])
        t = torch.full((x.shape[0], 1), discretized_time[0], dtype=x.dtype, device=x.device)
        x = self.consistency_function(x, t, conditions=conditions)
        
        for n in range(1, steps):
            noise = torch.randn_like(x, device=x.device)
            x_n = x + torch.sqrt(torch.square(discretized_time[n]) - self.epsilon**2) * noise
            t = t.fill_(discretized_time[n])
            x = self.consistency_function(x_n, t, conditions=conditions)
        return x

    def consistency_function(self, x, t, conditions=None, **kwargs):
        f = self.network(t, x, conditions)
        skip = self.sigma2 / ((t - self.epsilon) ** 2 + self.sigma2)
        out = torch.sqrt(self.sigma2) * (t - self.epsilon) / (torch.sqrt(self.sigma2 + t**2))

        return skip.unsqueeze(1) * x + out.unsqueeze(1) * f

    def loss(self, theta, context_data):
        self.current_step += 1
        with torch.enable_grad():
            current_num_steps = self._schedule_discretization()
            discretized_time = self._discretize_time(current_num_steps)
            p_mean = -1.1
            p_std = 2.0

            logits = torch.log(torch.erf((torch.log(discretized_time[1:]) - p_mean) / (math.sqrt(2.0) * p_std)) - torch.erf((torch.log(discretized_time[:-1]) - p_mean) / (math.sqrt(2.0) * p_std)))
            times = torch.distributions.Categorical(logits=logits).sample([theta.size(0)])
            t1 = discretized_time[times]
            t2 = discretized_time[times + 1]
            noise = torch.randn_like(theta, device=theta.device)

            teacher_out = self._forward_train(theta, noise, t1, conditions=context_data)
            teacher_out = teacher_out.detach()
            student_out = self._forward_train(theta, noise, t2, conditions=context_data)

            lam = 1 / (t2 - t1)
            loss = torch.mean(lam.unsqueeze(1) * (torch.sqrt((teacher_out - student_out)**2 + self.c_huber2) - self.c_huber))

        return loss

    def sample_batch(self, *context_data, batch_size: int = None, steps: int = 10):
        """
        Returns num_sample conditional samples for a batch of contexts by solving an ODE
        forwards in time.

        Parameters
        ----------
        *context_data: list[torch.Tensor]
            context data (e.g., gravitational-wave data)
        batch_size: int = None
            batch_size for sampling. If len(context_data) > 0, we automatically set
            batch_size = len(context_data[0]), so this option is only used for
            unconditional sampling.
        steps: int = 10
            Number of ODE steps to take during sampling.

        Returns
        -------
        torch.tensor
            the generated samples
        """
        self.network.eval()

        if len(context_data) == 0 and batch_size is None:
            raise ValueError("For unconditional sampling, the batch size needs to be set.")
        elif len(context_data) > 0:
            if batch_size is not None:
                raise ValueError("For conditional sampling, the batch_size cannot be set manually as it is automatically determined by the context_data.")
            batch_size = len(context_data[0])

        with torch.no_grad():
            latent_samples = torch.randn((batch_size, self.theta_dim), device=self.s0.device)
            samples = self._inverse(latent_samples, conditions=context_data[0] if len(context_data) > 0 else None, steps=steps)

        self.network.train()
        return samples

    def sample_and_log_prob_batch(self, *context_data):
        raise NotImplementedError

    def log_prob_batch(self, data, *context_data):
        raise NotImplementedError