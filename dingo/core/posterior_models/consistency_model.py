import torch
from torch import nn
from .base_model import Base
from dingo.core.nn.cfnets import create_cf_model

class ConsistencyModel(Base):
    """
    Class for consistency model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theta_dim = self.metadata["train_settings"]["model"]["posterior_kwargs"]["input_dim"]
        self.s0 = self.model_kwargs["posterior_kwargs"]["consistency_args"]["s0"]
        self.s1 = self.model_kwargs["posterior_kwargs"]["consistency_args"]["s1"]
        self.tmax = self.model_kwargs["posterior_kwargs"]["consistency_args"]["tmax"]
        self.epsilon = self.model_kwargs["posterior_kwargs"]["consistency_args"]["epsilon"]
        self.sigma2 = self.model_kwargs["posterior_kwargs"]["consistency_args"]["sigma2"]

        print("init successful!")

    def initialize_network(self):
        model_kwargs = {k: v for k, v in self.model_kwargs.items() if k != "type"}
        if self.initial_weights is not None:
            model_kwargs["initial_weights"] = self.initial_weights
        self.network = create_cf_model(**model_kwargs)

    def loss(self, data, context):
        """
        Compute the loss for a batch of data.

        Parameters
        ----------
        data: Tensor
        context: Tensor

        Returns
        -------
        loss: Tensor
            loss for the batch
        """
        
        return torch.tensor(0.0, requires_grad=True)

    def sample_batch(self, *context_data):
        """
        Sample a batch of data from the posterior model.

        Parameters
        ----------
        context: Tensor
        Returns
        -------
        batch: dict
            dictionary with batch data
        """
        pass

    def sample_and_log_prob_batch(self, *context_data):
        raise NotImplementedError

    def log_prob_batch(self, data, *context_data):
        raise NotImplementedError
    
