import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
sys.path.append("../..")
from dingo.core.posterior_models.consistency_model import ConsistencyModel

# Define the TwoMoonsSimulator class as described in the user's requirements
def twomoons_simulator(batch_size):
    r = np.random.normal(0.1, 0.01, size=(batch_size, 1))
    alpha = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, size=(batch_size, 1))
    theta = np.random.uniform(-1.0, 1.0, size=(batch_size, 2))

    c1 = -np.abs(theta[..., :1] + theta[..., 1:]) / np.sqrt(2.0) + r * np.cos(alpha) + 0.25
    c2 = (-theta[..., :1] + theta[..., 1:]) / np.sqrt(2.0) + r * np.sin(alpha)

    conditions = np.concatenate([c1, c2], axis=-1)

    return dict(theta=torch.tensor(theta, dtype=torch.float32), conditions=torch.tensor(conditions, dtype=torch.float32))

# Instantiate the ConsistencyModel
train_settings = {
    'data': {
        'inference_parameters': ['theta'],
        'total_budget': 20000,
        'train_fraction': 0.95
    },
    'model': {
        'posterior_kwargs': {
            'type': 'DenseResidualNet',
            'input_dim': 2,
            'context_dim': 2,
            'activation': 'gelu',
            'batch_norm': False,
            'context_with_glu': False,
            'dropout': 0.0,
            'hidden_dims': [
                256, 256, 256, 256, 256, 256,
            ],
            'sigma_min': 0.0001,
            'theta_with_glu': False,
            'time_prior_exponent': 1,
            'consistency_args': {
                's0': 10,
                's1': 50,
                'tmax': 200,
                'epsilon': 0.001,
                'sigma2': 1.0
            }
        }
    },
    'training': {
        'stage_0': {
            'epochs': 30,
            'batch_size': 64
        }
    },
    'local': {
        'device': 'cpu',
        'num_workers': 0,
    }
}
batch_size = train_settings['training']['stage_0']['batch_size']
num_training_batches = 512

total_steps = train_settings['training']['stage_0']['epochs'] * num_training_batches

train_data = twomoons_simulator(batch_size * num_training_batches)
train_theta, train_conditions = train_data['theta'], train_data['conditions']

# Prepare DataLoader
dataset = TensorDataset(train_theta, train_conditions)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = ConsistencyModel(metadata={'train_settings': train_settings}, initial_weights=None, device='cpu')
model.initialize_network()

# Define optimizer with Cosine Decay learning rate
initial_learning_rate = 5e-4
optimizer = optim.Adam(model.network.parameters(), lr=initial_learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

# Training loop
for epoch in range(train_settings['training']['stage_0']['epochs']):  # Number of epochs
    for batch_idx, (theta_batch, conditions_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        loss = model.loss(theta_batch, conditions_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate based on the scheduler
        print(f'{batch_idx}', end='\r')

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete!")

test_conditions = torch.zeros(1, 2)

num_samples = 10000
samples = torch.zeros(num_samples, 2)
for i in range(num_samples):
    samples[i] = model.sample_batch(test_conditions)

print(samples)

import matplotlib.pyplot as plt
plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.2)
plt.show()