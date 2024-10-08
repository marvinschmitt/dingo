import torch
from torch import nn
from .base_model import Base

class ConsistencyModel(Base):
    """
    Class for consistency model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

    def initialize_network(self):
        pass

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
        pass

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
    
