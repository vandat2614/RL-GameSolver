import torch
import torch.nn as nn
from typing import List, Type
import gymnasium
import yaml

ACTIVATION_MAP = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU,
    'Sigmoid': nn.Sigmoid,
    'Softmax': lambda: nn.Softmax(dim=-1),
    None : nn.Identity
}

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],  
        output_size: int,
        hidden_activations: List[Type[nn.Module]],  # Activations for hidden layers
        output_activation: Type[nn.Module],         # Activation for output layer
    ):
        super(NeuralNetwork, self).__init__()
        
        if len(hidden_sizes) != len(hidden_activations):
            raise ValueError("Number of hidden activations must match number of hidden layers")

        layers = []
        for i in range(len(hidden_sizes)):
            in_features = input_size if i == 0 else hidden_sizes[i - 1]
            out_features = hidden_sizes[i]
            activation = hidden_activations[i]
            layers.extend([
                nn.Linear(in_features, out_features, bias=True),
                activation()
            ])

        layers.extend([
            nn.Linear(hidden_sizes[-1], output_size, bias=True),
            nn.Identity() if output_activation is None else output_activation()
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    @classmethod
    def from_config(cls, config: dict, input_size : int, output_size : 12):
        model_config = config['model']
        
        hidden_activations = [ACTIVATION_MAP[act] for act in model_config['hidden_activations']]
        output_activation = ACTIVATION_MAP[model_config['output_activation']] 
        
        return cls(
            input_size=input_size,
            hidden_sizes=model_config['hidden_sizes'],
            output_size=output_size,
            hidden_activations=hidden_activations,
            output_activation=output_activation
        )
