import torch
import torch.nn as nn
from typing import List, Tuple, Type, Optional
import gymnasium

ACTIVATION_MAP = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU,
    'Sigmoid': nn.Sigmoid,
    'Softmax': lambda: nn.Softmax(dim=-1),
    None : nn.Identity
}

class CNN(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int, int],  # (height, width, channels)
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        embed_dims: List[int],
        conv_activations: List[Type[nn.Module]],
        poolings: List[Optional[nn.Module]]  # None means no pooling
    ):
        super(CNN, self).__init__()
        
        layers = []
        for i in range(len(kernel_sizes)):
            in_channels = input_size[2] if i == 0 else embed_dims[i - 1]
            out_channels = embed_dims[i]
            
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                bias=True
            )
            
            activation = conv_activations[i]()
            pooling = nn.Identity() if poolings[i] is None else poolings[i]()
            layers.extend([conv, activation, pooling])
        
        self.features = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ConvNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: List[Tuple[int, int, int]],  
        kernel_sizes: List[List[int]],           
        strides: List[List[int]],                 
        paddings: List[List[int]],                
        embed_dims: List[List[int]],              
        conv_activations: List[List[Type[nn.Module]]], 
        poolings: List[List[Optional[nn.Module]]], 
        hidden_sizes: List[int],                  
        hidden_activations: List[Type[nn.Module]], 
        output_size: int,                         
        output_activation: Optional[Type[nn.Module]] = None  
    ):
        super(ConvNeuralNetwork, self).__init__()
        
        
        self.feature_extractors = nn.ModuleList()
        for i in range(len(input_sizes)):
            conv_net = CNN(
                input_size=input_sizes[i],
                kernel_sizes=kernel_sizes[i],
                strides=strides[i],
                paddings=paddings[i],
                embed_dims=embed_dims[i],
                conv_activations=conv_activations[i],
                poolings=poolings[i]
            )
            self.feature_extractors.append(conv_net)
        
        feature_dim = self._caculate_feature_dim(input_sizes)
        
        fc_layers = []
        fc_layers.append(nn.Flatten())
        
        for i in range(len(hidden_sizes)):
            in_features = feature_dim if i == 0 else hidden_sizes[i - 1]
            out_features = hidden_sizes[i]

            fc_layers.extend([
                nn.Linear(in_features=in_features,
                          out_features=out_features,
                          bias=True),
                hidden_activations[i]()
            ])

        fc_layers.extend([
            nn.Linear(hidden_sizes[-1], output_size),
            nn.Identity() if output_activation is None else output_activation()
            ])
        
        self.classifier = nn.Sequential(*fc_layers)
    
    def _caculate_feature_dim(self, input_sizes: List[Tuple[int, int, int]]) -> int:
        feature_dim = 0
        for i, input_size in enumerate(input_sizes):
            dummy_input = torch.zeros(1, *input_size).permute(0, 3, 1, 2)
            with torch.no_grad():
                dummy_features = self.feature_extractors[i](dummy_input)
                feature_dim += dummy_features.numel()
        return feature_dim
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:

        features = []
        for i, input_tensor in enumerate(inputs):
            feature = self.feature_extractors[i](input_tensor)
            feature = feature.view(feature.size(0), -1)
            features.append(feature)
        
        features = torch.cat(features, dim=1)
        output = self.classifier(features)
        
        return output

    @classmethod
    def from_config(cls, config: dict, input_sizes: Optional[List[Tuple[int, int, int]]], output_size : int):
        model_config = config['model']
        kernel_sizes = model_config['kernel_sizes']             
        strides = model_config['strides']                        
        paddings = model_config['paddings']                      
        embed_dims = model_config['embed_dims']                  
        
        conv_activations = [
            [ACTIVATION_MAP[act] for act in branch] 
            for branch in model_config['conv_activations']
        ]
        
        poolings = [
            [POOLING_MAP[pool] for pool in branch]
            for branch in model_config['poolings']
        ]
        
        hidden_sizes = model_config['hidden_sizes']
        hidden_activations = [ACTIVATION_MAP[act] for act in model_config['hidden_activations']]
        output_activation = model_config['output_activation']
        
        return cls(
            input_sizes=input_sizes,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            embed_dims=embed_dims,
            conv_activations=conv_activations,
            poolings=poolings,
            hidden_sizes=hidden_sizes,
            hidden_activations=hidden_activations,
            output_size=output_size,
            output_activation=output_activation
        )
    

ACTIVATION_MAP = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "Softmax": nn.Softmax(dim=-1),  
    None: nn.Identity,
}

POOLING_MAP = {
    "MaxPool2d": lambda: nn.MaxPool2d(kernel_size=2, stride=2),
    "AvgPool2d": lambda: nn.AvgPool2d(kernel_size=2, stride=2),
    "AdaptiveAvgPool2d": lambda: nn.AdaptiveAvgPool2d((1, 1)),
    "AdaptiveMaxPool2d": lambda: nn.AdaptiveMaxPool2d((1, 1)),
    None: nn.Identity,
}

