"""
Copyright (C) [2023] [Tharuka Kasthuriarachchige]

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Paper: [FeDABoost: AdaBoost Enhanced Federated Learning]
Published in: 
"""

import torch
from typing import List
import torch.nn.functional as F

def fedAvg(global_model: torch.nn.Module, local_models: List[torch.nn.Module]) -> torch.nn.Module:
    """
    Federated averaging algorithm. Returns the updated global model.

    Parameters:
    ------------
    global_model: torch.nn.Module object; Global model.
    local_models: list; List of local models.

    Returns:
    ------------
    global_model: torch.nn.Module object; Updated global model.
    """
    # update global model parameters here
    state_dicts = [model.state_dict() for model in local_models]
    for key in global_model.track_layers.keys():
        global_model.track_layers[key].weight.data = torch.stack(
            [item[str(key) + ".weight"] for item in state_dicts]
        ).mean(dim=0)
        global_model.track_layers[key].bias.data = torch.stack(
            [item[str(key) + ".bias"] for item in state_dicts]
        ).mean(dim=0)
        # info here - https://discuss.pytorch.org/t/how-to-change-weights-and-bias-nn-module-layers/93065/2
    return global_model

def fedProx(global_model: torch.nn.Module, local_models: List[torch.nn.Module], mu: float) -> torch.nn.Module:
    """
    Federated Proximal algorithm. Returns the updated global model.

    Parameters:
    ------------
    global_model: torch.nn.Module object; global model.
    local_models: list; List of local models.
    mu: float; Proximal term coefficient.

    Returns:
    ------------
    global_model: torch.nn.Module object
        Updated global model.
    """
    state_dicts = [model.state_dict() for model in local_models]
    global_state_dict = global_model.state_dict()

    for key in global_state_dict.keys():
        
        stacked_tensors = torch.stack([state_dict[key] for state_dict in state_dicts], dim=0) # Stack the parameters from each local model
        averaged_tensor = torch.mean(stacked_tensors, dim=0) # Compute the mean of the stacked parameters
        
        global_state_dict[key] = averaged_tensor

    global_model.load_state_dict(global_state_dict)
    
    # Apply the proximal term adjustment
    for key in global_state_dict.keys():
        global_state_dict[key] -= mu * (global_state_dict[key] - global_model.state_dict()[key])
    
    global_model.load_state_dict(global_state_dict)
    
    return global_model


def weighted_avg(global_model: torch.nn.Module, local_models: List[torch.nn.Module], weights: List[float]) -> torch.nn.Module:
    """
    Average model parameters using weighted averaging.
    
    Parameters:
    ------------
    global_model: torch.nn.Module object
        Global model.
    local_models: list
        List of local models.
    weights: list
        List of weights for each local model.
    
    Returns:
    ------------
    global_model: torch.nn.Module object
        Updated global model.
    """

    state_dicts = [model.state_dict() for model in local_models]

    with torch.no_grad():  
        for key in global_model.state_dict().keys():
            stacked_params = torch.stack(
                [state_dict[key] * weights[i] for i, state_dict in enumerate(state_dicts)], dim=0
            )
            global_model.state_dict()[key].copy_(stacked_params.sum(dim=0))  

    return global_model


def boosted_avg(global_model: torch.nn.Module, local_models: List[torch.nn.Module], weights: List[float]) -> torch.nn.Module:
    """
    Average model parameters using weighted averaging, ignoring clients with negative weights.
    
    Parameters:
    ------------
    global_model: torch.nn.Module object
        Global model.
    local_models: list
        List of local models.
    weights: list
        List of weights for each local model.
    
    Returns:
    ------------
    global_model: torch.nn.Module object
        Updated global model.
    """
    
    valid_indices = [i for i, w in enumerate(weights) if w > 0]
    if not valid_indices:
        raise ValueError("No clients with positive weights available for aggregation.")
    
    valid_models = [local_models[i] for i in valid_indices]
    valid_weights_tensor = torch.tensor([weights[i] for i in valid_indices])
    
    # Normalize weights using softmax so that they sum to 1
    valid_weights = F.softmax(valid_weights_tensor, dim=0).tolist()
    
    state_dicts = [model.state_dict() for model in valid_models]

    with torch.no_grad():
        for key in global_model.state_dict().keys():
            # Aggregate parameters from each model using the normalized weights
            stacked_params = torch.stack(
                [state_dict[key] * valid_weights[i] for i, state_dict in enumerate(state_dicts)], dim=0
            )
            global_model.state_dict()[key].copy_(stacked_params.sum(dim=0))
    
    return global_model

""" 
from copy import deepcopy

def weighted_avg(global_model: torch.nn.Module, local_models: List[torch.nn.Module], weights: List[float])-> torch.nn.Module:

    weights = [weight / sum(weights) for weight in weights]
    
    new_model=deepcopy(global_model)
    set_to_zero_model_weights(new_model)

    clients_params = []
    for local_model in local_models:
        list_params=list(local_model.parameters())
        list_params=[tens_param.detach() for tens_param in list_params]
        clients_params.append(list_params)  

    for k,client_hist in enumerate(clients_params):
        
        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution=client_hist[idx].data*weights[k]
            layer_weights.data.add_(contribution)
            
    return new_model

def set_to_zero_model_weights(model):

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data) 
        
"""