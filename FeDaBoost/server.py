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
import random
import os
import logging
import copy
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from clients import Client
from aggregators import  fedProx, weighted_avg, fedaboost_avg

from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset

from sklearn.metrics import f1_score

from torch.utils.tensorboard import SummaryWriter

class Server:
    """
    The federated learning server class. 
    
    Parameters:
    ----------------
    rounds: int;
        Number of global rounds
    stratergy: callable;
        Averaging stratergy for federated learning.
    checkpt_path: str;
        Path to save checkpoints of the global model.
    log_dir: str;
        TensorBoard log directory.
    
    Methods:
    ----------------
    init_model(self, model: torch.nn.Module) -> None:
        Initialize the model for federated learning.
    connect_client(self, client: Client) -> None:
        Add a client for federated learning setup.
    sample_clients(self, num_clients: int) -> dict:
        Sample clients from the client dictionary.
    _aggregate(self, trained_clients, weights = None) -> None:
        Aggregate the models of the clients.
    _broadcast(self, model: torch.nn.Module, clients:list = None) -> None:
        Broadcast the model to the clients.
    _receive(self, client:callable) -> list:
        Receive the models from the clients.
    train(self, train_samples:dict, max_local_round:int, threshold:float, patience:int) -> torch.nn.Module:
        Train the model using federated learning.
    __check_convergence(self, global_loss: float, prev_global_loss: float) -> bool:
        Check if the training has converged.
    _collect_stats(self, local_loss: list, weights: list) -> dict:
        Collect training statistics.        
    """

    def __init__(self,rounds:int, stratergy:callable, checkpt_path:str=None, log_dir:str = 'runs') -> None:
        self.rounds = rounds
        self.stratergy = stratergy
        self.client_dict = {}
        self.checkpoint_path = checkpt_path
        print(f"Checkpoint Path: {self.checkpoint_path}")

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)


    def init_model(self, model: torch.nn.Module) -> None:
        """
        Initialize the model for federated learning.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            The global model.
        """
        self.global_model = model
        self.global_model.train()
        self.global_model.to("mps:0")


    def connect_client(self, client: Client) -> None:
        """
        Add a client for federated learning setup.

        Parameters:
        ----------------
        client_id: str;
            Client id
        """
        
        client_id = client.client_id
        self.client_dict[client_id] = client


    def sample_clients(self, num_clients: int) -> dict:
        """
        Sample clients from the client dictionary.

        Parameters:
        ----------------
        num_clients: int;
            Number of clients to sample

        Returns:
        ----------------
        sampled_clients: dict
            Dict of sampled clients
        """
        sampled_client_ids = np.random.choice(list(self.client_dict.keys()), num_clients, replace=False)
        sampled_clients = {client_id: self.client_dict[client_id] for client_id in sampled_client_ids}
  
        return sampled_clients


    def _aggregate(self, trained_clients, weights = None) -> None:
        """
        Aggregate the models of the clients.     

        Parameters:
        ----------------
        weights: list or None
            List of weights for the weighted average strategy. If None, standard strategies are used.

        Returns:
        ----------------
        model: torch.nn.Module object
            Aggregated model.
        updated: bool
            Indicates whether the global model was updated.
        """

        prev_params = [p.clone() for p in self.global_model.parameters()]
        client_models = [client.get_model() for client in trained_clients.values()]

        aggregation_functions = {
        "fedavg": weighted_avg,
        "fedprox": fedProx,
        "fedaboost-optima": weighted_avg,
        "ditto": weighted_avg,
        }

        aggregation_function = aggregation_functions.get(self.stratergy)

        if aggregation_function is None:
            raise ValueError(f"Unsupported aggregation strategy: {self.stratergy}")
        
        self.global_model = aggregation_function(self.global_model, client_models, weights)
        updated_params = [p.clone() for p in self.global_model.parameters()]
        updated = self._check_model_update(prev_params, updated_params)

        return self.global_model, updated


    def _broadcast(self, model: torch.nn.Module, clients:list = None) -> None:
        """
        Broadcast the model to the clients.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            Model to be broadcasted
        """
        model_state_dict = model.state_dict()

        if clients is None:
            clients = list(self.client_dict.keys())

        for client_id, client in self.client_dict.items():
            if client_id in clients:
                client.set_model(copy.deepcopy(model_state_dict))
                self.client_dict[client_id] = client
                print(f"Broadcasted model to client {client.client_id}")


    def _receive(self, client:callable) -> list:
        """
        Receive the models from the clients.

        Returns:
        ----------------
        models: list;
            List of models
        """
        print(f"Received model from client {client.client_id}")
        self.client_dict[client.client_id] = client


    def train(self, train_samples:dict, max_local_round:int, threshold:float, patience:int) -> torch.nn.Module:
        """
        Train the model using federated learning.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            Model to be trained

        Returns:
        ----------------
        model: torch.nn.Module object;
            Trained model
        """
        consecutive_no_update_rounds = 0
        consecutive_loss_change_rounds = 0

        for round in range(1,self.rounds+1):
            update_status = False
            print(f"\n | Global Training Round : {round} |\n")
            logging.info(f"\n | Global Training Round : {round} |\n")
            num_data_points = {}

            train_clients_list= train_samples[str(round)]
            train_clients = {client: self.client_dict[client] for client in train_clients_list}
            logging.info(f"Selected Clients: {train_clients.keys()}")
            self._broadcast(self.global_model, train_clients.keys())

            for client in train_clients.values():
                _ = client.train(round,max_local_round,threshold, patience)
                num_data_points[client.client_id] = client.get_num_datapoints()
                self._receive(client)

            total_data_points = sum(num_data_points[client.client_id] for client in train_clients.values())
            weights = [num_data_points[client.client_id] / total_data_points for client in train_clients.values()]
            
            logging.info(f"Weights used for aggregation: {weights}")
            self.global_model, update_status = self._aggregate(train_clients, weights=weights)
            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")
            print(f"Model Updated: {update_status}")

            if consecutive_loss_change_rounds == 5:
                print("The global model parameters have not been updated for 5 consecutive rounds, so the training has converged.")
                break
            
            if not update_status:
                consecutive_no_update_rounds += 1
                print("The global model parameters have not been updated, so the training has converged.")
            else:
                consecutive_no_update_rounds = 0

            if consecutive_no_update_rounds == 5:
                print("The global model parameters have not been updated for 5 consecutive rounds, so the training has converged.")
                break

        return self.global_model
    
    def __check_convergence(self, global_loss: float, prev_global_loss: float) -> bool:
        """
        Check if the training has converged.

        Parameters:
        ----------------
        global_loss: float
            Current global loss
        prev_global_loss: float
            Previous global loss

        Returns:
        ----------------
        bool
            True if converged, False otherwise
        """
        if abs(global_loss - prev_global_loss) < 0.0001:
            self.consecutive_loss_change_rounds += 1
            if self.consecutive_loss_change_rounds == 3:
                print("The global model parameters have not changed significantly for 3 consecutive rounds, training has converged.")
                return True
        else:
            self.consecutive_loss_change_rounds = 0
        return False
    
    def _collect_stats(self, local_loss: list, weights: list) -> dict:
        """
        Collect training statistics.

        Parameters:
        ----------------
        local_loss: list
            List of client losses
        weights: list
            List of weights for clients

        Returns:
        ----------------
        dict
            Dictionary containing training statistics
        """
        stats = {f"{client.client_id}-loss": local_loss[i] for i, client in enumerate(self.client_dict.values())}
        stats.update({f"{client.client_id}-weight": weights[i] for i, client in enumerate(self.client_dict.values())})
        return stats
    
    def _save_stats(self, stats: list, path: str) -> None:
        """
        Save training statistics to a CSV file.

        Parameters:
        ----------------
        stats: list
            List of training statistics
        path: str
            Path to save the CSV file

        Returns:
        ----------------
        None
        """
        stats_df = pd.DataFrame(stats)
        directory = os.path.dirname(path)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        stats_df.to_csv(path, index=False)

    def _check_model_update(self, prev_params: list, updated_params: list) -> bool:
        """
        Check if the model parameters have been updated.

        Parameters:
        ----------------
        prev_params: list
            List of previous model parameters
        updated_params: list
            List of updated model parameters

        Returns:
        ----------------
        bool
            True if updated, False otherwise
        """
        for prev_param, updated_param in zip(prev_params, updated_params):
            if not torch.equal(prev_param, updated_param):
                return True
        return False
    
    def save_checkpt(self, checkpoint: torch.nn.Module, ckptpath: str) -> None:
        """
        Saving the checkpoints.

        Parameters:
        ----------------
        checkpoint:
            Model at a specific checkpoint.
        ckptpath: str;
            Path to save the checkpoint. Default is None.
        """
        if os.path.exists(ckptpath):
            torch.save(
                checkpoint.state_dict(),
                ckptpath,
            )
        else:
            os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
            torch.save(
                checkpoint.state_dict(),
                ckptpath,
            )

class BoostingServer(Server):

    """
    The federated learning server class for fedaboost.
    Parameters:
    ----------------
    rounds: int;
        Number of global rounds
    stratergy: callable;
        Aggregation stratergy for federated learning.
    checkpt_path: str;
        Path to save checkpoints of the global model.
    log_dir: str;
        TensorBoard log directory.
    max_local_round: int;
        Maximum number of local rounds for each client.

    Methods:
    ----------------
    __aggregate(self, weights = []) -> None:
        Aggregate the models of the clients.
    train(self) -> None:
        Train the model using federated fedaboost-optima algorithm.
    
    """

    def __init__(self, rounds: int, strategy: callable, checkpt_path:str=None,  log_dir:str = 'runs', max_local_round = 10) -> None:
        super().__init__(rounds, strategy, checkpt_path, log_dir)
        self.max_local_round = max_local_round
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        logging.info("Boosting Server Initialized")


    def __aggregate(self,trained_clients, weights = None) -> None:
        """
        Aggregate the models of the clients.

        Parameters:
        ----------------
        weights: list or None
            List of weights for the weighted average strategy. 

        Returns:
        ----------------
        model: torch.nn.Module object
            Aggregated model.
        updated: bool
            Indicates whether the global model was updated.
        """
    
        prev_params = [p.clone() for p in self.global_model.parameters()]
        self.global_model = fedaboost_avg(self.global_model, trained_clients, weights)

        updated_params = [p.clone() for p in self.global_model.parameters()]
        updated = self._check_model_update(prev_params, updated_params)

        return self.global_model, updated


    def train(self, train_samples:dict, max_local_round:int, threshold:float, patience:int) -> torch.nn.Module:
        """
        Train the model using federated learning.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            Model to be trained

        Returns:
        ----------------
        model: torch.nn.Module object;
            Trained model
        """
        consecutive_no_update_rounds = 0
        weights_dict = {client: (1 / len(train_samples[str(1)])) for client in self.client_dict}

        for client in self.client_dict.values():
            client.set_weight(1/10) #/len(train_samples[str(1)])

        logging.info(f"Initial Weights: {weights_dict}")

        gamma_history = {}

        for round in range(1,self.rounds+1):
            update_status = False

            print(f"\n | Global Training Round : {round} |\n")
            logging.info(f"\n | Global Training Round : {round} |\n")

            alphas = {}
            train_clients_list= train_samples[str(round)]
            k = len(train_clients_list)
            train_clients = {client: self.client_dict[client] for client in train_clients_list}
            self._broadcast(self.global_model, train_clients.keys())

            for client in train_clients.values():
                _, alpha, gamma = client.train(round, max_local_round, k, threshold, patience)
                alphas[client.client_id] = alpha
                gamma_history[client.client_id] = gamma
                self._receive(client)
            logging.info(f"Alpha values used for aggregation: {alphas}")


            # align the client order
            client_ids = list(train_clients.keys())
            alpha_list = [alphas[cid] for cid in client_ids]  # consistent ordering
            #alpha_values = self.scaler.fit_transform(np.array(alpha_list).reshape(-1, 1)).flatten()

            # Normalize alpha values to sum to 1, comment this later
            alpha_tensor = torch.tensor(alpha_list, dtype=torch.float32)
            alpha_values = torch.softmax(alpha_tensor, dim=0).numpy()  # weights sum to 1

            client_models = [train_clients[cid].get_model() for cid in client_ids]
            self.global_model, update_status = self.__aggregate(client_models, alpha_values)
        
            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

            print(f"Model Updated: {update_status}")
            logging.info(f"Model Updated Successfully using Fedaboost weighted averaging: {update_status}")
            
            if not update_status:
                consecutive_no_update_rounds += 1
                print("The global model parameters have not been updated, so the training has converged.")
                logging.info("The global model parameters have not been updated, so the training has converged. Might be some issue with the model aggregation.")
            else:
                consecutive_no_update_rounds = 0

            if consecutive_no_update_rounds == 3:
                print("The global model parameters have not been updated for 5 consecutive rounds, so the training has converged.")
                logging.info("The global model parameters have not been updated for 3 consecutive rounds. Hence, stopping the training.")
                break
        
        #pd.DataFrame(gamma_history).to_csv(f"{self.checkpoint_path}/gamma_history.csv", index=False)
        with open(f"{self.checkpoint_path}/clients_gamma.json", 'w') as f:
            json.dump(gamma_history, f, indent=4)

        return self.global_model

class DittoServer(Server):
    """
    DittoServer class extends the base Server for Ditto Personalized FL.
    In Ditto, each client trains both a global model copy (for aggregation)
    and a personal model. The server still only manages the global model,
    using standard aggregation. The personal models remain on each client.

    Parameters (inherited from Server):
    -----------------------------------
    rounds: int
        Number of global training rounds.
    stratergy: callable
        Aggregation strategy for federated learning.
    checkpt_path: str, optional
        Path to save checkpoints of the global model.
    log_dir: str
        TensorBoard log directory.
    """

    def __init__(self, rounds: int, stratergy: callable, checkpt_path: str = None, log_dir: str = 'runs'):
        super().__init__(rounds, stratergy, checkpt_path, log_dir)
  
    def train(self, train_samples: dict, max_local_round: int, threshold: float, patience: int) -> torch.nn.Module:
        """
        Overridden train method. The main difference for Ditto is that clients 
        (DittoClient) will internally train both the global model copy and their 
        personal model. The server side remains mostly the same as standard FL.
        """
        self.global_model = super().train(train_samples, max_local_round, threshold, patience)
        return self.global_model
