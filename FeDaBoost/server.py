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

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from clients import Client
from aggregators import fedAvg, fedProx, weighted_avg
from utils import get_alpha, get_weights, influence_alpha, adjust_local_epochs, get_device, z_scores

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
    
    Methods:
    ----------------
    init_model(self, model: torch.nn.Module) -> None:
        Initialize the model for federated learning.
    connect_client(self, client: Client) -> None:
        Add a client for federated learning setup.
    __aggregate(self, weights = []) -> None:
        Aggregate the models of the clients.
    _broadcast(self, model: torch.nn.Module) -> None:
        Broadcast the model to the clients.
    _receive(self, client:callable) -> list:
        Receive the models from the clients.
    train(self) -> None:
        Train the model using federated learning.
    _collect_stats(self, local_loss: list, weights: list) -> dict:
        Collect training statistics.
    _save_stats(self, stats: list, path: str) -> None:
        Save training statistics to a CSV file.
    _check_model_update(self, prev_params: list, updated_params: list) -> bool:
        Check if the model parameters have been updated.
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
            Model to be trained

        Returns:
        ----------------
        None
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

        Returns:
        ----------------
        None
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
        list
            Dict of sampled clients
        """
        sampled_client_ids = np.random.choice(list(self.client_dict.keys()), num_clients, replace=False)
        sampled_clients = {client_id: self.client_dict[client_id] for client_id in sampled_client_ids}
  
        return sampled_clients


    def _aggregate(self,trained_clients, weights = None) -> None:
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
        "fedaboost-ranker": weighted_avg,
        "fedaboost-concord": weighted_avg,
        "ditto": weighted_avg,
        }

        aggregation_function = aggregation_functions.get(self.stratergy)

        if aggregation_function is None:
            raise ValueError(f"Unsupported aggregation strategy: {self.stratergy}")
        
        self.global_model = aggregation_function(self.global_model, client_models, weights)

        updated_params = [p.clone() for p in self.global_model.parameters()]

        updated = self._check_model_update(prev_params, updated_params)

        return self.global_model, updated


    def _broadcast(self, model: torch.nn.Module) -> None:
        """
        Broadcast the model to the clients.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            Model to be broadcasted
        """
        model_state_dict = model.state_dict()
        for client_id, client in self.client_dict.items():
            client.set_model(model_state_dict)
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

            self._broadcast(self.global_model)

            for client in train_clients.values():
                #client.set_model(self.global_model.state_dict())
                loss, f1 = client.evaluate()
                _ = client.train(round,max_local_round,threshold, patience)
                num_data_points[client.client_id] = client.get_num_datapoints()
                self._receive(client)

            total_data_points = sum(num_data_points[client.client_id] for client in train_clients.values())
            weights = [num_data_points[client.client_id] / total_data_points for client in train_clients.values()]
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

        Returns:
        ----------------
        None
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

class OptimaServer(Server):

    """
    The federated learning server class for fedaboost-optima.

    Parameters:
    ----------------
    rounds: int;
        Number of global rounds
        stratergy: callable;
        Averaging stratergy for federated learning.

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
        logging.info("FedaBoost-Optima Boosting Server Initialized")


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
        client_models = [client.get_model() for client in trained_clients.values()]
        self.global_model = weighted_avg(self.global_model, client_models, weights)

        updated_params = [p.clone() for p in self.global_model.parameters()]
        updated = self._check_model_update(prev_params, updated_params)
        return self.global_model, updated


    def train(self, train_samples:dict, max_local_round:int, threshold:float, patience:int, alpha_constant) -> torch.nn.Module:
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

        weights_dict = {client: (1 / len(self.client_dict)) for client in self.client_dict}

        for client in self.client_dict.values():
            client.set_weight(weights_dict[client.client_id])

        logging.info(f"Initial Weights: {weights_dict}")
        for round in range(1,self.rounds+1):
            update_status = False

            print(f"\n | Global Training Round : {round} |\n")
            logging.info(f"\n | Global Training Round : {round} |\n")

            alphas = {}
            train_clients_list= train_samples[str(round)]
            k = len(train_clients_list)
            train_clients = {client: self.client_dict[client] for client in train_clients_list}

            for client in train_clients.values():
                client.set_model(self.global_model.state_dict())

                _ = client.train(round, max_local_round,k, threshold, patience,)
                _, alpha = client.get_alpha(k)
                alphas[client.client_id] = alpha
                #logging.info(f"Client {client.client_id} Alpha: {alpha}")
                self._receive(client)


            logging.info(f"Alphas: {alphas}")
            final_alphas = influence_alpha(0.5, alphas, alpha_constant)
            logging.info(f"Final Alphas: {final_alphas}")
            alpha_values = self.scaler.fit_transform(np.array(list(alphas.values())).reshape(-1, 1))
            self.global_model, update_status = self.__aggregate(train_clients, alpha_values.flatten())

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

            print(f"Model Updated: {update_status}")
            logging.info(f"Model Updated Successfully using Fedaboost-Optima: {update_status}")

            self._broadcast(self.global_model)
            
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
        # Call the parent constructor
        super().__init__(rounds, stratergy, checkpt_path, log_dir)
        # Any additional Ditto-specific fields can be initialized here if needed
        # For example: self.personal_models = {}  # If you wanted the server to track them

    def train(self, train_samples: dict, max_local_round: int, threshold: float, patience: int) -> torch.nn.Module:
        """
        Overridden train method. The main difference for Ditto is that clients 
        (DittoClient) will internally train both the global model copy and their 
        personal model. The server side remains mostly the same as standard FL.

        """
        consecutive_no_update_rounds = 0

        for round_idx in range(1, self.rounds + 1):
            print(f"\n | Global Training Round : {round_idx} |\n")
            logging.info(f"\n | Global Training Round : {round_idx} |\n")

            train_clients_list = train_samples[str(round_idx)]
            train_clients = {cid: self.client_dict[cid] for cid in train_clients_list}
            logging.info(f"Selected Clients: {train_clients.keys()}")

            self._broadcast(self.global_model)

            # 3) Each selected client trains both:
            #    - The 'global model' copy (standard local FL updates)
            #    - Their personal model (Ditto approach)
            #    and returns the updated global model parameters
            num_data_points = {}
            for client in train_clients.values():
                # (a) Evaluate current global model (optional)
                pre_loss, pre_f1 = client.evaluate()
                
                # (b) DittoClient.train(...) -> (updated_global_model, updated_personal_model)
                updated_global, updated_personal = client.train(
                    global_round=round_idx,
                    max_local_round=max_local_round,
                    threshold=threshold,
                    patience=patience
                )

                personal_ckpt_path = f"{self.checkpoint_path}/personal_ckpts/ckpt_{round_idx}/{client.client_id}.pt"
                self.save_checkpt(updated_personal, personal_ckpt_path)
                
                # The updated global model is now in client.get_model() (or updated_global).
                # The personal model is in client.get_personal_model().
                
                # (c) Count local datapoints for weighting in the aggregator
                num_data_points[client.client_id] = client.get_num_datapoints()
                
                # (d) "Receive" - In your original code, you do this to update references
                self._receive(client)

            # 4) Aggregate the updated global models from all selected clients
            total_data_points = sum(num_data_points[cid] for cid in train_clients)
            weights = [num_data_points[cid] / total_data_points for cid in train_clients]
            self.global_model, update_status = self._aggregate(train_clients, weights=weights)

            # 5) Save a checkpoint
            ckpt_path = f"{self.checkpoint_path}/checkpoints/ckpt_{round_idx}.pt"
            self.save_checkpt(self.global_model, ckpt_path)
            print(f"Model Updated: {update_status}")

            # 6) Early stopping logic: If the global model was not updated, increment
            if not update_status:
                consecutive_no_update_rounds += 1
                print("The global model parameters have not been updated, so the training may be converging.")
            else:
                consecutive_no_update_rounds = 0

            if consecutive_no_update_rounds == 5:
                print("The global model parameters have not been updated for 5 consecutive rounds, stopping early.")
                break

        return self.global_model

    # If you want to evaluate personal models on the server side, you can define a helper:
    def evaluate_personal_models(self, client_ids=None):
        """
        Evaluate the personal models on a subset (or all) of clients.
        Typically, personal models stay local to each client in Ditto.

        Parameters:
        -----------
        client_ids: list or None
            If None, evaluate personal models for all connected clients.
            Otherwise, only evaluate for clients in the provided list.

        Returns:
        --------
        results: dict
            Mapping from client_id -> (loss, f1) for the personal model.
        """
        if client_ids is None:
            client_ids = list(self.client_dict.keys())

        results = {}
        for cid in client_ids:
            client = self.client_dict[cid]
            loss_avg, f1_avg = client.evaluate_personal_model()
            results[cid] = (loss_avg, f1_avg)
            print(f"Client {cid} [personal model] -> loss: {loss_avg:.4f}, f1: {f1_avg:.4f}")
        return results
