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
import os
import torch
import math
import logging
import copy
import numpy as np

from torch.utils.data import DataLoader
from utils import get_device
from datasets.mnist.preprocess import MNISTDataset
from datasets.femnist.preprocess import FEMNISTDataset
from sklearn.metrics import f1_score

class Client:
    """
    Client class for federated learning.

    Parameters:
    ------------
    client_id: str; client id
    train_dataset: torch.utils.data.Dataset object; training dataset
    test_dataset: torch.utils.data.Dataset object; validation dataset
    loss_fn: torch.nn.Module object; loss function
    batch_size: int; batch size
    learning_rate: float; learning rate
    weight_decay: float; weight decay
    local_model: torch.nn.Module object; model
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        loss_fn: torch.nn.Module,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        local_model: object = None,
    ) -> None:

        self.client_id: str = client_id
        self.loss_fn = loss_fn
        self.batch_size = batch_size

        self.traindl = DataLoader(
            train_dataset, batch_size, shuffle=True, drop_last=True
        )

        self.valdl = DataLoader(test_dataset, 8, shuffle=False, drop_last=True)
        self.optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.device = get_device()
        self.local_model = local_model
        self.local_model = local_model.to(self.device)
        self.train_dataset = train_dataset
        self.datapoints = len(train_dataset)

    def get_num_datapoints(self) -> int:
        """
        Get the number of samples in the training dataset.
        """
        return self.datapoints

    def set_model(self, model_weights) -> None:
        """
        Set the model for the client.

        Parameters:
        ------------
        model_weights: dict; state dictionary of model weights
        """
        self.local_model.load_state_dict(model_weights)

    def get_model(self) -> object:
        """
        Get the model of the client.

        Parameters:
        ------------
        None

        Returns:
        ------------
        model: torch.nn.Module object; model
        """
        return self.local_model

    def train(self, global_round:int, max_local_round:int, threshold:float, patience:int) -> tuple:
        """
        Training the model, using the fedaboost-optima strategy.

        Parameters:
        ------------
        model: torch.nn.Module object; model to be trained.
        loss_fn: torch.nn.Module object; loss function.

        Returns:
        ------------
        model: torch.nn.Module object; trained model.
        """
        previous_loss_avg = float('inf')  
        no_improvement_rounds = 0  

        print(f"Client: {self.client_id} \tTraining...")
        logging.info(f"Client: {self.client_id} \tTraining...")

        for epoch in range(max_local_round):
            print("\n")
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.local_model(x)

                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and isinstance(self.train_dataset, FEMNISTDataset):
                    y = y.view(-1)
                elif isinstance(self.train_dataset, MNISTDataset):
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1, 1)

                loss = self.loss_fn(outputs, y)
                self.local_model.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                batch_loss.append(loss.item())
    
            loss_avg = sum(batch_loss) / len(batch_loss)    
            print(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round}")
            logging.info(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round}")

            # Dynamic loss reduction evaluation.
            loss_reduction = previous_loss_avg - loss_avg
            if loss_reduction < threshold:
                no_improvement_rounds += 1
                print(f"Loss reduction below threshold ({loss_reduction:.6f}). No improvement rounds: {no_improvement_rounds}")
                logging.info(f"Loss reduction below threshold ({loss_reduction:.6f}). No improvement rounds: {no_improvement_rounds}")
            else:
                no_improvement_rounds = 0

            if no_improvement_rounds >= patience:
                print(f"Stopping early at local epoch {epoch + 1} due to no significant improvement.")
                logging.info(f"Stopping early at local epoch {epoch + 1} due to no significant improvement.")
                break

            previous_loss_avg = loss_avg

        return self.local_model

    def evaluate(self) -> tuple:
        """
        Evaluate the model with validation dataset.

        Parameters:
        ------------
        model: torch.nn.Module object; model to be evaluated
        loss_fn: torch.nn.Module object; loss function

        Returns:
        ------------
        loss_avg: float; average loss
        f1_avg: float; average F1 score
        """
        batch_loss = []
        all_preds = []
        all_labels = []

        for _, (x, y) in enumerate(self.valdl):
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.local_model(x)
            if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and isinstance(self.train_dataset, FEMNISTDataset):
                y = y.view(-1)
            elif isinstance(self.train_dataset, MNISTDataset):
                y = torch.argmax(y, dim=1)
            else:
                y = y.view(-1, 1)
            
            loss = self.loss_fn(outputs, y)
            batch_loss.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        loss_avg = sum(batch_loss) / len(batch_loss)
        f1_avg = f1_score(all_labels, all_preds, average='macro') 
        
        return loss_avg, f1_avg

class BoostingClient(Client):
    """
    Client class for federated learning.
    
    Parameters:
    ------------
    client_id: str; client id
    train_dataset: torch.utils.data.Dataset object; training dataset
    test_dataset: torch.utils.data.Dataset object; validation dataset
    loss_fn: torch.nn.Module object; loss function
    batch_size: int; batch size
    learning_rate: float; learning rate
    weight_decay: float; weight decay
    local_model: torch.nn.Module object; model
    local_round: int; number of local rounds
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        loss_fn: torch.nn.Module,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        local_model: object = None,
        num_classes: int = 10,
    ) -> None:
        
        super().__init__(
            client_id, 
            train_dataset, 
            test_dataset, 
            loss_fn, 
            batch_size, 
            learning_rate, 
            weight_decay, 
            local_model
        )

        self.eta = 0.1
        self.error_threshold = 0.3
        self.num_classes = num_classes
        self.loss_fn.gamma = 0

    def train(self, global_round, max_local_round,k, threshold=0.01, patience=2,) -> tuple:
        """
        Training the model, using the fedaboost-optima strategy.

        Parameters:
        ------------
        model: torch.nn.Module object; model to be trained.
        loss_fn: torch.nn.Module object; loss function.

        Returns:
        ------------
        model: torch.nn.Module object; trained model.
        """

        error_rate, alpha = self.get_alpha(k)
        self.weight = self.update_weight(alpha, performance_indicator = (error_rate > self.error_threshold))

        print(f"The client training is boosted by: {self.weight}")
        logging.info(f"The client training is boosted by: {self.weight}")

        if global_round == 1:
            pass
        else:
            if (error_rate > self.error_threshold):
                new_gamma = min((self.weight + self.loss_fn.gamma), 3)
                #self.loss_fn.update_gamma(new_gamma)

        logging.info(f"Client focal loss gamma: {self.loss_fn.gamma}")

        previous_loss_avg = float('inf')  
        no_improvement_rounds = 0  

        print(f"Client: {self.client_id} \tTraining...")
        logging.info(f"Client: {self.client_id} \tTraining...")

        

        for epoch in range(max_local_round):
            print("\n")
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.local_model(x)

                if isinstance(self.train_dataset, FEMNISTDataset):
                    y = y.view(-1)
                elif isinstance(self.train_dataset, MNISTDataset):
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1, 1)

                loss = self.loss_fn(outputs, y)
                self.local_model.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                batch_loss.append(loss.item())
    
            loss_avg = sum(batch_loss) / len(batch_loss)    
            print(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round} {self.loss_fn.gamma} {alpha}")
            logging.info(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round} {self.loss_fn.gamma} {alpha}")

            # Dynamic loss reduction evaluation.
            loss_reduction = previous_loss_avg - loss_avg

            if loss_reduction < threshold:
                no_improvement_rounds += 1
                print(f"Loss reduction below threshold ({loss_reduction:.6f}). No improvement rounds: {no_improvement_rounds}")
                logging.info(f"Loss reduction below threshold ({loss_reduction:.6f}). No improvement rounds: {no_improvement_rounds}")
            else:
                no_improvement_rounds = 0

            if no_improvement_rounds >= patience:
                print(f"Stopping early at local epoch {epoch + 1} due to no significant improvement.")
                logging.info(f"Stopping early at local epoch {epoch + 1} due to no significant improvement.")
                break

            previous_loss_avg = loss_avg            

        return self.local_model

    def __get_error_rate(self) -> float:
        """
        Evaluate the model on the validation dataset and return the error rate.

        Returns:
        ------------
        error_rate: float; error rate (proportion of incorrect predictions)
        """
        incorrect_preds = 0
        total_samples = 0

        for _, (x, y) in enumerate(self.traindl):
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.local_model(x)

            if isinstance(self.train_dataset, FEMNISTDataset):
                y = y.view(-1)
            elif isinstance(self.train_dataset, MNISTDataset):
                y = torch.argmax(y, dim=1)
            else:
                y = y.view(-1, 1)

            preds = torch.argmax(outputs, dim=1)

            incorrect_preds += (preds != y).sum().item()  
            total_samples += y.size(0)  

        # Error rate as the proportion of incorrect predictions
        error_rate = incorrect_preds / total_samples if total_samples > 0 else 0
        return error_rate
    
    def get_alpha(self,k):
        """
        Calculate adjusted weights for client in the FL setting, 
        giving higher weights to clients with lower errors with the global model.

        Parameters:
        - error: Error value for the client validation data with the global model.

        Returns:
        - Adjusted weight (alpha) for the client.
        """
        error_rate = self.__get_error_rate()
        logging.info(f"Client {self.client_id} Error rate: {error_rate}")

        if 0 < error_rate < 1:
            alpha = np.log((1 - error_rate) / (error_rate)) + np.log(k - 1)
        elif error_rate == 1:
            alpha = np.log((1 - (1-1e-6)) / (1-1e-6)) + np.log(k - 1)
        elif error_rate == 0:
            alpha = np.log((1 - 1e-6) / (1e-6)) + np.log(k - 1)
        else:
            raise ValueError("Error value must be in the range [0, 1].")

        return error_rate, alpha
    
    def set_weight(self, weight:float) -> None:
        """
        Set the weight for the client.

        Parameters:
        ------------
        weight: float; weight
        """
        self.weight = weight
        return self.weight

    def update_weight(self, alpha, performance_indicator=1) -> None:
        """
        Update the weights for the client.

        Parameters:
        ------------
        weight: float; weight
        """
        self.weight = self.weight * math.exp(float(self.eta) * -float(alpha) * int(performance_indicator))
        return self.weight



class DittoClient(Client):
    """
    DittoClient class extends the base Client for Ditto personalized FL.
    Maintains a separate `personal_model` and trains it with a
    regularization penalty towards the global (local_model) parameters.

    Parameters:
    ------------
    client_id: str
        Unique identifier of the client.
    train_dataset: torch.utils.data.Dataset
        The training dataset for this client.
    test_dataset: torch.utils.data.Dataset
        The test/validation dataset for this client.
    loss_fn: torch.nn.Module
        The loss function to use (e.g., CrossEntropyLoss).
    batch_size: int
        Batch size for local training.
    learning_rate: float
        Learning rate for local training of the global model.
    weight_decay: float
        Weight decay for local training of the global model.
    local_model: torch.nn.Module
        The initial global model (received from the server).
    personal_learning_rate: float
        Learning rate for training the personalized model.
    ditto_lambda: float
        The regularization strength used in Ditto to keep personal_model
        close to local_model.
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        loss_fn: torch.nn.Module,
        batch_size: int,
        learning_rate: float,                   # Global model learning rate
        weight_decay: float,
        local_model: object = None,           
        personal_learning_rate: float = 0.01,   # Personal model learning rate
        ditto_lambda: float = 0.1,
        personalized: bool = True,
        checkpt_path: str = None,                
    ) -> None:
        
        super().__init__(
            client_id,
            train_dataset,
            test_dataset,
            loss_fn,
            batch_size,
            learning_rate,
            weight_decay,
            local_model,
        )

        # Ditto-specific parameters
        self.ditto_lambda = ditto_lambda
        self.personalized = personalized
        self.checkpt_path = checkpt_path
        self.personal_lr = personal_learning_rate if self.personalized else learning_rate

        # Take Deep-copy the global/local model architecture as the personal model for Ditto
        self.personal_model = copy.deepcopy(self.local_model).to(self.device)
        
        # Personal optimizer (different LR is often used)
        self.personal_optimizer = torch.optim.SGD(
            self.personal_model.parameters(),
            lr=self.personal_lr,
            weight_decay=weight_decay,
        )

    def train(
        self,
        global_round: int,
        max_local_round: int,
        threshold: float,
        patience: int
    ):
        """
        Overrides the parent train method to:
          1) Train the global model using fedAvg's standard procedure.
          2) Train the personal_model with an L2 penalty to keep it close to global model.
        
        Returns:
            global_model: The updated global model and personal model.
        """
        # Standard FL local training on local_model (global model copy).
        super().train(global_round, max_local_round, threshold, patience)
        # Personal model training.
        self._train_personal_model(global_round, max_local_round)

        personal_ckpt_path = f"{self.checkpt_path}/personal_ckpts/{self.client_id}/round_{global_round}.pt"
        self._save_checkpt(self.personal_model, personal_ckpt_path)
        return self.local_model

    def _train_personal_model(self, global_round: int, max_local_round: int):
        """
        Trains the personal_model for Ditto. Uses an L2 penalty (with strength ditto_lambda)
        against the current local_model’s parameters (which serve as the “anchor”).
        """
        print(f"Client: {self.client_id} \tTraining personal model for Ditto...")
        logging.info(f"Client: {self.client_id} \tTraining personal model for Ditto...")

        self.personal_model.train()

        for epoch in range(max_local_round):
            batch_losses = []
            with torch.no_grad():
                anchor_params = [p.detach() for p in self.local_model.parameters()]

            for x, y in self.traindl:
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.personal_model(x)

                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and isinstance(self.train_dataset, FEMNISTDataset):
                    y = y.view(-1)
                elif isinstance(self.train_dataset, MNISTDataset):
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1, 1)

                loss = self.loss_fn(predictions, y)

                # L2 distance between personal_model & local_model
                ditto_penalty = 0.0
                for personal_param, anchor_param in zip(self.personal_model.parameters(), anchor_params):
                    ditto_penalty += torch.sum((personal_param - anchor_param) ** 2)
                
                ditto_penalty = self.ditto_lambda * 0.5 * ditto_penalty  # 0.5 is optional scaling
                total_loss = loss + ditto_penalty
                self.personal_optimizer.zero_grad()
                total_loss.backward()
                self.personal_optimizer.step()

                batch_losses.append(total_loss.item())

            avg_loss = sum(batch_losses) / len(batch_losses)
            print(f"Client: {self.client_id} \tEpoch (personal): {epoch+1} \tAvg Loss: {avg_loss:.6f} \tGlobal Round: {global_round}")
            logging.info(f"Client: {self.client_id} \tEpoch (personal): {epoch+1} \tAvg Loss: {avg_loss:.6f} \tGlobal Round: {global_round}")

    def _save_checkpt(self, checkpoint: torch.nn.Module, ckptpath: str) -> None:
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


