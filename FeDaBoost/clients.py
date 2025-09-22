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
import copy
import torch
import math
import logging
import copy
import numpy as np

from torch.utils.data import DataLoader
from utils import get_device
from datasets.mnist.preprocess import MNISTDataset
from datasets.femnist.preprocess import FEMNISTDataset
from datasets.celeba.preprocess import CELEBADataset
from datasets.cifar10.preprocess import CIFARDataset
from sklearn.metrics import f1_score

class Client:
    """
    Client class for federated learning.

    Parameters:
    ------------
    client_id: str; client id.
    train_dataset: torch.utils.data.Dataset object; training dataset.
    test_dataset: torch.utils.data.Dataset object; validation dataset.
    loss_fn: torch.nn.Module object; loss function.
    train_batch_size: int; train batch size.
    learning_rate: float; learning rate for clients.
    weight_decay: float; weight decay for optimizer.
    local_model: torch.nn.Module object; model.
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        loss_fn: torch.nn.Module,
        train_batch_size: int,
        test_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        local_model: object = None,
    ) -> None:

        self.client_id: str = client_id
        self.loss_fn = copy.deepcopy(loss_fn)
        self.batch_size = train_batch_size
        self.device = get_device()
        self.local_model = local_model
        self.local_model = local_model.to(self.device)
        self.train_dataset = train_dataset
        self.datapoints = len(train_dataset)
        

        self.traindl = DataLoader(
            train_dataset, train_batch_size, shuffle=True, drop_last=True
        )

        self.valdl = DataLoader(test_dataset, test_batch_size, shuffle=False, drop_last=True)

        if isinstance(self.train_dataset, CIFARDataset):
            self.optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,  # Momentum is often used in CIFAR-10 training
            )            
        else:
            self.optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        """ #temporary testing AdamW optimizer
        self.optimizer = torch.optim.AdamW(
                self.local_model.parameters(),
                lr=0.00001,
                weight_decay=1e-6,
            )
        """
        #temporary testing AdamW optimizer
        #self.optimizer = torch.optim.AdamW(
        #        self.local_model.parameters(),
        #        lr=0.0003,
        #        weight_decay=1e-6,
        #    )
        

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
        #print(f"Client: {self.client_id} \tReceiving global model...")
        #logging.info(f"Client: {self.client_id} \tReceiving global model...")
        
        prev_weights = copy.deepcopy(self.local_model.state_dict())
        
        try:
            update_status = self.local_model.load_state_dict(model_weights, strict=True)
            
            if update_status.missing_keys or update_status.unexpected_keys:
                logging.warning(f"Client: {self.client_id} encountered update issues: missing_keys={update_status.missing_keys}, unexpected_keys={update_status.unexpected_keys}. Reverting to previous weights.")
                self.local_model.load_state_dict(prev_weights)
            else:
                pass
                #logging.info(f"Client: {self.client_id} model updated successfully.")
        except Exception as e:
            logging.error(f"Client: {self.client_id} error during model update: {e}. Reverting to previous weights.")
            self.local_model.load_state_dict(prev_weights)

    def get_model(self) -> object:
        """
        Get the model of the client.
        """
        return self.local_model

    def train(self, global_round:int, max_local_round:int, threshold:float, patience:int) -> tuple:
        """
        Training the model, using the fedaboost-optima strategy.

        Parameters:
        ------------
        global_round: int; global round number.
        max_local_round: int; maximum number of local rounds, in case loss reduction threshold is not met.
        threshold: float; threshold for loss reduction.
        patience: int; number of patience rounds to wait for loss reduction.

        Returns:
        ------------
        model: torch.nn.Module object; trained model.
        """
        previous_loss_avg = float('inf')  
        no_improvement_rounds = 0  

        print(f"Client: {self.client_id} \tTraining...")
        logging.info(f"Client: {self.client_id} \tTraining...")

        #val_loss, val_f1 = self.evaluate()
        #print(f"Client: {self.client_id} \tInitial Validation Loss: {val_loss:.4f} \tInitial Validation F1: {val_f1:.4f}")
        #logging.info(f"Client: {self.client_id} \tInitial Validation Loss before training: {val_loss:.4f} \tInitial Validation F1: {val_f1:.4f}")

        self.local_model.train()
        for epoch in range(max_local_round):
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.local_model(x)
                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and isinstance(self.train_dataset, FEMNISTDataset):
                    y = y.view(-1)
                elif isinstance(self.train_dataset, MNISTDataset):
                    y = torch.argmax(y, dim=1)
                elif isinstance(self.train_dataset, CIFARDataset):
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1, 1)
                loss = self.loss_fn(outputs, y)
                self.local_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
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
            

        #val_loss, val_f1 = self.evaluate()
        #print(f"Client: {self.client_id} \t Updated Validation Loss: {val_loss:.4f} \tUpdated Validation F1: {val_f1:.4f}")
        #logging.info(f"Client: {self.client_id} \tUpdated Loss: {val_loss:.4f} \tUpdated Validation F1: {val_f1:.4f}")

        return self.local_model

    def evaluate(self) -> tuple:
        """
        Evaluate the client local model with validation dataset of the client.

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
            elif isinstance(self.train_dataset, CIFARDataset):
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
    client_id: str; client id.
    train_dataset: torch.utils.data.Dataset object; training dataset.
    test_dataset: torch.utils.data.Dataset object; validation dataset.
    loss_fn: torch.nn.Module object; loss function.
    train_batch_size: int; train batch size.
    learning_rate: float; learning rate for clients.
    weight_decay: float; weight decay for optimizer.
    local_model: torch.nn.Module object; model.
    num_classes: int; number of classes in the entire dataset.
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        loss_fn: torch.nn.Module,
        train_batch_size: int,
        test_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        local_model: object = None,
        num_classes: int = 10,
        eta = 0.01,  # Boosting learning rate
        error_threshold: float = 0.5,  # Error threshold for boosting
    ) -> None:
        
        super().__init__(
            client_id, 
            train_dataset, 
            test_dataset, 
            loss_fn, 
            train_batch_size,
            test_batch_size, 
            learning_rate, 
            weight_decay, 
            local_model
        )

        """
        self.optimizer = torch.optim.AdamW(
                self.local_model.parameters(),
                lr=0.0002,
                weight_decay=1e-6,
            )
        """
        """
        self.optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,  # Momentum is often used in CIFAR-10 training
            )
        """
        self.optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,  # Momentum is used in CIFAR-10 training
            nesterov=True,
            )
        
        """
        self.optimizer = torch.optim.SGD([
                {'params': local_model.conv1.parameters(), 'lr': 0.005},
                {'params': local_model.conv2.parameters(), 'lr': 0.005},
                {'params': local_model.gn1.parameters(), 'lr': 0.005},
                {'params': local_model.gn2.parameters(), 'lr': 0.005},
                {'params': local_model.fc1.parameters(), 'lr': learning_rate},
                {'params': local_model.fc2.parameters(), 'lr': learning_rate},
            ], momentum=0.9, weight_decay=0.001, nesterov=True,)           
        """

        self.loss_fn = copy.deepcopy(loss_fn)
        self.eta = 0.01  # Boosting learning rate
        self.eta = eta
        self.error_threshold = error_threshold
        self.loss_fn.gamma = 0.0
        self.num_classes = train_dataset.num_classes() if num_classes is None else num_classes

    def train(self, global_round, max_local_round,k, threshold=0.01, patience=2,) -> tuple:
        """
        Training the model, using the fedaboost-optima strategy.

        Parameters:
        ------------
        global_round: int; global round number.
        max_local_round: int; maximum number of local rounds, in case loss reduction threshold is not met.
        threshold: float; threshold for loss reduction.
        patience: int; number of patience rounds to wait for loss reduction.

        Returns:
        ------------
        model: torch.nn.Module object; trained model.
        alpha: float; alpha weight for the client aggregation in server.
        """

        previous_loss_avg = float('inf')  
        no_improvement_rounds = 0  

        error_rate, alpha = self.get_alpha()
        logging.info(f"Client {self.client_id} error rate Before start training: {error_rate}")
        print(f"Client {self.client_id} alpha for boosting weights: {alpha}")
        logging.info(f"Client {self.client_id} alpha for boosting weights: {alpha}")
        logging.info(f'Clients weights before update: {self.weight}')

        prev_weight = self.weight
        self.weight = self.update_weight(alpha, performance_indicator = (error_rate > self.error_threshold))
        logging.info(f'Clients weights after update: {self.weight}')
        logging.info(f'Clients weights change: {self.weight - prev_weight}')
        
        if (error_rate > self.error_threshold):
            print(f"The client training is boosted by: {self.weight}")
            logging.info(f"The client training is boosted by weight: {self.weight}")
            max_gamma = 3

            new_gamma = min(self.loss_fn.gamma + self.weight, max_gamma)
            self.loss_fn.update_gamma(new_gamma)
            logging.info(f"Client {self.client_id} gamma for training: {self.loss_fn.gamma}")
        else:
            logging.info(f"The client training is not boosted by weight: {self.weight}, because the error rate is less than the threshold.")
            logging.info(f"Client {self.client_id} gamma for training: {self.loss_fn.gamma}")


        print(f"Client: {self.client_id} \tTraining...")
        logging.info(f"Client: {self.client_id} \tTraining...")

        #val_loss, val_f1 = self.evaluate()
        #print(f"Client: {self.client_id} \tInitial Validation Loss: {val_loss:.4f} \tInitial Validation F1: {val_f1:.4f}")
        #logging.info(f"Client: {self.client_id} \tInitial Validation Loss before training: {val_loss:.4f} \tInitial Validation F1: {val_f1:.4f}")

        self.local_model.train()
        for epoch in range(max_local_round):
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad() 
                outputs = self.local_model(x)

                if isinstance(self.train_dataset, FEMNISTDataset):
                    y = y.view(-1)
                elif isinstance(self.train_dataset, MNISTDataset):
                    y = torch.argmax(y, dim=1)
                elif isinstance(self.train_dataset, CIFARDataset):
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1, 1)

                loss = self.loss_fn(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                batch_loss.append(loss.item())

    
            loss_avg = sum(batch_loss) / len(batch_loss)    
            print(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round} \tGamma: {self.loss_fn.gamma}")
            logging.info(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round} \tGamma: {self.loss_fn.gamma}")

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
            

        error_rate, alpha = self.get_alpha()
        alpha = np.clip(alpha, -3, 3)
        logging.info(f"Client {self.client_id} error rate After training: {error_rate}")
        #val_loss, val_f1 = self.evaluate()
        #print(f"Client: {self.client_id} \t Updated Validation Loss: {val_loss:.4f} \tUpdated Validation F1: {val_f1:.4f}")
        #logging.info(f"Client: {self.client_id} \tUpdated Loss: {val_loss:.4f} \tUpdated Validation F1: {val_f1:.4f}")

        return self.local_model, alpha, self.loss_fn.gamma

    def __get_error_rate(self, use_macro_f1: bool = False) -> float:
        """
        Evaluate the model on the validation dataset and return the error rate.

        Parameters:
        ----------------
        use_macro_f1: bool
            If True, use 1 - macro F1 score as the error.
            If False, use standard classification error rate.

        Returns:
        ----------------
        error_rate: float
            Error rate (proportion of incorrect predictions or 1 - F1)
        """
        self.local_model.eval()
        device = self.device

        incorrect_preds = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for _, (x, y) in enumerate(self.valdl):
                x, y = x.to(device), y.to(device)
                outputs = self.local_model(x)

                # Convert one-hot to class index if needed
                if y.dim() > 1:
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1)

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                incorrect_preds += (preds != y).sum().item()
                total_samples += y.size(0)

        if total_samples == 0:
            return 1.0  # Assume total error if no samples

        if use_macro_f1:
            f1 = f1_score(all_labels, all_preds, average='macro')
            error_rate = 1.0 - f1
        else:
            error_rate = incorrect_preds / total_samples

        return error_rate

    
    def get_alpha(self) -> float:
        """
        Calculate adjusted weight (alpha) for the client in the FL setting,
        giving higher weights to clients with lower errors with the global model.
        """
        error_rate = self.__get_error_rate(use_macro_f1=True)  # or False if you prefer
        eps = 1e-6
        error_rate = min(max(error_rate, eps), 1 - eps)
        alpha = np.log((1 - error_rate) / error_rate) + np.log(self.num_classes - 1)
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
        alpha = np.clip(alpha, -5, 5)  #Ensure alpha is not too small
        self.weight = self.weight * math.exp(float(self.eta) *  -float(alpha) * int(performance_indicator))
        return self.weight

class DittoClient(Client):
    """
    DittoClient class extends the base Client for Ditto personalized FL.
    Maintains a separate `personal_model` and trains it with a
    regularization penalty towards the global (local_model) parameters.

    Parameters:
    ------------
    client_id: str; client id.
    train_dataset: torch.utils.data.Dataset object; training dataset.
    test_dataset: torch.utils.data.Dataset object; validation dataset.
    loss_fn: torch.nn.Module object; loss function.
    train_batch_size: int; train batch size.
    test_batch_size: int; test batch size.
    learning_rate: float; learning rate for clients.
    weight_decay: float; weight decay for optimizer.
    local_model: torch.nn.Module object; model.
    personal_learning_rate: float; personal model learning rate.
    ditto_lambda: float; regularization strength for Ditto.
    personalized: bool; whether to use personalized learning rates.
    checkpt_path: str; path to save personal model checkpoints.
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        loss_fn: torch.nn.Module,
        train_batch_size: int,
        test_batch_size: int,
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
            train_batch_size,
            test_batch_size,
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
        conv_params = []
        fc_params = []

        for name, param in self.personal_model.named_parameters():
            if "conv" in name:
                conv_params.append(param)
            else:
                fc_params.append(param)

        self.personal_optimizer = torch.optim.SGD([
            {"params": conv_params, "lr": self.personal_lr * 2}, 
            {"params": fc_params, "lr": self.personal_lr}  
        ], weight_decay=0.0001, momentum=0.0)

        self.personal_optimizer = torch.optim.Adam(
            self.personal_model.parameters(),
            lr=self.personal_lr, 
            weight_decay=0.0001)


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
        self._save_checkpt(self.personal_model.eval(), personal_ckpt_path)

        return self.local_model

    def _train_personal_model(self, global_round: int, max_local_round: int):
        """
        Trains the personal_model for Ditto. Uses an L2 penalty (with strength ditto_lambda)
        against the current local_model’s parameters (which serve as the “anchor”).
        """
        print(f"Client: {self.client_id} \tTraining personal model for Ditto...")
        logging.info(f"Client: {self.client_id} \tTraining personal model for Ditto...")

        self.personal_model.train()

        #eval_loss, eval_f1 = self.evaluate_personal_model()
        #print(f"Client (personal): {self.client_id} \tInitial Validation Loss: {eval_loss:.6f} \tValidation F1: {eval_f1:.6f}")
        #logging.info(f"Client (personal): {self.client_id} \tInitial Validation Loss: {eval_loss:.6f} \tValidation F1: {eval_f1:.6f}")

        for epoch in range(max_local_round*1):
            batch_losses = []

            anchor_params = [p.detach() for p in self.local_model.parameters()]

            for x, y in self.traindl:
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.personal_model(x)

                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and isinstance(self.train_dataset, FEMNISTDataset):
                    y = y.view(-1)
                elif isinstance(self.train_dataset, MNISTDataset):
                    y = torch.argmax(y, dim=1)
                elif isinstance(self.train_dataset, CIFARDataset):
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1, 1)

                loss = self.loss_fn(predictions, y)

                # L2 distance between personal_model & local_model
                ditto_penalty = 0.0
                for personal_param, anchor_param in zip(self.personal_model.parameters(), anchor_params):
                    ditto_penalty += torch.nn.functional.mse_loss(personal_param, anchor_param, reduction="sum")

                ditto_penalty = self.ditto_lambda * 0.5 * ditto_penalty  # 0.5 is optional scaling
                total_loss = loss + ditto_penalty
                self.personal_optimizer.zero_grad()
                total_loss.backward()

                self.personal_optimizer.step()
                batch_losses.append(total_loss.item())

            avg_loss = sum(batch_losses) / len(batch_losses)
            print(f"Client: {self.client_id} \tEpoch (personal): {epoch+1} \tAvg Loss: {avg_loss:.6f} \tGlobal Round: {global_round}")
            logging.info(f"Client: {self.client_id} \tEpoch (personal): {epoch+1} \tAvg Loss: {avg_loss:.6f} \tGlobal Round: {global_round}")
        
        #eval_loss, eval_f1 = self.evaluate_personal_model()
        #print(f"Client: {self.client_id} \tFinal Validation Loss: {eval_loss:.6f} \tValidation F1: {eval_f1:.6f}")
        #logging.info(f"Client: {self.client_id} \tFinal Validation Loss: {eval_loss:.6f} \tValidation F1: {eval_f1:.6f}")
    
    def evaluate_personal_model(self) -> tuple:
        """
        Evaluate the personal model with the validation dataset.
        """

        batch_loss = []
        all_preds = []
        all_labels = []

        for _, (x, y) in enumerate(self.valdl):
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.personal_model(x)

            if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and isinstance(self.train_dataset, FEMNISTDataset):
                y = y.view(-1)
            elif isinstance(self.train_dataset, MNISTDataset):
                y = torch.argmax(y, dim=1)
            elif isinstance(self.train_dataset, CIFARDataset):
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
    
    def _save_checkpt(self, checkpoint: torch.nn.Module, ckptpath: str) -> None:
        """
        Saving the checkpoints.

        Parameters:
        ----------------
        checkpoint: Model at a specific checkpoint.
        ckptpath: str; Path to save the checkpoint. Default is None.
        """
        checkpoint.eval()
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


