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

Published in: 
"""
import os
import json
import time
import logging
import argparse
import configparser
from datetime import datetime
from enum import Enum
import torch
import copy

from clients import Client, BoostingClient, DittoClient
from server import Server, BoostingServer, DittoServer
from utils import get_device, get_client_ids

from models.kv import ShallowNN
from models.femnist import FEMNISTNet
from models.mnist import MNISTNet
from models.celeba import CELEBANet
from models.cifar10 import CIFAR10Net
from evals import FocalLoss, HybridLoss

from datasets.kv.preprocess import KVDataSet
from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset
from datasets.celeba.preprocess import CELEBADataset
from datasets.cifar10.preprocess import CIFARDataset

def load_config(config_path="config.cfg"):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def setup_logging(stratergy,dataset, timestamp):
    log_dir = '.logs'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'federation__{dataset}_{stratergy}_{timestamp}.log')
    
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return log_filename

def parse_arguments():
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--dataset", type=dataset_enum, default="cifar10", help="Choose a dataset from the available options; femnist, mnist, kv")
    parser.add_argument("--data_dir", type=str, default="datasets/cifar10/", help="Path to the data directory, expected to have train and test directories with names trainpt and testpt respectively.")
    parser.add_argument("--loss_function", type=str, default="FocalLoss", help="Choose a loss function from the available options; CrossEntropyLoss, FocalLoss, HybridLoss")
    parser.add_argument("--stratergy", type=str, default="fedaboost", help="Choose a federated learning stratergy from the available options; fedavg, fedprox, fedaboost")
    parser.add_argument("--log_summary", action="store_true")
    parser.add_argument("--global_rounds", type=int, default=30)
    parser.add_argument("--local_rounds", type=int, default=10)
    parser.add_argument("--save_ckpt", action="store_true")
    return parser.parse_args()

class Federation:
    """
    Class for federated learning.

    Parameters:
    ----------------
    client_ids: list;
        List of client ids.
    model: torch.nn.Module;
        Model to be trained.
    loss_fn: torch.nn.Module;
        Loss function.
    train_data_dir: str;
        Path to the training data directory.
    test_data_dir: str;
        Path to the testing data directory.
    num_classes: int;
        Number of classes in the dataset.
    global_rounds: int;
        Number of global rounds.
    stratergy: callable;
        Federated learning stratergy.
    learning_rate: float;
        Learning rate.
    train_batch_size: int;
        Batch size for training.
    test_batch_size: int;
        Batch size for testing.
    weight_decay: float;    
    """

    def __init__(
        self,
        client_ids: list,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        train_data_dir: str,
        test_data_dir: str,
        num_classes: int,
        global_rounds: int,
        stratergy: callable,
        learning_rate: float,
        train_batch_size: int,
        test_batch_size: int,
        weight_decay: float,
        eta: float,
        error_threshold: float,
    ) -> None:
        
        self.client_ids = client_ids
        self.num_classes = num_classes
        self.model = model
        self.loss_fn = loss_fn
        self.global_rounds = global_rounds
        self.stratergy = stratergy
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.weight_decay = weight_decay
        self.eta = eta
        self.error_threshold = error_threshold

        if stratergy == "fedaboost":
            self.server = BoostingServer(global_rounds, stratergy, checkpt_path=checkpt_path)
            self.server.init_model(model)

            # Set up the clients for fedaboost server
            for id in client_ids:
                self.server.connect_client(BoostingClient(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt"),
                    torch.load(f"{test_data_dir}/{id}.pt"),
                    self.loss_fn,
                    self.train_batch_size,
                    self.test_batch_size,
                    self.learning_rate,
                    self.weight_decay,
                    local_model= copy.deepcopy(self.model),
                    num_classes=self.num_classes,
                    eta=self.eta,
                    error_threshold=self.error_threshold,
                ))

        elif stratergy == "fedavg":
            self.server = Server(global_rounds,stratergy,checkpt_path=checkpt_path)
            self.server.init_model(model)

            # Set up the clients for fedavg server
            for id in client_ids:
                self.server.connect_client(Client(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt"),
                    torch.load(f"{test_data_dir}/{id}.pt"),
                    self.loss_fn,
                    self.train_batch_size,
                    self.test_batch_size,
                    self.learning_rate,
                    self.weight_decay,
                    local_model= copy.deepcopy(self.model),
                ))

        elif stratergy == "ditto":
            self.server = Server(global_rounds, stratergy, checkpt_path=checkpt_path)
            self.server.init_model(model)

            for id in client_ids:
                self.server.connect_client(
                    DittoClient(
                        client_id=id,
                        train_dataset=torch.load(f"{train_data_dir}/{id}.pt"),
                        test_dataset=torch.load(f"{test_data_dir}/{id}.pt"),
                        loss_fn=self.loss_fn,
                        train_batch_size=self.train_batch_size,
                        test_batch_size=self.test_batch_size,
                        learning_rate=self.learning_rate,
                        weight_decay=self.weight_decay,
                        local_model=copy.deepcopy(self.model),
                        personal_learning_rate=0.001,
                        ditto_lambda=0.1,
                        personalized=True,
                        checkpt_path=checkpt_path,

                    )
                )
        else:
                raise ValueError(f"Invalid stratergy. Choose from: {', '.join([stratergy.value for stratergy in Stratergy])}")

    def train(self, training_samples, max_local_round:int = 10, threshold:float = 0.01, patience = 2) -> tuple:
        """
        Training the federated learning model.

        Returns:
        ----------------
        trained_model: torch.nn.Module;
            Trained model.
        """
        print(threshold)
        print(type(threshold))
        if self.stratergy == "fedaboost":
            trained_model = self.server.train(training_samples, max_local_round, threshold=0.05, patience=5)
        else:
            trained_model = self.server.train(training_samples, max_local_round, threshold, patience)
        return trained_model

    def save_models(self, model: torch.nn.Module, ckptpath: str) -> None:
        """
        Saving the training stats and the model.

        Parameters:
        ----------------
        model:
            Trained model.
        ckptpath: str;
            Path to save the model and the training stats. Default is None.
        """
        if os.path.exists(ckptpath):
            torch.save(
                model.state_dict(),
                ckptpath,
            )
        else:
            os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
            torch.save(
                model.state_dict(),
                ckptpath,
            )

class Dataset(Enum):
    CELEBA = "celeba"
    FEMNIST = "femnist"
    MNIST = "mnist"
    KV = "kv"
    CIFAR10 = "cifar10"

class Stratergy(Enum):
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDABOOST = "fedaboost" 
    DITTO = "ditto"

def dataset_enum(dataset_str: str) -> str:
    """
    Returns the dataset enum.
    
    Parameters:
    ----------------
    dataset_str: str;
        Dataset string.
    """
    try:
        return Dataset(dataset_str.lower())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dataset. Choose from: {', '.join([dataset.value for dataset in Dataset])}")
    

if __name__ == "__main__":
    device = get_device()
    args = parse_arguments()
    config = load_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = setup_logging(args.stratergy, args.dataset, timestamp)

    loss_threshould = float(config['GENERAL']['loss_threshould'])
    patience = int(config['GENERAL']['patience'])

    global_rounds = args.global_rounds
    local_rounds = args.local_rounds
    epochs = global_rounds * local_rounds
    save_ckpt = args.save_ckpt
    stratergy = args.stratergy
    data_dir = args.data_dir
    dataset = args.dataset

    train_data_dir = f"{data_dir}/trainpt"
    test_data_dir = f"{data_dir}/testpt"

    if args.stratergy in ["fedaboost"] and args.loss_function != "FocalLoss":
        warning_message = (
            f"Warning: The strategy '{args.stratergy}' is selected without using 'FocalLoss' as the loss function. "
            f"It is recommended to use 'FocalLoss' for better performance."
        )
        logging.warning(warning_message)
        print(warning_message)  # Optionally print to console as well

    if args.loss_function == "HybridLoss":
        loss_fn = HybridLoss(focal_alpha=1, focal_gamma=2, focal_weight=0.7)
    elif args.loss_function == "FocalLoss":
        logging.info("Using Focal Loss")
        loss_fn = FocalLoss(alpha=1, gamma=0, reduction='mean')
    else:
        loss_fn = getattr(torch.nn, args.loss_function)()
    
    log_summary = args.log_summary
    checkpt_path = f"checkpt/{stratergy}/{dataset.name}/new_exp/v5/epoch_{epochs}/{global_rounds}_rounds_{local_rounds}_epochs_per_round/"
    client_ids = get_client_ids(train_data_dir)

    if args.dataset == Dataset.FEMNIST:
        model = FEMNISTNet(62)
        learning_rate = float(config['FEMNIST']['learning_rate'])
        train_batch_size = int(config['FEMNIST']['train_batch_size'])
        test_batch_size = int(config['FEMNIST']['test_batch_size'])
        weight_decay = float(config['FEMNIST']['weight_decay'])
        num_classes = int(config['FEMNIST']['num_classes'])
        training_samples = json.load(open(f"{data_dir}/training_samples_v1.json"))
        eta = float(config['FEMNIST']['eta'])
        error_threshold = float(config['FEMNIST']['error_threshold'])

    elif args.dataset == Dataset.CELEBA:
        model = CELEBANet()
        learning_rate = float(config['CELEBA']['learning_rate'])
        train_batch_size = int(config['CELEBA']['train_batch_size'])
        test_batch_size = int(config['CELEBA']['test_batch_size'])
        weight_decay = float(config['CELEBA']['weight_decay'])
        num_classes = int(config['CELEBA']['num_classes'])
        training_samples = json.load(open(f"{data_dir}/training_samples_v1.json"))
        error_threshold = float(config['CELEBA']['error_threshold'])

    elif args.dataset == Dataset.MNIST:
        model = MNISTNet()
        learning_rate = float(config['MNIST']['learning_rate'])
        train_batch_size = int(config['MNIST']['train_batch_size'])
        test_batch_size = int(config['MNIST']['test_batch_size'])
        weight_decay = float(config['MNIST']['weight_decay'])
        num_classes = int(config['MNIST']['num_classes'])
        training_samples = json.load(open(f"{data_dir}/training_samples.json"))
        eta = float(config['MNIST']['eta'])
        error_threshold = float(config['MNIST']['error_threshold'])

    elif args.dataset == Dataset.CIFAR10:
        model = CIFAR10Net()
        learning_rate = float(config['CIFAR10']['learning_rate'])
        train_batch_size = int(config['CIFAR10']['train_batch_size'])
        test_batch_size = int(config['CIFAR10']['test_batch_size'])
        weight_decay = float(config['CIFAR10']['weight_decay'])
        num_classes = int(config['CIFAR10']['num_classes'])
        training_samples = json.load(open(f"{data_dir}/training_samples_60.json"))
        eta = float(config['CIFAR10']['eta'])
        error_threshold = float(config['CIFAR10']['error_threshold'])

    elif args.dataset == Dataset.KV:
        model = ShallowNN(176)

    federation = Federation(
        client_ids,
        model,
        loss_fn,
        train_data_dir,
        test_data_dir,
        num_classes,
        global_rounds,
        stratergy,
        learning_rate,
        train_batch_size,
        test_batch_size,
        weight_decay,
        eta=eta,
        error_threshold=error_threshold,
    )

    print("Federation with clients " + ", ".join(client_ids))
    logging.info("Federation with clients " + ", ".join(client_ids))
    logging.info(f"Global rounds: {global_rounds}")
    logging.info(f"Local rounds: {local_rounds}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Loss function: {loss_fn}")
    logging.info(f"Stratergy: {stratergy}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Batch size: {train_batch_size}")
    logging.info(f"Weight decay: {weight_decay}")
    logging.info(f"Checkpoint path: {checkpt_path}")
    logging.info(f"Special Notes: ")
    start = time.time()
    trained_model = federation.train(training_samples, max_local_round=local_rounds, threshold=loss_threshould, patience=patience)
    model_path = f"{checkpt_path}/global_model.pth"
    federation.save_models(trained_model, model_path)
    print(
        "Approximate time taken to train",
        str(round((time.time() - start) / 60, 2)) + " minutes",
    )
    