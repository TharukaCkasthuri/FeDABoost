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
import numpy as np
import pickle
import cv2
import os
import argparse
import random
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from imutils import paths

def load(paths, verbose=-1):
    """
    Loads the images and labels from disk.

    Parameters:
    ------------
    paths: list of image paths
    verbose: whether to display progress

    Returns:
    ------------
        tuple of images and labels
    """
    data = list()
    labels = list()

    for (i, imgpath) in enumerate(paths):    
        im_gray = cv2.imread(imgpath , cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten() 
        label = imgpath.split(os.path.sep)[-2]
        data.append(image/255)
        labels.append(label)
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))

    return data, labels

def create_clients(image_list: list, label_list: list, num_clients: int, initial: str, save_dir: str, batch_size: int) -> dict:
    """
    Create clients using the given images and labels, and save each client's data into a directory.
    This version splits the sorted data (skewed by class) into many shards, assigns several shards per client,
    ensures that each client has at least two different classes by swapping shards if needed,
    and post-processes the data to guarantee that each class has at least two samples.

    Parameters:
    ------------
    image_list: list of numpy arrays.
    label_list: list of binary labels (assumed one-hot).
    num_clients: number of clients to create.
    initial: client name prefix (e.g., 'client').
    save_dir: the base directory to save client data.
    batch_size: minimum batch size requirement per client.

    Returns:
    ------------
    clients: dictionary of client data.
    """
    import os, pickle, random
    from collections import Counter
    import numpy as np

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Attempting to create {num_clients} clients from {len(image_list)} samples.")

    # Compute the class index for each sample.
    max_y = np.argmax(label_list, axis=-1)
    sorted_zip = sorted(zip(max_y, label_list, image_list), key=lambda x: x[0])
    # Create tuples (image, label, class) for later processing.
    data = [(x, y, cls) for cls, y, x in sorted_zip]

    # Determine the minimum number of samples per client.
    min_client_size = max(batch_size * 3, 1)
    max_clients_possible = len(data) // min_client_size
    if num_clients > max_clients_possible:
        print(f"Reducing clients from {num_clients} to {max_clients_possible} to ensure batch-size allocation.")
        num_clients = max_clients_possible

    client_names = [f"{initial}_{i+1}" for i in range(num_clients)]
    print("Client names:", client_names)

    # Create shards: 4 shards per client by default.
    num_shards = 4 * num_clients
    shard_size = len(data) // num_shards

    # Adjust the number of shards if shard_size is less than 2.
    if shard_size < 2:
        num_shards = len(data) // 2
        shard_size = len(data) // num_shards
        print(f"Adjusted number of shards to {num_shards} for a minimum shard size of 2.")

    shards = [data[i * shard_size: (i + 1) * shard_size] for i in range(num_shards)]
    random.shuffle(shards)

    # Assign shards to clients in a round-robin fashion.
    clients_shards = {client: [] for client in client_names}
    for i, shard in enumerate(shards):
        client = client_names[i % num_clients]
        clients_shards[client].append(shard)

    def get_client_classes(shard_list):
        """
        Get the set of classes for a client's assigned shards.
        """
        return set(shard[0][2] for shard in shard_list if shard)

    # Ensure each client has at least two different classes.
    for client in client_names:
        client_classes = get_client_classes(clients_shards[client])
        if len(client_classes) < 2:
            for other in client_names:
                if other == client:
                    continue
                for j, other_shard in enumerate(clients_shards[other]):
                    other_class = other_shard[0][2] if other_shard else None
                    if other_class not in client_classes:
                        clients_shards[client][0], clients_shards[other][j] = clients_shards[other][j], clients_shards[client][0]
                        client_classes = get_client_classes(clients_shards[client])
                        break
                if len(client_classes) >= 2:
                    break
            if len(client_classes) < 2:
                print(f"Warning: Client {client} could not get at least two different classes.")

    # Combine shards for each client and ensure each class has at least two samples.
    clients = {}
    for client in client_names:
        client_data = [item for shard in clients_shards[client] for item in shard]
        # Count the occurrences of each class.
        counts = Counter([cls for _, _, cls in client_data])
        for cls, count in counts.items():
            if count < 2:
                print(f"Client {client} has only {count} sample(s) for class {cls}; duplicating one sample to ensure at least two.")
                sample = next((item for item in client_data if item[2] == cls), None)
                if sample is not None:
                    client_data.append(sample)
        # Save only the (image, label) tuple.
        client_data = [(img, label) for img, label, cls in client_data]
        clients[client] = client_data

        # Save client data to disk.
        file_name = f"{client}.pkl"
        with open(os.path.join(save_dir, file_name), 'wb') as f:
            pickle.dump(client_data, f)

    print(f"Successfully created {num_clients} clients, each with at least two different classes and a skewed distribution.")
    return clients

def create_clients_dirichlet(
    image_list: list,
    label_list: list,
    num_clients: int,
    initial: str,
    save_dir: str,
    batch_size: int,
    alpha: float = 0.5
) -> dict:
    """
    Create clients using Dirichlet-based splitting without data duplication.
    Clients with too few samples are merged, and their data is redistributed to other clients.

    Parameters:
    ------------
    image_list: list of numpy arrays (flattened and normalized images).
    label_list: list (or array) of labels. If one-hot, we take the argmax.
    num_clients: number of clients to create.
    initial: prefix for client names (e.g., 'client').
    save_dir: directory to save client data.
    batch_size: batch size for training (used to determine minimum sample needs).
    alpha: Dirichlet concentration parameter. Lower alpha => more skewed distributions.

    Returns:
    ------------
    clients: dict mapping client names to their (image, label) lists.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Creating {num_clients} clients from {len(image_list)} samples using Dirichlet(alpha={alpha})...")

    # Convert one-hot labels to class indices if needed.
    if hasattr(label_list, 'ndim') and label_list.ndim > 1 and label_list.shape[1] > 1:
        max_y = np.argmax(label_list, axis=1)
    else:
        max_y = label_list

    classes = np.unique(max_y)
    class_indices = [np.where(max_y == c)[0] for c in classes]
    client_data_indices = [[] for _ in range(num_clients)]

    # Assign samples to clients using Dirichlet distribution
    for c_indices in class_indices:
        n_samples_for_class = len(c_indices)
        if n_samples_for_class == 0:
            continue

        # Dirichlet sampling to determine class proportions per client
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = np.round(proportions * n_samples_for_class).astype(int)
        
        # Adjust counts to match total available samples
        diff = n_samples_for_class - np.sum(counts)
        while diff > 0:
            idx = np.argmin(counts)
            counts[idx] += 1
            diff -= 1
        while diff < 0:
            idx = np.argmax(counts)
            if counts[idx] > 0:
                counts[idx] -= 1
                diff += 1

        # Shuffle class-specific indices and assign to clients
        np.random.shuffle(c_indices)
        start = 0
        for client_id, count in enumerate(counts):
            if count > 0:
                selected = c_indices[start : start + count]
                client_data_indices[client_id].extend(selected)
                start += count

    # Set minimum sample requirement
    min_samples_required = 2 * batch_size
    client_names = [f"{initial}_{i+1}" for i in range(num_clients)]
    clients = {}
    small_clients = []  # Stores clients that need merging

    for i, client_name in enumerate(client_names):
        indices = client_data_indices[i]
        if len(indices) < min_samples_required:
            print(f"Client {client_name} has {len(indices)} samples (requires {min_samples_required}), marking for merging.")
            small_clients.extend(indices)  # Store samples for redistribution
        else:
            client_data = [(image_list[idx], label_list[idx]) for idx in indices]
            file_name = os.path.join(save_dir, f"{client_name}.pkl")
            with open(file_name, 'wb') as f:
                pickle.dump(client_data, f)
            clients[client_name] = client_data

    # Redistribute samples from small clients
    print("\nRedistributing samples from small clients to maintain total dataset size...\n")
    valid_clients = list(clients.keys())
    np.random.shuffle(valid_clients)  # Shuffle to distribute fairly

    for idx in small_clients:
        assigned_client = random.choice(valid_clients)  # Pick a valid client
        clients[assigned_client].append((image_list[idx], label_list[idx]))  # Assign sample

    print(f"\nSuccessfully created {len(clients)} clients while preserving dataset size ({len(image_list)} samples).")
    return clients


class MNISTDataset(Dataset):
    """
    Custom dataset class for the training and validation dataset.
    """

    def __init__(self, data) -> None:
        self.data = data

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        --------
        length: int; length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        """
        Returns the item at the given index.

        Parameters:
        ------------
        idx: int; index of the item

        Returns:
        ------------
        x_train: torch.tensor object; input data
        y_train: torch.tensor object; label
        """
        image, label = self.data[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def num_classes(self) -> int:
        """
        Returns the number of unique classes in the dataset.

        Returns:
        ------------
        num_classes: int; number of unique labels in the dataset
        """
        # Extract labels from the dataset
        labels = [label for _, label in self.data]
        unique_classes = torch.unique(torch.tensor(labels))
        return len(unique_classes)
    

def build_dataset(data_dir, saving_dir) -> None:
    """
    Split the pickles into train and test, saving as a PyTorch dataset with stratified split.
    Only process clients that have at least two classes.

    Parameters:
    ------------
    data_dir: str; path to the pickle files
    saving_dir: str; path to save the PyTorch datasets

    Returns:
    ------------
    None
    """

    files = os.listdir(data_dir)
    ids = [file.split(".")[0] for file in files]
    files_path = [os.path.join(data_dir, file) for file in files]

    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    trainpt_dir = os.path.join(saving_dir, "trainpt")
    if not os.path.exists(trainpt_dir):
        os.makedirs(trainpt_dir)

    testpt_dir = os.path.join(saving_dir, "testpt")
    if not os.path.exists(testpt_dir):
        os.makedirs(testpt_dir)

    for id, pickle_file in zip(ids, files_path):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        labels = [label for _, label in data]
        # Convert one-hot encoded labels to integer labels.
        integer_labels = [np.argmax(label) for label in labels]

        # Check if the client has at least two classes.
        if len(set(integer_labels)) < 2:
            print(f"Skipping client with id {id} because it only has one class: {set(integer_labels)}")
            continue

        try:
            train_data, test_data = train_test_split(
                data, test_size=0.2, random_state=42, stratify=integer_labels
            )
        except ValueError:
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        except Exception as e:
            print(f"Error processing client {id}: {e}")
            continue

        train_dataset = MNISTDataset(train_data)
        test_dataset = MNISTDataset(test_data)

        torch.save(train_dataset, os.path.join(trainpt_dir, f"{id}.pt"))
        torch.save(test_dataset, os.path.join(testpt_dir, f"{id}.pt"))

        print(f"Saved {os.path.join(trainpt_dir, f'{id}.pt')} and {os.path.join(testpt_dir, f'{id}.pt')}")



def main():
    parser = argparse.ArgumentParser(description="Preprocess the MNIST dataset.")
    parser.add_argument("--num_clients", type=int, default=500)
    parser.add_argument("--image_path", type=str, default="/Users/tak/Documents/BTH/MNIST/trainingSet")
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha parameter")
    args = parser.parse_args()

    image_path = args.image_path
    image_paths = list(paths.list_images(image_path))
    image_list, label_list = load(image_paths, verbose=10000)

    #binarize the labels
    lb = LabelBinarizer()
    label_list = lb.fit_transform(label_list)

    #split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                        label_list, 
                                                        test_size=0.1, 
                                                        random_state=42)

    #create_clients(X_train, y_train, num_clients=args.num_clients, initial='client', save_dir = 'client_data', batch_size=32)
    create_clients_dirichlet(image_list, label_list, num_clients=args.num_clients,
               initial='client',
               save_dir='client_data',
               batch_size=32,
               alpha=args.alpha)
    #create_clients(X_train, y_train, num_clients=args.num_clients, initial='client', save_dir = 'client_data')
    build_dataset("/Users/tak/Documents/BTH/FeDABoost/FeDaBoost/datasets/mnist/client_data", "/Users/tak/Documents/BTH/FeDABoost/FeDaBoost/datasets/mnist")

if __name__ == "__main__":
    main()