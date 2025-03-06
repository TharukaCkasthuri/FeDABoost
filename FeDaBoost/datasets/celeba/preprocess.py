import os
import json
import torch
import argparse

from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms

def read_dir(data_dir:str)->tuple:
    """
    Reads data from the input directory.
    
    Parameters:
    ------------
    data_dir: str; path to the directory containing the data
    
    Returns:
    ------------
    clients: list; list of client ids
    groups: list; list of group ids
    data: dict; dictionary containing the data
    """

    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))

    return clients, groups, data


def read_data(train_data_dir:str, test_data_dir:str)->tuple:
    """
    Parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Parameters:
    ------------
    train_data_dir: str; path to the directory containing the training data
    test_data_dir: str; path to the directory containing the test data

    Returns:
    ------------
    clients: list of client ids
    groups: list of group ids; empty list if none found
    train_data: dictionary of train data
    test_data: dictionary of test data

    """

    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return test_clients, test_groups, train_data, test_data

def preprocess_data(data: dict, data_dir: str, transform=None) -> dict:
    """
    Preprocess the data by loading images and converting them and their labels to tensors.
    
    Args:
        data (dict): Dictionary with keys 'x' (list of image file names) and 'y' (list of labels).
        data_dir (str): Directory where the images are stored.
        transform: Optional torchvision transform(s) to apply to the images.
        
    Returns:
        dict: A dictionary with preprocessed 'x' and 'y', where 'x' is a tensor of images
              and 'y' is a tensor of labels.
    """
    images = []
    labels = []
    for file_name, label in zip(data['x'], data['y']):
        image_path = os.path.join(data_dir, file_name)

        image = Image.open(image_path).convert("RGB")
        if transform:
            image = transform(image)
        else:
            image = ToTensor()(image)
        images.append(image)
        
        # Convert label to int
        labels.append(int(float(label)))
        
    # Stack images into a single tensor (assumes all images are the same size)
    processed_x = torch.stack(images)
    processed_y = torch.tensor(labels, dtype=torch.long)
    return {'x': processed_x, 'y': processed_y}

class CELEBADataset(Dataset):
    """Custom Dataset for loading CelebA data."""

    def __init__(self, data: dict, transform=None) -> None:
        """
        Args:
            data: dictionary with keys 'x' and 'y' for inputs and labels.
            transform: Optional torchvision transforms to apply to the image.
        """
        self.x = data['x']
        self.y = data['y']
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.y)

    def __getitem__(self, idx) -> tuple:
        """
        Returns the item at the given index.

        Parameters:
        ------------
        idx: int; index of the item

        Returns:
        ------------
        A tuple (sample_x, sample_y) where:
            - sample_x: a torch.Tensor of the image data.
            - sample_y: a torch.Tensor of the label.
        """
        sample_x = self.x[idx]
        sample_y = self.y[idx]

        # If sample_x is a string, assume it's a path to an image file.
        if isinstance(sample_x, str):
            image = Image.open(sample_x).convert("RGB")
            if self.transform:
                sample_x = self.transform(image)
            else:
                sample_x = ToTensor()(image)
        else:
            sample_x.clone().detach()

        # Convert sample_y to int if it's a string.
        if isinstance(sample_y, str):
            sample_y = int(sample_y)
        sample_y = torch.tensor(sample_y, dtype=torch.float32).view(-1, 1)
        
        return sample_x, sample_y

    def num_classes(self) -> int:
        """
        Returns the number of unique classes in the dataset.
        """
        unique_classes = torch.unique(torch.tensor(self.y))
        return len(unique_classes)

def main():

    BATCH_SIZE = 16
    min_samples = 1 * BATCH_SIZE

    parser = argparse.ArgumentParser(description="CELEBA dataset preprocessing parameters")
    parser.add_argument("--train_data_dir", type=str, 
                        default=os.path.join('/Users/tak/Documents/BTH/','leaf','data','celeba','data','train'), 
                        help="Path to the training data directory")
    parser.add_argument("--test_data_dir", type=str, 
                        default=os.path.join('/Users/tak/Documents/BTH/','leaf','data','celeba','data','test'),
                        help="Path to the test data directory")
    parser.add_argument("--raw_data_dir", type=str, default=os.path.join('/Users/tak/Documents/BTH/','leaf','data','celeba','data','raw', 'img_align_celeba'), 
                        help="Path to the raw data directory")
    
    args = parser.parse_args()
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    raw_data_dir = args.raw_data_dir
    
    train_clients, train_groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            ])

    for client in train_clients:
        if len(train_data[client]['y']) > min_samples and len(test_data[client]['y']) > 2:
            print("Processing client: ", client)
            if not os.path.exists("./trainpt"):
                os.makedirs("./trainpt")
            if not os.path.exists("./testpt"):
                os.makedirs("./testpt")

            preprocessed_train_data = preprocess_data(train_data[client], raw_data_dir, transform=transform)
            preprocessed_test_data = preprocess_data(test_data[client], raw_data_dir, transform=transform)
            train_dataset = CELEBADataset(preprocessed_train_data)
            torch.save(train_dataset, "./trainpt/" + str(client) + ".pt")
            test_dataset = CELEBADataset(preprocessed_test_data)
            torch.save(test_dataset, "./testpt/" + str(client) + ".pt")

            train_path = f"./trainpt/{client}.pt"
            test_path  = f"./testpt/{client}.pt"

            if os.path.exists(train_path):
                torch.save(train_dataset, train_path)
            else:
                os.makedirs(os.path.dirname(train_path), exist_ok=True)
                torch.save(train_dataset, train_path)

            if os.path.exists(test_path):
                torch.save(test_dataset, test_path)
            else:
                os.makedirs(os.path.dirname(test_path), exist_ok=True)
                torch.save(test_dataset, test_path)
    
if __name__ == "__main__":
    main()