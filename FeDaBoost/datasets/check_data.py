import os
import argparse
import pickle
import numpy as np

def verify_clients(clients: dict) -> bool:
    """
    Verify that each client in the given dictionary has at least two different classes.
    Also prints the class distribution for each client to assess skewness.
    
    Parameters:
    -----------
    clients : dict
        A dictionary where each key is a client name and the value is a list of (image, label) tuples.
        The label is assumed to be either an integer or a one-hot encoded vector.
        
    Returns:
    --------
    bool
        True if all clients have at least two different classes, False otherwise.
    """
    all_clients_valid = True
    for client, data in clients.items():
        class_counts = {}
        for _, label in data:
            # If label is one-hot (list or numpy array), convert to class index
            if isinstance(label, (list, np.ndarray)):
                label_arr = np.array(label)
                cls = int(np.argmax(label_arr))
            else:
                cls = label
            class_counts[cls] = class_counts.get(cls, 0) + 1

        classes = list(class_counts.keys())
        print(f"Client '{client}' has classes: {classes}")
        print(f"Class distribution: {class_counts}\n")
        
        if len(classes) < 2:
            print(f"Error: Client '{client}' does not have at least two different classes!\n")
            all_clients_valid = False
    return all_clients_valid

def dataset_summary(clients: dict) -> dict:
    """
    Analyzes the dataset distribution to assess the level of non-IID-ness.

    Provides:
    - The number of clients
    - The total number of samples
    - The class distribution across all clients
    - The number of clients having 2, 3, 4, etc., different classes
    - The number of data points per client
    - The variance of class distributions per client to analyze non-IID-ness

    Parameters:
    -----------
    clients : dict
        A dictionary where each key is a client name and the value is a list of (image, label) tuples.

    Returns:
    --------
    dict
        A dictionary containing dataset statistics, including non-IID analysis.
    """
    total_samples = 0
    overall_class_counts = {}
    client_class_distribution = {}
    client_data_counts = {}
    client_class_variance = {}

    # Count class distribution and data points per client
    for client, data in clients.items():
        total_samples += len(data)
        client_data_counts[client] = len(data)
        class_counts = {}

        for _, label in data:
            if isinstance(label, (list, np.ndarray)):
                label = int(np.argmax(label))
            class_counts[label] = class_counts.get(label, 0) + 1
            overall_class_counts[label] = overall_class_counts.get(label, 0) + 1

        num_classes = len(class_counts)
        client_class_distribution[client] = num_classes

        # Compute variance of class distribution for non-IID analysis
        class_sizes = np.array(list(class_counts.values()))
        variance = np.var(class_sizes) if len(class_sizes) > 1 else 0
        client_class_variance[client] = variance

    # Count number of clients with 2, 3, 4, etc., classes
    class_count_distribution = {}
    for num_classes in client_class_distribution.values():
        class_count_distribution[num_classes] = class_count_distribution.get(num_classes, 0) + 1

    # Generate readable class count summary
    class_count_summary = [
        f"{count} clients have {num_classes} classes"
        for num_classes, count in sorted(class_count_distribution.items())
    ]

    # Compute average variance of class distributions
    avg_variance = np.mean(list(client_class_variance.values()))

    summary = {
        "num_clients": len(clients),
        "total_samples": total_samples,
        "class_distribution": overall_class_counts,
        "client_class_summary": class_count_summary,
        "client_data_counts": client_data_counts,
        "client_class_variance": client_class_variance,
        "avg_class_variance": avg_variance  # Higher variance = more non-IID
    }

    return summary


def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", type=str, default="cifar10/client_data", help="Path to the directory containing the pickled MNIST data.")
    args = parser.parse_args()

    data_dir = args.data_dir

    files = os.listdir(data_dir)
    ids = [file.split(".")[0] for file in files]
    files_path = [os.path.join(data_dir, file) for file in files]

    clients = {}
    for id, pickle_file in zip(ids, files_path):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            clients[id] = data

    valid = verify_clients(clients)
    if valid:
        print("All clients have at least two different classes.")
    else:
        print("Some clients do not have at least two different classes.")

    summary = dataset_summary(clients)
    print("\nDataset summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
        
if __name__ == "__main__":
    main()