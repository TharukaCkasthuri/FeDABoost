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