import os

def create_log_dir(algo_name, **params):
    """Create a logger for storing the training/testing metrics."""
    i = 0
    os.makedirs("logs", exist_ok=True)
    
    parametres = ""
    
    for key, value in params.items():
        parametres += f"{key}_{value}_"
    
    if not os.path.exists(f"logs/{algo_name}"):
        os.makedirs(f"logs/{algo_name}")
    while os.path.exists(f"logs/{algo_name}/{parametres}{i}") and os.listdir(f"logs/{algo_name}/{parametres}{i}"):
        i += 1
    os.makedirs(f"logs/{algo_name}/{parametres}{i}", exist_ok=True)
    
    return f"logs/{algo_name}/{parametres}{i}"

def save_results(save_dir, results, **params):
    """Save the results to a file."""
    with open(f"{save_dir}/results.txt", "w") as f:
        f.write("Parameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Results:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")