from core.env import Env as BaseEnv
from core.vis.logger import Logger as BaseLogger
from core.base_scenario import BaseScenario
import random
import torch
from eval.metrics.metrics import SuccessRate, AvgLatency

import os
import numpy as np

class Scenario(BaseScenario):
    
    def __init__(self, config_file, dataset, flag):
        """
        :param flag: '25N50E', '50N50E', '100N150E' or 'MilanCityCenter'
        """
        assert dataset in ['Topo4MEC', 'Pakistan'], f"Invalid dataset={dataset}"
        if dataset == 'Topo4MEC':
            assert flag in ['25N50E', '50N50E', '100N150E', 'MilanCityCenter'], \
            f"Invalid flag={flag}"
        elif dataset == 'Pakistan':
            assert flag in ['Tuple30K', 'Tuple50K', 'Tuple100K'], \
            f"Invalid flag={flag}"
        super().__init__(config_file)
        
        # # Load the test dataset (not recommended)
        # data = pd.read_csv(f"{ROOT_PATH}/{flag}/testset.csv")
        # self.testset = list(data.iloc[:].values)
    
    def status(self):
        pass

class Checkpoint:
    """A simple class to manage checkpoints."""
    
    def __init__(self, path):
        self.path = os.path.join(path, "checkpoints")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, policy, epoch):
        policy.save(os.path.join(self.path, f"checkpoint_epoch_{epoch}.pt"))

    def load(self, policy, epoch):
        policy.load(os.path.join(self.path, f"checkpoint_epoch_{epoch}.pt"))



def create_env(config):
    """Create and return an environment instance."""
    dataset = config["env"]["dataset"]
    flag = config["env"]["flag"]
    scenario = Scenario(config_file=f"eval/benchmarks/{dataset}/data/{flag}/config.json", 
                        dataset=dataset, flag=flag)
    env = Env(scenario, config_file="core/configs/env_config_null.json", verbose=False, refresh_rate=config["env"].get("refresh_rate", 1))
    return env

def error_handler(error: Exception):
    """Customized error handler for different types of errors."""
    errors = ['DuplicateTaskIdError', 'NetworkXNoPathError', 'IsolatedWirelessNode', 'NetCongestionError', 'InsufficientBufferError']
    message = error.args[0][0]
    if message in errors:
        pass
    else:
        raise
    
class Logger(BaseLogger):
    """
    Custom logger class that extends the BaseLogger to include additional functionalities.
    """
    def __init__(self, config):
        super().__init__(config)
        self.log_dir = os.path.join(self.log_dir, "DQRL")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.best_epoch = 0
        self.best_score = np.inf

    def is_best(self, score, epoch):
        """Check if the current score is the best."""
        if score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.log_file.write(f"New best score: {score} at epoch {epoch}\n")
            return True
        return False
        
class Env(BaseEnv):
    """
    Custom environment class that extends the BaseEnv to include additional functionalities.
    """
    def __init__(self, scenario, config_file=None, verbose=True, refresh_rate=1):
        super().__init__(scenario, config_file=config_file, verbose=verbose)
        self.max_total_time = 0
        self.max_total_energy = 0
        self.refresh_rate = refresh_rate
        
def set_seed(seed):
    """Set the random seed for reproducibility across all libraries."""
    # Set Python's random module seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seed
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
        
# Set a random seed for reproducibility.

def update_metrics(logger: Logger, env: Env, config: dict):
    m1 = SuccessRate()
    m2 = AvgLatency()
    ttr = m1.eval(env.logger)
    avg_latency = m2.eval(env.logger)
    avg_power = env.avg_node_power()  # Convert to mW
    
    if "eval" in config and "lambda" in config["eval"]:

        score = (ttr * config["eval"]["lambda"][0] + avg_latency / env.max_total_time * config["eval"]["lambda"][1] + avg_power / env.max_total_energy * config["eval"]["lambda"][2]) * 100
        logger.update_metric("score", score)

    else:
        print(config)
        score = None

    logger.update_metric('TaskThrowRate', ttr *100)
    logger.update_metric('AvgLatency', avg_latency/(1-ttr) if ttr < 1 else np.inf)  # Avoid division by zero
    logger.update_metric("AvgPower", avg_power/(1-ttr) * 1000 if ttr < 1 else np.inf)  # Convert to mW
    
    
    return ttr, avg_latency, avg_power, score
