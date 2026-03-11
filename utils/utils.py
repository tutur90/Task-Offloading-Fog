from core.env import Env as BaseEnv
from core.logger import Logger as BaseLogger
from core.base_scenario import BaseScenario
import random
import torch
from eval.metrics.metrics import SuccessRate, AvgLatency, AvgEnergy, get_metrics

import os
import numpy as np

class Scenario(BaseScenario):
    
    def __init__(self, config_file, dataset, flag):
        """
        :param flag: '25N50E', '50N50E', '100N150E' or 'MilanCityCenter'
        """
        assert dataset in ['Topo4MEC', 'Pakistan', 'Synthetic'], f"Invalid dataset={dataset}"
        if dataset == 'Topo4MEC':
            assert flag in ['25N50E', '50N50E', '100N150E', 'MilanCityCenter'], \
            f"Invalid flag={flag}"
        # elif dataset == 'Pakistan':
            # assert flag in ['Tuple30K', 'Tuple50K', 'Tuple100K'], \
            # f"Invalid flag={flag}"
        # elif dataset == 'Synthetic':
        #     assert flag in ['small', 'medium', 'large'], \
        #     f"Invalid flag={flag}"
        super().__init__(config_file)
        
        # # Load the test dataset (not recommended)
        # data = pd.read_csv(f"{ROOT_PATH}/{flag}/testset.csv")
        # self.testset = list(data.iloc[:].values)
    
    def status(self):
        pass

class Checkpoint:
    """A simple class to manage checkpoints."""

    def __init__(self, path, keep_last_n=1):
        self.path = os.path.join(path, "checkpoints")
        self.keep_last_n = keep_last_n  # None means keep all
        self._saved_epochs = []
        os.makedirs(self.path, exist_ok=True)

    def save(self, policy, epoch):
        policy.save(os.path.join(self.path, f"checkpoint_epoch_{epoch}.pt"))
        self._saved_epochs.append(epoch)
        if self.keep_last_n is not None:
            while len(self._saved_epochs) > self.keep_last_n:
                old_epoch = self._saved_epochs.pop(0)
                for ext in (".pt", ".npz"):
                    old_path = os.path.join(self.path, f"checkpoint_epoch_{old_epoch}{ext}")
                    if os.path.exists(old_path):
                        os.remove(old_path)

    def load(self, policy, epoch):
        policy.load(os.path.join(self.path, f"checkpoint_epoch_{epoch}.pt"))



def create_env(config):
    """Create and return an environment instance."""
    dataset = config["env"]["dataset"]
    flag = config["env"]["flag"]
    scenario = Scenario(config_file=f"eval/benchmarks/{dataset}/data/{flag}/config.json", 
                        dataset=dataset, flag=flag)
    env = Env(scenario, config_file="core/configs/env_config_null.json", verbose=False, refresh_rate=config["env"].get("refresh_rate", 1))

    if "eval" in config and "expected_max_latency" in config["eval"]:
        env.max_total_time = config["eval"]["expected_max_latency"]
    if "eval" in config and "expected_max_energy" in config["eval"]:
        env.max_total_energy = config["eval"]["expected_max_energy"]
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

        os.makedirs(self.log_dir, exist_ok=True)
        self.best_epoch = 0
        self.best_score = np.inf
        self.early_stopping_counter = 0

    def is_best(self, score, epoch):
        """Check if the current score is the best."""
        if score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self._logger.info(f"New best score: {score} at epoch {epoch+1}")
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
    # 
    
    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
        
# Set a random seed for reproducibility.



def update_metrics(logger: Logger, env: Env, config: dict, metrics=None):

    if metrics is None:
        ttr, avg_latency, avg_power, score = get_metrics(env, config)
    else:
        ttr, avg_latency, avg_power, score = metrics

    logger.update_metric('TaskDropRate', ttr * 100)
    logger.update_metric('AvgLatency', avg_latency)
    logger.update_metric("AvgPower", avg_power)
    logger.update_metric("Score", score)

    return ttr, avg_latency, avg_power, score
