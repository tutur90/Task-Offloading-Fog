import logging
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import multiprocessing


class _RunFormatter(logging.Formatter):
    """Plain format for this run's structured logger; rich format for all other sources."""
    def __init__(self, run_logger_name: str):
        super().__init__()
        self._run_name = run_logger_name
        self._rich = logging.Formatter("%(levelname)s | %(name)s | %(message)s")

    def format(self, record):
        if record.name == self._run_name:
            return record.getMessage()
        return self._rich.format(record)


class Logger:
    """
    Logger creates a unique log directory and writes log messages immediately to a log.txt file.
    It stores logged data (rows with Epoch, Mode, Metric, Value) internally, prints each update,
    and provides methods to plot the results and save them to a CSV.
    """
    def __init__(self, config):
        """
        Initializes the Logger with the given configuration.

        Args:
            config (dict): The configuration dictionary. Must contain:
                - env: with keys "dataset" and "flag"
                - policy: the policy name (string)
                - training: training parameters (e.g., num_epoch, batch_size, lr, gamma, epsilon, etc.)
        """
        self.config = config
        self.dataset = config["env"]["dataset"]
        self.flag = config["env"]["flag"]
        self.policy = config["policy"]
        self.training_config = config.get("training", {})
        
        # Create log directory in the form: logs/<dataset>/<flag>/<policy>/<timestamp>_<tuned_params>
        self.log_dir = self.create_log_dir(
            self.dataset, self.flag, self.policy,
            worker_id=config.get("worker_id"),
            tuned_params=config.get("tuned_params", {}),
        )
        self.log_file_path = os.path.join(self.log_dir, "log.txt")
        self.csv_file_path = os.path.join(self.log_dir, "result.csv")

        # Single file + stream handler shared by all loggers for this run.
        # _RunFormatter uses plain format for structured output, rich format for policy logs.
        fmt = _RunFormatter(self.log_dir)
        self._file_handler = logging.FileHandler(self.log_file_path, mode="w")
        self._file_handler.setFormatter(fmt)
        self._stream_handler = logging.StreamHandler()
        self._stream_handler.setFormatter(fmt)

        # Named logger for structured (epoch/mode/metric) output — does not propagate.
        self._logger = logging.getLogger(self.log_dir)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.addHandler(self._file_handler)
        self._logger.addHandler(self._stream_handler)

        # Route 'policies' namespace to the same handlers (rich format via _RunFormatter).
        _policies_logger = logging.getLogger("policies")
        _policies_logger.setLevel(logging.INFO)
        _policies_logger.propagate = False
        _policies_logger.addHandler(self._file_handler)
        _policies_logger.addHandler(self._stream_handler)

        self._write_header()
        
        # Use rows as the sole storage for logged data.
        # Each row is a dictionary with keys: "Epoch", "Mode", "Metric", "Value".
        self.rows = []
        self.current_epoch = None
        self.current_mode = None
        
        self.best_epoch = 0
        self.best_score = np.inf  # Initialize to positive infinity for minimization tasks.

    @staticmethod
    def create_log_dir(dataset, flag, policy, worker_id=None, tuned_params=None):
        """
        Creates a unique log directory with the format:
            logs/<dataset>/<flag>/<policy>/<timestamp>_t<id>_<param_tags>
        e.g. logs/Topo4MEC/25N50E/MLP/0216_143022_t3_dm128_nl3

        Args:
            dataset (str): Dataset name.
            flag (str): Flag name.
            policy (str): The policy name.
            worker_id (int, optional): Trial/worker number.
            tuned_params (dict, optional): Only the tuned hyperparameters
                (e.g. {"model.d_model": 128, "model.n_layers": 3}).
        """
        base_dir = os.path.join("logs", dataset, flag, policy)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        timestamp = datetime.now().strftime("%m%d_%H%M%S")

        # Build tag from only the tuned params: first 2 letters + value
        # e.g. worker_id=3, d_model=128, n_layers=3 -> "t3_dm128_nl3"
        tag_parts = []
        if worker_id is not None:
            tag_parts.append(f"t_{worker_id}")
        for k, v in (tuned_params or {}).items():
            short = k.split(".")[-1].replace("_", "")[:2]
            tag_parts.append(f"{short}{v}")

        tag = "_".join(tag_parts) if tag_parts else ""
        dir_name = f"{timestamp}_{tag}" if tag else timestamp
        log_dir = os.path.join(base_dir, dir_name)

        os.makedirs(log_dir)
        return log_dir

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    def _write_header(self):
        """
        Writes header information (configuration details) to the log file.
        """
        header = f"{self._now()}\n"

        for key, value in self.config.items():
            if isinstance(value, dict):
                header += f"{key}:\n"
                for k, v in value.items():
                    header += f"    {k}: {v}\n"
            else:
                header += f"{key}: {value}\n"

        self._logger.info(header)

    def update_epoch(self, epoch):
        """
        Updates the current epoch and writes a header line to the log file.

        Args:
            epoch (int): The current epoch (0-indexed; will be logged as 1-indexed).
        """
        self.current_epoch = epoch + 1
        self._logger.info(f"\n{self._now()} Epoch {self.current_epoch}/{self.training_config['num_epochs']}")

    def update_mode(self, mode):
        """
        Updates the current mode (e.g., "Training" or "Testing") and writes it to the log file.

        Args:
            mode (str): The current mode.
        """
        self.current_mode = mode
        self._logger.info(f"   Mode: {mode}")

    def update_metric(self, metric, value):
        """
        Logs a metric value under the current epoch and mode. If current_epoch or current_mode
        is None, an empty string is stored instead. The logged value is stored internally,
        written to the log file, printed, and appended as a row for CSV export (immediately).

        Args:
            metric (str): The metric name.
            value (float): The metric value.
        """
        
        if metric is None or value is None:
            return
        
        # Use empty string if current_epoch or current_mode is not set.
        epoch_val = self.current_epoch if self.current_epoch is not None else ""
        mode_val = self.current_mode if self.current_mode is not None else ""
        
        row = {
            "Epoch": epoch_val,
            "Mode": mode_val,
            "Metric": metric,
            "Value": value
        }
        self.rows.append(row)
        self._logger.info(f"       {metric}: {value:.3e}")
        # Write this row immediately to CSV.
        self._append_to_csv(row)

    def _append_to_csv(self, row):
        """
        Appends a single row to the CSV file. If the CSV file is empty, writes the header first.
        """
        file_exists = os.path.exists(self.csv_file_path) and os.path.getsize(self.csv_file_path) > 0
        with open(self.csv_file_path, "a", newline="") as csvfile:
            fieldnames = ["Epoch", "Mode", "Metric", "Value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def plot(self, display=False, excluded_modes=[], excluded_metrics=[], dpi=400,
            log_eps: float | dict | None = {'TaskDropRate': 1.,
                                            'AvgLatency': 1., 
                                            'AvgPower': 1.,
                                            'Score': 1.,
                                            'PopTaskDropRate': 1.,
                                            'PopAvgLatency': 1.,
                                            'PopAvgPower': 1.,}, 
            metric_groups = [
                ['TaskDropRate', 'AvgLatency', 'AvgPower'],
                ['PopTaskDropRate', 'PopAvgLatency', 'PopAvgPower'],
                ['AvgSigmaW', 'AvgSigmaB'],
                ['CDMean', 'CDMean+Std', 'CDMean-Std', 'CDMin'],
                ['AvgLoss', 'AvgGradNorm', 'AvgReward'],
            ]):
        """
        Plots the logged metrics over epochs.

        Layout: one row per group, one column per metric in that group.
        All modes (Train, Val, …) are overlaid as separate lines on the same subplot.

        Args:
            display (bool): Whether to call plt.show().
            excluded_modes (list): Modes to skip entirely.
            excluded_metrics (list): Metrics to skip entirely.
            log_eps (float | dict[str, float] | None): log(x + eps) scale configuration.
                - None: linear for all metrics (default).
                - float: apply log(x + eps) with that eps to all metrics.
                - dict[str, float]: per-metric eps, e.g.
                    {'AvgLoss': 1e-8, 'AvgLatency': 1e-4}
                  Metrics not listed stay linear.
            metric_groups (list[list[str]]): Ordered groups of metric names.
                Metrics present in the data but not in any group are collected into
                an extra group appended at the end.
                Defaults to:
                    [['TaskDropRate', 'AvgLatency', 'AvgPower'],
                     ['AvgLoss', 'AvgGradNorm']]
        """

        df = pd.DataFrame(self.rows)

        try:
            df['Epoch_num'] = pd.to_numeric(df['Epoch'], errors='raise')
        except Exception:
            df['Epoch_num'] = df.groupby(["Mode", "Metric"]).cumcount() + 1

        modes = [m for m in df['Mode'].unique() if m not in excluded_modes]
        if not modes:
            print("No modes to plot after excluding specified modes.")
            return

        all_metrics = [m for m in df['Metric'].unique() if m not in excluded_metrics]
        if not all_metrics:
            print("No metrics to plot after excluding specified metrics.")
            return

        # Build actual groups: only keep metrics present in the data.
        grouped = set()
        actual_groups = []
        for group in metric_groups:
            present = [m for m in group if m in all_metrics]
            if present:
                actual_groups.append(present)
                grouped.update(present)

        # Remaining metrics: each gets its own individual subplot row.
        for m in all_metrics:
            if m not in grouped:
                actual_groups.append([m])

        if not actual_groups:
            print("No metrics to plot.")
            return

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for gi, group in enumerate(actual_groups):
            num_cols = len(group)
            fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 4), squeeze=False)
            
            

            for j, metric in enumerate(group):
                ax = axes[0][j]
                for k, mode in enumerate(modes):
                    subset = df[(df['Mode'] == mode) & (df['Metric'] == metric)]
                    if subset.empty:
                        continue
                    
                    size = 40/max(subset['Epoch_num'].max(), 40)
                    ax.plot(subset['Epoch_num'], subset['Value'],
                            marker='o', label=mode, color=colors[k % len(colors)], markersize=6*size, linewidth=1.5*size)
                ax.set_title(metric)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                eps = log_eps.get(metric) if isinstance(log_eps, dict) else log_eps
                if eps is not None:
                    mmin = subset['Value'].min()
                    if mmin <= 0:
                        print(f"Warning: Minimum value for metric '{metric}' is {mmin:.3e}, which is <= 0. ")
                    ax.set_yscale('function', functions=(lambda x: np.log(x + eps), lambda y: np.exp(y) - eps))
                ax.legend()

            plt.tight_layout()
            plot_path = os.path.join(self.log_dir, f"score_plot_{gi}.png")
            plt.savefig(plot_path, dpi=dpi)
            if display:
                plt.show()
            plt.close(fig)
        

    def save_csv(self):
        """
        Saves the logged results to a CSV file in the log directory.
        """
        df = pd.DataFrame(self.rows)
        df.to_csv(self.csv_file_path, index=False)

    def close(self):
        """
        Closes the log file.
        """
        
        self._logger.info(f"\n{self._now()} Best Epoch: {self.best_epoch+1}, Best Value: {self.best_score:.4f}")
        _policies_logger = logging.getLogger("policies")
        _policies_logger.removeHandler(self._file_handler)
        _policies_logger.removeHandler(self._stream_handler)
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)
