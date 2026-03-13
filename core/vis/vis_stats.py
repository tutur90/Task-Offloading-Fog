import os
import re
from core.env import Env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SUCCESS = 0

_PREFIX_ORDER = {'c': 0, 'f': 1, 'e': 2}

def _node_rank(name: str) -> int:
    """Scalar sort rank for a node name (c < f < e, then numeric index)."""
    m = re.match(r'([a-zA-Z]+)(\d+)', str(name))
    if m:
        return _PREFIX_ORDER.get(m.group(1), 99) * 100000 + int(m.group(2))
    return 99 * 100000

def _node_sort_key(series: pd.Series) -> pd.Series:
    """key= for sort_values on a Series of node names."""
    return series.map(_node_rank)

def _link_sort_key(series: pd.Series) -> pd.Series:
    """key= for sort_values on a Series of 'src-->dst' link strings."""
    def link_rank(link: str) -> int:
        parts = str(link).split('-->')
        src = _node_rank(parts[0].strip()) if len(parts) > 0 else 99 * 100000
        dst = _node_rank(parts[1].strip()) if len(parts) > 1 else 99 * 100000
        return src * 10 ** 7 + dst
    return series.map(link_rank)

class VisStats:
    def __init__(self, save_path: str, display_numbers: bool = False, log_eps: float | None = None):
        """
        Initialize with a path where figures will be saved.

        Parameters:
            save_path (str): Directory where plots will be saved.
            display_numbers (bool): Whether to annotate bars with their numeric values.
            log_eps (float | None): If set, applies log(x + eps) y-axis scale to all
                bar charts. None means linear scale (default).
                Example: log_eps=1e-8
        """
        self.save_path = os.path.join(save_path, "figs")
        os.makedirs(self.save_path, exist_ok=True)
        self.task_info = {}
        self.node_info = {}
        self.display_numbers = display_numbers
        self.log_eps = log_eps

    def get_stats(self, env: Env):
        """Extract task and node statistics from the environment."""
        task_list = []
        
        # Get index values for various logged attributes.
        statue_code_idx = env.logger.get_value_idx("status_code")
        node_names_idx = env.logger.get_value_idx("node_names")
        time_list_idx = env.logger.get_value_idx("time_list")
        energy_list_idx = env.logger.get_value_idx("energy_list")
        
        for task_id, val in env.logger.task_info.items():
            src_name, dst_name = val[node_names_idx]
            if val[statue_code_idx] == SUCCESS:
                transmission_time = val[time_list_idx][0]  # Task transmission time.
                wait_time = val[time_list_idx][1]          # Task wait time.
                execution_time = val[time_list_idx][2]
                total_time = sum(val[time_list_idx])       # Sum of transmission, wait, and execution times.
                status = 'SUCCESS'
            else:
                transmission_time = 0
                wait_time = 0
                execution_time = 0
                total_time = 0
                status = val[time_list_idx][0]  # Error code.
                
            transmission_energy = val[energy_list_idx][0]
            execution_energy = val[energy_list_idx][1]
            total_energy = sum(val[energy_list_idx])
            
            task_list.append([task_id, src_name, dst_name, f'{src_name}-->{dst_name}', status,
                              transmission_time, wait_time, execution_time, total_time,
                              transmission_energy, execution_energy, total_energy])
        self.task_info = pd.DataFrame(task_list, 
                                      columns=['Task ID', 'Source', 'Destination', 'Link', 'Status',
                                               'Trans Time', 'Wait Time', 'Exe Time', 'Time',
                                               'Trans Energy', 'Exe Energy', 'Energy'])

        # Process node information.
        node_list = []
        for node_id, val in env.logger.node_info.items():
            node_name = env.scenario.node_id2name[node_id]
            energy, cpu_freq, clock = val
            max_cpu_freq = env.scenario.get_node(node_name).max_cpu_freq
            node_list.append([node_name, clock, energy, cpu_freq, max_cpu_freq])
        self.node_info = pd.DataFrame(node_list,
                                      columns=['Node Name', 'Clock', 'Energy', 'CPU Freq', 'Max CPU Freq'])\
                                     .sort_values(by='Node Name', key=_node_sort_key).reset_index(drop=True)

    def _apply_log_scale(self, ax):
        """Apply log(x + eps) y-axis scale if log_eps is set."""
        if self.log_eps is not None:
            eps = self.log_eps
            ax.set_yscale('function', functions=(lambda x: np.log(x + eps), lambda y: np.exp(y) - eps))

    def vis(self, env: Env, dpi: int = 400):
        """Generate and save several visualizations based on the current environment stats."""
        self.get_stats(env)

        # Helper function to annotate bars in an axis.
        def annotate_bars(ax, fmt="{:.0f}", fontsize=10):
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(fmt.format(height),
                            (p.get_x() + p.get_width() / 2., height),
                            ha="center", va="bottom", fontsize=fontsize)

        # 1. Bar chart: Total tasks vs. successful tasks per link (overlay style).
        task_counts = self.task_info.groupby('Link')['Status'].agg(
            Total='size',
            Success=lambda x: (x == 'SUCCESS').sum()
        ).reset_index().sort_values(by='Link', key=_link_sort_key).reset_index(drop=True)
        f, ax = plt.subplots(figsize=(10, 6))
        plt.xticks(rotation=45, fontsize=10)
        sns.barplot(x="Link", y="Total", data=task_counts, label="Total", color="lightgray", ax=ax)
        sns.barplot(x="Link", y="Success", data=task_counts, label="Success", color="red", ax=ax)
        self._apply_log_scale(ax)
        if self.display_numbers:
            annotate_bars(ax, fmt="{:.0f}")
        sns.despine(left=True, bottom=True)
        ax.set_title('Task Offloading Statistics')
        ax.legend()
        plt.tight_layout()
        f.savefig(os.path.join(self.save_path, 'task_offloading_statistics.png'), dpi=dpi)
        plt.close(f)

        # 2. Pie chart: Distribution of error types.
        errors = self.task_info[self.task_info['Status'] != 'SUCCESS']
        error_counts = errors.groupby('Status').size()
        f, ax = plt.subplots(figsize=(8, 8))
        ax.pie(error_counts, labels=error_counts.index, autopct='%1.1f%%')
        ax.set_title('Type of Errors')
        plt.tight_layout()
        f.savefig(os.path.join(self.save_path, 'error_distribution.png'), dpi=dpi)
        plt.close(f)

        # 3. Bar chart: Average latency per link (for successful tasks).
        latency = self.task_info[self.task_info['Status'] == 'SUCCESS']\
                      .groupby('Link')[['Trans Time', 'Wait Time', 'Exe Time', 'Time']].mean()\
                      .reset_index().sort_values(by='Link', key=_link_sort_key).reset_index(drop=True)
        latency_melt = latency.melt(id_vars='Link', var_name='Latency Type', value_name='Average')
        f, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=latency_melt, x='Link', y='Average', hue='Latency Type', ax=ax)
        self._apply_log_scale(ax)
        ax.set_title('Average Latency per Link')
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        if self.display_numbers:
            annotate_bars(ax, fmt="{:.1f}")
        f.savefig(os.path.join(self.save_path, 'avg_latency_per_link.png'), dpi=dpi)
        plt.close(f)

        # 4. Bar chart: Energy consumption per node.
        energy = self.task_info.groupby('Destination')[['Trans Energy', 'Exe Energy']].sum()\
                     .reset_index().sort_values(by='Destination', key=_node_sort_key).reset_index(drop=True)

        energy = energy.merge(self.node_info, left_on='Destination', right_on='Node Name', suffixes=('_task', '_node'))
        energy['Idle Energy'] = energy['Energy'] - energy["Trans Energy"] - energy["Exe Energy"]
        energy_melt = energy[['Node Name', 'Trans Energy', 'Exe Energy', 'Idle Energy', 'Energy']].melt(
            id_vars='Node Name', var_name='Energy Type', value_name='Total')

        f, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=energy_melt, x='Node Name', y='Total', hue='Energy Type', ax=ax)
        self._apply_log_scale(ax)
        ax.set_title('Energy Consumption per Node')
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        f.savefig(os.path.join(self.save_path, 'energy_consumption_per_node.png'), dpi=dpi)
        plt.close(f)

        # 5. Bar chart: Power consumption per node (Power = Energy/Clock).
        # First, include the clock in the melt so we can compute power per energy type.
        
        energy_melt = energy[['Node Name', 'Clock', 'Trans Energy', 'Exe Energy', 'Idle Energy', 'Energy']].melt(
            id_vars=['Node Name', 'Clock'], var_name='Energy Type', value_name='Total')
        energy_melt['Power'] = energy_melt['Total'] / energy_melt['Clock']
        f, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=energy_melt, x='Node Name', y='Power', hue='Energy Type', ax=ax)
        self._apply_log_scale(ax)
        ax.set_title('Power Consumption per Node')
        ax.set_ylabel('Power (Energy / Clock)')
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        if self.display_numbers:
            annotate_bars(ax, fmt="{:.1f}")
        f.savefig(os.path.join(self.save_path, 'power_consumption_per_node.png'), dpi=dpi)
        plt.close(f)
        
        energy_melt = energy[['Node Name', 'Clock', 'Trans Energy', 'Exe Energy', 'Idle Energy', 'Energy']].melt(
            id_vars=['Node Name', 'Clock'], var_name='Energy Type', value_name='Total')
        energy_melt['Power'] = energy_melt['Total'] / energy_melt['Clock']
        
        for node_name in energy_melt['Node Name'].unique():

            mask = energy_melt['Node Name'] == node_name
            energy_melt.loc[mask, "Power"] = energy_melt.loc[mask, "Power"] / (self.task_info['Destination'] == node_name).sum()

        f, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=energy_melt, x='Node Name', y='Power', hue='Energy Type', ax=ax)
        self._apply_log_scale(ax)
        ax.set_title('Power Consumption per Node')
        ax.set_ylabel('Power (Energy / Clock)')
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        if self.display_numbers:
            annotate_bars(ax, fmt="{:.1f}")
        f.savefig(os.path.join(self.save_path, 'power_consumption_per_node_per_task.png'), dpi=dpi)
        plt.close(f)

        # 5b. Bar chart: Avg active power per successfully offloaded task per node.
        #     Active power = (Trans Energy + Exe Energy) / Clock / num_successful_tasks
        success_counts = self.task_info[self.task_info['Status'] == 'SUCCESS']\
                             .groupby('Destination').size().rename('SuccessCount').reset_index()
        power_per_task = energy[['Node Name', 'Clock', 'Trans Energy', 'Exe Energy']]\
                             .merge(success_counts, left_on='Node Name', right_on='Destination', how='left')
        power_per_task['SuccessCount'] = power_per_task['SuccessCount'].fillna(0)
        power_per_task['Trans Power / Task'] = power_per_task['Trans Energy'] / power_per_task['Clock'] / power_per_task['SuccessCount'].replace(0, float('nan'))
        power_per_task['Exe Power / Task']   = power_per_task['Exe Energy']   / power_per_task['Clock'] / power_per_task['SuccessCount'].replace(0, float('nan'))
        power_per_task_melt = power_per_task[['Node Name', 'Trans Power / Task', 'Exe Power / Task']]\
                                  .melt(id_vars='Node Name', var_name='Power Type', value_name='Power / Task')
        f, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=power_per_task_melt, x='Node Name', y='Power / Task', hue='Power Type', ax=ax)
        self._apply_log_scale(ax)
        ax.set_title('Avg Active Power per Successfully Offloaded Task')
        ax.set_ylabel('Power / Task')
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        if self.display_numbers:
            annotate_bars(ax, fmt="{:.2e}")
        f.savefig(os.path.join(self.save_path, 'avg_power_per_successful_task.png'), dpi=dpi)
        plt.close(f)

        # 6. Bar chart: CPU frequency per node (overlay style).
        f, ax = plt.subplots(figsize=(10, 6))
        plt.xticks(rotation=45, fontsize=10)
        # Normalize CPU frequency by clock.
        self.node_info['CPU Freq'] = self.node_info['CPU Freq'] / self.node_info['Clock']
        sns.barplot(x="Node Name", y="Max CPU Freq", data=self.node_info, label="Max CPU Freq", color="lightgray", ax=ax)
        sns.barplot(x="Node Name", y="CPU Freq", data=self.node_info, label="CPU Freq", color="red", ax=ax)
        self._apply_log_scale(ax)
        if self.display_numbers:
            annotate_bars(ax, fmt="{:.0f}")
        sns.despine(left=True, bottom=True)
        ax.set_title('CPU Frequency per Node')
        ax.legend()
        plt.tight_layout()
        f.savefig(os.path.join(self.save_path, 'cpu_frequency_per_node.png'), dpi=dpi)
        plt.close(f)

        # 7. Bar chart: Percent CPU frequency per node.
        self.node_info['Percent CPU Freq'] = (self.node_info['CPU Freq'] / self.node_info['Max CPU Freq']) * 100
        
        f, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=self.node_info, x='Node Name', y='Percent CPU Freq', ax=ax)
        self._apply_log_scale(ax)
        ax.set_title('Percent CPU Frequency per Node')
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        if self.display_numbers:
            annotate_bars(ax, fmt="{:.1f}")
        f.savefig(os.path.join(self.save_path, 'percent_cpu_frequency_per_node.png'), dpi=dpi)
        plt.close(f)

        print(f"Figures saved in {self.save_path}")
