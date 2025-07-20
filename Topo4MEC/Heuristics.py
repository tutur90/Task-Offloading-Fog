"""
This script demonstrates how to use the Pakistan dataset.
"""

import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd

from core.env import Env
from core.task import Task
from core.vis import *
from core.vis.vis_stats import VisStats
from eval.benchmarks.Topo4MEC.scenario import Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency  # metric
from policies.demo.demo_greedy import GreedyPolicy
from policies.demo.demo_random import DemoRandom
from policies.demo.demo_round_robin import RoundRobinPolicy

from utils import create_env, error_handler, set_seed, update_metrics
from utils import Logger, Checkpoint


import yaml


# Global statistics for different error types
dup_task_id_error = []
net_no_path_error = []
isolated_wireless_node_error = []
net_cong_error = []
insufficient_buffer_error = []

def error_handler(error: Exception):
    """Customized error handler for different types of errors."""
    errors = ['DuplicateTaskIdError', 'NetworkXNoPathError', 'IsolatedWirelessNode', 'NetCongestionError', 'InsufficientBufferError']
    message = error.args[0][0]
    if message in errors:
        pass
    else:
        raise

def main():
    
    config_name = "Greedy"  # Change this to the desired policy name

    config_path = f"Topo4MEC/configs/Heuristics/{config_name}.yaml"

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize the logger.
    logger = Logger(config)
    
    # Create the environment.
    env = create_env(config)
    
    
    # Load the test dataset.
    flag = config["env"]["flag"]
    dataset = config["env"]["dataset"]
    data = pd.read_csv(f"eval/benchmarks/{dataset}/data/{flag}/testset.csv")
        # Load train and test datasets.
    # data = pd.read_csv(f"eval/benchmarks/Topo4MEC/data/25N50E/testset.csv")

    # Init the policy.
    if config["algo"] == "DemoGreedy":
        policy = GreedyPolicy()
    elif config["algo"] == "DemoRandom":
        policy = DemoRandom()
    elif config["algo"] == "DemoRoundRobin":
        policy = RoundRobinPolicy()
    else:
        raise ValueError("Invalid policy name.")

    # Begin the simulation.
    until = 0
    launched_task_cnt = 0
    
    
    
    for i, task_info in data.iterrows():
        generated_time = task_info['GenerationTime']
        task = Task(task_id=task_info['TaskID'],
                    task_size=task_info['TaskSize'],
                    cycles_per_bit=task_info['CyclesPerBit'],
                    trans_bit_rate=task_info['TransBitRate'],
                    ddl=task_info['DDL'],
                    src_name=task_info['SrcName'],
                    task_name=task_info['TaskName'])

        while True:
            # Catch completed task information.
            while env.done_task_info:
                item = env.done_task_info.pop(0)
            
            if env.now >= generated_time:
                dst_id = policy.act(env, task)  # offloading decision
                dst_name = env.scenario.node_id2name[dst_id]
                env.process(task=task, dst_name=dst_name)
                launched_task_cnt += 1
                break

            # Execute the simulation with error handler.
            until += env.refresh_rate
            try:
                env.run(until=until)
            except Exception as e:
                error_handler(e)



    # Continue the simulation until the last task successes/fails.
    while env.task_count < launched_task_cnt:
        until += env.refresh_rate
        try:
            env.run(until=until)
        except Exception as e:
            pass

    # Evaluation
    print("\n===============================================")
    print("Evaluation:")
    print("===============================================\n")


    # Update metrics.
    update_metrics(logger, env, config)
    
    # Stats Visualization
    vis = VisStats(logger.log_dir)
    vis.vis(env)
    
    env.close()


if __name__ == '__main__':
    main()
