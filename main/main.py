"""
This script demonstrates how to run the DQRLPolicy.

Oh, wait a moment. It seems that extra effort is required to make this method work. The current version 
is for reference only, and contributions are welcome.
"""

import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from tqdm import tqdm
import yaml


from core.task import Task
from core.vis import *
from core.vis.vis_stats import VisStats

from eval.metrics.metrics import SuccessRate, AvgLatency
from policies.dqrl.mlp_policy import MLPPolicy
from policies.dqrl.taskformer_policy import TaskFormerPolicy
from policies.heuristics.greedy import GreedyPolicy
from policies.heuristics.random import  RandomPolicy
from policies.heuristics.round_robin import RoundRobinPolicy


import numpy as np

from utils import create_env, error_handler, set_seed, update_metrics
from utils import Logger, Checkpoint



def run_epoch(config, policy, data: pd.DataFrame, train=True, lambda_=(1, 1, 1
                                                                       ), max_total_time=0, max_total_energy=0,
              ):
    """
    Run one simulation epoch over the provided task data.
    lambda_ = (fail, time, energy) if time is more important than energy, then lambda_ = (_, 1, 0) and vice versa.

    For each task:
      - Wait until the task's generation time.
      - Obtain the current state and select an action via the policy.
      - Schedule the task for processing.
      - Once processed, record the next state and compute the reward.
      - Store the transition for policy training.
      
    Every 'batch_size' tasks, update the policy.
    """

    m1 = SuccessRate()
    m2 = AvgLatency()
    
    env = create_env(config)
    
    until = 0
    launched_task_cnt = 0
    last_task_id = None
    pbar = tqdm(data.iterrows(), total=len(data))
    stored_transitions = {}
    number_in_batch = config.get("training", {}).get("batch_size", 32)

    env.max_total_time = max_total_time
    env.max_total_energy = max_total_energy

    for i, task_info in pbar:
        generated_time = task_info['GenerationTime']
        task = Task(task_id=task_info['TaskID'],
                    task_size=task_info['TaskSize'],
                    cycles_per_bit=task_info['CyclesPerBit'],
                    trans_bit_rate=task_info['TransBitRate'],
                    ddl=task_info['DDL'] ,
                    src_name=task_info['SrcName'] if 'SrcName' in task_info else 'e0',
                    task_name=task_info['TaskName'])

        # Wait until the simulation reaches the task's generation time.
        while True:
            while env.done_task_info:
                item = env.done_task_info.pop(0)
            
            if env.now >= generated_time:
                # Get action and current state from the policy.
                action, state = policy.act(env, task, train=train)
                dst_name = env.scenario.node_id2name[action]
                env.process(task=task, dst_name=dst_name)
                launched_task_cnt += 1
                number_in_batch -= 1

                # Update previous transition with the new state's observation.
                if last_task_id is not None and train:
                    prev_state, prev_action, _ = stored_transitions[last_task_id]
                    stored_transitions[last_task_id] = (prev_state, prev_action, state)
                
                break
            
            until += env.refresh_rate
            
            try:
                env.run(until=until)
            except Exception as e:
                # print(f"Error: {e}")
                error_handler(e)
                
        

            
        if train:
            done = False  # Each task is treated as an individual episode.
            last_task_id = task.task_id
            stored_transitions[last_task_id] = (state, action, None)
            
            # Process stored transitions if the task has been completed.
            for task_id, (state, action, next_state) in list(stored_transitions.items()):
                if task_id in env.logger.task_info:
                    val = env.logger.task_info[task_id]
                    if val[0] == 0:
                        task_trans_time, task_wait_time, task_exe_time = val[2]
                        total_time = task_trans_time + task_wait_time + task_exe_time
                        task_trans_energy, task_exe_energy = val[3]
                        total_energy = task_trans_energy + task_exe_energy
                        # env.max_total_time = max(env.max_total_time, total_time)
                        # env.max_total_energy = max(env.max_total_energy, total_energy)
                        reward = - ((lambda_[1] * total_time/env.max_total_time) + (lambda_[2] * total_energy/env.max_total_energy))
                    else:
                        reward = -lambda_[0]
                        
                    reward = reward * config["training"].get("reward_scale", 1.0)
                    policy.store_transition(state, action, reward, next_state, done)
                    del stored_transitions[task_id]
            # Update the policy every batch_size tasks during training.
            if number_in_batch < 1:
                r1 = m1.eval(env.logger) * 100  # Convert to percentage
                r2 = m2.eval(env.logger)
                e = env.avg_node_power()
                pbar.set_postfix({"SR": f"{r1:.3f}", "L": f"{r2:.3f}", "E": f"{e:.3f}"})
                policy.update()
                number_in_batch = np.random.randint(config["training"]["batch_size"]//2, config["training"]["batch_size"])
                # print(f"Policy updated at task {i}, next update in {number_in_batch} tasks.")
                
    if train and stored_transitions:
        policy.update()

    # Continue simulation until all tasks are processed.
    while env.task_count < launched_task_cnt:
        until += env.refresh_rate
        try:
            env.run(until=until)
        except Exception as e:
            error_handler(e)
            
    return env

def train(config, policy,  train_data, valid_data, logger, checkpoint, max_total_energy=0, max_total_time=0):
    """ Train the policy using the provided training data and validate it using the validation data. """
    for epoch in range(config["training"]["num_epochs"]):

        logger.update_epoch(epoch)

        # Training phase.
        
        logger.update_mode('Training')

        env = run_epoch(config, policy, train_data, train=True, lambda_=config["training"]["lambda"], max_total_time=max_total_time, max_total_energy=max_total_energy)

        update_metrics(logger, env, config)
        
        max_total_time = env.max_total_time
        max_total_energy = env.max_total_energy

        env.close()
        

        
        # Validation phase.

        logger.update_mode('Validation')
        

        env = run_epoch(config, policy, valid_data, train=False)
        
        env.max_total_energy = max_total_energy
        env.max_total_time = max_total_time

        score = update_metrics(logger, env, config)
        
        if logger.is_best(score[3], epoch):
            checkpoint.save(policy, epoch)

        env.close()

        policy.epsilon *= config["training"]["epsilon_decay"]
        
        for param_group in policy.optimizer.param_groups:
            param_group['lr'] *= config["training"]["lr_decay"]
        if config["algo"] == "TaskFormer":
            # Decay the learning rate of the TaskFormer model.
            for param_group in policy.optimizer.param_groups:
                param_group['lr'] *= config["training"]["lr_decay"]

    return max_total_energy, max_total_time

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run DQRL Policy")
    parser.add_argument('--config', type=str, default='configs/DQRL/MLP.yaml', help='Path to the config file.')
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()
    
    
    # config_name = "MLP"
    # config_name = "MLP"  # or "TaskFormer"
    config_name = "Heuristics/Greedy"  # or "Random", "RoundRobin"
    config_name = "Heuristics/RoundRobin"
    # config_name = "DQRL/NOTE-S"
    # config_name = "DQRL/TaskFormer-S" 
    # config_name = "DQRL/NodeFormer-S"
    # config_name = "DQRL/MLP" 

    config_path = f"main/configs/Pakistan/{config_name}.yaml"

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    set_seed(config.get("seed", 42))

    logger = Logger(config)



    env = create_env(config)
    
    
    if "training" in config.keys():
        
        checkpoint = Checkpoint(logger.log_dir)

        valid_size = config["training"].get("valid_size", 0.2)

        # Load train and test datasets.
        train_data = pd.read_csv(f"eval/benchmarks/{config['env']['dataset']}/data/{config['env']['flag']}/trainset.csv")
        train_data, valid_data = train_data.iloc[:int(len(train_data)*(1-valid_size))], train_data.iloc[int(len(train_data)*(1-valid_size)):]
        valid_data["GenerationTime"] = valid_data["GenerationTime"] - valid_data["GenerationTime"].min()  # Normalize generation time
        
    test_data = pd.read_csv(f"eval/benchmarks/{config['env']['dataset']}/data/{config['env']['flag']}/testset.csv")
    
    #         # Load train and test datasets.
    # train_data = pd.read_csv(f"eval/benchmarks/Topo4MEC/data/25N50E/trainset.csv")
    # test_data = pd.read_csv(f"eval/benchmarks/Topo4MEC/data/25N50E/testset.csv")

    if config["algo"] == "MLP":
        policy = MLPPolicy(env=env, config=config)
    elif config["algo"] == "TaskFormer":
        policy = TaskFormerPolicy(env=env, config=config)
    elif config["algo"] == "Greedy":
        policy = GreedyPolicy()
    elif config["algo"] == "Random":
        policy = RandomPolicy()
    elif config["algo"] == "RoundRobin":
        policy = RoundRobinPolicy()
    else:
        raise ValueError("Invalid policy name.")

    max_total_time = config.get("eval", {}).get("expected_max_latency", 0)
    max_total_energy = config.get("eval", {}).get("expected_max_energy", 0)


    if "training" in config.keys():
        max_total_energy, max_total_time = train(config, policy, train_data, valid_data, logger, checkpoint, max_total_energy, max_total_time)
        checkpoint.load(policy, logger.best_epoch)
        
    print(f"Max total energy: {max_total_energy}, Max total time: {max_total_time}")

    logger.update_mode('Testing')
    env = run_epoch(config, policy, test_data, train=False)
    
    env.max_total_energy = max_total_energy
    env.max_total_time = max_total_time
    
    update_metrics(logger, env, config)


    logger.plot()
    logger.save_csv()
    

    
    logger.close()
    env.close()
    
    vis_stats = VisStats(save_path=logger.log_dir)
    vis_stats.vis(env)


if __name__ == '__main__':
    main()
