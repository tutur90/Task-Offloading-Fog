
import os
import sys
from tqdm import tqdm
import time

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd

from core.env import Env
from core.task import Task
from core.vis import *


from eval.benchmarks.Pakistan.scenario import Scenario

from eval.metrics.metrics import SuccessRate, AvgLatency  # metric
from policies.drl_policy import DRLPolicy
# from policies.dqrl_policy import DQRLPolicy


num_epoch = 20

batch_size = 256

def run_epoch(env, policy, data, refresh_rate=1, train=True):   
    
    m1 = SuccessRate()
    
    m2 = AvgLatency()
            
    last_task_id = 0
    until = 0
    pbar = tqdm(data.iterrows(), total=len(data))
    for i, task_info in pbar:

        # header = ['TaskName', 'GenerationTime', 'TaskID', 'TaskSize', 'CyclesPerBit', 
        #           'TransBitRate', 'DDL', 'SrcName']  # field names
        generated_time = task_info['GenerationTime']
        task = Task(task_id=task_info['TaskID'],
                    task_size=task_info['TaskSize'],
                    cycles_per_bit=task_info['CyclesPerBit'],
                    trans_bit_rate=task_info['TransBitRate'],
                    ddl=task_info['DDL'],
                    src_name='e0',
                    task_name=task_info['TaskName'],
                    )

        while True:
            # Catch the returned info of completed tasks
            while env.done_task_info:
                item = env.done_task_info.pop(0)
                # print(f"[{item[0]}]: {item[1:]}")

            if env.now >= generated_time:
                dst_id = policy.act(env, task)  # offloading decision
                
                # print(f"Task {task.task_id} is offloaded to {env.scenario.node_id2name[dst_id]}")

                dst_name = env.scenario.node_id2name[dst_id]
                env.process(task=task, dst_name=dst_name)
                time.sleep(0.1)
                # print(dst_name)
                break

            # Execute the simulation with error handler
            try:
                env.run(until=until)
            except Exception as e:
                pass

            until += refresh_rate
            
        if (i+1) % batch_size == 0 and train:
            # print(f"Training batch {i//batch_size}...")

            keys = list(env.logger.task_info.keys())
            
            new_keys = {key: env.logger.task_info[key] for key in keys[last_task_id+1:]}
            
            last_task_id = len(keys)-1
            
            r1 = m1.eval(new_keys)
            r2 = m2.eval(new_keys)
            
            pbar.set_postfix({"AvgLatency": f"{r2:.3f}", "SuccessRate": f"{r1:.3f}"})

            policy.backward(r2)
            

    # Continue the simulation until the last task successes/fails.
    while env.process_task_cnt < len(data):
        until += refresh_rate
        try:
            env.run(until=until)
        except Exception as e:
            pass
    
    return env

def create_env(scenario):
    env = Env(scenario, config_file="core/configs/env_config_null.json", refresh_rate=1, verbose=False)
    return env
    

def main():

    flag = 'Tuple30K'
    # flag = 'Tuple50K'
    # flag = 'Tuple100K'
    
    refresh_rate = 0.001
    
    # Create the Env
    scenario=Scenario(config_file=f"eval/benchmarks/Pakistan/data/{flag}/config.json", flag=flag)
    env = create_env(scenario)

    # Load the test dataset
    test_data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{flag}/testset.csv")
    train_data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{flag}/trainset.csv")

    # Init the policy
    # policy = RandomPolicy()
    header = [
            # 'TaskName', 'GenerationTime', 'TaskID', 
              'TaskSize', 
              'CyclesPerBit', 
              'TransBitRate', 
              'DDL', 
            #   'SrcName'
              ]
    
    policy = DRLPolicy(env=env, lr=1e-4)
    
    
    m1 = SuccessRate()
    m2 = AvgLatency()

    # Train the policy
    
    for epoch in range(num_epoch):
        
        print(f"Epoch {epoch+1}/{num_epoch}")

        env = create_env(scenario)
        
        env = run_epoch(env, policy, train_data, refresh_rate,)
        

        r2 = m2.eval(env.logger.task_info)
        
        

        print(f"Training: The average latency per task: {r2:.4f}")
        print(f"Training: The success rate of all tasks: {m1.eval(env.logger.task_info):.4f}")

        env.close()
        
        env = create_env(scenario)
        
        env = run_epoch(env, policy, test_data, train=False)

        r2 = m2.eval(env.logger.task_info)
        r1 = m1.eval(env.logger.task_info)
        
        
        print(f"Testing: The average latency per task: {r2:.4f}")
        print(f"Testing: The success rate of all tasks: {r1:.4f}")

        print("===============================================")
        
        env.close()


if __name__ == '__main__':
    main()