"""Example on how to use the Topo4MEC dataset.
"""
import os
import sys
import decimal

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd

from core.env import Env
from core.task import Task
from core.vis import *
from core.utils import create_log_dir
from core.vis.vis_stats import VisStats
from core.utils import save_results

from eval.benchmarks.Pakistan.scenario import Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency  # metric
from policies.demo.demo_random import DemoRandom  # policy
from policies.demo.demo_greedy import GreedyPolicy  # policy
from policies.demo.demo_roundrobin import RoundRobinPolicy  # policy

from core.vis.plot_score import PlotScore

def main():
    flag = 'Tuple30K'
    # flag = 'Tuple50K'
    # flag = 'Tuple100K'
    
    refresh_rate = 0.001
    
    algo_name = "Greedy"
    
    # Create the Env
    scenario=Scenario(config_file=f"eval/benchmarks/Pakistan/data/{flag}/config.json", flag=flag)
    env = Env(scenario, config_file="core/configs/env_config_null.json", refresh_rate=refresh_rate, verbose=False, dec_place=3)

    # Load the test dataset
    data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{flag}/testset.csv")
    # Init the policy
    if algo_name == 'Random':
        policy = DemoRandom(env)
    elif algo_name == 'Greedy':
        policy = GreedyPolicy(env)
    elif algo_name == 'RoundRobin':
        policy = RoundRobinPolicy(env)

    
    path_dir = create_log_dir(algo_name, flag=flag)
    
    # Begin Simulation
    until = 0
    for i, task_info in data.iterrows():
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
                dst_name = env.scenario.node_id2name[dst_id]
                env.process(task=task, dst_name=dst_name)
                break

            # Execute the simulation with error handler
            try:
                env.run(until=until)
            except Exception as e:
                pass

            until += refresh_rate

    # Continue the simulation until the last task successes/fails.
    while env.process_task_cnt < len(data):
        until += refresh_rate
        try:
            env.run(until=until)
        except Exception as e:
            pass

    # Evaluation
    print("\n===============================================")
    print("Evaluation:")
    print("===============================================\n")

    print("-----------------------------------------------")
    m1 = SuccessRate()
    r1 = m1.eval(env.logger.task_info)
    print(f"The success rate of all tasks: {r1:.4f}")
    print("-----------------------------------------------\n")

    print("-----------------------------------------------")
    m2 = AvgLatency()
    r2 = m2.eval(env.logger.task_info)
    print(f"The average latency per task: {r2:.4f}")
    
    e = env.avg_node_energy()
    print(f"The average energy consumption per node: {e:.4f}")
    print("-----------------------------------------------\n")

    env.close()
    
    # Stats Visualization
    vis = VisStats(path_dir)
    vis.vis(env)
    
    # Save the results
    save_results(path_dir, {'SuccessRate': r1, 'AvgLatency': r2, 'EnergyConsumption': e}, algo_name=algo_name, flag=flag)
    
    


if __name__ == '__main__':
    main()
