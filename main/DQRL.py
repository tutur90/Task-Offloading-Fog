import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np

# Set up the module search path.
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.env import Env
from core.task import Task
from core.vis import *
from core.utils import create_log_dir
from core.utils import save_results   
from core.vis.vis_stats import VisStats

from eval.benchmarks.Pakistan.scenario import Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency
from policies.dqrl_policy import DQRLPolicy
from policies.qtrans_policy import QTransPolicy
from policies.qrnn_policy import QRNNPolicy
from core.vis.plot_score import PlotScore

# Global parameters
num_epoch = 40
batch_size = 256

def run_epoch(scenario, policy, data: pd.DataFrame, refresh_rate=1, train=True):
    """
    Run one simulation epoch over the provided task data.

    For each task:
      - Wait until the task's generation time.
      - Obtain the current state and select an action via the policy.
      - Schedule the task for processing.
      - Once processed, record the next state and compute the reward.
      - Store the transition for policy training.
      
    Every 'batch_size' tasks, update the policy.
    """
    
    env = create_env(scenario, refresh_rate=refresh_rate)
    
    m1 = SuccessRate()
    m2 = AvgLatency()
    last_task_id = None
    until = 0
    pbar = tqdm(data.iterrows(), total=len(data))
    stored_transitions = {}

    for i, task_info in pbar:
        generated_time = task_info['GenerationTime']
        task = Task(task_id=task_info['TaskID'],
                    task_size=task_info['TaskSize'],
                    cycles_per_bit=task_info['CyclesPerBit'],
                    trans_bit_rate=task_info['TransBitRate'],
                    ddl=task_info['DDL'],
                    src_name='e0',
                    task_name=task_info['TaskName'])

        # Wait until the simulation reaches the task's generation time.
        while True:
            while env.done_task_info:
                env.done_task_info.pop(0)
            if env.now >= generated_time:
                # Get action and current state from the policy.
                action, state = policy.act(env, task, eval=not train)
                dst_name = env.scenario.node_id2name[action]
                env.process(task=task, dst_name=dst_name)

                # Update previous transition with the new state's observation.
                if last_task_id is not None and train:
                    prev_state, prev_action, _ = stored_transitions[last_task_id]
                    stored_transitions[last_task_id] = (prev_state, prev_action, state)
                break
            try:
                env.run(until=until)
            except Exception:
                pass
            until += refresh_rate
            
        if train:
            # Compute the reward and store the transition.

            done = False  # Each task is treated as an individual episode.
            last_task_id = task.task_id
            stored_transitions[last_task_id] = (state, action, None)
            
            # print(env.logger.task_info)

            # Process stored transitions if the task has been completed.
            for task_id, (state, action, next_state) in list(stored_transitions.items()):
                if task_id in env.logger.task_info:
                    val = env.logger.task_info[task_id]
                    if val[0] == 0:
                        task_trans_time, task_wait_time, task_exe_time = val[1]
                        total_time = task_trans_time + task_wait_time + task_exe_time
                        reward = -total_time
                    else:
                        reward = -1e6
                    policy.store_transition(state, action, reward, next_state, done)
                    del stored_transitions[task_id]

            # Update the policy every batch_size tasks during training.
            if (i + 1) % batch_size == 0:
                r1 = m1.eval(env.logger.task_info)
                r2 = m2.eval(env.logger.task_info)
                pbar.set_postfix({"AvgLatency": f"{r2:.3f}", "SuccessRate": f"{r1:.3f}"})
                policy.update()

    # Continue simulation until all tasks are processed.
    while env.process_task_cnt < len(data):
        until += refresh_rate
        try:
            env.run(until=until)
        except Exception:
            pass
    return env

def create_env(scenario, refresh_rate=1):
    """Create and return an environment instance."""
    return Env(scenario, config_file="core/configs/env_config_null.json", refresh_rate=refresh_rate, verbose=False)



def main():
    flag = 'Tuple30K'
    policy_name = 'DQRL'
    
    refresh_rate = 0.001  # Simulation time refresh rate
    scenario = Scenario(config_file=f"eval/benchmarks/Pakistan/data/{flag}/config.json", flag=flag)
    
    # Load train and test datasets.
    train_data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{flag}/trainset.csv")
    test_data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{flag}/testset.csv")
    
    params = {
        "flag": flag,
        "policy_name": policy_name,
        "num_epoch": num_epoch,
        "batch_size": batch_size
    }
    
    
    # Initialize the policy.
    env = create_env(scenario, refresh_rate
                     )
    if policy_name == 'QTrans':
        
        model_params = {
            "lr": 1e-3,
            "epsilon": 0.2,
            "d_model": 64,
            "nhead": 4,
            "n_layers": 6,
            "d_ff": 256
        }
        
        policy = QTransPolicy(env=env, lr=model_params['lr'], epsilon=model_params['epsilon'], d_model=model_params['d_model'], nhead=model_params['nhead'], n_layers=model_params['n_layers'], d_ff=model_params['d_ff'])
    elif policy_name == 'DQRL':
        model_params = {
            "lr": 1e-3,
            "epsilon": 0.2,
            "d_model": 128,
            "include_link_obs": True,
            "gamma": 0
        }
        policy = DQRLPolicy(env=env, lr=model_params['lr'], epsilon=model_params['epsilon'], d_model=model_params['d_model'], incl_link_obs=model_params['include_link_obs'], gamma=model_params['gamma'])
    elif policy_name == 'QRNN':
        model_params = {
            "lr": 1e-3,
            "epsilon": 0.2,
            "hidden_size": 128,
            "gamma": 0.99
        }
        policy = QRNNPolicy(env=env, lr=model_params['lr'], epsilon=model_params['epsilon'], d_model=model_params['hidden_size'], gamma=model_params['gamma'])
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    

    m1 = SuccessRate()
    m2 = AvgLatency()
    
    plotter = PlotScore(metrics=['SuccessRate', 'AvgLatency'], modes=['Training', 'Testing'])
    
    for epoch in range(num_epoch):
        print(f"Epoch {epoch+1}/{num_epoch}")
        
        # Training phase.

        env = run_epoch(scenario, policy, train_data, refresh_rate=refresh_rate, train=True)
        print(f"Training - AvgLatency: {m2.eval(env.logger.task_info):.4f}, SuccessRate: {m1.eval(env.logger.task_info):.4f}")
        plotter.append(mode='Training', metric='SuccessRate', value=m1.eval(env.logger.task_info))
        plotter.append(mode='Training', metric='AvgLatency', value=m2.eval(env.logger.task_info))
        env.close()
        
        policy.update_epsilon(0.98)
        
        # Testing phase.
        env = run_epoch(scenario, policy, test_data, refresh_rate=refresh_rate, train=False)
        print(f"Testing  - AvgLatency: {m2.eval(env.logger.task_info):.4f}, SuccessRate: {m1.eval(env.logger.task_info):.4f}")
        print("===============================================")
        plotter.append(mode='Testing', metric='SuccessRate', value=m1.eval(env.logger.task_info))
        plotter.append(mode='Testing', metric='AvgLatency', value=m2.eval(env.logger.task_info))
        env.close()
        
    
    # Final testing phase.
    print("Final Testing Phase")

    env = run_epoch(scenario, policy, test_data, refresh_rate=refresh_rate, train=False)
    r1 = m1.eval(env.logger.task_info)
    r2 = m2.eval(env.logger.task_info
                    )   
    e = env.scenario.avg_node_energy()
    
    
    print(f"Testing  - AvgLatency: {r2:.4f}, SuccessRate: {r1:.4f}, Energy: {e:.4f}")
    
    env.close()
    
    log_dir = create_log_dir(policy_name, flag=flag, num_epoch=num_epoch, batch_size=batch_size)
    
    
    vis_stats = VisStats(save_path=log_dir)
    vis_stats.vis(env)

    plotter.plot(num_epoch, save_dir=log_dir)
    
    # save_results(log_dir, {
    #     "SuccessRate": m1.eval(env.logger.task_info),
    #     "AvgLatency": m2.eval(env.logger.task_info),
    #     "EnergyConsumption": e
    # }, algo_name=policy_name, flag=flag, num_epoch=num_epoch, batch_size=batch_size)
    
    plotter.save_results(log_dir, params=params, model_params=model_params)
    
    
    

if __name__ == '__main__':
    main()
