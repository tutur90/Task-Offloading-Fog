import pandas as pd
from tqdm import tqdm

from core.task import Task
from core.env import Env
from utils.utils import create_env, error_handler
from eval.metrics.metrics import SuccessRate, AvgLatency, AvgEnergy, get_metrics
from policies.dql.base_policy import DQNPolicy

def update_transitions(policy: DQNPolicy, env: Env, stored_transitions: dict, config: dict):
    
    done = False  # Each task is treated as an individual episode.

    for task_id, (state, action, next_state) in list(stored_transitions.items()):
        if task_id in env.logger.task_info and next_state is not None:
            val = env.logger.task_info[task_id]
            if val[0] == 0:
                latency = sum(val[2])
                energy = sum(val[3])
                tdr = -1
                
            else:
                latency = None
                energy = None
                tdr = 1
                
            reward = policy.norm_reward([tdr, latency, energy], config["training"]["lambda"])
                
            policy.store_transition(state, action, reward, next_state, done)
            del stored_transitions[task_id]
    # Update the policy every update_freq tasks during training.
            loss = policy.update()
            


def run_epoch(config: dict, policy: DQNPolicy, data: pd.DataFrame,      train=True,  ):
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

    
    env = create_env(config)
    
    log_freq = config.get("training", {}).get("log_freq", 200)
    disp_progress = log_freq > 0
    
    until = 0
    launched_task_cnt = 0
    last_task_id = None
    pbar = tqdm(data.iterrows(), total=len(data)) if disp_progress else data.iterrows()
    stored_transitions = {}

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
            
            update_transitions(policy, env, stored_transitions, config)
            
        if  disp_progress and i % log_freq == 0:
            
            tdr, avg_latency, avg_energy, score = get_metrics(env, config)
            pbar.set_description(f"TTR: {tdr*100:.3e} - L: {avg_latency:.3e} - E: {avg_energy:.3e} - S: {score:.3e}")
                # print(f"Policy updated at task {i}, next update in {number_in_batch} tasks.")
                

    # Continue simulation until all tasks are processed.
    while env.task_count < launched_task_cnt:
        until += env.refresh_rate
        try:
            env.run(until=until)
        except Exception as e:
            error_handler(e)
            
    update_transitions(policy, env, stored_transitions, config)
    
    return env
