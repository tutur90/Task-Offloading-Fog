import pandas as pd
from tqdm import tqdm

from core.task import Task
from core.env import Env
from utils.utils import create_env, error_handler
from eval.metrics.metrics import SuccessRate, AvgLatency


def run_epoch(config, policy, data: pd.DataFrame, train=True, 
              lambda_=(1, 1, 1), max_total_time=0, max_total_energy=0,
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
    update_freq = config.get("training", {}).get("update_freq", 32)

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
                update_freq -= 1

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
                        env.max_total_energy = env.max_total_energy*0.999 + total_energy*0.001
                        env.max_total_time = env.max_total_time*0.999 + total_time*0.001

                        reward = - ((lambda_[1] * total_time/env.max_total_time) + (lambda_[2] * total_energy/env.max_total_energy))
                    else:
                        reward = -lambda_[0]
                        
                    reward = reward * config["training"].get("reward_scale", 1.0)
                    policy.store_transition(state, action, reward, next_state, done)
                    del stored_transitions[task_id]
            # Update the policy every update_freq tasks during training.
            if update_freq < 1:
                r1 = m1.eval(env.logger) * 100  # Convert to percentage
                r2 = m2.eval(env.logger)
                e = env.avg_node_power()
                pbar.set_postfix({"SR": f"{r1:.3f}", "L": f"{r2:.3f}", "E": f"{e:.3f}"})
                policy.update()
                update_freq = config.get("training", {}).get("update_freq", 32)
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
