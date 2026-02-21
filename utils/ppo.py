import pandas as pd
from tqdm import tqdm

from core.task import Task
from core.env import Env
from utils.utils import create_env, error_handler
from eval.metrics.metrics import get_metrics
from policies.ppo.tpto_policy import TPTOPolicy


def _flush_pending(policy: TPTOPolicy, env: Env, pending: dict, rollout: list, config: dict):
    """Move completed pending transitions into the rollout buffer with their rewards."""
    for task_id, entry in list(pending.items()):
        if task_id in env.logger.task_info and entry["next_state"] is not None:
            val = env.logger.task_info[task_id]
            if val[0] == 0:
                latency = sum(val[2])
                energy  = sum(val[3])
                tdr     = 0
            else:
                latency = 0
                energy  = 0
                tdr     = 1

            reward = -policy.norm_reward([tdr, latency, energy], config["training"]["lambda"])
            entry["reward"] = reward
            rollout.append(entry)
            del pending[task_id]


def run_epoch_ppo(config: dict, policy: TPTOPolicy, data: pd.DataFrame, train: bool = True):
    """
    On-policy PPO training loop.  Mirrors utils/dql.py::run_epoch but collects
    rollouts for PPO updates instead of filling a DQN replay buffer.

    Returns the finished Env instance (same interface as run_epoch).
    """
    env = create_env(config)

    rollout_size   = config["training"].get("rollout_size", 512)
    log_freq       = config.get("training", {}).get("log_freq", 200)
    disp_progress  = log_freq > 0

    until              = 0
    launched_task_cnt  = 0
    last_task_id       = None

    rollout_buffer: list = []   # completed transitions ready for PPO update
    pending: dict        = {}   # task_id → transition entry awaiting completion

    pbar = tqdm(data.iterrows(), total=len(data)) if disp_progress else data.iterrows()

    for i, task_info in pbar:
        generated_time = task_info["GenerationTime"]
        task = Task(
            task_id=task_info["TaskID"],
            task_size=task_info["TaskSize"],
            cycles_per_bit=task_info["CyclesPerBit"],
            trans_bit_rate=task_info["TransBitRate"],
            ddl=task_info["DDL"],
            src_name=task_info.get("SrcName", "e0"),
            task_name=task_info["TaskName"],
        )

        # Advance simulation until task's generation time
        while True:
            # Drain any done-task notifications
            while env.done_task_info:
                env.done_task_info.pop(0)

            if env.now >= generated_time:
                action, log_prob, value, state = policy.act(env, task, train=train)
                dst_name = env.scenario.node_id2name[action]
                env.process(task=task, dst_name=dst_name)
                launched_task_cnt += 1

                # Attach the new observation as next_state for the previous task
                if last_task_id is not None and train:
                    pending[last_task_id]["next_state"] = state

                break

            until += env.refresh_rate
            try:
                env.run(until=until)
            except Exception as e:
                error_handler(e)

        if train:
            last_task_id = task.task_id
            pending[task.task_id] = {
                "state":      state,
                "action":     action,
                "log_prob":   log_prob,
                "value":      value,
                "reward":     0.0,
                "next_state": None,
                "done":       False,
            }
            _flush_pending(policy, env, pending, rollout_buffer, config)

            # PPO update when enough data is collected
            if len(rollout_buffer) >= rollout_size:
                policy.update(rollout_buffer)
                rollout_buffer.clear()

        if disp_progress and i % log_freq == 0:
            tdr, avg_latency, avg_energy, score = get_metrics(env, config)
            pbar.set_description(
                f"TTR: {tdr*100:.3e} - L: {avg_latency:.3e} - E: {avg_energy:.3e} - S: {score:.3e}"
            )

    # Wait for all launched tasks to complete
    while env.task_count < launched_task_cnt:
        until += env.refresh_rate
        try:
            env.run(until=until)
        except Exception as e:
            error_handler(e)

    if train:
        _flush_pending(policy, env, pending, rollout_buffer, config)
        if rollout_buffer:
            policy.update(rollout_buffer)

    return env
