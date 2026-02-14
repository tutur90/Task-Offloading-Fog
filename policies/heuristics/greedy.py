from core.env import Env
from core.task import Task
from policies.base_policy import BasePolicy

import numpy as np
class GreedyPolicy(BasePolicy):
    """A simple greedy policy that selects the node with the minimal 
    predicted total time (transmission + computation)."""
    def __init__(self, env, config):
        super().__init__(env, config)
        
        self.obs_type = config["obs_type"] if "obs_type" in config else "cpu"
        
    def _make_observation(self, env: Env, task: Task, obs_type=["cpu", "buffer", "bw"]):
        """
        Returns a flat observation vector.
        For instance, we return the free CPU frequency for each node.
        """
        
        
        obs = np.zeros((len(env.scenario.get_nodes()), len(obs_type)), dtype=np.float32)
        
        for i, node_name in enumerate(env.scenario.get_nodes()):
            if "cpu" in obs_type:
                obs[env.scenario.node_name2id[node_name], obs_type.index("cpu")] = env.scenario.get_node(node_name).free_cpu_freq 
            if "buffer" in obs_type:
                obs[env.scenario.node_name2id[node_name], obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw" in obs_type:
                # Get the bandwidth for the link associated with the task
                src_node =  "e0"
                if node_name != src_node:
                    obs[env.scenario.node_name2id[node_name], obs_type.index("bw")] = min(link.free_bandwidth for link in env.scenario.infrastructure.get_shortest_links(src_node, node_name))
                else:
                    obs[env.scenario.node_name2id[node_name], obs_type.index("bw")] = max(link.free_bandwidth for link in env.scenario.infrastructure.get_links().values())


        if task is None:
            task_obs = np.zeros(4, dtype=np.float32)
        else:
            task_obs = np.array([
                task.task_size,
                task.cycles_per_bit,
                task.trans_bit_rate,
                task.ddl,
            ], dtype=np.float32)


        return obs, task_obs

    def act(self, env, task, **kwargs):
        """
        Greedily choose the node that yields the lowest estimated total latency.

        Args:
            env (Env): The environment (for accessing node data, if needed).
            task (Task): The current task to be scheduled/offloaded.

        Returns:
            int: The selected action (index of the chosen node).
        """
        best_node = None
        best_latency = float('inf')
        
        obs = self._make_observation(env, task)[0][:,["cpu", "buffer", "bw"].index(self.obs_type)]  # Get the CPU frequency part of the observation
        
        best_node = np.argmax(obs)  # Select the node with the minimum value in the selected observation type

        return best_node, None
