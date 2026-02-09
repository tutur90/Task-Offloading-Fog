from policies.base_policy import BasePolicy


class RoundRobinPolicy(BasePolicy):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.idx = 0

    def act(self, env, task, **kwargs):
        """
        Selects the next node in a round-robin fashion.
        """
        self.idx = (self.idx + 1) % len(env.scenario.get_nodes())
        return self.idx, None
