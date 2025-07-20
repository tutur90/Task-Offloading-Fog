from ..base_policy import BasePolicy

import random


class RandomPolicy(BasePolicy):

    def act(self, env, task, **kwargs):
        return random.randint(0, len(env.scenario.get_nodes()) - 1)
