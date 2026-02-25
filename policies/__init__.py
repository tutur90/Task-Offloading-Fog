from policies.dql.note_policy import NOTEPolicy, DuelingNOTEPolicy
from policies.dql.mlp_policy import MLPPolicy, DuelingMLPPolicy
from policies.dql.base_policy import DQNPolicy
from policies.dql.opo_policy import OPOPolicy
from policies.dql.nate_policy import NATEPolicy
from policies.dql.natd_policy import NATDPolicy


from policies.ga.nsga_policy import NSGA2Policy
from policies.ga.npga_policy import NPGAPolicy

from policies.heuristics.random import RandomPolicy
from policies.heuristics.greedy import GreedyPolicy
from policies.heuristics.round_robin import RoundRobinPolicy

from policies.ppo.tpto_policy import TPTOPolicy


policies = {
    "MLP": MLPPolicy,
    "T-MLP": MLPPolicy,
    
    "DuelingMLP": DuelingMLPPolicy,
    
    "NOTE": NOTEPolicy,
    "DuelingNOTE": DuelingNOTEPolicy,
    "T-NOTE": NOTEPolicy,
    
    "NATE": NATEPolicy,
    
     "NATD": NATDPolicy,
    
    "OPO": OPOPolicy,
    "NPGA": NPGAPolicy,
    "NSGA2": NSGA2Policy,
    "Greedy": GreedyPolicy,
    "Random": RandomPolicy,
    "RoundRobin": RoundRobinPolicy,
    "TPTO": TPTOPolicy,
}