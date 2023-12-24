from ._switch import SwitchPolicy, ConcreteSwitchPolicy
from ._cg import CGPolicy
from ._variance_cg import VarianceCGPolicy


class MeanVarianceSwitchPolicy(SwitchPolicy):
    def __init__(self, switch_iter: int):
        first_policy = CGPolicy()
        second_policy = VarianceCGPolicy()
        super().__init__(first_policy, second_policy, switch_iter)
