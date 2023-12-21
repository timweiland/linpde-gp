from ._policy import ConcretePolicy, Policy


class ConcreteSwitchPolicy(ConcretePolicy):
    def __init__(
        self,
        gp_params,
        first_policy: ConcretePolicy,
        second_policy: ConcretePolicy,
        switch_iter: int,
    ):
        self._gp_params = gp_params
        self._first_policy = first_policy
        self._second_policy = second_policy
        self._switch_iter = switch_iter

    def __call__(self, solver_state, rng=None):
        if solver_state.iteration < self._switch_iter:
            return self._first_policy(solver_state, rng=rng)
        else:
            return self._second_policy(solver_state, rng=rng)


class SwitchPolicy(Policy):
    """
    A policy that switches between two other policies based on a specified iteration.

    Args:
        first_policy (Policy): The first policy to use before the switch.
        second_policy (Policy): The second policy to use after the switch.
        switch_iter (int): The iteration at which to switch between the two policies.
    """

    def __init__(self, first_policy: Policy, second_policy: Policy, switch_iter: int):
        self._first_policy = first_policy
        self._second_policy = second_policy
        self._switch_iter = switch_iter

    def get_concrete_policy(self, gp_params, **kwargs):
        first_concrete_policy = self._first_policy.get_concrete_policy(
            gp_params, **kwargs
        )
        second_concrete_policy = self._second_policy.get_concrete_policy(
            gp_params, **kwargs
        )
        return ConcreteSwitchPolicy(
            gp_params, first_concrete_policy, second_concrete_policy, self._switch_iter
        )
