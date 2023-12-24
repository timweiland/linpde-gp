from .._solver_state import SolverState

class Logger:
    def __init__(self):
        self._gp_params = None
    
    def start(self, gp_params):
        self._gp_params = gp_params
    
    def __call__(self, solver_state: SolverState):
        raise NotImplementedError()
    
    def finish(self):
        raise NotImplementedError()