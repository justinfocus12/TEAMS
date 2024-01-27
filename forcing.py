from abc import ABC, abstractmethod

class Forcing:
    def __init__(self, init_time, term_time, *args, **kwargs):
        self.init_time = init_time
        self.term_time = term_time # inclusive
        return

class ImpulsiveForcing(Forcing):
    def __init__(self, init_time, term_time, impulse_times, impulses):
        super().__init__(init_time, term_time)
        self.impulse_times = impulse_times
        self.impulses = impulses
        return

class AdditiveTendencyForcing(Forcing):
    def __init__(self, init_time, term_time, tendency_pert, tendency_type):
        super().__init__(init_time, term_time)
        self.tendency_pert = tendency_pert
        self.tend

