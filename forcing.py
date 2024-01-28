from abc import ABC, abstractmethod

class Forcing(ABC):
    def __init__(self, init_time, fin_time, *args, **kwargs):
        self.init_time = init_time
        self.fin_time = fin_time # inclusive
        return

class ImpulsiveForcing(Forcing):
    def __init__(self, init_time, fin_time, impulse_times, impulses):
        super().__init__(init_time, fin_time)
        self.impulse_times = impulse_times
        self.impulses = impulses
        return


class AdditiveTendencyForcing(Forcing):
    def __init__(self, init_time, fin_time, params):
        super().__init__(init_time, fin_time)
        self.

