from abc import ABC, abstractmethod

class Forcing(ABC):
    def __init__(self, init_time, fin_time, *args, **kwargs):
        self.init_time = init_time
        self.fin_time = fin_time # inclusive
        return

class ImpulsiveForcing(Forcing):
    def __init__(self, impulse_times, impulses, fin_time):
        super().__init__(impulse_times[0], fin_time)
        self.impulse_times = impulse_times
        self.impulses = impulses
        return

class WhiteNoiseForcing(Forcing):
    def __init__(self, init_time, fin_time, reseed_times, seeds):
        super().__init__(init_time, fin_time)
        assert reseed_times[0] == init_time
        self.reseed_times = reseed_times
        self.seeds = seeds
        return

#class AdditiveTendencyForcing(Forcing):
#    def __init__(self, init_time, fin_time, params):
#        super().__init__(init_time, fin_time)
#        self.

