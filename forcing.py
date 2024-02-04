from abc import ABC, abstractmethod

# TODO add plotting methods for each kind of forcing, e.g., overlay the impulse times atop a trajectory timeseries

class Forcing(ABC):
    def __init__(self, init_time, fin_time, *args, **kwargs):
        self.init_time = init_time
        self.fin_time = fin_time # inclusive
        return
    @abstractmethod
    def get_forcing_times(self):
        pass

class ImpulsiveForcing(Forcing):
    def __init__(self, impulse_times, impulses, fin_time):
        super().__init__(impulse_times[0], fin_time)
        self.impulse_times = impulse_times
        self.impulses = impulses
        return
    def get_forcing_times(self):
        return self.impulse_times

class WhiteNoiseForcing(Forcing):
    def __init__(self, reseed_times, seeds, fin_time):
        super().__init__(reseed_times[0], fin_time)
        self.reseed_times = reseed_times
        self.seeds = seeds
        return
    def get_forcing_times(self):
        return self.reseed_times

#class AdditiveTendencyForcing(Forcing):
#    def __init__(self, init_time, fin_time, params):
#        super().__init__(init_time, fin_time)
#        self.

