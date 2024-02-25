from abc import ABC, abstractmethod
from functools import reduce
import numpy as np

# TODO add plotting methods for each kind of forcing, e.g., overlay the impulse times atop a trajectory timeseries

class Forcing(ABC):
    def __init__(self, init_time, fin_time, *args, **kwargs):
        self.init_time = init_time
        self.fin_time = fin_time # inclusive
        return
    @abstractmethod
    def get_forcing_times(self):
        pass

class OccasionalVectorForcing(Forcing):
    # For when the integrator transforms a vector of real numbers into a perturbation at a point in time
    def __init__(self, init_time, fin_time, force_times, forces):
        assert len(force_times) == len(forces)
        if len(forces) > 0:
            assert init_time <= min(force_times)
            assert max(force_times) < fin_time
        self.force_times = force_times
        self.forces = forces
        super().__init__(init_time, fin_time)
        return
    def get_forcing_times(self):
        return self.force_times

class OccasionalReseedForcing(Forcing):
    # For when the integrator performs its own random number generation based on input seeds
    # For when the integrator transforms a vector of real numbers into a perturbation at a point in time
    def __init__(self, init_time, fin_time, reseed_times, seeds):
        assert len(reseed_times) == len(seeds)
        if len(seeds) > 0:
            assert init_time <= min(reseed_times)
            assert max(reseed_times) < fin_time
        self.reseed_times = reseed_times
        self.seeds = seeds
        super().__init__(init_time, fin_time)
        return
    def get_forcing_times(self):
        return self.reseed_times

class SuperposedForcing(Forcing):
    def __init__(self,frc_list):
        self.frc_list = frc_list
        init_time = min([f.init_time for f in frc_list])
        fin_time = max([f.fin_time for f in frc_list])
        self.forcing_times = list(np.sort(reduce(np.union1d, [f.get_forcing_times() for f in self.frc_list])))
        super().__init__(init_time, fin_time)
        return
    def get_forcing_times(self):
        return self.forcing_times


# Need a composite forcing, a summation of a white noise and impulsive term

#class ImpulsiveForcing(Forcing):
#    def __init__(self, impulse_times, impulses, fin_time):
#        super().__init__(impulse_times[0], fin_time)
#        self.impulse_times = impulse_times
#        self.impulses = impulses
#        return
#    def get_forcing_times(self):
#        return self.impulse_times
#    def __str__(self):
#        s = f'timespan = ({self.init_time},{self.fin_time}); imptimes = {self.impulse_times}'
#        return s
#
#class WhiteNoiseForcing(Forcing):
#    def __init__(self, reseed_times, seeds, fin_time):
#        super().__init__(reseed_times[0], fin_time)
#        self.reseed_times = reseed_times
#        self.seeds = seeds
#        return
#    def get_forcing_times(self):
#        return self.reseed_times
#
#class ContinuousTimeForcing(Forcing):
#    def __init__(self, init_time, fin_time, reseed_times, seeds):
#        assert len(reseed_times) == len(seeds)
#        if len(reseed_times) > 0:
#            assert (min(reseed_times) >= init_time) and (max(reseed_times) < fin_time)
#        super().__init__(init_time, fin_time)
#        self.reseed_times = reseed_times
#        self.seeds = seeds
#        return
#    def get_forcing_times(self):
#        return self.reseed_times
#
#
#class MultiplicativeTendencyForcing(Forcing):
#    # Multiply an ODE tendency by a number between 0 and 1
#    def __init__(self, reseed_times, seeds, fin_time):
#        super().__init__(reseed_times[0], fin_time)
#        self.reseed_times = reseed_times
#        self.seeds = seeds
#        return
#    def get_forcing_times(self):
#        return self.reseed_times



#class AdditiveTendencyForcing(Forcing):
#    def __init__(self, init_time, fin_time, params):
#        super().__init__(init_time, fin_time)
#        self.

