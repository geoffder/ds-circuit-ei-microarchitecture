from neuron import h
import numpy as np


class NetQuanta:
    def __init__(self, syn, weight, delay=0.0):
        self.con = h.NetCon(None, syn, 0.0, 0.0, weight)
        self.delay = delay  # NetCon does not impose delay on *scheduled* events
        self._events = []

    @property
    def weight(self):
        return self.con.weight[0]

    @property
    def events(self):
        return self._events

    @weight.setter
    def weight(self, w):
        self.con.weight[0] = w

    @events.setter
    def events(self, ts):
        self._events = [t + self.delay for t in ts]

    def clear_events(self):
        self._events = []

    def add_event(self, t):
        self._events.append(t + self.delay)

    def add_quanta(self, quanta, dt, t0, jitters=None):
        i = len(self._events)
        self._events += [None] * max(0, np.sum(quanta))
        jitters = np.zeros(len(quanta)) if jitters is None else jitters
        t = t0 + self.delay
        for n, j in zip(quanta, jitters):
            jittered = t + j
            for _ in range(n):
                self._events[i] = jittered  # type:ignore
                i += 1
            t += dt

    def initialize(self):
        """Schedule events in the NetCon object. This must be called within the
        function given to h.FInitializeHandler in the model running class/functions.
        """
        for ev in self._events:
            self.con.event(ev)
