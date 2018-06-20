import bisect
import event as ev


class SimulationTime:
    """Used to keep track of the current simulation time.
    """
    def __init__(self, time=0.0):
        """Used to keep track of the current simulation time.
        """
        self.time = time

    def __str__(self):
        """
        @rtype: str
        """
        return '%8.4f' % self.time


class SimulationCalendar:
    """Simulation calendar that gets populated with events.
    """
    def __init__(self, calendar=None):
        """Simulation calendar that gets populated with events.
        """
        if calendar is None:
            calendar = []
        self.calendar = calendar
        """@type : list[event.Event]"""
        self.cal_event_trace = []  # SIM_STATS ('Time', 'Name', 'Method')
        """@type : list[(float, str, str)]"""

    def __str__(self):
        """
        @rtype: str
        """
        cal = ''
        for evnt in self.calendar:
            cal += evnt.__str__()

        return cal

    def add_event(self, objct, method, args, occurrence_time):
        """Inserts the event into the (already sorted) simulation
        calendar.  The custom sort defined in the Event class is
        used.
        @type method: str
        @type args: list
        @type occurrence_time: float
        @param objct: The object that has an event occur
        @param method: The method of the object
        @param args: A list of arguments to the method
        @param occurrence_time: The occurrence time of the event
        """
        evnt = ev.Event(objct, method, args, occurrence_time)
        bisect.insort(self.calendar, evnt)

    def handle_event(self, scn, sim_time):
        """Updates sim_time.time to reflect the current simulation
        time (at which the event occurs), and removes the event from
        the simulation calendar.
        @type scn: scenario.Scenario
        @type sim_time: sim_engine.SimulationTime
        """
        sim_time.time = self.calendar[0].occurrence_time

        next_event = self.calendar.pop(0)

        if scn.rep_analysis_lvl in ['all']:
            self.cal_event_trace.append((next_event.occurrence_time,  # SIM_STATS
                                         next_event.object.log_label,  # SIM_STATS
                                         next_event.method))  # SIM_STATS
        next_event.handle_event()
