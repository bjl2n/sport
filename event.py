# noinspection PyShadowingBuiltins
class Event:
    """
    The Event class defines an object that holds events that will be placed on the
    simulation calendar and executed at some point in the future.
    An event is defined by the objct, method, args, and occurrence_time.
    @param objct: The object that has an event occur.
    @type method: str
    @param method: The method of the object.
    @type args: list
    @param args: A list of arguments to the method.
    @type occurrence_time: float
    @param occurrence_time: The occurrence time of the event.
    """
    # EVENT_METHODS is a dictionary containing all methods that can be placed on
    # the simulation calendar.  The value in each (key, value) pair is the
    # priority of the method relative to other methods in the dictionary.  If
    # two methods have the same priority then whichever was put on the simulation
    # calendar first will occur first.
    EVENT_METHODS = {'child_node_daily_demand': -1,
                     'demand_event': -1,
                     'remove_stale_demand_event': -1,
                     'arrive_at_child_node': 0,
                     'arrive_at_parent_node': 0,
                     'tanker_finished_unloading': 0,
                     'tanker_finished_loading': 0,
                     'daily_FSP_update': 0.1,
                     'load_tanker': 1,
                     'unload_tanker': 1,
                     'tanker_to_unloading_queue': 2.1,
                     'tanker_to_available_queue': 2.1,
                     'daily_plan': 2.2,
                     'execute_loaded_tanker_TMR': 2.4,
                     'execute_unloaded_tanker_TMR': 2.4,
                     'submit_loading_plan': 2.4,
                     'calculate_lead_time': 2.5}

    def __init__(self, objct, method, args, occurrence_time):
        """The Event class defines an object that holds events that will be placed on the
        simulation calendar and executed at some point in the future.
        An event is defined by the object, method, args, and occurrence_time.
        @param objct: The object that has an event occur.
        @type method: str
        @param method: The method of the object.
        @type args: list
        @param args: A list of arguments to the method.
        @type occurrence_time: float
        @param occurrence_time: The occurrence time of the event.
        """
        self.object = objct
        self.method = method
        self.args = args
        self.occurrence_time = occurrence_time

    def __str__(self):
        """
        @rtype: str
        """
        return '%8.3f: %s - %s\n' \
               % \
               (self.occurrence_time, self.object.log_label,
                self.method)

    def __cmp__(self, other):
        """Custom comparison used to order events in the simulation
        calendar.  Events are first sorted by occurrence_time.
        If one or more events have the same occurrence_time then
        self.method_comp is used to sort those events.
        @type other: event.Event
        @rtype: int
        """
        if self.occurrence_time < other.occurrence_time:
            return -1
        if self.occurrence_time > other.occurrence_time:
            return 1

        # At this point occurrence_time is equal, so we need to
        # differentiate the events based on the method called
        return self.method_comp(self.method, other.method)

    def method_comp(self, method_one, method_two):
        """Use the ordering defined in the EVENT_METHODS dictionary to order two events
        (defined by methods) that have the same occurrence_time.  Both methods must
        be present in the EVENT_METHODS dictionary.
        @type method_one: str
        @type method_two: str
        @rtype: int
        """
        assert (method_one in self.EVENT_METHODS) and (method_two in self.EVENT_METHODS), \
            'One of %s or %s has not been defined in event.EVENT_METHODS\n' % \
            (method_one, method_two)

        if self.EVENT_METHODS[method_one] < self.EVENT_METHODS[method_two]:
            return -1
        if self.EVENT_METHODS[method_one] > self.EVENT_METHODS[method_two]:
            return 1
        return 0

    def handle_event(self):
        """Execute object.method(args) at occurrence_time.
        """
        getattr(self.object, self.method)(*self.args)
