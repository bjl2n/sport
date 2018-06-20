import numpy as np


# noinspection PyPep8Naming
class LoadingPoint:
    """LoadingPoint is a resource used to load and unload tankers.  Each LoadingPoint
    has a unique name, an upload rate, a download rate, a tanker (if the LoadingPoint
    is currently being used to load or unload a tanker), a last_updated time, a
    finish_time and an action (upload or download) if the LP has a tanker.
    @type name: str
    @param name: Name of loading point
    @type node_name: str
    @param node_name: Name of node that the LP belongs to
    @type fuel_type: str
    @param fuel_type: Type of fuel the LP uploads/downloads
    @type upload_rate: float
    @param upload_rate: Rate at which fuel can be loaded (gal/hr)
    @type download_rate: float
    @param download_rate: Rate at which fuel can be unloaded (gal/hr)
    @type tanker: tanker.Tanker
    @param tanker: Tanker that is currently occupying the LP (if any)
    @type last_updated: float
    @param last_updated: Last time the amount on board the tanker was updated.
    @type finish_time: float
    @param finish_time: Time at which the loading point will be free (-1 indicates it is free)
    @type action: str
    @type action: One of 'upload' or 'download'
    """
    def __init__(self, name, node_name, fuel_type, upload_rate, download_rate, tanker=None,
                 last_updated=-1, finish_time=-1, action=None):
        """LoadingPoint is a resource used to load and unload tankers.  Each LoadingPoint
        has a unique name, an upload rate, a download rate, a tanker (if the LoadingPoint
        is currently being used to load or unload a tanker), a last_updated time, a
        finish_time and an action (upload or download) if the LP has a tanker.
        @type name: str
        @param name: Name of loading point
        @type node_name: str
        @param node_name: Name of node that the LP belongs to
        @type fuel_type: str
        @param fuel_type: Type of fuel the LP uploads/downloads
        @type upload_rate: float
        @param upload_rate: Rate at which fuel can be loaded (gal/hr)
        @type download_rate: float
        @param download_rate: Rate at which fuel can be unloaded (gal/hr)
        @type tanker: tanker.Tanker
        @param tanker: Tanker that is currently occupying the LP (if any)
        @type last_updated: float
        @param last_updated: Last time the amount on board the tanker was updated.
        @type finish_time: float
        @param finish_time: Time at which the loading point will be free (-1 indicates it is free)
        @type action: str
        @type action: One of 'upload' or 'download'
        """
        self.log_label = '_'.join([node_name, fuel_type, name])
        self.name = name
        self.node_name = node_name
        self.fuel_type = fuel_type
        self.upload_rate = float(upload_rate)
        self.download_rate = float(download_rate)
        self.tanker = tanker
        self.last_updated = last_updated
        self.finish_time = finish_time
        self.action = action
        # SIM_STATS ('Time', 'Tanker_Name', 'Action', 'Fuel_Amount')
        self.util_trace = []  # SIM_STATS
        """@type : list[(float, str, str, float)]"""

    def __str__(self):
        """
        @rtype: str
        """
        tanker_name = 'None'
        action = 'None'
        if self.tanker is not None:
            tanker_name = self.tanker.name
            action = self.action

        return 'LP %s:  Load rate: %.1f, Unload rate: %.1f, Occupied by: %s, Action: %s, Finish time: %.3f\n' % \
               (self.name.upper(), self.upload_rate, self.download_rate, tanker_name, action, self.finish_time)

    def is_available(self):
        """Return True if the loading point is currently available.
        @rtype: bool
        """
        if self.tanker is None:
            return True
        return False

    def hold_LP(self, sim_time, tanker, fuel_amount, action):
        """Hold the LP while a tanker is loading or unloading fuel_amount of fuel.
        action is either 'upload' or 'download'.
        @type sim_time: sim_engine.SimulationTime
        @type tanker: tanker.Tanker
        @type fuel_amount: float
        @type action: str
        """
        assert self.tanker is None, 'There is a tanker occupying this LP already'
        assert getattr(self, action + '_rate') != 0, '%s CANNOT %s %s fuel!' % \
                                                     (self.log_label, action, tanker.fuel_type)
        self.tanker = tanker
        self.action = action
        self.last_updated = sim_time.time
        self.finish_time = sim_time.time + (fuel_amount / getattr(self, action + '_rate'))

        tanker.status_trace.append((sim_time.time, 'on', self.name, self.node_name, fuel_amount))  # SIM_STATS
        if action == 'upload':
            fuel_amount *= -1
        self.util_trace.append((sim_time.time, tanker.name, 'on', fuel_amount))  # SIM_STATS

    def release_LP(self, sim_time):
        """Release the LP once a tanker is done loading or unloading.
        @type sim_time: sim_engine.SimulationTime
        """
        assert self.tanker is not None, 'There is no tanker occupying this LP'

        self.util_trace.append((sim_time.time, self.tanker.name, 'off', np.nan))  # SIM_STATS

        self.tanker = None
        self.action = None
        self.last_updated = -1
        self.finish_time = -1
