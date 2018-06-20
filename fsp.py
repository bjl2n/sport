import math
import misc
import numpy as np
import numpy.random as npr
import stats
import tanker as tkr


# noinspection PyPep8Naming
class FuelStoragePoint:
    """Holds a given fuel type at a node.  All FuelStoragePoints store a specific type
    of fuel, have a minimum and maximum amount of storage, have a stockage objective,
    keep track of the amount of fuel on hand (which does not account for the storage
    minimum), and keep track of when the next tanker can load or unload.
    @type node_name: str
    @param node_name: (Unique) name of node that this FSP is attached to.
    @type fuel_type: str
    @param fuel_type: Type of fuel being stored.
    @type LPs: list[lp.LoadingPoint]
    @param LPs: List of Loading Points.
    @type storage_min: float
    storage_min: Minimum amount of fuel that has to be carried (gal).
    @type storage_max: float
    @param storage_max: Maximum amount of fuel that can be carried (gal).
    @type stockage_obj: float
    @param stockage_obj: Stockage objective - how much should be stocked at all times
        if possible (gal).
    @type on_hand: float
    @param on_hand: The amount of fuel currently available (NOTE: this does not account
        for the storage_min constraint) (gal).
    @type upload_wait_until: float
    @param upload_wait_until: Time stamp that gets set when a tanker wants to load
        but there is not enough fuel.
    @type download_wait_until: float
    @type download_wait_until: Time stamp that gets set when a tanker wants to unload
        but there is not enough storage space.
    """
    def __init__(self, node_name, fuel_type, LPs, storage_min, storage_max,
                 stockage_obj, on_hand, upload_wait_until=-1, download_wait_until=-1):
        """Holds a given fuel type at a node.  All Fuel Storage Points store a specific type
        of fuel, have a minimum and maximum amount of storage, have a stockage objective,
        keep track of the amount of fuel on hand (which does not account for the storage
        minimum), and keep track of when the next tanker can load or unload.
        @type node_name: str
        @param node_name: (Unique) name of node that this FSP is attached to.
        @type fuel_type: str
        @param fuel_type: Type of fuel being stored.
        @type LPs: list[lp.LoadingPoint]
        @param LPs: List of Loading Points.
        @type storage_min: float
        storage_min: Minimum amount of fuel that has to be carried (gal).
        @type storage_max: float
        @param storage_max: Maximum amount of fuel that can be carried (gal).
        @type stockage_obj: float
        @param stockage_obj: Stockage objective - how much should be stocked at all times
            if possible (gal).
        @type on_hand: float
        @param on_hand: The amount of fuel currently available (NOTE: this does not account
            for the storage_min constraint) (gal).
        @type upload_wait_until: float
        @param upload_wait_until: Time stamp that gets set when a tanker wants to load
            but there is not enough fuel.
        @type download_wait_until: float
        @type download_wait_until: Time stamp that gets set when a tanker wants to unload
            but there is not enough storage space.
        """
        self.log_label = node_name + '_FSP_' + fuel_type
        self.name = 'FSP_' + fuel_type
        self.fuel_type = fuel_type
        self.LPs = LPs
        self.storage_min = float(storage_min)
        self.storage_max = float(storage_max)
        self.SO = float(stockage_obj)
        self.on_hand = float(on_hand)
        self.upload_wait_until = upload_wait_until
        self.download_wait_until = download_wait_until
        self.loading_queue = []
        """@type loading_queue: list[tanker.Tanker]"""
        self.unloading_queue = []
        """@type unloading_queue: list[tanker.Tanker]"""
        self.inv_trace = [(0.0, self.on_hand)]  # SIM_STATS ('Time', 'On_Hand')
        """@type inv_trace: list[(float, float)]"""
        self.loadingQ_trace = [(0.0, 0)]  # SIM_STATS ('Time', 'Num_Tankers')
        """@type loadingQ_trace: list[(float, int)]"""
        self.unloadingQ_trace = [(0.0, 0)]  # SIM_STATS ('Time', 'Num_Tankers')
        """@type unloadingQ_trace: list[(float, int)]"""
        # Keeps track of the latest time that fuel started loading on a tanker at the Parent
        # node and arrived at and got through the soak yard at the Child node
        self.latest_loading_start_time = 0.0

    def __str__(self):
        """
        @rtype: str
        """
        fsp_str = 'Fuel type:  %s\nStorage:  Min: %.1f  Max: %.1f  On hand: %.1f\n' % \
                  (self.fuel_type.upper(), self.storage_min, self.storage_max, self.on_hand)
        for lp in self.LPs:
            fsp_str += lp.__str__()
        fsp_str += '\n'

        return fsp_str

    def update_loadingQ_trace(self, sim_time, delta):  # SIM_STATS
        """Add the change (delta) in loading_queue status to the loadingQ_trace.
        @type sim_time: sim_engine.SimulationTime
        @type delta: int
        """
        last_update_time, last_Q_size = self.loadingQ_trace[-1]
        if last_update_time == sim_time.time:
            self.loadingQ_trace.pop(-1)
        self.loadingQ_trace.append((sim_time.time, last_Q_size + delta))  # SIM_STATS

    def update_unloadingQ_trace(self, sim_time, delta):  # SIM_STATS
        """Add the change (delta) in unloading_queue status to the unloadingQ_trace.
        @type sim_time: sim_engine.SimulationTime
        @type delta: int
        """
        last_update_time, last_Q_size = self.unloadingQ_trace[-1]
        if last_update_time == sim_time.time:
            self.unloadingQ_trace.pop(-1)
        self.unloadingQ_trace.append((sim_time.time, last_Q_size + delta))  # SIM_STATS

    def update_on_hand(self, scn, sim_time, amount):
        """amount represents the gallons of fuel that will be added or subtracted from
        the current on_hand value.  Checks in load_tanker and unload_tanker should
        prevent the assert statements from triggering.  The function uses the
        scenario fuel_epsilon tolerance to correct floating point inaccuracies.
        @type scn: scenario.Scenario
        @type sim_time: sim_engine.SimulationTime
        @type amount: float
        @param amount: Amount of fuel to be added or subtracted from the current
        on_hand value.
        """
        assert self.on_hand + amount >= self.storage_min - scn.fuel_epsilon, \
            'Attempting to load more fuel from %s than there is available\n%.5f\n%.5f\n.%5f\n%.5f' % (
                self.log_label, self.on_hand, amount, self.storage_max, scn.fuel_epsilon)
        assert self.on_hand + amount <= self.storage_max + scn.fuel_epsilon, \
            'Attempting to unload more fuel to %s than there is storage available\n%.5f\n%.5f\n.%5f\n%.5f' % (
                self.log_label, self.on_hand, amount, self.storage_max, scn.fuel_epsilon)

        self.on_hand += amount
        on_hand_to_nearest_gallon = math.floor(self.on_hand + 0.5)

        if abs(on_hand_to_nearest_gallon - self.on_hand) < scn.fuel_epsilon:
            self.on_hand = on_hand_to_nearest_gallon
        if self.inv_trace[-1][0] == sim_time.time:            # SIM_STATS
            self.inv_trace.pop(-1)                            # SIM_STATS
        self.inv_trace.append((sim_time.time, self.on_hand))  # SIM_STATS

    def update_FSP(self, scn, sim_time, force_update):
        """Updates the on_hand values at the FSP if time has elapsed since the last
        update AND there has been a change in the inventory level.  If force_update
        is True, on_hand is updated no matter what.
        ASSUMPTION: Uploading and downloading of fuel occurs linearly between updates.
        @type scn: scenario.Scenario
        @type sim_time: sim_engine.SimulationTime
        @type force_update: bool
        """
        FSP_delta = 0
        for LP in self.LPs:
            if LP.tanker is not None:
                time_since_last_update = sim_time.time - LP.last_updated
                if time_since_last_update > 0:
                    fuel_delta = time_since_last_update * getattr(LP, LP.action + '_rate')

                    LP.last_updated = sim_time.time
                    # Update tanker on_board and FSP on_hand
                    if LP.action == 'upload':
                        LP.tanker.update_on_board(scn, LP.tanker.on_board + fuel_delta)
                        FSP_delta -= fuel_delta
                    elif LP.action == 'download':
                        LP.tanker.update_on_board(scn, LP.tanker.on_board - fuel_delta)
                        FSP_delta += fuel_delta
                    else:
                        assert False, 'LP.action is neither upload or download'
        if (FSP_delta != 0) or force_update:
            self.update_on_hand(scn, sim_time, FSP_delta)

    def is_LP_available(self, action):
        """Return a free LP that is able to perform the specified action ('download'
        or 'upload' rate is greater than 0), otherwise return None.
        @type action: str
        @rtype: lp.LoadingPoint | None
        """
        assert action in ['download', 'upload'], 'action must be either \'download\' or \'upload\''

        for LP in self.LPs:
            if LP.is_available() and (getattr(LP, action + '_rate') > 0):
                return LP
        return None

    def can_process_tanker(self, sim_time, fuel_amount, rate):
        """Return a tuple where the first entry is True if the upload or download of
        fuel_amount at the given rate will work, False if not.  The second entry is
        a float that is non-zero if the first entry is False and corresponds to the
        earliest anticipated time that the first entry would be True.
        NOTE: rate is negative if the action is upload, positive if the action is download.
        @param sim_time: sim_engine.SimulationTime
        @param fuel_amount: float
        @param rate: float
        @return: (boolean, float)
        """
        finish_time = sim_time.time + (fuel_amount / abs(rate))

        on_point = [(finish_time, rate)]
        for LP in self.LPs:
            if not LP.is_available():
                if LP.action == 'upload':
                    on_point.append((LP.finish_time, -LP.upload_rate))
                elif LP.action == 'download':
                    on_point.append((LP.finish_time, LP.download_rate))
                else:
                    assert False, '%s action must be either upload or download' % LP.log_label
        on_point.sort(key=lambda x: x[0])  # sort on finish_time in ascending order

        start_time = sim_time.time
        projected_on_hand = self.on_hand
        net_rate = sum([i[1] for i in on_point])
        for finish_time, LP_rate in on_point:
            projected_on_hand += net_rate * (finish_time - start_time)
            if (projected_on_hand < self.storage_min) or (projected_on_hand > self.storage_max):
                return False, finish_time

            start_time = finish_time
            net_rate -= LP_rate

        assert net_rate == 0, 'All occupied LPs were not considered'

        return True, 0

    def load_tanker(self, scn, sim_cal, sim_time, the_node):
        """If the loading_queue has at least one tanker waiting, attempt to load
        the tanker.  If no LP is free or if a LP is free but the fuel cannot be
        loaded, schedule a new load_tanker event some time later.
        ASSUMPTION: Each tanker will be filled to capacity.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type the_node: parent_node.ParentNode | child_node.ChildNode
        """
        if self.loading_queue:
            LP = self.is_LP_available('upload')
            if LP and (self.upload_wait_until <= sim_time.time):
                self.upload_wait_until = -1

                self.update_FSP(scn, sim_time, False)
                if the_node.subnetwork_role == 'parent':  # Enforces Shortest Processing Time order
                    tanker = sorted(self.loading_queue, key=lambda tnkr: tnkr.on_board)[0]
                else:  # Enforces First In, First Out order
                    tanker = self.loading_queue[0]

                fully_loaded = True
                fuel_needed = tanker.capacity - tanker.on_board
                # Load the tanker in stages if the fuel needed exceeds what can be stored.
                if fuel_needed > (self.storage_max - self.storage_min):
                    fuel_needed = (self.storage_max - self.storage_min) / 2.0
                    fully_loaded = False

                tanker_can_load, wait_time = self.can_process_tanker(sim_time, fuel_needed, -LP.upload_rate)
                if tanker_can_load:
                    # Force an update to take care of the case that there has been
                    # no update since the last daily update_FSP call
                    self.update_FSP(scn, sim_time, True)
                    self.loading_queue.remove(tanker)
                    self.update_loadingQ_trace(sim_time, -1)  # SIM_STATS

                    LP.hold_LP(sim_time, tanker, fuel_needed, 'upload')

                    sim_cal.add_event(the_node, 'tanker_finished_loading',
                                      [scn, sim_cal, sim_time, LP, tanker, fully_loaded],
                                      sim_time.time + tanker.load_time(LP, fuel_needed))
                else:  # Wait five minutes and try again
                    self.upload_wait_until = sim_time.time + (5.0 / 60)  # wait_time
                    sim_cal.add_event(self, 'load_tanker',
                                      [scn, sim_cal, sim_time, the_node],
                                      self.upload_wait_until)

    def unload_tanker(self, scn, sim_cal, sim_time, the_node):
        """If the unloading_queue has at least one tanker waiting, attempt to
        unload the tanker.  If no LP is free or if a LP is free but the fuel
        cannot be unloaded, schedule a new unload_tanker event some time later.
        ASSUMPTION: Each tanker will be completely emptied.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type the_node: parent_node.ParentNode | child_node.ChildNode
        """
        if self.unloading_queue:
            LP = self.is_LP_available('download')
            if LP and (self.download_wait_until <= sim_time.time):
                self.download_wait_until = -1

                self.update_FSP(scn, sim_time, False)
                # Enforce Shortest Processing Time order
                tanker = sorted(self.unloading_queue, key=lambda tnkr: tnkr.on_board)[0]

                fully_unloaded = True
                storage_needed = tanker.on_board
                # Unload the tanker in stages if the space needed exceeds the storage capacity.
                if storage_needed > (self.storage_max - self.storage_min):
                    storage_needed = (self.storage_max - self.storage_min) / 2.0
                    fully_unloaded = False

                can_unload_tanker, wait_time = self.can_process_tanker(sim_time, storage_needed, LP.download_rate)
                if can_unload_tanker:
                    # Force an update to take care of the case that there has been
                    # no update since the last daily update_FSP call
                    self.update_FSP(scn, sim_time, True)
                    self.unloading_queue.remove(tanker)
                    self.update_unloadingQ_trace(sim_time, -1)  # SIM_STATS

                    LP.hold_LP(sim_time, tanker, storage_needed, 'download')

                    sim_cal.add_event(the_node, 'tanker_finished_unloading',
                                      [scn, sim_cal, sim_time, LP, tanker, fully_unloaded],
                                      sim_time.time + tanker.unload_time(LP, storage_needed))
                else:  # Wait five minutes and try again
                    self.download_wait_until = sim_time.time + (5.0 / 60)   # wait_time
                    sim_cal.add_event(self, 'unload_tanker',
                                      [scn, sim_cal, sim_time, the_node],
                                      self.download_wait_until)


# noinspection PyPep8Naming
class ParentNodeFSP(FuelStoragePoint):
    """ParentNodeFSP is a subclass of FuelStoragePoint that is meant to be used with
    the ParentNode class.
    """
    def __init__(self, node_name, fuel_type, LPs, storage_min, storage_max,
                 stockage_obj, on_hand, upload_wait_until=-1, download_wait_until=-1):
        """ParentNodeFSP is a subclass of FuelStoragePoint that is meant to be used with
        the ParentNode class.
        """
        FuelStoragePoint.__init__(self, node_name, fuel_type, LPs, storage_min, storage_max,
                                  stockage_obj, on_hand, upload_wait_until, download_wait_until)
        self.daily_amount_loaded = []  # Records the amount of fuel loaded from the FSP each day


# noinspection PyPep8Naming
class ChildNodeFSP(FuelStoragePoint):
    """ChildNodeFSP is a subclass of FuelStoragePoint.  It contains functions to handle the
    daily consumption of fuel at a ChildNode.  The ChildNodeFSP contains information
    about the distribution of the daily consumption rate (DC_distribution), a random number
    stream (rndstrm) to compute a seed for the ChildNodeFSP's random streams (DC_rndstrm).
    The additional parameters necessary are given below.  The variable gamma keeps track of
    the number of type gamma tankers (i.e. fuel demands) that have occurred at this FSP.
    @type node_type: str
    @param node_type: One of 'intermediate' or 'end user'.
    @type DC_distribution: collections.OrderedDict[int: list | tuple]
    @param DC_distribution: The keys of DC_distribution represent the last day of the simulation
        time horizon that the distribution DC_distribution[key] is in effect for. If node_type =
        'end user', DC_distribution[key] is a list/tuple with distribution information. If
        node_type = 'intermediate', DC_distribution[key] is a list of floats representing the
        empirical distribution of demand.
    """
    def __init__(self, node_name, node_type, fuel_type, LPs, storage_min, storage_max, stockage_obj,
                 on_hand, DC_distribution, DC_realization_idx, rep_rndstrm,
                 upload_wait_until=-1, download_wait_until=-1):
        """ChildNodeFSP is a subclass of FuelStoragePoint.  It contains functions to handle the
        daily consumption of fuel at a ChildNode.  The ChildNodeFSP contains information
        about the distribution of the daily consumption rate (DC_distribution), a random number
        stream (rndstrm) to compute a seed for the ChildNodeFSP's random streams (DC_rndstrm).
        The additional parameters necessary are given below.  The variable gamma keeps track of
        the number of type gamma tankers (i.e. fuel demands) that have occurred at this FSP.
        @type node_type: str
        @param node_type: One of 'intermediate' or 'end user'.
        @type DC_distribution: collections.OrderedDict[int: list | tuple]
        @param DC_distribution: The keys of DC_distribution represent the last day of the simulation
            time horizon that the distribution DC_distribution[key] is in effect for. If node_type =
            'end user', DC_distribution[key] is a list/tuple with distribution information. If
            node_type = 'intermediate', DC_distribution[key] is a list of floats representing the
            empirical distribution of demand.
        """
        FuelStoragePoint.__init__(self, node_name, fuel_type, LPs, storage_min, storage_max,
                                  stockage_obj, on_hand, upload_wait_until, download_wait_until)
        self.gamma = 0
        assert node_type.lower() in ['intermediate', 'end user'], 'Node type %s is not recognized' % node_type
        self.node_type = node_type
        self.DC_distribution = DC_distribution
        # SIM_STATS (time_requested, time_fulfilled, amount_fulfilled)
        self.demand_trace = []  # SIM_STATS
        """@type : list[(float, float, float)]"""
        self.total_fuel_used = 0.0
        """@type total_fuel_used: float"""
        self.rndstrm = npr.RandomState(misc.bounded_integer_seed(rep_rndstrm))
        """@type : numpy.random.RandomState"""
        self.DC_rndstrm = npr.RandomState(misc.bounded_integer_seed(self.rndstrm))
        self.DC_realization_idx = DC_realization_idx
        # print self.log_label, self.DC_realization_idx
        """@type : numpy.random.RandomState"""
        self.daily_amount_demanded = []  # Records the amount of fuel demanded from the FSP each day

    def __str__(self):
        """
        @rtype: str
        """
        fsp_str = 'FSP Fuel type:  %s\nStorage:  Min: %.1f  Max: %.1f  On hand: %.1f\n' % \
                  (self.fuel_type.upper(), self.storage_min, self.storage_max, self.on_hand)
        for lp in self.LPs:
            fsp_str += lp.__str__()
        fsp_str += '\n'

        return fsp_str

    def get_current_distribution(self, scn, start_time):
        """Returns the demand distribution in effect at start_time.
        @type scn: scenario.Scenario
        @type start_time: float
        """
        assert self.node_type == 'end user', 'Node must be an End User node to access this function'
        for end_time in sorted(self.DC_distribution.keys()):
            if start_time <= ((end_time + scn.warm_up) * 24.0):
                return self.DC_distribution[end_time]
        assert False, 'No distribution defined for the current simulation time'

    def generate_daily_demand(self, scn, start_time):
        """Return a list containing the daily demand (broken into discrete chunks)
        for the 24 hours following the given start_time (in hours).
        @type scn: scenario.Scenario
        @type start_time: float
        @rtype daily_demand: list[(float, float)]
        """
        if self.node_type.lower() == 'intermediate':
            day_idx = int(math.ceil(start_time / 24.0))
            total_demand = self.DC_distribution[str(day_idx)][self.DC_realization_idx]
        else:
            current_DC_distribution = self.get_current_distribution(scn, start_time)
            dist_name = current_DC_distribution[0]
            dist_params = current_DC_distribution[1:]

            if dist_name.lower() == 'constant':
                total_demand = dist_params[0]
            else:
                total_demand = getattr(self.DC_rndstrm, dist_name)(*dist_params)

        self.daily_amount_demanded.append(total_demand)

        if total_demand < scn.fuel_epsilon:
            return []

        # Number of discrete (individual) demands that the daily consumption is divided into
        num_chunks = self.DC_rndstrm.random_integers(2, 6, 1)[0]

        # Sorted list of uniform [0,1) floats representing the percentage of the
        # day that has elapsed before each demand occurs
        pct_of_day_gone = self.DC_rndstrm.random_sample(num_chunks)
        pct_of_day_gone.sort()

        # Normalized list of uniform [0, 1) floats representing the percentage of
        # total_demand that is demanded at each discrete time point
        pct_of_total_demand = self.DC_rndstrm.random_sample(num_chunks)
        pct_of_total_demand /= sum(pct_of_total_demand)

        daily_consumption = []
        for pct_day_gone, pct_consumption in zip(pct_of_day_gone, pct_of_total_demand):
            daily_consumption.append((pct_day_gone * 24.0, pct_consumption * total_demand))
        return daily_consumption

    def expected_daily_demand(self, scn, start_time):
        """Return the expected daily demand (in gallons).
        ASSUMPTION: The distribution of demand being used at start_time is used
        for the next 24 hours.
        @type scn: scenario.Scenario
        @type start_time: float
        @rtype: float
        """
        if self.node_type.lower() == 'intermediate':
            day_idx = int(math.ceil(start_time / 24.0))
            return np.mean(self.DC_distribution.get(str(day_idx), [0]))
        else:
            return stats.mean(self.get_current_distribution(scn, start_time))

    def expected_demand(self, scn, sim_time, num_days):
        """Return the expected demand (in gallons) for the next num_days.
        @type scn: scenario.Scenario
        @type sim_time: sim_engine.SimulationTime
        @type num_days: float
        @rtype: float
        """
        partial_day, full_days = math.modf(num_days)
        start_time = sim_time.time
        total_expected_demand = 0
        for _ in range(int(full_days)):
            total_expected_demand += self.expected_daily_demand(scn, start_time)
            start_time += 24.0

        partial_day_start_time = sim_time.time + (num_days * 24.0)
        total_expected_demand += partial_day * self.expected_daily_demand(scn, partial_day_start_time)

        return total_expected_demand

    def demand_event(self, scn, sim_cal, sim_time, demand_amount, daily_consumption, cNode):
        """Add the demand event (modeled as a type gamma tanker with capacity equal to
        demand_amount) to the child node loading queue.  Schedule the removal of the demand
        event if it has not begun to be fulfilled after a set amount of time (stale_after).
        Finally, if there is another demand event during the day schedule it's occurrence.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type demand_amount: float
        @type daily_consumption: list[(float, float)]
        @type cNode: child_node.ChildNode
        """
        gamma_tanker = tkr.Tanker('HNTg_%s' % self.gamma, 'gamma', demand_amount, self.fuel_type)
        self.loading_queue.append(gamma_tanker)
        self.update_loadingQ_trace(sim_time, 1)  # SIM_STATS
        gamma_tanker.status_trace.append((sim_time.time, 'on', 'loading_Q', self.name, np.nan))  # SIM_STATS
        self.gamma += 1

        sim_cal.add_event(self, 'load_tanker',
                          [scn, sim_cal, sim_time, cNode],
                          sim_time.time)

        # The demand event grows stale (is no longer needed) after a fixed amount of time
        sim_cal.add_event(self, 'remove_stale_demand_event',
                          [sim_time, gamma_tanker],
                          sim_time.time + scn.stale_after)

        if daily_consumption:
            time_delta, demand_amount = daily_consumption.pop(0)
            sim_cal.add_event(self, 'demand_event',
                              [scn, sim_cal, sim_time, demand_amount, daily_consumption, cNode],
                              sim_time.time + time_delta)

    def remove_stale_demand_event(self, sim_time, gamma_tanker):
        """Removes the stale demand event by removing gamma_tanker from the loading_queue.
        @type sim_time: sim_engine.SimulationTime
        @type gamma_tanker: tanker.Tanker
        @rtype: None
        """
        for tanker in self.loading_queue:
            if tanker.name == gamma_tanker.name:
                self.loading_queue.remove(tanker)
                self.update_loadingQ_trace(sim_time, -1)  # SIM_STATS
                demand_not_satisfied = tanker.capacity - tanker.on_board
                self.demand_trace.append((tanker.status_trace[0][0], sim_time.time, -demand_not_satisfied))  # SIM_STATS
                del tanker
                return None
