import misc
import numpy as np
import numpy.random as npr
import stats


# noinspection PyPep8Naming
class Node:
    """Each node has a unique name, type ('depot', 'intermediate', 'end user'), a list of
    Fuel Storage Points, a dictionary of convoy travel time (CTT) distributions, a
    dictionary of Transportation Movement Request (TMR) distributions, the probability that
    a tanker will be placed on a convoy, queues (soak_yard_queue, loaded_queue,
    staged_loaded_queue, unloaded_queue, and staged_unloaded_queue) that are formed from
    dictionaries, and a random number stream used to compute a seed for the different
    random streams needed by this class.
    @type name: str
    @param name: Name of the node.
    @type node_type: str
    @param node_type: Type of node; one of 'depot', 'intermediate', or 'end user'.
    @type FSPs: dict[str, fsp.ParentNodeFSP | fsp.ChildNodeFSP]
    @param FSPs: Dictionary of Fuel Storage Points.
    @type CTT_distributions: dict[str, list]
    @param CTT_distributions: Dictionary containing the convoy travel time from this node to
        all other nodes connected to this node.  Each dictionary value is a list where
        CTT_distributions[dest_node][0] is the name of the distribution (see stats.py for a
        list of acceptable distributions), while CTT_distributions[dest_node][1:] are the
        parameters of the distribution (units of days).
    @type TMR_distributions: dict
    @param TMR_distributions: Each dictionary value is a list where
        TMR_distributions[child_node_name/parent_node_name][0] is the name of the distribution and
        TMR_distributions[child_node_name/parent_node_name[1:] are the parameters of the distribution
        (units of days) of the
        a) time between loading plan and occurrence of the TMR (sending loaded tankers to their
        destination child node), or
        b) time between occurrences of TMRs (sending unloaded tankers back to their parent node).
    @type prob_on_convoy: float
    @param prob_on_convoy: The probability of a tanker being placed on a convoy at
        this particular node.
    @type rep_rndstrm: numpy.random.RandomState
    @param rep_rndstrm: The random number stream for the replication.
    """
    def __init__(self, name, node_type, FSPs, CTT_distributions, TMR_distributions, prob_on_convoy, rep_rndstrm):
        """Each node has a unique name, type ('depot', 'intermediate', 'end user'), a list of
        Fuel Storage Points, a dictionary of convoy travel time (CTT) distributions, a
        dictionary of Transportation Movement Request (TMR) distributions, the probability that
        a tanker will be placed on a convoy, queues (soak_yard_queue, loaded_queue,
        staged_loaded_queue, unloaded_queue, and staged_unloaded_queue) that are formed from
        dictionaries, and a random number stream used to compute a seed for the different
        random streams needed by this class.
        @type name: str
        @param name: Name of the node.
        @type node_type: str
        @param node_type: Type of node; one of 'depot', 'intermediate', or 'end user'.
        @type FSPs: dict[str, fsp.ParentNodeFSP | fsp.ChildNodeFSP]
        @param FSPs: Dictionary of Fuel Storage Points.
        @type CTT_distributions: dict[str, list]
        @param CTT_distributions: Dictionary containing the convoy travel time from this node to
            all other nodes connected to this node.  Each dictionary value is a list where
            CTT_distributions[dest_node][0] is the name of the distribution (see stats.py for a
            list of acceptable distributions), while CTT_distributions[dest_node][1:] are the
            parameters of the distribution (units of days).
        @type TMR_distributions: dict
        @param TMR_distributions: Each dictionary value is a list where
            TMR_distributions[child_node_name/parent_node_name][0] is the name of the distribution and
            TMR_distributions[child_node_name/parent_node_name[1:] are the parameters of the distribution
            (units of days) of the
            a) time between loading plan and occurrence of the TMR (sending loaded tankers to their
            destination child node), or
            b) time between occurrences of TMRs (sending unloaded tankers back to their parent node).
        @type prob_on_convoy: float
        @param prob_on_convoy: The probability of a tanker being placed on a convoy at
            this particular node.
        @type rep_rndstrm: numpy.random.RandomState
        @param rep_rndstrm: The random number stream for the replication.
        """
        self.log_label = name
        self.name = name
        self.node_type = node_type
        self.subnetwork_role = ''  # This parameter gets set in the ParentNode and ChildNode class definitions
        self.FSPs = FSPs
        self.CTT_distributions = CTT_distributions
        self.TMR_distributions = TMR_distributions
        self.prob_on_convoy = prob_on_convoy
        self.soak_yard_queue = {}
        """@type : dict[str, tanker.Tanker]"""
        self.loaded_queue = {}
        """@type : dict[str, tanker.Tanker]"""
        self.staged_loaded_queue = {}
        """@type : dict[str, tanker.Tanker]"""
        self.unloaded_queue = {}
        """@type : dict[str, tanker.Tanker]"""
        self.staged_unloaded_queue = {}
        """@type : dict[str, tanker.Tanker]"""
        self.rndstrm = npr.RandomState(misc.bounded_integer_seed(rep_rndstrm))
        """@type : numpy.random.RandomState"""
        self.on_convoy_rndstrm = npr.RandomState(misc.bounded_integer_seed(self.rndstrm))
        """@type : numpy.random.RandomState"""
        self.CTT_rndstrms = {}
        """@type : dict[str, numpy.random.RandomState]"""
        for dest_node in CTT_distributions.iterkeys():
            self.CTT_rndstrms[dest_node] = npr.RandomState(misc.bounded_integer_seed(self.rndstrm))
        self.TMR_rndstrm = npr.RandomState(misc.bounded_integer_seed(self.rndstrm))
        """@type : numpy.random.RandomState"""
        self.TMR_trace = []  # SIM_STATS ('Time', 'Num_Tankers')
        """@type : list[(float, int)]"""

    def __str__(self):
        """Returns a string with the name of the node and all FSPs.
        @rtype : str
        """
        node_str = 'Node %s\n' % self.name
        for x in self.FSPs.itervalues():
            node_str += x.__str__()
        return node_str

    def tanker_to_soak_yard(self, scn, sim_cal, sim_time, tanker):
        """Places tanker on soak_yard_queue for 24 hours, then adds a call
        to Node.tanker_to_unloading_queue() to the simulation calendar.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type tanker: tanker.Tanker
        """
        self.soak_yard_queue[tanker.name] = tanker
        tanker.status_trace.append((sim_time.time, 'on', 'soak_yard_Q', self.name, np.nan))  # SIM_STATS

        sim_cal.add_event(self, 'tanker_to_unloading_queue',
                          [scn, sim_cal, sim_time, tanker],
                          sim_time.time + 24)

    def tanker_to_unloading_queue(self, scn, sim_cal, sim_time, tanker):
        """Removes tanker from soak_yard_queue, places it on the appropriate unloading_queue,
        and then adds a call to Node.unload_tanker() to the simulation calendar.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type tanker: tanker.Tanker
        """
        del self.soak_yard_queue[tanker.name]
        FSP = self.FSPs[tanker.fuel_type]
        FSP.unloading_queue.append(tanker)
        FSP.update_unloadingQ_trace(sim_time, 1)  # SIM_STATS
        tanker.status_trace.append((sim_time.time, 'on', 'unloading_Q', FSP.name, np.nan))  # SIM_STATS

        FSP.latest_loading_start_time = max(FSP.latest_loading_start_time, tanker.most_recent_loading_start_time())

        sim_cal.add_event(FSP, 'unload_tanker',
                          [scn, sim_cal, sim_time, self],
                          sim_time.time)

    def tanker_finished_unloading(self, scn, sim_cal, sim_time, LP, tanker, fully_unloaded):
        """When a tanker is finished unloading:
        1. Update the FSP and release the LP,
        2. a. If the tanker is fully unloaded place the tanker on the staged_unloaded_queue,
           b. If the tanker is not fully unloaded place the tanker back on
              unloading_queue, and
        3. Check to see if there is another tanker that needs to be loaded or unloaded.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type LP: lp.LoadingPoint
        @type tanker: tanker.Tanker
        @type fully_unloaded: bool
        """
        FSP = self.FSPs[tanker.fuel_type]
        FSP.update_FSP(scn, sim_time, False)

        LP.release_LP(sim_time)

        if fully_unloaded:
            tanker.going_to = tanker.coming_from
            tanker.coming_from = self
            tanker.TMR_info = None

            self.staged_unloaded_queue[tanker.name] = tanker
            tanker.status_trace.append((sim_time.time, 'on', 'staged_unloaded_Q', self.name, np.nan))  # SIM_STATS
        else:
            FSP.unloading_queue.insert(0, tanker)
            FSP.update_unloadingQ_trace(sim_time, 1)  # SIM_STATS
            tanker.status_trace.append((sim_time.time, 'on', 'unloading_Q', FSP.name, np.nan))  # SIM_STATS

        self.perform_queue_check(scn, sim_cal, sim_time, FSP)

    def perform_queue_check(self, scn, sim_cal, sim_time, FSP):
        """Checks the loading queue and then the unloading queue and adds
        a call to either load or unload a tanker.
        NOTE: Loading of fuel is prioritized by checking the loadingQ first.
        @param scn: scenario.Scenario
        @param sim_cal: sim_engine.Calendar
        @param sim_time: sim_engine.SimulationTime
        @param FSP: fsp.ParentNodeFSP | fsp.ChildNodeFSP
        @rtype: None
        """
        if len(FSP.loading_queue) > 0:
            sim_cal.add_event(FSP, 'load_tanker',
                              [scn, sim_cal, sim_time, self],
                              sim_time.time)
        if len(FSP.unloading_queue) > 0:
            sim_cal.add_event(FSP, 'unload_tanker',
                              [scn, sim_cal, sim_time, self],
                              sim_time.time)

    def on_convoy(self):
        """Returns True if the tanker was placed on the convoy, False if it was not.
        @rtype: bool
        """
        if self.on_convoy_rndstrm.uniform(0, 1, 1) <= self.prob_on_convoy:
            return True
        return False

    def generate_CTT(self, scn, dest_node_name):
        """Returns the travel time (in hours) of travelling between self and the
        dest_node_name node.  Note that the distribution parameters are in units
        of days and must be converted to hours.
        @type scn: scenario.Scenario
        @type dest_node_name: str
        @param dest_node_name: The name of the node that will be travelled to.
        @rtype: float
        """
        assert dest_node_name in self.CTT_distributions.keys(), '%s is not a destination node.' % dest_node_name

        if scn.lower_bound:
            return stats.minimum(self.CTT_distributions[dest_node_name]) * 24.0
        else:
            distribution = self.CTT_distributions[dest_node_name][0]
            params = self.CTT_distributions[dest_node_name][1:]
            if distribution.lower() == 'constant':
                return params[0] * 24.0
            else:  # Return a random variate from the given distribution
                return getattr(self.CTT_rndstrms[dest_node_name], distribution)(*params) * 24.0

    def expected_CTT(self, scn, dest_node_name):
        """Returns the expected travel time (in hours) of travelling between self
        and the dest_node_name node.  Note that the distribution parameters are in units
        of days and must be converted to hours.
        @type scn: scenario.Scenario
        @type dest_node_name: str
        @param dest_node_name: The name of the node that will be travelled to.
        @rtype: float
        """
        if scn.lower_bound:
            return stats.minimum(self.CTT_distributions[dest_node_name]) * 24.0
        else:
            return stats.mean(self.CTT_distributions[dest_node_name]) * 24.0

    def generate_time_to_TMR(self, TMR_type):
        """Returns a random variate from the specified TMR distribution.
        If TMR_type == 'loaded', the variate returned represents the time (in hours)
        until the TMR sending loaded tankers to the destination child nodes will occur.
        If TMR_type == 'unloaded', the variate returned represents the time (in hours)
        until the TMR sending unloaded tankers back to their parent nodes will occur.
        Note that the distribution parameters are in units of days and must be converted
        to hours.
        @type TMR_type: str
        @rtype: float
        """
        dist_name = self.TMR_distributions[TMR_type][0]
        dist_params = self.TMR_distributions[TMR_type][1:]
        if dist_name.lower() == 'constant':
            return dist_params[0] * 24.0
        else:  # Return a random variate from the given distribution
            return getattr(self.TMR_rndstrm, dist_name)(*dist_params) * 24.0

    def expected_time_to_TMR(self, TMR_type):
        """Returns the expected value (in hours) of the specified TMR distribution.
        If TMR_type == 'loaded', the variate returned represents the expected time (in hours)
        until a TMR sending loaded tankers to the destination child nodes will occur.
        If TMR_type == 'unloaded', the variate returned represents the expected time (in hours)
        until a TMR sending unloaded tankers back to their parent nodes will occur.
        Note that the distribution parameters are in units of days and must be converted
        to hours.
        @type TMR_type: str
        @rtype: float
        """
        return stats.mean(self.TMR_distributions[TMR_type]) * 24.0

    def daily_FSP_update(self, scn, sim_cal, sim_time):
        """Called once every 24 hours to update the FSPs at this node.
        Schedules the next round of FSP updates 24 hours later.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        """
        for FSP in self.FSPs.values():
            FSP.update_FSP(scn, sim_time, True)

            if self.subnetwork_role.lower() == 'parent':
                # Sets the on_hand amount at the FSP to the maximum level at the
                # end of each day - this is used to simulate unlimited storage
                # at a ParentNode FSP.
                FSP.daily_amount_loaded.append(FSP.storage_max - FSP.on_hand)  # SIM_STATS
                FSP.update_on_hand(scn, sim_time, FSP.storage_max - FSP.on_hand)

        sim_cal.add_event(self, 'daily_FSP_update',
                          [scn, sim_cal, sim_time],
                          sim_time.time + 24.0)
