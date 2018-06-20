import node
import numpy as np
import opt_alg
import tmr


# noinspection PyPep8Naming
class ParentNode(node.Node):
    """ParentNode is a subclass of Node but also takes an additional child_nodes parameter.
    This class also contains queues (maintenance_queue, available_queue, en_route_queue)
    formed from dictionaries, and a list of TMRs that have occurred.
    @type child_nodes: list[child_node.ChildNode]
    """
    def __init__(self, name, node_type, FSPs, CTT_distributions, TMR_distributions,
                 prob_on_convoy, rep_rndstrm, child_nodes):
        """ParentNode is a subclass of Node but also takes an additional child_nodes parameter.
        This class also contains queues (maintenance_queue, available_queue, en_route_queue)
        formed from dictionaries, and a list of TMRs that have occurred.
        @type child_nodes: list[child_node.ChildNode]
        """
        node.Node.__init__(self, name, node_type, FSPs, CTT_distributions, TMR_distributions,
                           prob_on_convoy, rep_rndstrm)
        self.subnetwork_role = 'parent'
        self.child_nodes = child_nodes
        self.maintenance_queue = {}
        """@type : dict[str, tanker.Tanker]"""
        self.available_queue = {}
        """@type : dict[str, tanker.Tanker]"""
        self.en_route_queue = {}
        """@type : dict[str, tanker.Tanker]"""
        self.executed_TMRs = []
        """@type : list[tmr.TMR]"""
        self.TMR_idx = 0
        self.fuel_requirements_trace = []  # SIM_STATS ['Time', 'Fuel_Requests', 'Available_Tankers']
        """@type : list[[float, str, str]]"""
        # SIM_STATS ['Time', 'TMR_Index', 'Tanker_Name', 'Fuel_Type',
        # SIM STATS 'Fuel_Amount', 'Destination', 'TMR_Occurrence_Time']
        self.loading_plans_trace = []  # SIM_STATS
        """@type : list[(float, int, str, str, float, str, float)]"""
        # SIM_STATS ('Time', 'Name', 'Destination', 'Fuel_Type', 'Fuel_Amount', 'TMR_Occurrence_Time')
        self.tankers_sent_trace = []  # SIM_STATS
        """@type : list[(float, str, str, str, float, float)]"""

    def __str__(self):
        """
        @rtype: str
        """
        return self.node_type.title() + ' (Parent) ' + node.Node.__str__(self)

    def arrive_at_parent_node(self, scn, sim_cal, sim_time, tanker):
        """Sends type beta tankers to maintenance.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type tanker: tanker.Tanker
        """
        assert tanker.coming_from in self.child_nodes, '%s came from %s' % (tanker, tanker.coming_from.name)
        assert tanker.on_board < scn.fuel_epsilon, '%s arrived from %s carrying fuel' % \
                                                   (tanker, tanker.coming_from.name)

        tanker.status_trace.append((sim_time.time, 'arrived at', self.name, np.nan, np.nan))  # SIM_STATS
        self.tanker_to_maintenance(sim_cal, sim_time, tanker)

    def tanker_to_maintenance(self, sim_cal, sim_time, tanker):
        """Keeps tanker in maintenance for 12 hours then adds the tanker to availableQ.
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type tanker: tanker.Tanker
        """
        self.maintenance_queue[tanker.name] = tanker
        tanker.status_trace.append((sim_time.time, 'on', 'maintenance_Q', self.name, np.nan))  # SIM_STATS
        sim_cal.add_event(self, 'tanker_to_available_queue',
                          [sim_time, tanker],
                          sim_time.time + 12)

    def tanker_to_available_queue(self, sim_time, tanker):
        """Removes tanker from maintenance_queue and adds it to available_queue.
        @type sim_time: sim_engine.SimulationTime
        @type tanker: tanker.Tanker
        """
        del self.maintenance_queue[tanker.name]
        self.available_queue[tanker.name] = tanker
        tanker.status_trace.append((sim_time.time, 'on', 'available_Q', self.name, np.nan))  # SIM_STATS

    def tanker_finished_loading(self, scn, sim_cal, sim_time, LP, tanker, fully_loaded):
        """When a tanker is finished loading:
        1. Update the FSP and release the LP,
        2. a. If the tanker is fully loaded place the tanker on either the loaded_queue
              or the staged queue,
           b. If the tanker is not fully loaded place the tanker back on loading_queue,
           and
        3. Check to see if there is another tanker that needs to be loaded or unloaded.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type LP: lp.LoadingPoint
        @type tanker: tanker.Tanker
        @type fully_loaded: bool
        """
        FSP = self.FSPs[tanker.fuel_type]
        FSP.update_FSP(scn, sim_time, False)

        LP.release_LP(sim_time)

        if fully_loaded:
            if tanker.TMR_info in self.executed_TMRs:  # The TMR for this tanker has already occurred
                self.loaded_queue[tanker.name] = tanker
                tanker.status_trace.append((sim_time.time, 'on', 'loaded_Q', self.name, np.nan))  # SIM_STATS
            else:
                self.staged_loaded_queue[tanker.name] = tanker
                tanker.status_trace.append((sim_time.time, 'on', 'staged_loaded_Q', self.name, np.nan))  # SIM_STATS
        else:
            FSP.loading_queue.insert(0, tanker)
            FSP.update_loadingQ_trace(sim_time, 1)  # SIM_STATS
            tanker.status_trace.append((sim_time.time, 'on', 'loading_Q', FSP.name, np.nan))  # SIM_STATS

        self.perform_queue_check(scn, sim_cal, sim_time, FSP)

    def daily_plan(self, scn, pol, sim_cal, sim_time):
        """Generating a TMR involves first generating a loading plan.  Then given the
        CURRENTLY available tankers at the ParentNode (i.e. those tankers presently on
        available_queue), determine what tankers will be loaded with what fuel and
        where those loaded tankers will be sent.
        In addition, schedule the submission of the loading_plan and the occurrence
        of the TMR.
        @type scn: scenario.Scenario
        @type pol: policy.Policy
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        """
        # The fuel_requirements will be generated ONLY IF there is at least one
        # tanker on available_queue AND the loading_plan will be determined
        # ONLY IF there is a request for fuel at at least one child node.
        tanker_list = []
        for tanker in self.available_queue.itervalues():
            tanker_list.append(tanker)

        fuel_requirements = []
        if tanker_list:
            fuel_requirements = opt_alg.forecast_fuel_requirements(scn, pol, self, sim_time)

        if fuel_requirements:
            TMR_occurrence_time = sim_time.time + self.generate_time_to_TMR('loaded')
            TMR_info = tmr.TMR(self.name, sim_time.time, TMR_occurrence_time)
            loading_plan = opt_alg.generate_loading_plan(scn, sim_time, self, fuel_requirements, tanker_list, TMR_info)
            self.TMR_idx += 1

            sim_cal.add_event(self, 'submit_loading_plan',
                              [scn, sim_cal, sim_time, loading_plan, TMR_info],
                              sim_time.time)
            sim_cal.add_event(self, 'execute_loaded_tanker_TMR',
                              [scn, sim_cal, sim_time, TMR_info],
                              TMR_occurrence_time)

        sim_cal.add_event(self, 'daily_plan',
                          [scn, pol, sim_cal, sim_time],
                          sim_time.time + 24)

    def submit_loading_plan(self, scn, sim_cal, sim_time, loading_plan, TMR_info):
        """Initiate the loading of all tankers as specified in the loading_plan
        by taking them off the available_queue and placing them on the loading_queue.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type loading_plan: list[(tanker.Tanker, str, child_node.ChildNode)]
        @type TMR_info: tmr.TMR
        """
        for tanker, fuel_type, cNode in loading_plan:
            del self.available_queue[tanker.name]

            # Update tanker with loading_plan information
            tanker.coming_from = self
            tanker.going_to = cNode
            tanker.TMR_info = TMR_info

            FSP = self.FSPs[fuel_type]
            FSP.loading_queue.append(tanker)
            FSP.update_loadingQ_trace(sim_time, 1)  # SIM_STATS
            tanker.status_trace.append((sim_time.time, 'on', 'loading_Q', FSP.name, np.nan))  # SIM_STATS

            sim_cal.add_event(FSP, 'load_tanker',
                              [scn, sim_cal, sim_time, self],
                              sim_time.time)

    def execute_loaded_tanker_TMR(self, scn, sim_cal, sim_time, current_TMR):
        """Each tanker that has been loaded by sim_time.time sits on either the loaded_queue
        (either a tanker that was done loading when its associated TMR occurred but did not
        get sent out or a tanker that only finished loading after its associated TMR occurred),
        or on the staged_loaded_queue (a tanker that is done loading before its associated TMR has
        occurred).
        Tankers on staged_loaded_queue with tanker.TMR_info.occurrence_time equal to
        current_TMR_info.occurrence_time can either be sent out or sent to the loaded_queue.
        Any tanker already on loaded_queue will be sent out if there is at least one tanker
        on the staged_loaded_queue that is being sent to the same child node.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type current_TMR: tmr.TMR
        """
        # Each time a TMR occurs, compute the convoy travel time from the IMNode to each child node
        convoy_travel_times = {}
        for cNode in self.child_nodes:
            convoy_travel_times[cNode.name] = self.generate_CTT(scn, cNode.name)

        # staged_loaded_queue can be left non-empty (holding tankers that have finished loading
        # but whose TMR has yet to occur).
        tankers_sent_to = set()
        temp_queue = []
        for tanker in self.staged_loaded_queue.values():
            assert (tanker.capacity - tanker.on_board) < scn.fuel_epsilon, \
                '%s is NOT carrying a full load (only %0.1f of %0.1f) %3.3f' % \
                (tanker.name, tanker.on_board, tanker.capacity, sim_time.time)
            if tanker.TMR_info == current_TMR:
                if self.on_convoy():
                    self.tankers_sent_trace.append((sim_time.time, tanker.name, tanker.going_to.name,  # SIM_STATS
                                                    tanker.fuel_type, tanker.capacity,  # SIM_STATS
                                                    tanker.TMR_info.occurrence_time))  # SIM_STATS
                    tankers_sent_to.add(tanker.going_to.name)
                    self.send_loaded_tanker_to_child_node(scn, sim_cal, sim_time, tanker, convoy_travel_times)
                else:
                    del self.staged_loaded_queue[tanker.name]
                    temp_queue.append(tanker)  # Place on temporary queue - will be placed on loaded_queue later

        # Each tanker on loaded_queue is only sent out if there is at least one tanker being sent
        # to the same child node on the TMR
        for tanker in self.loaded_queue.values():
            if tanker.going_to.name in tankers_sent_to:
                assert (tanker.capacity - tanker.on_board) < scn.fuel_epsilon, \
                    '%s is NOT carrying a full load' % tanker.name
                self.tankers_sent_trace.append((sim_time.time, tanker.name, tanker.going_to.name,  # SIM_STATS
                                                tanker.fuel_type, tanker.capacity,  # SIM_STATS
                                                tanker.TMR_info.occurrence_time))  # SIM_STATS

                self.send_loaded_tanker_to_child_node(scn, sim_cal, sim_time, tanker, convoy_travel_times)

        # Place any tankers that did not get put on a convoy on loaded_queue.  This cannot
        # be done before the above loop otherwise those tankers that were probabilistically
        # not placed on the convoy WILL be sent out
        for tanker in temp_queue:
            self.loaded_queue[tanker.name] = tanker
            tanker.status_trace.append((sim_time.time, 'on', 'loaded_Q', self.name, np.nan))  # SIM_STATS

        self.executed_TMRs.append(current_TMR)

    def send_loaded_tanker_to_child_node(self, scn, sim_cal, sim_time, tanker, convoy_travel_times):
        """Remove the tanker from the queue it was on (staged_loaded_queue or loaded_queue)
        and add it to en_route_queue.  Add an event to sim_cal to send the tanker
        to the destination child node.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type tanker: tanker.Tanker
        @type convoy_travel_times: dict[str, float]
        """
        if tanker.name in self.staged_loaded_queue:
            del self.staged_loaded_queue[tanker.name]
        else:
            del self.loaded_queue[tanker.name]

        self.en_route_queue[tanker.name] = tanker
        tanker.status_trace.append((sim_time.time, 'sent to', tanker.going_to.name, self.name, np.nan))  # SIM_STATS

        sim_cal.add_event(tanker.going_to, 'arrive_at_child_node',
                          [scn, sim_cal, sim_time, self, tanker],
                          sim_time.time + convoy_travel_times[tanker.going_to.name])

    def update_fuel_requirements_trace(self, sim_time, fuel_requirements, tanker_list):  # SIM_STATS
        """For SIM_STATS add the current day's fuel requests and available tankers
        information to the IMNode fuel_requirements_trace.
        @type sim_time: sim_engine.SimulationTime
        @type fuel_requirements: list[(child_node.ChildNode, str, float, float)]
        @type tanker_list: list[tanker.Tanker]
        """
        fuel_reqs_str = ''
        for entry in fuel_requirements:
            fuel_reqs_str += entry[0].name + '\t' + entry[1] + '\t' + \
                str(entry[2]) + '\t' + str(entry[3]) + '\n'

        avail_tankers_str = ''
        for entry in tanker_list:
            avail_tankers_str += entry.name + '\t' + str(entry.capacity) + '\n'

        self.fuel_requirements_trace.append([sim_time.time, fuel_reqs_str, avail_tankers_str])  # SIM_STATS

    def update_loading_plans_trace(self, sim_time, loading_plan, the_TMR):  # SIM_STATS
        """For SIM_STATS add the current day's loading plan information to the IMNode
        loading_plans_trace.
        @type sim_time: sim_engine.SimulationTime
        @type loading_plan: list[(tanker.Tanker, str, child_node.ChildNode)]
        @type the_TMR: tmr.TMR
        """
        for entry in loading_plan:
            self.loading_plans_trace.append((sim_time.time, self.TMR_idx, entry[0].name, entry[1],  # SIM_STATS
                                            entry[0].capacity, entry[2].name, the_TMR.occurrence_time))  # SIM_STATS
