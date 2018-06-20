import node
import numpy as np


# noinspection PyPep8Naming
class ChildNode(node.Node):
    """ChildNode is a subclass of Node.
    """
    # noinspection PyUnresolvedReferences
    def __init__(self, name, node_type, FSPs, CTT_distributions, TMR_distributions,
                 prob_on_convoy, rep_rndstrm):
        """ChildNode is a subclass of Node.
        """
        node.Node.__init__(self, name, node_type, FSPs, CTT_distributions, TMR_distributions, prob_on_convoy,
                           rep_rndstrm)
        self.subnetwork_role = 'child'
        self.lead_time = {}
        """@type lead_time: dict[str, float]"""

    def __str__(self):
        """
        @rtype: str
        """
        return self.node_type.title() + ' (Child) ' + node.Node.__str__(self)

    def arrive_at_child_node(self, scn, sim_cal, sim_time, pNode, tanker):
        """Removes type beta tanker from parent_node.en_route_queue, places
        it on unloading_queue, and sends the tanker to the soak yard.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type pNode: parent_node.ParentNode
        @type tanker: tanker.Tanker
        """
        assert tanker.on_board > 0, '%s arrived at child node %s empty' % (tanker.name, self.name)
        del pNode.en_route_queue[tanker.name]
        tanker.status_trace.append((sim_time.time, 'arrived at', self.name, np.nan, np.nan))  # SIM_STATS

        self.tanker_to_soak_yard(scn, sim_cal, sim_time, tanker)

    def tanker_finished_loading(self, scn, sim_cal, sim_time, LP, tanker, fully_loaded):
        """When a tanker is finished loading:
        1. Update the FSP and release the LP,
        2. a. If the tanker is fully loaded delete the tanker,
           b. If the tanker is not fully loaded place the tanker back on loading_queue,
           and
        3. Check to see if there is another tanker that needs to be loaded or unloaded.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type LP: lp.LoadingPoint
        @type tanker: tanker.Tanker
        @type fully_loaded: bool
        @return:
        """
        FSP = self.FSPs[tanker.fuel_type]
        FSP.update_FSP(scn, sim_time, False)

        LP.release_LP(sim_time)

        if not fully_loaded:
            # Check to see if the relevant remove_stale_demand_event is still on the simulation calendar
            event_occurred = True
            max_occurrence_time = sim_time.time + scn.stale_after
            for event in [event for event in sim_cal.calendar if event.occurrence_time <= max_occurrence_time]:
                if (event.method == 'remove_stale_demand_event') and (event.args[1].name == tanker.name):
                    event_occurred = False
                    break

            if event_occurred:  # This should count as a stock-out
                demand_not_satisfied = tanker.capacity - tanker.on_board
                FSP.demand_trace.append((tanker.status_trace[0][0], sim_time.time, -demand_not_satisfied))  # SIM_STATS
                del tanker
            else:  # There is still a chance the demand could be satisfied
                FSP.loading_queue.insert(0, tanker)
                FSP.update_loadingQ_trace(sim_time, 1)  # SIM_STATS
                tanker.status_trace.append((sim_time.time, 'on', 'loading_Q', FSP.name, np.nan))  # SIM_STATS
        else:
            FSP.demand_trace.append((tanker.status_trace[0][0], sim_time.time, tanker.on_board))  # SIM_STATS
            FSP.total_fuel_used += tanker.on_board
            del tanker

        self.perform_queue_check(scn, sim_cal, sim_time, FSP)

    def execute_unloaded_tanker_TMR(self, scn, sim_cal, sim_time, pNode):
        """Sends out every tanker that did not go out with the last unloaded
        tanker TMR.  Each new tanker that has been unloaded by the current
        simulation time is probabilistically added to the TMR.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type pNode: parent_node.ParentNode
        """
        # Once per TMR, compute the convoy travel time from this node to the parent_node
        convoy_travel_time = self.generate_CTT(scn, pNode.name)

        num_tankers_sent = 0
        # All tankers that were not sent out with the last TMR are sent out
        for tanker in self.unloaded_queue.values():
            self.send_unloaded_tanker_to_parent_node(scn, sim_cal, sim_time, tanker, convoy_travel_time)
            num_tankers_sent += 1

        for tanker in self.staged_unloaded_queue.values():
            if self.on_convoy():
                self.send_unloaded_tanker_to_parent_node(scn, sim_cal, sim_time, tanker, convoy_travel_time)
                num_tankers_sent += 1
            else:
                del self.staged_unloaded_queue[tanker.name]

                self.unloaded_queue[tanker.name] = tanker
                tanker.status_trace.append((sim_time.time, 'on', 'unloaded_Q', self.name, np.nan))  # SIM_STATS

        self.TMR_trace.append((sim_time.time, num_tankers_sent))  # SIM_STATS

        time_to_next_TMR = self.generate_time_to_TMR('unloaded')
        sim_cal.add_event(self, 'execute_unloaded_tanker_TMR',
                          [scn, sim_cal, sim_time, pNode],
                          sim_time.time + time_to_next_TMR)

    def send_unloaded_tanker_to_parent_node(self, scn, sim_cal, sim_time, tanker, convoy_travel_time):
        """Removes the tanker from the queue it was on (staged_unloaded_queue or
        unloaded_queue), and add an event to sim_cal to send it back to the parent node.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        @type tanker: tanker.Tanker
        @type convoy_travel_time: float
        """
        # The tanker could have come from the staged_unloaded_queue or the unloaded_queue
        if tanker.name in self.staged_unloaded_queue:
            del self.staged_unloaded_queue[tanker.name]
        else:
            del self.unloaded_queue[tanker.name]

        tanker.status_trace.append((sim_time.time, 'sent to', tanker.going_to.name, self.name, np.nan))  # SIM_STATS

        sim_cal.add_event(tanker.going_to, 'arrive_at_parent_node',
                          [scn, sim_cal, sim_time, tanker],
                          sim_time.time + convoy_travel_time)

    def calculate_lead_time(self, scn, euNode_name, fuel_type):
        """Calculates the lead time (in days) for the End User node--fuel type combination
        and stores it in the lead_time dictionary.
        @type scn: scenario.Scenario
        @type euNode_name: str
        @type fuel_type: str
        @rtype: None
        """
        self.lead_time[euNode_name + ':' + fuel_type] = scn.warm_up - \
                                                        (self.FSPs[fuel_type].latest_loading_start_time / 24.0)

    def child_node_daily_demand(self, scn, sim_cal, sim_time):
        """For each fuel_type at the ChildNode, divide the daily consumption into
        hourly amounts and add a demand_event to sim_cal.  Also add an event to
        sim_cal a day later to call child_node_daily_demand again.
        @type scn: scenario.Scenario
        @type sim_cal: sim_engine.SimulationCalendar
        @type sim_time: sim_engine.SimulationTime
        """
        for FSP in self.FSPs.itervalues():
            daily_demand_list = FSP.generate_daily_demand(scn, sim_time.time)

            if daily_demand_list:
                time_delta, demand_amount = daily_demand_list.pop(0)
                sim_cal.add_event(FSP, 'demand_event',
                                  [scn, sim_cal, sim_time, demand_amount, daily_demand_list, self],
                                  sim_time.time + time_delta)

        # Add a daily demand update one day later
        sim_cal.add_event(self, 'child_node_daily_demand',
                          [scn, sim_cal, sim_time],
                          sim_time.time + 24)
