import math
import operator


# noinspection PyPep8Naming
def calc_fuel_destined_for(pNode, cNode, fuel_type):
    """Return the amount of fuel_type that is destined for the child node, i.e.:
    1. To be loaded at the parent node,
    2. Is being loaded at the parent node,
    3. Has been loaded at the parent node, OR
    4. Is en-route to the child node,
    5. Is in the soak yard at the child node,
    6. To be unloaded at the child node, and
    7. Is being unloaded at the child node.
    ASSUMPTION: Tanker will be filled to capacity at the parent node.
    ASSUMPTION: Tanker will be completely unloaded at the child node.
    @type pNode: parent_node.ParentNode
    @type cNode: child_node.ChildNode
    @type fuel_type: str
    @rtype destined_for: float
    """
    destined_for = 0.0
    for tanker in pNode.FSPs[fuel_type].loading_queue:  # To be loaded at the parent node
        if tanker.going_to.name == cNode.name:
            destined_for += tanker.capacity  # ASSUMPTION: tanker will be filled to capacity

    for LP in pNode.FSPs[fuel_type].LPs:  # Is being loaded at the parent
        if (not LP.is_available()) and (LP.tanker.going_to.name == cNode.name):
            destined_for += LP.tanker.capacity  # ASSUMPTION: tanker will be filled to capacity

    for tanker in pNode.staged_loaded_queue.itervalues():  # Has been staged at the parent node
        if (tanker.going_to.name == cNode.name) and (tanker.fuel_type == fuel_type):
            destined_for += tanker.on_board

    for tanker in pNode.loaded_queue.itervalues():  # Has been loaded at the parent node
        if (tanker.going_to.name == cNode.name) and (tanker.fuel_type == fuel_type):
            destined_for += tanker.on_board

    for tanker in pNode.en_route_queue.itervalues():  # Is en route to the child node
        if (tanker.going_to.name == cNode.name) and (tanker.fuel_type == fuel_type):
            destined_for += tanker.on_board

    for tanker in cNode.soak_yard_queue.itervalues():  # Stuck in the soak yard at the child node
        if tanker.fuel_type == fuel_type:
            destined_for += tanker.on_board

    for tanker in cNode.FSPs[fuel_type].unloading_queue:  # To be unloaded at the child node
        destined_for += tanker.on_board  # ASSUMPTION: tanker will be completely unloaded

    for LP in cNode.FSPs[fuel_type].LPs:  # Is being unloaded at the child node
        if (not LP.is_available()) and (LP.action == 'download'):
            destined_for += LP.tanker.on_board  # ASSUMPTION: tanker will be completely unloaded

    return destined_for


# noinspection PyPep8Naming
def forecast_fuel_requirements(scn, pol, pNode, sim_time):
    """Return a list that contains (for each child node--fuel_type combination that
    satisfies certain conditions) the amount of fuel of that fuel type to send to
    the child node, along with the priority score for that child node--fuel_type
    combination (larger priority score equals higher priority).
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type pNode: parent_node.ParentNode
    @type sim_time: sim_engine.SimulationTime
    @return: fuel_requirements
    @rtype: list[(child_node.ChildNode, str, float, float)]
    """
    # Update all FSPs at the parent node
    for FSP in pNode.FSPs.itervalues():
        FSP.update_FSP(scn, sim_time, False)

    if scn.fuel_req_setting == 'alpha':
        fuel_requirements = forecast_fuel_reqs_alpha(scn, pol, pNode, sim_time)
    elif scn.fuel_req_setting == 'delta':
        fuel_requirements = forecast_fuel_reqs_delta(scn, pol, pNode, sim_time)
    else:
        assert False, 'Unknown fuel_req_setting: %s' % scn.fuel_req_setting

    return fuel_requirements


# noinspection PyPep8Naming
def forecast_fuel_reqs_alpha(scn, pol, pNode, sim_time):
    """Return a list that contains (for each child node--fuel type combination where
    the expected_on_hand amount was negative) the amount of fuel of that fuel type
    to send to the child, along with the priority score for that child node--fuel type
    combination (larger priority score equals higher priority).
    This forecast DOES NOT take into account any stockout information (either
    probability of stockout or severity of stockout from the previous policy).
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type pNode: parent_node.ParentNode
    @type sim_time: sim_engine.SimulationTime
    @return: fuel_requirements
    @rtype: list[(child_node.ChildNode, str, float, float)]
    """
    days_to_next_convoy = 1.0
    fuel_requirements = []
    for cNode in pNode.child_nodes:
        # Earliest expected time (in days) a tanker from a loading plan submitted
        # days_to_next_convoy later will reach the child node.
        # ASSUMPTION: Enough fuel of each type of fuel required is on hand at the parent node.
        # ASSUMPTION: Enough tankers to load all fuel of each type required are available
        # at the parent node.
        # ASSUMPTION: All tankers will be loaded by the time the TMR is executed.
        # ASSUMPTION: Fuel is available for use as soon as it has spent the 24 hours
        # in the soak yard at the child node.
        expected_days_to_child = days_to_next_convoy + \
                                 (24 + pNode.expected_CTT(scn, cNode.name) +
                                  pNode.expected_time_to_TMR('loaded')) / 24.0

        for FSP in cNode.FSPs.itervalues():
            FSP.update_FSP(scn, sim_time, False)

            # +1 in FSP.expected_demand to account for demand during time in soak_yard
            expected_on_hand = (FSP.on_hand + calc_fuel_destined_for(pNode, cNode, FSP.fuel_type)
                                - FSP.expected_demand(scn, sim_time, expected_days_to_child + 1))

            safety_stock_days = scn.base_safety_stock_days
            safety_stock_days += pol.adj_factor.get((cNode.name, FSP.fuel_type), 0)

            safety_stock = max(0,
                               FSP.expected_demand(scn, sim_time, safety_stock_days)
                               * math.copysign(1, safety_stock_days))

            if expected_on_hand < (FSP.storage_min + safety_stock):
                amount_to_send = FSP.SO - FSP.storage_min
                expected_next_single_day_demand = FSP.expected_demand(scn, sim_time, 1)
                if expected_next_single_day_demand > scn.fuel_epsilon:
                    priority_score = (((FSP.storage_min + safety_stock) - expected_on_hand)
                                      / expected_next_single_day_demand)
                else:
                    priority_score = 0.0
                fuel_requirements.append((cNode, FSP.fuel_type, amount_to_send, priority_score))

    return fuel_requirements


# noinspection PyPep8Naming
def forecast_fuel_reqs_delta(scn, pol, pNode, sim_time):
    """Return a list that contains (for each child node--fuel_type combination where
    the expected_on_hand amount was negative) the amount of fuel of that fuel type
    to send to the child node, along with the priority score for that child node--fuel_type
    combination (larger priority score equals higher priority).
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type pNode: parent_node.ParentNode
    @type sim_time: sim_engine.SimulationTime
    @return: fuel_requirements
    @rtype: list[(child_node.ChildNode, str, float, float)]
    """
    days_to_next_convoy = 1.0
    fuel_requirements = []
    for cNode in pNode.child_nodes:
        # Earliest expected time (in days) a tanker from a loading plan submitted
        # days_to_next_convoy later will reach the child node.
        # ASSUMPTION: Enough fuel of each type of fuel required is on hand at the parent node.
        # ASSUMPTION: Enough tankers to load all fuel of each type required are available
        # at the parent node.
        # ASSUMPTION: All tankers will be loaded by the time the TMR is executed.
        # ASSUMPTION: Fuel is available for use as soon as it has spent the 24 hours
        # in the soak yard at the child node.
        expected_days_to_child = days_to_next_convoy + \
                                 (24 + pNode.expected_CTT(scn, cNode.name) +
                                  pNode.expected_time_to_TMR('loaded')) / 24.0
        expected_arrival_time = sim_time.time + (expected_days_to_child * 24.0)

        for FSP in cNode.FSPs.itervalues():
            FSP.update_FSP(scn, sim_time, False)

            # This is the expected amount of demand that was not satisfied under the last policy during
            # the time interval [sim_time.time, expected_arrival_time + 24.0], i.e., during the time the
            # convoy is travelling to the child node plus the 24 hours the fuel spends in the soak_yard.
            expected_stocked_out_demand = expected_stock_out(scn, pol, sim_time.time,
                                                             (expected_arrival_time + 24.0), cNode.name, FSP)

            # +1 in FSP.expected_demand to account for demand during time in soak_yard
            expected_on_hand = (FSP.on_hand + calc_fuel_destined_for(pNode, cNode, FSP.fuel_type)
                                - FSP.expected_demand(scn, sim_time, expected_days_to_child + 1)
                                - expected_stocked_out_demand)

            safety_stock_days = scn.base_safety_stock_days
            safety_stock_days += pol.adj_factor.get((cNode.name, FSP.fuel_type), 0)

            safety_stock = max(0,
                               FSP.expected_demand(scn, sim_time, safety_stock_days)
                               * math.copysign(1, safety_stock_days))

            if expected_on_hand < (FSP.storage_min + safety_stock):
                amount_to_send = FSP.SO - FSP.storage_min
                expected_next_single_day_demand = FSP.expected_demand(scn, sim_time, 1)
                if expected_next_single_day_demand > scn.fuel_epsilon:
                    priority_score = (((FSP.storage_min + safety_stock) - expected_on_hand)
                                      / expected_next_single_day_demand)
                else:
                    priority_score = 0.0
                fuel_requirements.append((cNode, FSP.fuel_type, amount_to_send, priority_score))

    return fuel_requirements


# noinspection PyPep8Naming
def expected_stock_out(scn, pol, start_time, end_time, cNode_name, FSP):
    """Return the expected amount of demand that was not satisfied for the given
    child node--fuel type combination during the time interval [start_time, end_time].
    NOTE: Because of the way get_index_list() works, an interval of up to length
    scn.analysis_interval_length may be omitted from the list returned by
    get_index_list, and so expected_stock_out is really a lower bound on the
    expected amount of demand not satisfied.
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type start_time: float
    @type end_time: float
    @type cNode_name: str
    @type FSP: fsp.ChildNodeFSP
    @return: expected_DNS
    @rtype: float
    """
    expected_DNS = 0
    if pol.idx > 0:
        for idx in get_index_list(scn, pol, cNode_name, FSP, start_time, end_time):
            pol_idx = pol.lineage[FSP.fuel_type][-1]
            expected_DNS += (pol.history[pol_idx].cNode_PSSO[(cNode_name, FSP.fuel_type, 'PSO')][idx][3] *
                             pol.history[pol_idx].cNode_PSSO[(cNode_name, FSP.fuel_type, 'amount')][idx])
    return expected_DNS


# noinspection PyPep8Naming
def get_index_list(scn, pol, cNode_name, FSP, start_time, end_time):
    """Return a list with indices corresponding to the entries in
    cNode_PSSO[(cNode_name, FSP.fuel_type, 'int_start')] from the previous
    policy that fall into the time interval [start_time, end_time].
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type cNode_name: str
    @type FSP: fsp.ChildNodeFSP
    @type start_time: float
    @type end_time: float
    @return: list[int]
    """
    index_list = []
    pol_idx = pol.lineage[FSP.fuel_type][-1]
    interval_start_times = pol.history[pol_idx].cNode_PSSO[(cNode_name, FSP.fuel_type, 'int_start')]
    for i, interval_start_time in enumerate(interval_start_times):
        # The whole interval (of length scn.analysis_interval_length) must fall into
        # the time interval [start_time, end_time]
        if (start_time <= interval_start_time) and \
                ((interval_start_time + scn.analysis_interval_length) <= end_time):
            index_list.append(i)

    return index_list


# noinspection PyPep8Naming
def generate_loading_plan(scn, sim_time, pNode, fuel_requirements, tanker_list, TMR_info):
    """Returns a loading plan.
    @type scn: scenario.Scenario
    @type sim_time: sim_engine.SimulationTime
    @type pNode: parent_node.ParentNode
    @type fuel_requirements: list[(child_node.ChildNode, str, float, float)]
    @type tanker_list: list[tanker.Tanker]
    @type TMR_info: tmr.TMR
    @return: loading_plan
    @rtype: list[(tanker.Tanker, str, child_node.ChildNode)]
    """
    # Sort fuel_requirements by the priority_score of each entry.  The larger the
    # priority score the more urgent the need, so the most urgent will be at
    # the start of the list.
    fuel_requirements.sort(key=lambda x: x[3], reverse=True)

    # The largest capacity tankers should be filled first, so they should be at
    # the start of the list.
    tanker_list.sort(key=operator.attrgetter('capacity'), reverse=True)

    if scn.rep_analysis_lvl in ['all']:
        pNode.update_fuel_requirements_trace(sim_time, fuel_requirements, tanker_list)  # SIM_STATS

    if scn.load_plan_setting == 'greedy':
        loading_plan = greedy_loading_plan(fuel_requirements, tanker_list)
    elif scn.load_plan_setting == 'non_greedy':
        loading_plan = non_greedy_loading_plan(scn, fuel_requirements, tanker_list)
    else:
        assert False, 'Unknown load_plan_setting: %s' % scn.load_plan_setting

    # if scn.rep_analysis_lvl in ['all']:
    #     pNode.update_loading_plans_trace(sim_time, loading_plan, TMR_info)  # SIM_STATS
    pNode.update_loading_plans_trace(sim_time, loading_plan, TMR_info)  # SIM_STATS

    return loading_plan


# noinspection PyPep8Naming
def greedy_loading_plan(fuel_requirements, tanker_list):
    """Greedy heuristic to allocate fuel types and amounts contained on fuel_requirements
    list to the tankers on the tanker_list. fuel_requirements contains tuples of
    (cNode, fuel_type, amount_to_send, priority_score), while tanker_list contains
    the tankers that are currently on available_queue.
    @type fuel_requirements: list[(child_node.ChildNode, str, float, float)]
    @type tanker_list: list[tanker.Tanker]
    @return: loading_plan
    @rtype: list[(tanker.Tanker, str, child_node.ChildNode)]
    """
    loading_plan = []
    for cNode, fuel_type, amount_to_send, p_s in fuel_requirements:
        amount_left = amount_to_send

        fuel_type_tanker_list = [tnkr for tnkr in tanker_list if tnkr.fuel_type == fuel_type]
        for tanker in fuel_type_tanker_list:
            loading_plan.append((tanker, fuel_type, cNode))
            amount_left -= tanker.capacity
            tanker_list.remove(tanker)
            if amount_left < 0:
                break

    return loading_plan


# noinspection PyPep8Naming
def non_greedy_loading_plan(scn, fuel_requirements, tanker_list):
    """Attempts to satisfy all the fuel fuel_requirements
    ASSUMPTION:  All tankers on tanker_list have the same capacity.
    @type scn: scenario.Scenario
    @type fuel_requirements: list[(child_node.ChildNode, str, float, float)]
    @type tanker_list: list[tanker.Tanker]
    @return: loading_plan
    @rtype: list[(tanker.Tanker, str, child_node.ChildNode)]
    """
    for tanker in tanker_list:
        if tanker_list[0].capacity != tanker.capacity:
            assert False, 'All tankers must have the same capacity for this algorithm to work'

    loading_plan = []
    for fuel_type in scn.fuel_cNode_info.iterkeys():
        fuel_type_requirements = [fuel_requirement_as_tankers(requirement, tanker_list[0].capacity)
                                  for requirement in fuel_requirements if requirement[1] == fuel_type]
        # Sort on num_tankers needed for the fuel requirement, descending order
        fuel_type_requirements.sort(key=lambda x: x[1], reverse=True)

        fuel_type_tanker_list = [tanker for tanker in tanker_list if tanker.fuel_type == fuel_type]

        index = 0
        while fuel_type_tanker_list and fuel_type_requirements:
            if fuel_type_requirements[index][1] > 0:
                tanker = fuel_type_tanker_list.pop()
                cNode = fuel_type_requirements[index][0]
                fuel_type_requirements[index][1] -= 1
                loading_plan.append((tanker, fuel_type, cNode))

                if fuel_type_requirements[index][1] == 0:
                    fuel_type_requirements.pop(index)

            if fuel_type_requirements:
                index += 1
                index %= len(fuel_type_requirements)

    return loading_plan


# noinspection PyPep8Naming,PyUnusedLocal
def fuel_requirement_as_tankers(fuel_request, tanker_capacity):
    """Return a cNode and the number of tankers required for the fuel requirement passed in.
    @type fuel_request: (child_node.ChildNode, str, float, float)
    @type tanker_capacity: float
    @rtype: list[child_node.ChildNode, int]
    """
    cNode, _, amount_to_send, _ = fuel_request
    if amount_to_send < tanker_capacity:
        return [cNode, 1]

    num_tankers = int(amount_to_send / float(tanker_capacity))
    if (amount_to_send - (num_tankers * tanker_capacity)) > (0.25 * tanker_capacity):
        num_tankers += 1

    return [cNode, num_tankers]
