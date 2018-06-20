import constants as const
from itertools import izip
import matplotlib.pyplot as plt
import misc
import numpy as np
import pandas as pd


# noinspection PyPep8Naming
def replication_analysis(scn, pol, the_seed, rep_data):
    """Analysis of a single simulation replication.  Depending on the value
    of rep_analysis_lvl, different amounts of analysis are performed.
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type the_seed: int
    @type rep_data: (sim_engine.SimulationCalendar, parent_node.ParentNode,
                        dict[str, child_node.ChildNode], list[Tanker])
    @param rep_data: A tuple containing instances of the simulation calendar (sim_cal),
        the parent node instance (sim_pNode), a dictionary of instances of child nodes
        (sim_cNodes), and a list of instances of tankers (sim_tankers).
    """
    rep_path = pol.path + '/' + str(the_seed)
    misc.create_replication_dir(pol.path, rep_path)

    sim_cal, sim_pNode, sim_cNodes, sim_tankers = rep_data
    if scn.rep_analysis_lvl in ['plots', 'data', 'all']:
        plot_tanker_utilization_histogram(scn, sim_pNode, sim_tankers, rep_path)

        for FSP in sim_pNode.FSPs.itervalues():
            plot_FSP_traces(scn, FSP, rep_path, sim_pNode.name, 'parent', False)
        for cNode in sim_cNodes.itervalues():
            for FSP in cNode.FSPs.itervalues():
                plot_FSP_traces(scn, FSP, rep_path, cNode.name, 'child', True)
                # plot_FSP_traces(scn, FSP, rep_path, cNode.name, 'child', False)

    if scn.rep_analysis_lvl in ['data', 'all']:
        directories = ["/FSP_inventory", "/FSP_queues", "/FSP_demand", "/LPs", "/tankers"]
        misc.create_replication_sub_dirs(pol.path, rep_path, directories)

        write_all_FSP_inventory_traces(sim_pNode, sim_cNodes, rep_path)
        write_all_cNode_FSP_demand_traces(sim_cNodes, rep_path)
        write_all_FSP_queue_traces(sim_pNode, sim_cNodes, rep_path)
        write_all_LP_utilization_traces(sim_pNode, sim_cNodes, rep_path)
        write_all_cNode_FSP_PNS(scn, sim_cNodes, rep_path)

    if scn.rep_analysis_lvl in ['all']:
        directories = ["/nodes"]
        misc.create_replication_sub_dirs(pol.path, rep_path, directories)

        write_cNode_TMR_traces(sim_cNodes, rep_path)
        write_tanker_traces(sim_tankers, rep_path)

        file_name = rep_path + '/nodes/' + sim_pNode.name + '_fuel_requirements_trace.txt'
        with open(file_name, 'w') as wrtr:
            for time, loading_plan, available_tankers in sim_pNode.fuel_requirements_trace:
                wrtr.write(str(time) + '\n')
                wrtr.write(loading_plan)
                wrtr.write(available_tankers)

        columns = ['Time', 'TMR_Index', 'Tanker_Name', 'Fuel_Type', 'Fuel_Amount', 'Destination', 'TMR_Occurrence_Time']
        file_name = rep_path + '/nodes/' + sim_pNode.name + '_loading_plans_trace.txt'
        pd.DataFrame(data=sim_pNode.loading_plans_trace, columns=columns).to_csv(file_name, index=False)

        columns = ['Time', 'Name', 'Destination', 'Fuel_Type', 'Fuel_Amount', 'TMR_Occurrence_Time']
        file_name = rep_path + '/nodes/' + sim_pNode.name + '_tankers_sent_trace.txt'
        pd.DataFrame(data=sim_pNode.tankers_sent_trace, columns=columns).to_csv(file_name, index=False)

        # columns = ['Time', 'Name', 'Method']
        # file_name = rep_path + '/calendar_event_trace.txt'
        # pd.DataFrame(data=sim_cal.cal_event_trace, columns=columns).to_csv(file_name, index=False)

    if scn.rep_analysis_lvl not in ['plots', 'data', 'all']:
        assert False, 'Replication analysis level not recognized.'

    print 'Replication %s analysis completed' % str(the_seed)


# noinspection PyPep8Naming
def plot_FSP_traces(scn, FSP, rep_path, node_name, node_subnetwork_role, plot_SO):
    """Plots the on-hand levels, and loading and unloading queue sizes
    on a single plot for the FSP.
    @type scn: scenario.Scenario
    @type FSP: fsp.ParentNodeFSP | fsp.ChildNodeFSP
    @type rep_path: str
    @type node_name: str
    @type node_subnetwork_role: str
    @type plot_SO: bool
    """
    fig, subplot = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex='all',
                                gridspec_kw={'height_ratios': [6, 2.5, 2, 2]})

    plot_FSP_inventory_trace_subplot(scn, subplot[0], FSP, 'On-Hand Inventory\n (thousands of gallons)')
    plot_FSP_LP_traces_subplot(scn, subplot[1], FSP)
    plot_FSP_queueing_trace_subplot(scn, subplot[2], FSP, 'loadingQ', 'Loading\nQueue Size')
    plot_FSP_queueing_trace_subplot(scn, subplot[3], FSP, 'unloadingQ', 'Unloading\nQueue Size')

    if (node_subnetwork_role == 'child') and plot_SO:
        start_times, end_times = calc_SO_intervals(scn, FSP, cluster=True, time_gap=(12 / 24.0))
        for x1, x2 in zip(start_times, end_times):
            subplot[0].axvspan(x1, x2, alpha=0.35, color=const.FIREBRICK_hex)
            subplot[1].axvspan(x1, x2, alpha=0.35, color=const.FIREBRICK_hex)
            subplot[2].axvspan(x1, x2, alpha=0.35, color=const.FIREBRICK_hex)
            subplot[3].axvspan(x1, x2, alpha=0.35, color=const.FIREBRICK_hex)

    subplot[0].set_title('%s, %s' % (node_name, FSP.fuel_type))
    subplot[3].set_xlabel('Time (days)')
    fig.tight_layout()
    if (node_subnetwork_role == 'child') and plot_SO:
        plt.savefig(rep_path + '/' + node_name + '_' + FSP.fuel_type + '_SO' + scn.plot_type, transparent=True)
    else:
        plt.savefig(rep_path + '/' + node_name + '_' + FSP.fuel_type + scn.plot_type, transparent=True)
    plt.close('all')


# noinspection PyPep8Naming
def plot_FSP_inventory_trace_subplot(scn, subplot, FSP, y_label):
    """Plot the on-hand inventory trace of the given FSP on the subplot.
    @type scn: scenario.Scenario
    @type subplot:
    @type FSP: fsp.ParentNodeFSP | fsp.ChildNodeFSP
    @type y_label: str
    """
    time_vals = [time / 24.0 for time, _ in FSP.inv_trace]
    on_hand_vals = [on_hand / 1000.0 for _, on_hand in FSP.inv_trace]
    max_x = max(time_vals)

    subplot.hlines(FSP.storage_max / 1000.0, 0, max_x, colors=const.FUSCHIA_hex, linewidth=1, linestyles='-.')
    subplot.hlines(FSP.storage_min / 1000.0, 0, max_x, colors=const.FUSCHIA_hex, linewidth=1, linestyles='-.')
    subplot.plot(time_vals, on_hand_vals, color=const.QUEEN_BLUE_hex, linewidth=1.5)
    subplot.set_xlim(0, scn.stop_time / 24.0)
    subplot.set_ylim(0, (FSP.storage_max + 500) / 1000.0)
    subplot.yaxis.set_label_position("right")
    subplot.set_ylabel(y_label)
    subplot.hlines(FSP.SO / 1000.0, 0, max_x, colors=const.ORANGE_PEEL_hex, linewidth=1.5, linestyles='--')
    subplot.grid(True)


# noinspection PyPep8Naming
def plot_FSP_LP_traces_subplot(scn, subplot, FSP):
    """Plot the LP traces of the given FSP on the subplot.
    @type scn: scenario.Scenario
    @type subplot:
    @type FSP: fsp.ParentNodeFSP | fsp.ChildNodeFSP
    """
    start_times, end_times, y_pos, color = [], [], [], []
    y_val = 0
    for LP in FSP.LPs:
        num_on, num_off = 0, 0
        for time, _, action, amount in LP.util_trace:
            if action == 'on':
                num_on += 1
                start_times.append(time / 24.0)
                y_pos.append(y_val)
                if amount < 0:
                    color.append(const.BOYSENBERRY_rgb)
                else:
                    color.append(const.BLACK_OLIVE_rgb)
            elif action == 'off':
                num_off += 1
                end_times.append(time / 24.0)

        if num_off < num_on:
            num_off += 1
            end_times.append(scn.stop_time / 24.0)
        assert num_on == num_off, 'Issue in r_analysis.get_LP_intervals'

        y_val += 1

    subplot.hlines(y_pos, start_times, end_times, color=color, linewidth=10)
    subplot.set_ylim(-0.5, y_val-0.5)
    subplot.set_xlim(0, scn.stop_time / 24.0)
    subplot.set_yticklabels([])
    subplot.yaxis.set_label_position("right")
    subplot.set_ylabel('Loading Point Usage')
    subplot.grid(True)


# noinspection PyPep8Naming
def plot_FSP_queueing_trace_subplot(scn, subplot, FSP, Q_type, y_label):
    """Plot the queuing trace of the given FSP on the subplot.
    @type scn: scenario.Scenario
    @type subplot:
    @type FSP: fsp.ParentNodeFSP | fsp.ChildNodeFSP
    @type Q_type: str
    @type y_label: str
    """
    if Q_type == 'loadingQ':
        Q_trace = FSP.loadingQ_trace
        color = const.BOYSENBERRY_hex
    else:
        Q_trace = FSP.unloadingQ_trace
        color = const.BLACK_OLIVE_hex

    time_vals = [time / 24.0 for time, _ in Q_trace]
    onQ_vals = [onQ for _, onQ in Q_trace]
    subplot.step(time_vals, onQ_vals, where='post', color=color, linewidth=1.5)
    subplot.set_ylim(0, max(onQ_vals) + 1)
    subplot.set_xlim(0, scn.stop_time / 24.0)
    subplot.yaxis.set_label_position("right")
    subplot.set_ylabel(y_label)
    subplot.grid(True)


# noinspection PyPep8Naming
def calc_SO_intervals(scn, FSP, cluster, time_gap):
    """Return two lists containing the start and end time (in days) of each
    stock out interval.  If cluster is True, then two stock out intervals
    are clustered (merged into one interval) if the end of the first interval
    and the start of the second interval occur within time_gap days of each
    other.  time_gap is in units of days.
    @type scn: scenario.Scenario
    @type FSP: fsp.ChildNodeFSP
    @type cluster: bool
    @type time_gap: float
    @return: (list[float], list[float], list[int])
    """
    SO_interval_start, SO_interval_end = [], []
    for request_time, fulfill_time, amount in FSP.demand_trace:
        if amount < scn.fuel_epsilon:
            SO_interval_start.append(request_time / 24.0)
            SO_interval_end.append(fulfill_time / 24.0)
            try:
                if cluster and ((SO_interval_start[-1] - SO_interval_end[-2]) < time_gap):
                    SO_interval_start.pop()
                    SO_interval_end.pop()
                    SO_interval_end[-1] = fulfill_time / 24.0
            except IndexError:
                pass

    return SO_interval_start, SO_interval_end


# noinspection PyPep8Naming
def write_all_FSP_inventory_traces(sim_pNode, sim_cNodes, rep_path):
    """Write out the (on-hand) inventory trace for each FSP at each node (parent and child).
    @type sim_pNode: parent_node.ParentNode
    @type sim_cNodes: dict[str, child_node.ChildNode]
    @type rep_path: str
    """
    columns = ['Time', 'On_Hand']
    # Parent node
    for FSP in sim_pNode.FSPs.values():
        file_name = rep_path + '/FSP_inventory/' + FSP.log_label + '_invt_trace.txt'
        pd.DataFrame(data=FSP.inv_trace, columns=columns).to_csv(file_name, index=False)
    # Child nodes
    for cNode in sim_cNodes.values():
        for FSP in cNode.FSPs.itervalues():
            file_name = rep_path + '/FSP_inventory/' + FSP.log_label + '_invt_trace.txt'
            pd.DataFrame(data=FSP.inv_trace, columns=columns).to_csv(file_name, index=False)


# noinspection PyPep8Naming
def write_all_cNode_FSP_demand_traces(sim_cNodes, rep_path):
    """Write out the demand trace for each FSP at each child node.
    @type sim_cNodes: dict[str, child_node.ChildNode]
    @type rep_path: str
    """
    columns = ['Time_Requested', 'Time_Fulfilled', 'Amount_Fulfilled']
    for cNode in sim_cNodes.values():
        for FSP in cNode.FSPs.values():
            file_name = rep_path + '/FSP_demand/' + FSP.log_label + '_dmnd_trace.txt'
            pd.DataFrame(data=FSP.demand_trace, columns=columns).to_csv(file_name, index=False)


# noinspection PyPep8Naming
def write_all_FSP_queue_traces(sim_pNode, sim_cNodes, rep_path):
    """Write out both the loading and unloading queue traces for all FSPs
    at all nodes.
    @type sim_pNode: parent_node.ParentNode
    @type sim_cNodes: dict[str, child_node.ChildNode]
    @type rep_path: str
    """
    columns = ['Time', 'Num_Tankers']
    # Parent node
    for FSP in sim_pNode.FSPs.values():
        file_name = rep_path + '/FSP_queues/' + FSP.log_label + '_ldQ_trace.txt'
        pd.DataFrame(data=FSP.loadingQ_trace, columns=columns).to_csv(file_name, index=False)

        file_name = rep_path + '/FSP_queues/' + FSP.log_label + '_unldQ_trace.txt'
        pd.DataFrame(data=FSP.unloadingQ_trace, columns=columns).to_csv(file_name, index=False)
    # Child nodes
    for cNode in sim_cNodes.values():
        for FSP in cNode.FSPs.values():
            file_name = rep_path + '/FSP_queues/' + FSP.log_label + '_ldQ_trace.txt'
            pd.DataFrame(data=FSP.loadingQ_trace, columns=columns).to_csv(file_name, index=False)

            file_name = rep_path + '/FSP_queues/' + FSP.log_label + '_unldQ_trace.txt'
            pd.DataFrame(data=FSP.unloadingQ_trace, columns=columns).to_csv(file_name, index=False)


# noinspection PyPep8Naming
def write_all_LP_utilization_traces(sim_pNode, sim_cNodes, rep_path):
    """Write out the utilization trace for each LP at each FSP at each
    node (parent and child).
    @type sim_pNode: parent_node.ParentNode
    @type sim_cNodes: dict[str, child_node.ChildNode]
    @type rep_path: str
    """
    columns = ['Time', 'Tanker_Name', 'Action', 'Fuel_Amount']
    # Parent node
    for FSP in sim_pNode.FSPs.values():
        for LP in FSP.LPs:
            if LP.util_trace:
                file_name = rep_path + '/LPs/' + sim_pNode.name + \
                    '_' + FSP.name + '_' + LP.log_label + '_util_trace.txt'
                pd.DataFrame(data=LP.util_trace, columns=columns).to_csv(file_name, index=False)
    # Child nodes
    for cNode in sim_cNodes.values():
        for FSP in cNode.FSPs.values():
            for LP in FSP.LPs:
                if LP.util_trace:
                    file_name = rep_path + '/LPs/' + cNode.name + \
                        '_' + FSP.name + '_' + LP.log_label + '_util_trace.txt'
                    pd.DataFrame(data=LP.util_trace, columns=columns).to_csv(file_name, index=False)


# noinspection PyPep8Naming
def write_cNode_TMR_traces(sim_cNodes, rep_path):
    """Write out the TMR traces for each child node, not including the times when
    no tankers were sent out.
    @type sim_cNodes: dict[str, child_node.ChildNode]
    @type rep_path: str
    """
    columns = ['Time', 'Num_Tankers']
    for cNode in sim_cNodes.values():
        file_name = rep_path + '/nodes/' + cNode.name + '_TMR_trace.txt'
        df = pd.DataFrame(data=cNode.TMR_trace, columns=columns)
        df[df['Num_Tankers'] > 0].to_csv(file_name, index=False)


def write_tanker_traces(sim_tankers, rep_path):
    """For each tanker, write out the tanker trace.
    @type sim_tankers: list[tanker.Tanker]
    @type rep_path: str
    """
    columns = ['Time', 'Action', 'Loc_1', 'Loc_2', 'Fuel_Amount']
    for tanker in sim_tankers:
        file_name = rep_path + '/tankers/' + tanker.name + '_stat_trace.txt'
        pd.DataFrame(data=tanker.status_trace, columns=columns).to_csv(file_name, index=False)


# noinspection PyPep8Naming
def plot_tanker_utilization_histogram(scn, sim_pNode, sim_tankers, rep_path):
    """For each type of tanker and each type of fuel combination, plot a histogram
    displaying the proportion of tankers that made different numbers of trips, where
    a trip is defined as being sent from point A to point B and returning to point A.
    @type scn: scenario.Scenario
    @type sim_pNode: parent_node.ParentNode
    @type sim_tankers: list[tanker.Tanker]
    @type rep_path: str
    """
    columns = ['Time', 'Action', 'Loc_1', 'Loc_2', 'Fuel_Amount']
    for tanker_type in ['beta']:
        for fuel_type in scn.fuel_cNode_info.keys():
            usage = []
            for tanker in sim_tankers:
                if (tanker.tanker_type == tanker_type) and (tanker.fuel_type == fuel_type):
                    df = pd.DataFrame(data=tanker.status_trace, columns=columns)
                    df = df[(df['Action'] == 'arrived at') & (df['Loc_1'] == sim_pNode.name)]
                    usage.append(len(df))

            df2 = pd.DataFrame({'Count': usage})
            df2.plot(kind='hist', bins=np.arange(-0.5, max(usage) + 1, 1), alpha=0.5)
            plt.title('Type B Tankers, ' + fuel_type)
            plt.xlabel('Number of times sent to child node')
            plt.legend().remove()
            plt.ylabel('Number of tankers')
            plt.xlim(-0.75, max(usage) + 0.75)
            plt.savefig(rep_path + '/type_' + tanker_type + '_' +
                        fuel_type + '_tankers' + scn.plot_type, transparent=True)
            plt.close('all')


# noinspection PyPep8Naming
def write_all_cNode_FSP_PNS(scn, sim_cNodes, rep_path):
    """Calculate the proportion of demand not satisfied (PNS) for each fuel type at
    each child node, and write it out to a file.
    @type scn: scenario.Scenario
    @type sim_cNodes: dict[str, child_node.ChildNode]
    @type rep_path: str
    """
    with open(rep_path + '/FSP_demand/prop_NS.txt', 'w') as wrtr:
        for cNode in sim_cNodes.values():
            for FSP in cNode.FSPs.values():
                prop_stocked_out = calc_prop_stocked_out(scn, FSP)
                wrtr.write('%s\t%s\t%.2f\n' % (cNode.name, FSP.fuel_type, prop_stocked_out * 100))


# noinspection PyPep8Naming
def calc_prop_stocked_out(scn, FSP):
    """Return the proportion of demand that was not satisfied (stocked out)
    at the FSP during the planning horizon.
    @type scn: scenario.Scenario
    @type FSP: fsp.ChildNodeFSP
    @return: float
    """
    total_amount_demanded = 0
    amount_stocked_out = 0
    start_time = scn.warm_up * 24.0
    end_time = (scn.warm_up + scn.planning_horizon) * 24.0
    for time_requested, time_fulfilled, amnt in FSP.demand_trace:
        if start_time <= time_requested <= end_time:  # Ensure that only statistics during the planning horizon are
            total_amount_demanded += abs(amnt)
            if amnt < 0.0:
                amount_stocked_out += abs(amnt)

    prop_stocked_out = 0.0
    if total_amount_demanded > 0:
        prop_stocked_out = amount_stocked_out / float(total_amount_demanded)

    return prop_stocked_out


def calc_overlapping_intervals(all_interval_times):
    """Given a list of interval starts and interval ends, calculate the start and
    end of intervals during which one or more, two or more, three or more, etc.
    of the passed in intervals overlap.  The start and end points of the resultant
    intervals are stored in a dictionary where the key indicates the number of
    passed in intervals that overlapped during that time period, and the value is
    a list of either starts or ends of intervals during which the 'key' number of
    passed in intervals overlapped.
    TEST [(1, 's'), (1.5, 's'), (2, 's'), (3.5, 'e'), (4.5, 'e'), (6.5, 's'),
    (6.5, 's'), (7.5, 'e'), (9, 's'), (10, 'e'), (10, 'e'), (11, 'e'), (11, 's'),
    (12, 'e'), (12, 's'), (13, 'e'), (14, 's'), (14, 's'), (15, 'e'), (15, 'e'),
    (15, 's'), (15, 's'), (16, 'e'), (16, 'e')]
    ANS {1: [1, 4.5, 10], 2: [1.5, 3.5, 7.5, 14], 3: [2, 6.5, 9]}
        {1: [1.5, 6.5, 13], 2: [2, 4.5, 9, 16], 3: [3.5, 7.5, 10]
    @type all_interval_times: list[(float, str)]
    @rtype: (dict[int, list[float]], dict[int, list[float]])
    """
    overlapping_interval_starts = {}
    overlapping_interval_ends = {}

    all_interval_times = sorted(all_interval_times)
    count = 1
    if all_interval_times:
        previous_time, _ = all_interval_times.pop(0)
        for current_time, elt_type in all_interval_times:
            if (count != 0) and (previous_time != current_time):
                try:
                    # This takes care of the case where one or more intervals stop
                    # simultaneously and then the same number start again simultaneously
                    # back to back, so that it gets recorded as a single continuous
                    # interval
                    if overlapping_interval_ends[count][-1] == previous_time:
                        overlapping_interval_ends[count].pop(-1)
                        overlapping_interval_ends[count].append(current_time)
                    else:
                        overlapping_interval_starts[count].append(previous_time)
                        overlapping_interval_ends[count].append(current_time)
                except KeyError:
                    overlapping_interval_starts[count] = [previous_time]
                    overlapping_interval_ends[count] = [current_time]

            if elt_type == 's':
                count += 1
            else:
                count -= 1
            previous_time = current_time

    for key in overlapping_interval_starts:
        assert len(overlapping_interval_starts[key]) == len(overlapping_interval_ends[key]), \
            'Issue in r_analysis.calc_overlapping_intervals()'

    return overlapping_interval_starts, overlapping_interval_ends


# noinspection PyPep8Naming
def calc_expected_nzQL(scn, FSP):
    """Returns the expected length (in days) of intervals during the planning
    horizon during which the unloading queue had at least 1 tanker on it.
    @type scn: scenario.Scenario
    @type FSP: fsp.ParentNodeFSP | fsp.ChildNodeFSP
    @return: float
    """
    start_time = scn.warm_up * 24.0
    end_time = (scn.warm_up + scn.planning_horizon) * 24.0
    PH_unloadingQ_trace = [data for data in FSP.unloadingQ_trace if start_time <= data[0] <= end_time]
    unloadingQ_interval_times = calc_Q_intervals(scn, PH_unloadingQ_trace)
    if unloadingQ_interval_times:
        total_length = 0.0
        for start_time_tup, end_time_tup in get_pairs(unloadingQ_interval_times):
            total_length += (end_time_tup[0] - start_time_tup[0]) / 24.0

        return total_length / (len(unloadingQ_interval_times) / 2.0)

    return 0.0


# noinspection PyPep8Naming
def calc_Q_intervals(scn, queue_trace):
    """Return a single list that contains the start times and end times of
    intervals during which the queue had at least one tanker.  Because the
    functions update_loadingQ_trace and update_unloadingQ_trace guarantee
    that each time stamp in the queue_trace is unique, there will be no
    zero length intervals.
    TEST: trace = [(0, 0), (1,1), (3, 2), (4, 3), (5, 0), (7, 5), (10, 3), (11, 2), (12, 1)]
    ANS: [(1, 's'), (5, 'e'), (7, 's'), (12, 'e')]
    TEST: [(0, 0), (1,1), (3, 2), (4, 3), (5, 0), (7, 5), (10, 3), (11, 0), (12, 0)]
    ANS: [(1, 's'), (5, 'e'), (7, 's'), (11, 'e')]
    TEST: [(0, 0), (10, 0)]
    ANS: []
    @type scn: scenario.Scenario
    @type queue_trace: list[(float, int)]
    @rtype: list[(float, int)]
    """
    interval_times = []
    num_starts, num_ends = 0, 0
    for time, Q_length in queue_trace:
        if (Q_length > 0) and (num_starts == num_ends):
            num_starts += 1
            interval_times.append((time, 's'))
        elif (Q_length == 0) and (num_starts > num_ends):
            num_ends += 1
            interval_times.append((time, 'e'))

    # Takes care of the case where the queue length was non-zero at the
    # end of the simulation
    if num_starts > num_ends:
        num_ends += 1
        interval_times.append((scn.stop_time, 'e'))

    assert num_starts == num_ends, 'Q_interval lists of different sizes'

    return interval_times


def get_pairs(iterable):
    """Given an iterable (list, etc.), returns an iterator that provides
    the pairs of elements in the iterable.
    s -> (s0, s1), (s2, s3), (s4, s5), ...
    """
    a = iter(iterable)

    return izip(a, a)
