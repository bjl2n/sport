import constants as const
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import r_analysis


def policy_analysis(scn, pol, seed_range):
    """Analyses the current policy output to determine the new adjustment factors
    and policy lineage.
    @type pol: policy.Policy
    @type scn: scenario.Scenario
    @type seed_range: list[int]
    """
    # Replication level analysis
    if scn.rep_analysis_lvl is not None:
        for the_seed in seed_range:
            r_analysis.replication_analysis(scn, pol, the_seed, pol.rep_dict[the_seed])

    pol.calc_policy_stats(scn)

    new_adj_factor, new_lineage = {}, {}
    for fuel_type in scn.fuel_cNode_info.keys():
        if cycle_detected(pol, fuel_type):
            pol = best_policy_so_far(pol, fuel_type)
            if scn.adj_factor_setting == 'omicron':
                scn.adj_factor_setting = 'iota'
                calc_new_adj_factor_iota(scn, pol, new_adj_factor, fuel_type)
            elif scn.adj_factor_setting == 'iota':
                scn.adj_factor_setting = 'omicron'
                calc_new_adj_factor_omicron(scn, pol, new_adj_factor, fuel_type)
        elif ESO_score_increased(pol, fuel_type):
            pol = best_policy_so_far(pol, fuel_type)
            if scn.adj_factor_setting == 'omicron':
                calc_new_adj_factor_omicron(scn, pol, new_adj_factor, fuel_type, multiplier=1.5)
            elif scn.adj_factor_setting == 'iota':
                calc_new_adj_factor_iota(scn, pol, new_adj_factor, fuel_type, multiplier=1.5)
        else:
            if scn.adj_factor_setting == 'omicron':
                calc_new_adj_factor_omicron(scn, pol, new_adj_factor, fuel_type)
            elif scn.adj_factor_setting == 'iota':
                calc_new_adj_factor_iota(scn, pol, new_adj_factor, fuel_type)

        new_lineage[fuel_type] = calc_lineage(pol, fuel_type)

    return new_adj_factor, new_lineage


# noinspection PyPep8Naming
def calc_new_adj_factor_iota(scn, pol, new_adj_factor, fuel_type, multiplier=3.5):
    """Checks to see if the average length of time during which the unloading
    queue at the child node is non-zero is greater than 2 days.  If so, this
    child node can afford to have a smaller safety stock.
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type new_adj_factor: dict[(str, str), float]
    @type fuel_type: str
    @type multiplier: float
    """
    all_avg_PNS = []
    for cNode_name in scn.cNode_fuel_info.iterkeys():
        all_avg_PNS.append((pol.cNode_PNS[(cNode_name, fuel_type)], cNode_name))

    all_avg_PNS.sort(key=lambda x: x[0])  # Sort from smallest to largest

    nzQL_epsilon = 1.5
    for elt_idx, all_avg_PNS_elt in enumerate(all_avg_PNS):
        _, cNode_name = all_avg_PNS_elt
        delta = 0.0

        if elt_idx <= int(len(all_avg_PNS) * 0.1):  # Child nodes that can have a smaller safety stock
            if pol.node_nzQL[(cNode_name, fuel_type)] > nzQL_epsilon:
                delta -= 0.40 * multiplier
            else:
                delta -= 0.35 * multiplier

        elif int(len(all_avg_PNS) * 0.8) <= elt_idx:  # Child nodes that need a larger safety stock
            if pol.node_nzQL[(cNode_name, fuel_type)] > nzQL_epsilon:
                delta = 0.15 * multiplier
            else:
                delta = 0.20 * multiplier

        else:  # Not doing anything to these child nodes
            pass

        new_adj_factor[(cNode_name, fuel_type)] = pol.adj_factor.get((cNode_name, fuel_type), 0.0) + delta


# noinspection PyPep8Naming
def calc_new_adj_factor_omicron(scn, pol, new_adj_factor, fuel_type, multiplier=3.5):
    """Checks to see if the expected stockout is increasing, and tries to jump away.
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type new_adj_factor: dict[(str, str), float]
    @type fuel_type: str
    @type multiplier: float
    @rtype: None
    """
    all_avg_PNS = []
    for cNode_name in scn.cNode_fuel_info.iterkeys():
        all_avg_PNS.append((pol.cNode_PNS[(cNode_name, fuel_type)], cNode_name))

    all_avg_PNS.sort(key=lambda x: x[0])  # Sort from smallest to largest

    nzQL_epsilon = 1.5
    for elt_idx, all_avg_PNS_elt in enumerate(all_avg_PNS):
        _, cNode_name = all_avg_PNS_elt
        delta = 0.0

        if elt_idx <= int(len(all_avg_PNS) * 0.1):  # Child nodes that can have a smaller safety stock
            if pol.node_nzQL[(cNode_name, fuel_type)] > nzQL_epsilon:
                delta -= 0.45 * multiplier
            else:
                delta -= 0.35 * multiplier

        elif int(len(all_avg_PNS) * 0.8) <= elt_idx:  # Child nodes that need a larger safety stock
            if stays_constant(pol, 'cNode_PNS', cNode_name, fuel_type, 4, 0.05):
                pass
            else:
                if pol.node_nzQL[(cNode_name, fuel_type)] > nzQL_epsilon:
                    delta -= 0.05 * multiplier
                else:
                    delta = 0.15 * multiplier

        else:  # Not doing anything to these child nodes
            pass

        new_adj_factor[(cNode_name, fuel_type)] = pol.adj_factor.get((cNode_name, fuel_type), 0.0) + delta


# noinspection PyPep8Naming
def stays_constant(pol, data_type, cNode_name, fuel_type, look_back, tolerance_range):
    """Checks to see if the data over the last look_back number of policies
    has remained within the tolerance range provided.
    @type pol: policy.Policy
    @type data_type: str
    @type cNode_name: str
    @type fuel_type: str
    @type tolerance_range: float
    @type look_back: int
    @rtype: bool
    """
    dict_key = (cNode_name, fuel_type)
    if len(pol.lineage.get(fuel_type, [])) > (look_back + 1):
        for i in range(1, look_back + 1):
            pol_idx = pol.lineage[fuel_type][-1 - i]
            if abs(getattr(pol, data_type)[dict_key] - getattr(pol.history[pol_idx], data_type)[dict_key]) \
                    > tolerance_range:
                return False
        return True
    return False


# noinspection PyPep8Naming
def ESO_score_increased(pol, fuel_type):
    """
    @type pol: policy.Policy
    @type fuel_type: str
    @rtype: bool
    """
    if len(pol.lineage.get(fuel_type, [])) > 2:
        one_back, two_back = pol.lineage[fuel_type][-1], pol.lineage[fuel_type][-2]
        if (pol.score[fuel_type]['ESO'] > pol.history[one_back].score[fuel_type]['ESO']) and \
                (pol.history[one_back].score[fuel_type]['ESO'] > pol.history[two_back].score[fuel_type]['ESO']):
            return True
    return False


def cycle_detected(pol, fuel_type):
    """
    @type pol: policy.Policy
    @type fuel_type: str
    @rtype: bool
    """
    for old_pol in pol.history:
        if (old_pol.score[fuel_type]['ESO'] == pol.score[fuel_type]['ESO']) and \
                (old_pol.score[fuel_type]['SO_Int'] == pol.score[fuel_type]['SO_Int']):
            return True

    return False


# noinspection PyPep8Naming
def best_policy_so_far(pol, fuel_type):
    """
    @type pol: policy.Policy
    @type fuel_type: str
    @rtype: policy.Policy
    """
    best_ESO_score = 100000000
    best_pol_idx = None
    for idx, old_pol in enumerate(pol.history):
        if old_pol.score[fuel_type]['ESO'] <= best_ESO_score:
            best_ESO_score = old_pol.score[fuel_type]['ESO']
            best_pol_idx = idx

    return pol.history[best_pol_idx]


def calc_lineage(pol, fuel_type):
    """Returns an updated list of indices that reflect the history of the next policy.
    @type pol: policy.Policy
    @type fuel_type: str
    @rtype: list[int]
    """
    return pol.lineage.get(fuel_type, []) + [pol.idx]


# noinspection PyPep8Naming
def aggregate_SO_intervals(scn, rep_dict, cluster, time_gap):
    """Return a dictionary containing the stock out intervals aggregated over all
    replications for each child node--fuel type combination.  The data can optionally
    be clustered according to a time_gap criterion, where if a stock out occurs
    within time_gap of another stock out, the two stock outs are considered as
    one.  time_gap is in units of days.
    @type scn: scenario.Scenario
    @type rep_dict: dict[int, (sim_engine.SimulationCalendar, parent_node.ParentNode,
                                dict[str, child_node.ChildNode], list[Tanker])]
    @type cluster: bool
    @type time_gap: float
    @rtype aggregated_data: dict[(str, str), dict[str, list[float]]]
    """
    aggregated_data = {}
    rep_index = 1
    for _, _, sim_cNodes, _ in rep_dict.itervalues():
        for cNode in sim_cNodes.itervalues():
            for fuel_type, FSP in cNode.FSPs.iteritems():
                start_times, end_times = r_analysis.calc_SO_intervals(scn, FSP, cluster, time_gap)
                y = [rep_index for _ in range(len(end_times))]
                try:
                    aggregated_data[(cNode.name, fuel_type)]['x_start'] += start_times
                    aggregated_data[(cNode.name, fuel_type)]['x_end'] += end_times
                    aggregated_data[(cNode.name, fuel_type)]['y'] += y
                except KeyError:
                    aggregated_data[(cNode.name, fuel_type)] = {'x_start': start_times,
                                                                'x_end': end_times,
                                                                'y': y}
        rep_index += 1

    return aggregated_data


# noinspection PyPep8Naming
def plot_all_replication_SOs(scn, pol, aggregated_data):
    """Plot the stock outs for a each fuel type at all child nodes on a single plot.
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type aggregated_data: dict[(str, str), dict[str, list[float]]]
    """
    l_width = calc_line_width(scn.num_reps)
    for fuel_type, cNode_names in scn.fuel_cNode_info.iteritems():
        num_plots = len(cNode_names)
        if num_plots > 1:
            fig, subplot = plt.subplots(num_rows=num_plots, num_cols=1, sharex='all')
            for i, cNode_name in enumerate(cNode_names):
                data = aggregated_data[(cNode_name, fuel_type)]
                subplot[i].hlines(data['y'], data['x_start'], data['x_end'],
                                  linewidth=l_width, colors=const.FIREBRICK_hex)
                subplot[i].set_ylim(0.5, scn.num_reps + 0.5)
                subplot[i].yaxis.set_label_position('right')
                subplot[i].set_ylabel(cNode_name)
                subplot[i].set_xlim(0, scn.stop_time / 24.0)
                subplot[i].grid(True)
            subplot[0].set_title('Stock outs of ' + fuel_type)
            subplot[num_plots - 1].set_xlabel('Time (days)')
            fig.text(0.04, 0.5, 'Replication number', ha='center', va='center', rotation='vertical')
        else:
            cNode_name = cNode_names[0]
            fig, subplot = plt.subplots(1)
            data = aggregated_data[(cNode_name, fuel_type)]
            subplot.hlines(data['y'], data['x_start'], data['x_end'],
                           linewidth=l_width, colors=const.FIREBRICK_hex)
            subplot.set_ylim(0.5, scn.num_reps + 0.5)
            subplot.yaxis.set_label_position('right')
            subplot.set_ylabel(cNode_name)
            subplot.set_xlim(0, scn.stop_time / 24.0)
            subplot.grid(True)
            subplot.set_title('Stock outs of ' + fuel_type)
            subplot.set_xlabel('Time (days)')
            fig.text(0.04, 0.5, 'Replication number', ha='center', va='center', rotation='vertical')
        plt.savefig(pol.path + '/' + fuel_type + '_SO_' + scn.plot_type, transparent=True)
        plt.close('all')


def calc_line_width(n):
    """Calculates the line width for the replication stock out plots
    based on n, the number of simulation replications.
    @type n: int
    @rtype: float
    """
    if n < 21:
        return 4
    if n < 31:
        return 3
    if n < 41:
        return 2
    if n < 51:
        return 1.5
    if n < 101:
        return 1.25
    return 1


# noinspection PyPep8Naming
def policy_score_progression_plot(scn, policy_history):
    """Plots a single plot for each fuel type that contains the policy scores.
    The color of the point representing each score grows darker as the policy
    number increases, making it possible to track the progression of the scores.
    @type scn: scenario.Scenario
    @type policy_history: list[policy.Policy]
    """
    color_scheme = [const.LIGHTEST_BLUE_rgb, const.LIGHT_BLUE_rgb, const.MEDIUM_BLUE_rgb,
                    const.DARK_BLUE_rgb, const.DARKEST_BLUE_rgb]
    pols_per_category = math.ceil(len(policy_history) / float(len(color_scheme)))

    for fuel_type in policy_history[0].score.keys():
        ESO, SO_Ints, rgba_colors, alphas = [], [], [], []
        for pol in policy_history:
            ESO.append(pol.score[fuel_type]['ESO'])
            SO_Ints.append(pol.score[fuel_type]['SO_Int'])
            rgba_colors.append(tuple(color_scheme[int(pol.idx / pols_per_category)] + [0.75]))

        plt.scatter(ESO, SO_Ints, color=rgba_colors, s=70, marker='o')
        plt.savefig(scn.output_path + '/policy_progression_' + fuel_type + scn.plot_type, transparent=True)
        plt.close()


def calc_average_days_between_convoys(data):
    """Returns the average number of days between convoys where data is a list of
    times (in units of days) on which convoys were sent.
    @param data: list[float]
    @return: float
    """
    data.sort()
    days_between_convoys = []
    for current_day, next_day in itertools.izip(data, data[1:]):
        days_between_convoys.append(next_day - current_day)

    if days_between_convoys:
        return np.mean(days_between_convoys)
    else:
        return 0.0


# noinspection PyPep8Naming
def calc_expected_stock_out(scn, rgba_PSOs, interval_starts, conditional_SO_amounts, fuel_type, cNode_names):
    """
    @type scn: scenario.Scenario
    @type rgba_PSOs: dict
    @type interval_starts: dict
    @type conditional_SO_amounts: dict
    @type fuel_type: str
    @type cNode_names: list[str]
    @rtype: float
    """
    start_time = scn.warm_up * 24.0
    end_time = (scn.warm_up + scn.planning_horizon) * 24.0
    expected_SO = 0
    for cNode_name in cNode_names:
        key = (cNode_name, fuel_type)
        for int_start_time, rgba, SO_amount in zip(interval_starts[key], rgba_PSOs[key], conditional_SO_amounts[key]):
            if start_time <= int_start_time <= end_time:
                prob_SO = rgba[3]
                expected_SO += prob_SO * SO_amount

    return expected_SO / float(len(cNode_names))


# noinspection PyPep8Naming
def calc_SO_interval_metric(scn, interval_starts, fuel_type, cNode_names):
    """
    @type scn: scenario.Scenario
    @type interval_starts: dict
    @type fuel_type: str
    @type cNode_names: list[str]
    @rtype: float
    """
    start_time = scn.warm_up * 24.0
    end_time = (scn.warm_up + scn.planning_horizon) * 24.0
    cNode_SO_interval_metrics = []
    for cNode_name in cNode_names:
        key = (cNode_name, fuel_type)

        reduced_int_starts = [start for start in interval_starts[key] if start_time <= start <= end_time]
        if reduced_int_starts:
            interval_group_lengths = []
            group_length = 0
            for start, next_start in zip(reduced_int_starts, reduced_int_starts[1:]):
                group_length += 1
                if (start + scn.analysis_interval_length) != next_start:
                    interval_group_lengths.append(group_length)
                    group_length = 0

            # Deal with the last element in the list
            group_length += 1
            interval_group_lengths.append(group_length)

            cNode_SO_interval_metrics.append(sum(interval_group_lengths) / float(len(interval_group_lengths)))
        else:
            cNode_SO_interval_metrics.append(0)

    return sum(cNode_SO_interval_metrics) / float(len(cNode_names))


# noinspection PyPep8Naming
def SO_interval_rgba_color(scn, num_overlapping_intervals, prob):
    """Return the rgba color of the stockout interval based on the number of
    overlapping intervals (i.e. prob), where the alpha value is the probability
    of stock out for that interval.
    @type scn: scenario.Scenario
    @type num_overlapping_intervals: int
    @type prob: float
    @return: (float, float, float, float)
    """
    if scn.num_reps > 3:
        if num_overlapping_intervals <= int((scn.num_reps * 0.5) + 0.5):
            return tuple(const.BOYSENBERRY_rgb + [prob])  # lowest likelihood of stock out
        elif num_overlapping_intervals > int((scn.num_reps * 0.8) + 0.5):
            return tuple(const.BLACK_OLIVE_rgb + [prob])  # highest likelihood of stock out
        else:
            return tuple(const.FIREBRICK_rgb + [prob])  # medium likelihood of stock out

    # When there are only three replications and the non-overlapping
    # intervals are being plotted.
    if scn.num_reps == 3:
        if num_overlapping_intervals == 1:
            tuple(const.BOYSENBERRY_rgb + [prob])
        elif num_overlapping_intervals == 2:
            tuple(const.FIREBRICK_rgb + [prob])

    # When there are only two replications and the non-overlapping
    # intervals are being plotted.
    if (scn.num_reps == 2) and (num_overlapping_intervals == 1):
        return tuple(const.FIREBRICK_rgb + [prob])

    return tuple(const.BLACK_OLIVE_rgb + [prob])


# noinspection PyPep8Naming
def calc_prob_OHLvls_stats(scn, rep_dict, scale='Stockage Objective'):
    """Return three dictionaries containing the on-hand level data aggregated
    over all replications for each child node--fuel type--stockage objective
    level combination.  The first dictionary contains the rgba values of the
    overlapped intervals, and the second and third dictionaries contain the
    start and end times of each distinct overlapping interval.
    @type scn: scenario.Scenario
    @type rep_dict: dict[int, (sim_engine.SimulationCalendar, parent_node.ParentNode,
                                dict[str, child_node.ChildNode], list[Tanker])]
    @type scale: str
    """
    aggregated_data = {}
    for _, _, sim_cNodes, _ in rep_dict.itervalues():
        for cNode in sim_cNodes.itervalues():
            for fuel_type, FSP in cNode.FSPs.iteritems():
                for OHLvl_color in ['green', 'amber', 'red', 'black']:
                    OHLvl_interval_times = calc_OHLvl_intervals(scn, FSP, OHLvl_color, scale)
                    try:
                        aggregated_data[(cNode.name, fuel_type, OHLvl_color)] += OHLvl_interval_times
                    except KeyError:
                        aggregated_data[(cNode.name, fuel_type, OHLvl_color)] = OHLvl_interval_times

    rgba_OH_lvls = {}
    interval_starts = {}
    interval_ends = {}
    for key in aggregated_data.iterkeys():
        _, _, SO_lvl = key
        rgba_OH_lvls[key], interval_starts[key], interval_ends[key] = \
            overlapping_OHLvls_intervals(scn, aggregated_data[key], SO_lvl)

    return rgba_OH_lvls, interval_starts, interval_ends


# noinspection PyPep8Naming
def calc_OHLvl_intervals(scn, FSP, OHLvl_color, scale):
    """Return a single list that contains the start times and end times
    of intervals during which the on-hand level corresponds to the
    OHLvl_color passed in for the given FSP.
    @type scn: scenario.Scenario
    @type FSP: fsp.ParentNodeFSP | fsp.ChildNodeFSP
    @type OHLvl_color: str
    @type scale: str
    @rtype interval_times: list[(float, str)]
    """
    interval_times = []
    num_starts, num_ends = 0, 0

    lower_limit, upper_limit = OHLvls_color_map(scn, FSP, scale)[OHLvl_color]
    for time, on_hand in FSP.inv_trace:
        if lower_limit <= on_hand < upper_limit:
            if num_starts == num_ends:
                interval_times.append((time / 24.0, 's'))
                num_starts += 1
        else:
            if num_starts > num_ends:
                interval_times.append((time / 24.0, 'e'))
                num_ends += 1
    if num_starts > num_ends:
        interval_times.append((scn.stop_time / 24.0, 'e'))
        num_ends += 1

    return interval_times


# noinspection PyPep8Naming
def OHLvls_color_map(scn, FSP, scale):
    """Return a dictionary that maps the on-hand levels to colors.
    @type scn: scenario.Scenario
    @type FSP: fsp.ParentNodeFSP | fsp.ChildNodeFSP
    @type scale: str
    @return: dict[str, (float, float)]
    """
    if scale == 'Stockage Objective':
        on_hand_lvl_map = {'green': (FSP.SO * 0.75, FSP.storage_max + scn.fuel_epsilon),
                           'amber': (FSP.SO * 0.5, FSP.SO * 0.75),
                           'red': (FSP.SO * 0.25, FSP.SO * 0.5),
                           'black': (FSP.storage_min, FSP.SO * 0.25)
                           }
    elif scale == 'Days of Supply':
        # TODO This map defaults to the DOS based on the first daily demand distribution
        on_hand_lvl_map = {'green': (FSP.expected_daily_demand(scn, 0.1) * 10, FSP.storage_max + scn.fuel_epsilon),
                           'amber': (FSP.expected_daily_demand(scn, 0.1) * 8, FSP.expected_daily_demand(scn, 0.1) * 10),
                           'red': (FSP.expected_daily_demand(scn, 0.1) * 6, FSP.expected_daily_demand(scn, 0.1) * 8),
                           'black': (FSP.storage_min, FSP.expected_daily_demand(scn, 0.1) * 6)
                           }
    else:
        assert False, 'Unknown scale for On-Hand Level plots.'

    return on_hand_lvl_map


# noinspection PyPep8Naming
def overlapping_OHLvls_intervals(scn, OHLvl_interval_times, OHLvl_color):
    """Given the set of intervals over all replications for a particular stockage
    objective, calculate the start and end times of distinct overlapping intervals.
    A distinct overlapping interval is an interval during which n and only n
    on hand level intervals overlapped. The first returned list contains the rgba
    value of the overlapped interval, and the second and third lists contain the
    start and end times of each distinct overlapping interval.
    @type scn: scenario.Scenario
    @param OHLvl_interval_times: list[(float, str)]
    @type OHLvl_color: str
    @rtype: list[tuple], list[float], list[float]
    """
    int_starts, int_ends = r_analysis.calc_overlapping_intervals(OHLvl_interval_times)

    interval_rgba, interval_starts, interval_ends = [], [], []
    for count in int_starts.iterkeys():
        interval_starts += int_starts[count]
        interval_ends += int_ends[count]
        prob = count / float(scn.num_reps)
        interval_rgba += [OHLvl_to_rgba(OHLvl_color, prob) for _ in range(len(int_starts[count]))]

    return interval_rgba, interval_starts, interval_ends


# noinspection PyPep8Naming
def OHLvl_to_rgba(OHLvl_color, prob):
    """Return the rgba value corresponding to the given on-hand level--probability combination.
    @param OHLvl_color: str
    @param prob: float
    @return: tuple(float)
    """
    if OHLvl_color == 'green':
        return tuple(const.BOTTLE_GREEN_rgb + [prob])

    if OHLvl_color == 'amber':
        return tuple(const.TANGERINE_rgb + [prob])

    if OHLvl_color == 'red':
        return tuple(const.DARK_RED_rgb + [prob])

    return tuple(const.BLACK_OLIVE_rgb + [prob])


# noinspection PyPep8Naming
def calc_expected_OHLvls_stats(scn, rep_dict, time_delta, scale='Stockage Objective'):
    """Return three dictionaries containing the on-hand level data aggregated
    over all replications for each child node--fuel type--stockage objective
    level combination.  The first dictionary contains the rgba values of the
    overlapped intervals, and the second and third dictionaries contain the
    start and end times of each distinct overlapping interval.
    @type scn: scenario.Scenario
    @type rep_dict: dict[int, (sim_engine.SimulationCalendar, parent_node.ParentNode,
                                dict[str, child_node.ChildNode], list[Tanker])]
    @type time_delta: float
    @type scale: str
    """
    aggregated_data = {}
    cNode_fuel_type_to_FSP_map = {}
    for _, _, sim_cNodes, _ in rep_dict.itervalues():
        for cNode in sim_cNodes.itervalues():
            for fuel_type, FSP in cNode.FSPs.iteritems():
                cNode_fuel_type_to_FSP_map[(cNode.name, fuel_type)] = FSP
                rep_time_weighted_averages = time_weighted_expected_OHLvls_intervals(FSP, time_delta)
                try:
                    aggregated_data[(cNode.name, fuel_type)].append(rep_time_weighted_averages)
                except KeyError:
                    aggregated_data[(cNode.name, fuel_type)] = [rep_time_weighted_averages]

    rgba_exp_OH_lvls = {}
    for key, rep_twa_list in aggregated_data.iteritems():
        pol_time_weighted_averages = list(np.array(rep_twa_list).mean(axis=0))

        FSP = cNode_fuel_type_to_FSP_map[key]
        rgba_exp_OH_lvls[key] = \
            [expected_OHLvl_to_rgba(scn, FSP, scale, exp_OH_lvl) for exp_OH_lvl in pol_time_weighted_averages]

    return rgba_exp_OH_lvls


# noinspection PyPep8Naming
def expected_OHLvl_to_rgba(scn, FSP, scale, exp_OHLvl):
    """Returns the rgba color for the expected on-hand level for the
    specific FSP.
    @param scn: scenario.Scenario
    @param FSP: fsp.ParentNodeFSP | fsp.ChildNodeFSP
    @param scale: str
    @param exp_OHLvl: float
    @return: tuple
    """
    the_color_map = OHLvls_color_map(scn, FSP, scale)

    for color, limits in the_color_map.iteritems():
        lower_limit, upper_limit = limits
        if lower_limit <= exp_OHLvl < upper_limit:
            return OHLvl_to_rgba(color, 0.7)


# noinspection PyPep8Naming
def time_weighted_expected_OHLvls_intervals(FSP, time_delta):
    """Calculates the time weighted average on-hand level over each interval
    of length time_delta.  This method takes into account the fact that the
    FSP inventory trace may contain multiple entries with the same timestamp
    by taking the trace end to be the LAST entry with the same timestamp.
    @type FSP: fsp.ParentNodeFSP | fsp.ChildNodeFSP
    @type time_delta: float
    @return: list[float]
    """
    time_weighted_averages = []
    interval_end = time_delta
    twa = 0
    idx = 0
    while True:
        trace_start, trace_start_OHLvl = FSP.inv_trace[idx]
        try:
            trace_end, __ = FSP.inv_trace[idx + 1]
        except IndexError:  # This should only occur at the end of the trace.
            break

        idx_delta = 2
        while True:
            try:
                if trace_end == FSP.inv_trace[idx + idx_delta][0]:
                    trace_end, __ = FSP.inv_trace[idx + idx_delta]
                else:
                    idx += idx_delta - 1
                    break
                idx_delta += 1
            except IndexError:  # This should only occur at the end of the trace.
                idx += 1
                break

        if trace_end < interval_end:
            twa += ((trace_end - trace_start) / float(time_delta)) * trace_start_OHLvl
        else:
            twa += ((interval_end - trace_start) / float(time_delta)) * trace_start_OHLvl
            time_weighted_averages.append(twa)

            if trace_end == interval_end:
                twa = 0.0
                interval_end += time_delta
            else:
                while True:
                    if trace_end >= (interval_end + time_delta):
                        time_weighted_averages.append(trace_start_OHLvl)
                        interval_end += time_delta
                        twa = 0.0
                    else:
                        twa = ((trace_end - interval_end) / float(time_delta)) * trace_start_OHLvl
                        interval_end += time_delta
                        break

    return time_weighted_averages


def write_policy_data(scn, policy_history):
    """Writes out to text files information about the policies.
    @type scn: scenario.Scenario
    @type policy_history: list[policy.Policy]
    """
    adj_fac_history = {}
    for fuel_type, cNode_names in scn.fuel_cNode_info.iteritems():
        for cNode_name in cNode_names:
            adj_fac_history[(cNode_name, fuel_type)] = []

    for pol in policy_history:
        for fuel_type, cNode_names in scn.fuel_cNode_info.iteritems():
            for cNode_name in cNode_names:
                adj_fac_history[(cNode_name, fuel_type)].append(pol.adj_factor.get((cNode_name, fuel_type), -10000))

    file_name = 'adj_fac_progression.txt'
    with open(scn.output_path + '/' + file_name, 'w') as wrtr:
        for fuel_type, cNode_names in scn.fuel_cNode_info.iteritems():
            for cNode_name in cNode_names:
                wrtr.write('%s, %s: ' % (fuel_type, cNode_name))
                for adj_fac in adj_fac_history[(cNode_name, fuel_type)]:
                    wrtr.write('%0.3f, ' % adj_fac)
                wrtr.write('\n')
