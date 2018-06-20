import matplotlib.pyplot as plt
import misc
import numpy as np
import numpy.random as npr
import p_analysis
import r_analysis
import stats


# noinspection PyPep8Naming,PyTypeChecker
class Policy:
    """
    @param idx: Unique integer that identifies the policy.
    @type idx: int
    @param path: Path that specifies where any output related to the
        current policy will be put.
    @type path: str
    @param adj_factor: Dictionary of floats that represent the adjustment factor for this
        policy.
    @type adj_factor: dict[(str, str), float]
    @param lineage: A dictionary containing a list of indices of policies that preceded
        this policy for each fuel_type.
    @type lineage: dict[str, list[int]]
    @param damping_factor: A real number in the interval (0, 1] used to
        dampen the effect of policy adjustments as the policy index increases.
    @type damping_factor: float
    @param history: A list of all previous policies.
    @type history: list[policy.Policy]
    @param rndstrm_seed: The seed that initiates the random stream for the policy instance.
    @type rndstrm_seed: int
    @type rep_dict: dict[int, (sim_engine.SimulationCalendar, parent_node.ParentNode,
                        dict[str, child_node.ChildNode], list[Tanker])]
    """
    def __init__(self, idx, path, adj_factor, lineage, damping_factor, history,
                 rndstrm_seed=9823747, rep_dict=None):
        """
        @param idx: Unique integer that identifies the policy.
        @type idx: int
        @param path: Path that specifies where any output related to the
            current policy will be put.
        @type path: str
        @param adj_factor: Dictionary of floats that represent the adjustment factor for this
            policy.
        @type adj_factor: dict[(str, str), float]
        @param lineage: A dictionary containing a list of indices of policies that preceded
            this policy for each fuel_type.
        @type lineage: dict[str, list[int]]
        @param damping_factor: A real number in the interval (0, 1] used to
            dampen the effect of policy adjustments as the policy index increases.
        @type damping_factor: float
        @param history: A list of all previous policies.
        @type history: list[policy.Policy]
        @param rndstrm_seed: The seed that initiates the random stream for the policy instance.
        @type rndstrm_seed: int
        @type rep_dict: dict[int, (sim_engine.SimulationCalendar, parent_node.ParentNode,
                                    dict[str, child_node.ChildNode], list[Tanker])]
        """
        self.idx = int(idx)
        self.path = path
        self.adj_factor = adj_factor
        self.lineage = lineage
        self.damping_factor = float(damping_factor)
        self.history = history
        self.rep_dict = rep_dict
        self.node_DBC = None  # Days between convoys for each node
        self.node_nzQL = None  # Average expected length of intervals during which unloading queue length was > 0
        self.cNode_PNS = None  # Average proportion of demand not satisfied for each fuel type at each child node
        self.cNode_PSSO = {}  # Probability and severity of stock outs and when for each fuel type at each child node
        self.cNode_LT = None  # Average lead time and the raw lead time data for each fuel type
        self.score = {}
        self.pNode_PD = None  # Proxy demand data for each parent node--fuel type combination
        self.cNode_AD = None  # Actual demand data for each child node--fuel type combination
        if rndstrm_seed:
            self.rndstrm = npr.RandomState(rndstrm_seed + int(idx))
        else:
            self.rndstrm = npr.RandomState(int(idx))

    def calc_policy_stats(self, scn):
        """Calculates a number of policy level statistics and potentially
        outputs the statistics.
        @type scn: scenario.Scenario
        """
        self.node_DBC = self.calc_average_DBC(scn)
        self.node_nzQL = self.calc_average_nzQL(scn)
        self.cNode_PNS = self.calc_average_PNS(scn)
        self.cNode_LT = self.collate_euNode_lead_times()
        self.pNode_PD = self.collate_pNode_proxy_demand(scn)
        self.cNode_AD = self.collate_cNode_actual_demand()

        PSSO_stats = self.calc_PSSO_stats(scn, scn.analysis_interval_length)
        PSSO_rgba, PSSO_interval_starts, PSSO_sev_categories, conditional_SO_amounts = PSSO_stats
        for pns_key in self.cNode_PNS.iterkeys():
            cNode_name, fuel_type = pns_key
            self.cNode_PSSO[(cNode_name, fuel_type, 'PSO')] = PSSO_rgba[pns_key]
            self.cNode_PSSO[(cNode_name, fuel_type, 'int_start')] = PSSO_interval_starts[pns_key]
            self.cNode_PSSO[(cNode_name, fuel_type, 'amount')] = conditional_SO_amounts[pns_key]

        self.calc_policy_score(scn, PSSO_stats)

        if scn.pol_analysis_lvl in ['plots', 'all']:
            for fuel_type in scn.fuel_cNode_info.keys():
                self.PSSO_plot(scn, fuel_type, PSSO_rgba, PSSO_interval_starts, PSSO_sev_categories)

                # prob_OHLvls_stats = p_analysis.calc_prob_OHLvls_stats(scn, self.rep_dict)
                # OHLvls_rgba, OHLvls_interval_starts, OHLvls_interval_ends = prob_OHLvls_stats
                # self.prob_OHLvls_plot(scn, fuel_type, OHLvls_rgba, OHLvls_interval_starts, OHLvls_interval_ends)

                exp_OHLvls_rgba = p_analysis.calc_expected_OHLvls_stats(scn, self.rep_dict,
                                                                        scn.analysis_interval_length)
                self.expected_OHLvls_plot(scn, fuel_type, exp_OHLvls_rgba)

            # agg_stock_out_interval_data = p_analysis.aggregate_SO_intervals(scn, self.rep_dict, True, 12 / 24.0)
            # p_analysis.plot_all_replication_SOs(scn, self, agg_stock_out_interval_data)

        if scn.pol_analysis_lvl in ['all']:
            misc.create_dir(self.path)
            self.write_policy_stats()

        if scn.pol_analysis_lvl not in [None, 'plots', 'all']:
            assert False, 'Policy analysis level not recognized.'

    def write_policy_stats(self):
        if self.adj_factor:
            with open(self.path + '/adj_factors.txt', 'w') as wrtr:
                # Sort on fuel_type, then cNode_name
                for key in sorted(self.adj_factor.keys()[:], key=lambda x: (x[1], x[0])):
                    wrtr.write('%s\t%.4f\n' % (key, self.adj_factor[key]))

        with open(self.path + '/avg_PNS.txt', 'w') as wrtr:
            # Sort on fuel_type, then cNode_name
            for key in sorted(self.cNode_PNS.keys()[:], key=lambda x: (x[1], x[0])):
                wrtr.write('%s\t%.4f\n' % (key, self.cNode_PNS[key]))

        with open(self.path + '/avg_nzQL.txt', 'w') as wrtr:
            # Sort on fuel_type, then cNode_name
            for key in sorted(self.node_nzQL.keys()[:], key=lambda x: (x[1], x[0])):
                wrtr.write('%s\t%.4f\n' % (key, self.node_nzQL[key]))

        with open(self.path + '/avg_DBC.txt', 'w') as wrtr:
            for key in sorted(self.node_DBC.keys()[:]):
                wrtr.write('%s\t%.4f\n' % (key, self.node_DBC[key]))

    def calc_average_DBC(self, scn):
        """Return a dictionary that contains the average days between convoys (DBC)
        for each child node.
        @type scn: scenario.Scenario
        @rtype: dict[str, float]
        """
        days_convoy_was_sent, rep_avg_DBC = {}, {}

        for cNode_name in scn.cNode_fuel_info.iterkeys():
            days_convoy_was_sent[cNode_name] = set()
            rep_avg_DBC[cNode_name] = []

        for _, sim_pNode, _, _ in self.rep_dict.itervalues():
            for time, __, going_to, __, __, __ in sim_pNode.tankers_sent_trace:
                days_convoy_was_sent[going_to].add(int(time / 24.0))

            for cNode_name in scn.cNode_fuel_info.iterkeys():
                rep_avg_DBC[cNode_name].append(
                    p_analysis.calc_average_days_between_convoys(list(days_convoy_was_sent[cNode_name])))
                days_convoy_was_sent[cNode_name].clear()

        average_DBC = {}
        """@type average_DBC: dict[str, float]"""

        for cNode_name in scn.cNode_fuel_info.iterkeys():
            average_DBC[cNode_name] = np.mean(rep_avg_DBC[cNode_name])

        return average_DBC

    def calc_average_nzQL(self, scn):
        """Calculates the average expected length (in days) during the planning horizon
        of intervals during which the unloading queue had at least 1 tanker on it for
        each child node--fuel type combination.
        @type scn: scenario.Scenario
        @rtype: dict[(str, str), float]
        """
        collated_rep_nzQL = {}
        for _, _, sim_cNodes, _ in self.rep_dict.itervalues():
            for cNode_name, cNode in sim_cNodes.iteritems():
                for fuel_type, FSP in cNode.FSPs.iteritems():
                    rep_expected_nzQL = r_analysis.calc_expected_nzQL(scn, FSP)
                    try:
                        collated_rep_nzQL[(cNode_name, fuel_type)].append(rep_expected_nzQL)
                    except KeyError:
                        collated_rep_nzQL[(cNode_name, fuel_type)] = [rep_expected_nzQL]

        average_nzQL = {}
        """@type average_nzQL: dict[(str, str), float]"""

        for cNode_fuel_type_comb, collated_rep_data in collated_rep_nzQL.iteritems():
            average_nzQL[cNode_fuel_type_comb] = np.mean(collated_rep_data)

        return average_nzQL

    def calc_average_PNS(self, scn):
        """Return a dictionary containing the average proportion of demand not satisfied
        (stocked out) during the planning horizon for each child node--fuel type
        combination given the rep_dict.
        @type scn: scenario.Scenario
        @rtype: dict[(str, str), float]
        """
        collated_rep_PNS = {}
        for _, _, sim_cNodes, _ in self.rep_dict.itervalues():
            for cNode_name, cNode in sim_cNodes.iteritems():
                for fuel_type, FSP in cNode.FSPs.iteritems():
                    rep_PNS = r_analysis.calc_prop_stocked_out(scn, FSP)
                    try:
                        collated_rep_PNS[(cNode_name, fuel_type)].append(rep_PNS)
                    except KeyError:
                        collated_rep_PNS[(cNode_name, fuel_type)] = [rep_PNS]

        average_PNS = {}
        """@type average_PNS: dict[(str, str), float]"""

        for cNode_fuel_type_comb, collated_rep_data in collated_rep_PNS.iteritems():
            average_PNS[cNode_fuel_type_comb] = np.mean(collated_rep_data)

        return average_PNS

    def collate_euNode_lead_times(self):
        """Returns a dictionary with the collated lead time (in days) over all
        replications for each End User node--fuel type combination.
        @rtype: dict[str, list[float]]
        """
        the_LTs = {}
        """@type the_LTs: dict[str, list[float]]"""
        for _, _, sim_cNodes, _ in self.rep_dict.itervalues():
            for cNode_name, cNode in sim_cNodes.iteritems():
                for euNode_fuel, lead_time in cNode.lead_time.iteritems():
                    the_LTs[euNode_fuel] = the_LTs.get(euNode_fuel, []) + [lead_time]

        return the_LTs

    def collate_pNode_proxy_demand(self, scn):
        """Returns a dictionary containing the proxy demand for each fuel type
        at the parent node for the policy.  The proxy demand is collated over
        all replications.
        @type scn: scenario.Scenario
        @rtype: dict[str, list[list[float]]]
        """
        proxy_demand = {}
        """@type proxy_demand: dict[str, list[list[float]]]"""

        if scn.demand_proxy == 'fuel_sent':
            for _, sim_pNode, _, _ in self.rep_dict.itervalues():
                all_fuel_proxy_data = self.get_fuel_sent_proxy_data(scn, sim_pNode)
                for fuel_type in sim_pNode.FSPs.iterkeys():
                    proxy_demand[fuel_type] = proxy_demand.get(fuel_type, []) + [all_fuel_proxy_data[fuel_type]]
        elif scn.demand_proxy == 'amount_loaded':
            for _, sim_pNode, _, _ in self.rep_dict.itervalues():
                for fuel_type, FSP in sim_pNode.FSPs.iteritems():
                    proxy_demand[fuel_type] = proxy_demand.get(fuel_type, []) + [FSP.daily_amount_loaded]
        else:
            assert False, 'Proxy type (%s) unknown.' % scn.demand_proxy

        return proxy_demand

    @staticmethod
    def get_fuel_sent_proxy_data(scn, sim_pNode):
        """
        @type scn: scenario.Scenario
        @type sim_pNode: parent_node.ParentNode
        @rtype: dict[str, list[list[float]]]
        """
        daily_fuel_sent = {}
        """@type daily_fuel_sent: dict[str, list[list[float]]]"""

        for fuel_type in scn.pNode_fuel_info:
            daily_fuel_sent[fuel_type] = [0 for _ in range(scn.warm_up + scn.planning_horizon + scn.post_ph)]

        for time, _, _, fuel_type, fuel_amount, _ in sim_pNode.tankers_sent_trace:
            daily_fuel_sent[fuel_type][int(time / 24.0)] += fuel_amount

        return daily_fuel_sent

    def collate_cNode_actual_demand(self):
        """Returns a dictionary containing the actual demand for each fuel type
        at each child node for the policy.  The actual demand is collated over
        all replications.
        @rtype: dict[str, dict[str, list[list[float]]]]
        """
        actual_demand = {}
        """@type actual_demand: dict[str, dict[str, list[float]]]"""

        first_replication = True
        for _, _, sim_cNodes, _ in self.rep_dict.itervalues():
            for cNode in sim_cNodes.itervalues():
                if first_replication:
                    actual_demand[cNode.name] = {}
                for fuel_type, FSP in cNode.FSPs.iteritems():
                    try:
                        actual_demand[cNode.name][fuel_type].append(FSP.daily_amount_demanded)
                    except KeyError:
                        actual_demand[cNode.name][fuel_type] = [FSP.daily_amount_demanded]

            first_replication = False

        return actual_demand

    def calc_PSSO_stats(self, scn, time_delta):
        """Return three dictionaries, rgba_PSOs, interval_starts, severity_categories.  The simulation
        time horizon is divided in time intervals that are time_delta units long. For each
        child node--fuel type combination, rgba_PSOs contains a list of rgba color values for each interval
        during which there is a non-zero probability of stock out. intervals_starts contains the start
        time of each interval during which there is a non-zero probability of stock out, and
        severity_categories contains the (normalized) severity category of the stock out during that
        time interval.
        @type scn: scenario.Scenario
        @type time_delta: float
        """
        num_intervals = int(scn.stop_time / time_delta)

        # Initialize dictionaries
        counts, PSO = {}, {}
        amounts, cat_amounts = {}, {}
        breakpoints = {}
        for cNode_name in scn.cNode_fuel_info.iterkeys():
            for fuel_type in scn.cNode_fuel_info[cNode_name]:
                key = (cNode_name, fuel_type)
                counts[key] = {}
                PSO[key] = {}
                amounts[key] = {}
                cat_amounts[key] = {}

                # TODO These breakpoints need to be time dependent for graphical output to be correct
                if scn.cNode_fuel_dist[cNode_name, 'node_type'] == 'intermediate':
                    mean_value = np.mean([np.mean(data) for data in
                                          scn.cNode_fuel_dist[(cNode_name, fuel_type)].itervalues()])
                else:
                    mean_value = np.mean([stats.mean(dist_info) for dist_info in
                                          scn.cNode_fuel_dist[(cNode_name, fuel_type)].itervalues()])

                adj_fac = time_delta / 24.0
                breakpoints[(cNode_name, fuel_type, 'small')] = mean_value * 0.40 * adj_fac
                breakpoints[(cNode_name, fuel_type, 'med')] = mean_value * 1.0 * adj_fac
                breakpoints[(cNode_name, fuel_type, 'large')] = mean_value * 1.75 * adj_fac

                for interval_idx in range(num_intervals):
                    counts[key][interval_idx] = 0.0
                    amounts[key][interval_idx] = 0.0

        # Populate dictionaries with counts
        for _, _, sim_cNodes, _ in self.rep_dict.itervalues():
            for cNode in sim_cNodes.itervalues():
                for fuel_type, FSP in cNode.FSPs.iteritems():
                    last_interval_idx = -1
                    for demand_time, __, amount in FSP.demand_trace:
                        if amount < 0:
                            interval_idx = int(demand_time / time_delta)
                            if interval_idx != last_interval_idx:
                                counts[(cNode.name, fuel_type)][interval_idx] += 1
                                last_interval_idx = interval_idx
                            amounts[(cNode.name, fuel_type)][interval_idx] += -amount

        # Calculate averages for the count and amount data then categorize the
        # averaged amount data
        rgba_PSOs = {}
        interval_starts = {}
        severity_categories = {}
        conditional_SO_amounts = {}
        for cNode_name, fuel_type in counts.iterkeys():
            key = (cNode_name, fuel_type)
            rgba_PSOs[key] = []
            interval_starts[key] = []
            severity_categories[key] = []
            conditional_SO_amounts[key] = []
            for interval_idx in counts[(cNode_name, fuel_type)].iterkeys():
                if counts[key][interval_idx] > 0:  # Only calculate metrics for intervals where stockouts occurred
                    rgba_PSOs[key].append(p_analysis.SO_interval_rgba_color(scn, counts[key][interval_idx],
                                                                            counts[key][interval_idx] /
                                                                            float(scn.num_reps)))
                    interval_starts[key].append(interval_idx * time_delta)

                    # The average calculated here is conditional on a stock out occurring, which is
                    # why the divisor is the counts instead of the number of replications.
                    conditional_SO_amounts[key].append(amounts[key][interval_idx] / counts[key][interval_idx])

                    # Categorize the severity of the conditional stockout amounts
                    if conditional_SO_amounts[key][-1] > breakpoints[(cNode_name, fuel_type, 'large')]:
                        severity_categories[key].append(18)
                    elif conditional_SO_amounts[key][-1] > breakpoints[(cNode_name, fuel_type, 'med')]:
                        severity_categories[key].append(13)
                    elif conditional_SO_amounts[key][-1] > breakpoints[(cNode_name, fuel_type, 'small')]:
                        severity_categories[key].append(9)
                    else:
                        severity_categories[key].append(6)

        return rgba_PSOs, interval_starts, severity_categories, conditional_SO_amounts

    def calc_policy_score(self, scn, PSSO_stats):
        """The policy score is calculated for each fuel type and is multidimensional.
        @type scn: scenario.Scenario
        @type PSSO_stats: tuple
        @rtype policy_score: float
        """
        rgba_PSOs, interval_starts, _, conditional_SO_amounts = PSSO_stats

        for fuel_type, cNode_names in scn.fuel_cNode_info.iteritems():
            self.score[fuel_type] = \
                {'ESO': p_analysis.calc_expected_stock_out(scn, rgba_PSOs, interval_starts,
                                                           conditional_SO_amounts, fuel_type, cNode_names),
                 'SO_Int': p_analysis.calc_SO_interval_metric(scn, interval_starts, fuel_type, cNode_names)}

    def PSSO_plot(self, scn, fuel_type, PSO_rgba, interval_starts, sev_categories):
        """Plot the probability and severity of stock out plot for the given fuel type.
        @type scn: scenario.Scenario
        @type fuel_type: str
        @type PSO_rgba:
        @type interval_starts:
        @type sev_categories:
        @return: None
        """
        interval_length = scn.analysis_interval_length / 24.0
        y_level = 0
        y_labels = []
        for cNode_name in reversed(scn.fuel_cNode_info[fuel_type]):
            key = (cNode_name, fuel_type)

            y_levels = np.ones((len(interval_starts[key]),), dtype=np.int) * y_level
            int_starts = [(i / 24.0) for i in interval_starts[key]]
            int_ends = [(i + interval_length) for i in int_starts]
            if len(y_levels) > 0:
                plt.hlines(y_levels, int_starts, int_ends, linewidth=sev_categories[key], colors=PSO_rgba[key])

            y_labels.append(cNode_name)
            y_level += 1

        plt.ylim(-0.5, y_level - 0.5)
        plt.xlim(scn.warm_up, scn.warm_up + scn.planning_horizon)
        # plt.xlim(0, int(scn.stop_time / 24.0))
        plt.yticks(range(len(y_labels)), y_labels)
        plt.grid(True)
        plt.xlabel('Time (days)')
        plt.title('Prob. & Severity of Stock-Outs, %s, COA %i (%i, %.2f)' %
                  (fuel_type, self.idx, self.score[fuel_type]['ESO'], self.score[fuel_type]['SO_Int']))
        plt.tight_layout()
        plt.savefig(scn.output_path + '/' + fuel_type + '_PSSO_COA' + str(self.idx) + scn.plot_type,
                    transparent=True)
        plt.close('all')

    def prob_OHLvls_plot(self, scn, fuel_type, interval_rgba, interval_starts, interval_ends):
        """Plot the probabilistic version of the on-hand levels plot for the given fuel type.
        @type scn: scenario.Scenario
        @type fuel_type: str
        @type interval_rgba:
        @type interval_starts:
        @type interval_ends:
        @return: None
        """
        on_hand_level_order = ['black', 'red', 'amber', 'green']
        y_level = -0.225
        y_labels = []
        for cNode_name in reversed(scn.fuel_cNode_info[fuel_type]):
            y_levels, int_starts, int_ends, colors = [], [], [], []
            for on_hand_level in on_hand_level_order:
                OH_key = (cNode_name, fuel_type, on_hand_level)
                y_levels += [y_level for _ in range(len(interval_starts[OH_key]))]
                int_starts += interval_starts[OH_key]
                int_ends += interval_ends[OH_key]
                colors += interval_rgba[OH_key]
                y_level += 0.15

            plt.hlines(y_levels, int_starts, int_ends, linewidth=5, colors=colors)

            y_labels.append(cNode_name)
            y_level += 0.40

        plt.ylim(-0.5, y_level - 0.25)
        plt.xlim(scn.warm_up, scn.warm_up + scn.planning_horizon)
        # plt.xlim(0, int(scn.stop_time / 24.0))
        plt.yticks(range(len(y_labels)), y_labels)
        plt.grid(True)
        plt.xlabel('Time (days)')
        plt.title('On-Hand Levels, %s, COA %i (%i, %.2f)' %
                  (fuel_type, self.idx, self.score[fuel_type]['ESO'], self.score[fuel_type]['SO_Int']))
        plt.tight_layout()
        plt.savefig(scn.output_path + '/' + fuel_type + '_OHLvl_COA' + str(self.idx) + scn.plot_type,
                    transparent=True)
        plt.close('all')

    def expected_OHLvls_plot(self, scn, fuel_type, interval_rgba):
        """Plots the expected on-hand levels plot.
        """
        interval_length = scn.analysis_interval_length / 24.0
        y_level = 0
        y_labels = []
        for cNode_name in reversed(scn.fuel_cNode_info[fuel_type]):
            key = (cNode_name, fuel_type)

            y_levels = np.ones((len(interval_rgba[key]),), dtype=np.int) * y_level
            int_starts = [interval_length * i for i in range(len(interval_rgba[key]))]
            interval_ends = [i + interval_length for i in int_starts]
            plt.hlines(y_levels, int_starts, interval_ends, linewidth=10, colors=interval_rgba[key])

            y_labels.append(cNode_name)
            y_level += 1

        plt.ylim(-0.5, y_level - 0.5)
        plt.xlim(scn.warm_up, scn.warm_up + scn.planning_horizon)
        # plt.xlim(0, int(scn.stop_time / 24.0))
        plt.yticks(range(len(y_labels)), y_labels)
        plt.grid(True)
        plt.xlabel('Time (days)')
        plt.title('Exp. On-Hand Levels, %s, COA %i (%i, %.2f)' %
                  (fuel_type, self.idx, self.score[fuel_type]['ESO'], self.score[fuel_type]['SO_Int']))
        plt.tight_layout()
        plt.savefig(scn.output_path + '/' + fuel_type + '_ExpOHLvl_COA' + str(self.idx) + scn.plot_type,
                    transparent=True)
        plt.close('all')
