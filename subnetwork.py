import constants as const
import json
import matplotlib.pyplot as plt
import misc
import numpy as np
import pandas as pd


plt.style.use('ggplot')


def set_tanker_constraint(scn):
    """Sets the number of tankers in the subnetwork based on the
    compute_avg_tanker_days_needed() computation.
    that in the scenario.
    @type scn: scenario.Scenario
    @rtype: None
    """
    # SET UP TANKER INFORMATION BASED ON NODE POSITION IN THEATER NETWORK
    first_lvl_im_nodes = ['Gardez', 'Sharana', 'Ghazni', 'Fenty']
    second_lvl_im_nodes = ['Phoenix', 'Salerno']
    depot_node = ['Marmal']
    pNode_name = scn.scenario_data['SUPPLY_CHAIN']['PARENT_NODE']['node_name']

    for fuel_type in scn.pNode_fuel_info:
        avg_tanker_days_needed = misc.compute_avg_tanker_days_needed(scn, fuel_type)
        base_num_tankers = max(10.0, avg_tanker_days_needed / (scn.stop_time / 24.0))

        if pNode_name in first_lvl_im_nodes:
            num_tankers = int(base_num_tankers * 1.5)
        elif pNode_name in second_lvl_im_nodes:
            num_tankers = int(base_num_tankers * 4.75)
        elif pNode_name in depot_node:
            num_tankers = int(base_num_tankers * 5.75)
        else:
            assert False, 'Unknown node in network'

        # TODO Generalize to handle different tanker types
        assert len(scn.scenario_data['SUPPLY_CHAIN']['TANKER_TYPES'][fuel_type]) == 1, \
            'Too many %s tanker types' % fuel_type
        scn.scenario_data['SUPPLY_CHAIN']['TANKER_TYPES'][fuel_type][0]['num_tankers'] = num_tankers


def end_of_subnetwork_simulation(scn, pol_repo):
    """Performs all subnetwork level updates following the simulation of the
    subnetwork.
    @type scn: scenario.Scenario
    @type pol_repo: list[policy.Policy]
    """
    PS_indices = {}  # Policy indices in the pareto set
    for fuel_type in scn.pNode_fuel_info:
        PS_indices[fuel_type] = get_pareto_set_indices(pol_repo, fuel_type)

    PS_lead_time_data = get_pareto_set_lead_time_data(scn, pol_repo, PS_indices)
    write_lead_times(scn, PS_lead_time_data)

    cNode_actual_demand_data = get_pareto_set_actual_demand_data(scn, pol_repo, PS_indices)
    write_cNode_actual_demand_data(scn, cNode_actual_demand_data)

    pNode_proxy_demand_data = get_pareto_set_proxy_demand_data(scn, pol_repo, PS_indices)
    write_pNode_proxy_demand_data(scn, pNode_proxy_demand_data)
    plot_proxy_demand_distribution(scn, pNode_proxy_demand_data, PS_lead_time_data)


def get_pareto_set_indices(pol_repo, fuel_type):
    """Returns a list of indices of policies that form the pareto set.
    @type pol_repo: list[policy.Policy]
    @type fuel_type: str
    @rtype: list[int]
    """
    all_candidates = [(pol.idx, pol.score[fuel_type]['ESO'], pol.score[fuel_type]['SO_Int']) for pol in pol_repo]
    return [tuple_entry[0] for tuple_entry in get_pareto_set(all_candidates)]


def get_pareto_set(all_candidates):
    """Returns the pareto set of scores
    @param all_candidates: A list of tuples where the first tuple entry uniquely
        identifies that tuple, while the remaining entries correspond to the
        scores in each dimension of a multi-dimensional score for that tuple.
    @type all_candidates: list[tuple]
    @rtype: list[tuple]
    """
    dominated_set = []
    pareto_set = []
    remaining = all_candidates
    while remaining:
        candidate = remaining[0]
        new_remaining = []
        for other in remaining[1:]:
            [new_remaining, dominated_set][dominates(candidate, other)].append(other)
        if not any(dominates(other, candidate) for other in new_remaining):
            pareto_set.append(candidate)
        else:
            dominated_set.append(candidate)
        remaining = new_remaining
    return pareto_set


# noinspection PyPep8Naming
def dominates(champion, contender):
    """Returns True if the contender is equal to or better than the
    champion in every dimension.
    @type champion: tuple
    @param champion: A tuple where the first entry uniquely identifies the tuple
        and the remaining entries correspond to the scores in each dimension
        of a multi-dimensional score.
    @type contender: tuple
    @param contender: A tuple where the first entry uniquely identifies the tuple
        and the remaining entries correspond to the scores in each dimension
        of a multi-dimensional score.
    @rtype: bool
    """
    return all(chmp <= cont for chmp, cont in zip(champion[1:], contender[1:]))


# noinspection PyUnresolvedReferences
def get_pareto_set_proxy_demand_data(scn, pol_repo, PS_policy_indices):
    """Returns a dictionary that contains the day by day proxy demand over all
    simulation replications for each fuel type.
    @type scn: scenario.Scenario
    @type pol_repo: list[policy.Policy]
    @type PS_policy_indices: dict[str, list[int]]
    @rtype: dict[str, dict[int, list[float]]]
    """
    proxy_demand_data = {}
    """@type emp_demand_data: dict[str, dict[int, list[float]]]"""

    for fuel_type in scn.pNode_fuel_info:
        for pol_idx in PS_policy_indices[fuel_type]:
            for data in pol_repo[pol_idx].pNode_PD[fuel_type]:
                proxy_demand_data[fuel_type] = proxy_demand_data.get(fuel_type, []) + [data]

    for fuel_type in proxy_demand_data.iterkeys():
        proxy_demand_data[fuel_type] = {day_idx: data for day_idx, data in
                                        zip(range(0, int(scn.stop_time / 24.0)),
                                            zip(*proxy_demand_data[fuel_type]))}
    return proxy_demand_data


# noinspection PyPep8Naming
def write_pNode_proxy_demand_data(scn, pNode_proxy_demand_data):
    """Writes out the subnetwork parent node proxy distribution of demand data.
    @type scn: scenario.Scenario
    @type pNode_proxy_demand_data: dict[str, dict[int, list[float]]]
    @rtype: None
    """
    file_path = './sim_output/' + scn.theater_name + '/empirical_dist/' + scn.subnetwork_name + '.json'
    with open(file_path, 'w') as outfile:
        json.dump(pNode_proxy_demand_data, outfile)


def get_pareto_set_actual_demand_data(scn, pol_repo, PS_policy_indices):
    """Returns a dictionary that contains the day by day proxy demand over all
    simulation replications for each fuel type.
    @type scn: scenario.Scenario
    @type pol_repo: list[policy.Policy]
    @type PS_policy_indices: dict[str, list[int]]
    @rtype: dict[str, dict[int, list[float]]]
    """
    actual_demand_data = {}
    """@type actual_demand_data: dict[str, dict[int, list[float]]]"""

    for cNode_name, fuel_types in scn.cNode_fuel_info.iteritems():
        actual_demand_data[cNode_name] = {}
        for fuel_type in fuel_types:
            for pol_idx in PS_policy_indices[fuel_type]:
                for data in pol_repo[pol_idx].cNode_AD[cNode_name][fuel_type]:
                    actual_demand_data[cNode_name][fuel_type] = actual_demand_data[cNode_name].get(fuel_type, []) \
                                                                + [data]

    for cNode_name, fuel_types in scn.cNode_fuel_info.iteritems():
        for fuel_type in fuel_types:
            temp_dict = {day_idx: data for day_idx, data in
                         zip(range(0, int(scn.stop_time / 24.0)), zip(*actual_demand_data[cNode_name][fuel_type]))}
            actual_demand_data[cNode_name][fuel_type] = temp_dict

    return actual_demand_data


# noinspection PyPep8Naming
def write_cNode_actual_demand_data(scn, hist_demand_data):
    """Writes out all subnetwork child node actual distribution of demand data.
    @type scn: scenario.Scenario
    @type hist_demand_data:
    @rtype: None
    """
    for cNode_name, all_fuel_data in hist_demand_data.iteritems():
        file_path = './sim_output/' + scn.theater_name + '/historical_dist/' + cNode_name + '.json'
        with open(file_path, 'w') as outfile:
            json.dump(all_fuel_data, outfile)


def get_pareto_set_lead_time_data(scn, pol_repo, PS_indices):
    """Returns a dictionary of the subnetwork parent node--End User node--fuel type
    combination lead times.
    @type scn: scenario.Scenario
    @type pol_repo: list[policy.Policy]
    @type PS_indices: dict[str: list[int]]
    @rtype: dict[str, list[float]]
    """
    _, node_names = misc.get_subnetwork_details(scn.subnetwork_name)
    pNode_name = node_names[0]

    lead_time_data = {}
    for fuel_type in scn.pNode_fuel_info:
        for PS_pol_idx in PS_indices[fuel_type]:
            for euNode_fuel_type_comb, LTs in pol_repo[PS_pol_idx].cNode_LT.iteritems():
                if fuel_type in euNode_fuel_type_comb:
                    key = pNode_name + ':' + euNode_fuel_type_comb
                    lead_time_data[key] = lead_time_data.get(key, []) + LTs

    return lead_time_data


def write_lead_times(scn, lead_time_data):
    """Writes out the subnetwork parent node--fuel type combination lead times.
    @type scn: scenario.Scenario
    @type lead_time_data: dict[]
    @rtype: None
    """
    _, node_names = misc.get_subnetwork_details(scn.subnetwork_name)
    file_path = './sim_output/' + scn.theater_name + '/lead_time/' + node_names[0] + '.json'
    with open(file_path, 'w') as outfile:
        json.dump(lead_time_data, outfile)


def plot_proxy_demand_distribution(scn, proxy_demand_data, lead_time_data):
    """
    @type scn: scenario.Scenario
    @type proxy_demand_data: dict[str, dict[int, list[float]]]
    @type lead_time_data: dict[str, list[float]]
    @rtype: None
    """
    for fuel_type, proxy_data in proxy_demand_data.iteritems():
        data = {}
        fuel_type_LTs = []
        euNode_keys = []
        for pNode_euNode_fuel, LTs in lead_time_data.iteritems():
            if fuel_type in pNode_euNode_fuel:
                euNode_keys.append(pNode_euNode_fuel)
                fuel_type_LTs += LTs
        lead_time = int(max(fuel_type_LTs) + 0.5)

        for end_of_day, the_days_data in proxy_data.iteritems():
            if scn.warm_up - lead_time <= int(end_of_day) <= (scn.warm_up + scn.planning_horizon):
                data[int(end_of_day) - scn.warm_up] = the_days_data

        x_values = sorted(data.keys())
        emp_data_means = [np.mean(data[day]) / 1000.0 for day in x_values]

        df = pd.DataFrame({'x': x_values, 'mean': emp_data_means})
        df['rm'] = df['mean'].rolling(window=5).mean()

        x, y, rgba = [], [], []
        for end_of_day, days_data in data.iteritems():
            summarized_x, summarized_y, rgba_vals = summarize_daily_data(int(end_of_day), days_data,
                                                                         const.BOYSENBERRY_rgb)
            x += summarized_x
            y += summarized_y
            rgba += rgba_vals

        min_x, max_x, max_y = min(x), max(x), max(y)

        plt.figure(figsize=(20, 12))

        # Plot demand data
        labeling = 'Loaded'
        if scn.demand_proxy == 'fuel_sent':
            labeling = 'Sent'
        plt.scatter(x, y, c=rgba, s=50, label='Fuel ' + labeling)
        plt.scatter(x_values, emp_data_means, c=const.BLACK_OLIVE_rgb, marker='v', alpha=0.4,
                    label='Average Fuel' + labeling)
        plt.plot(df['x'], df['rm'], c=const.QUEEN_BLUE_rgb, linewidth=3, alpha=0.5, label='5-Day Rolling Mean')

        # Plot lead time data
        ctr = 1
        node_ordering = []
        eu_node_order = ['Hughie', 'Mustang', 'Monti', 'Connelly', 'Wright', 'Gamberi',
                         'Bostick', 'Garda', 'Shinwar', 'Najil', 'Falcon']
        for eu_node_name in eu_node_order:
            for pNode_euNode_fuel in euNode_keys:
                if eu_node_name in pNode_euNode_fuel:
                    node_ordering.append(pNode_euNode_fuel)
                    break

        for pNode_euNode_fuel in node_ordering:
            # noinspection PyTypeChecker
            LT_vals = [-f for f in lead_time_data[pNode_euNode_fuel]]
            plt.scatter([-f for f in lead_time_data[pNode_euNode_fuel]],
                        np.ones(len(lead_time_data[pNode_euNode_fuel])) * max_y * (1 - ctr * 0.025),
                        c=const.ORANGE_PEEL_rgb, s=60, marker='.', alpha=0.15)
            if ctr == 1:
                plt.scatter(sum(LT_vals) / float(len(LT_vals)), max_y * (1 - ctr * 0.025),
                            c=const.INDIAN_RED_rgb, s=70, marker='h', label='Estimated End User Node Lead Time')
            else:
                plt.scatter(sum(LT_vals) / float(len(LT_vals)), max_y * (1 - ctr * 0.025),
                            c=const.INDIAN_RED_rgb, s=70, marker='h')

            euNode_name = pNode_euNode_fuel.split(':')[1]
            x_coord = -min(lead_time_data[pNode_euNode_fuel]) + 2
            y_coord = max_y * (1 - ctr * 0.025)
            plt.annotate(euNode_name, xy=(x_coord, y_coord), xytext=(x_coord, y_coord), size='large', va='center',
                         color=const.BLACK_OLIVE_rgb + [0.8])
            ctr += 1

        # Plot limits
        plt.xlim(min_x - 1, max_x + 1)
        plt.ylim(max_y * -0.005, max_y * 1.05)

        # Plot labels + title
        plt.ylabel('Fuel Loaded (thousands of gallons)')
        plt.xlabel('Planning Horizon (days)')

        subnetwork_idx, node_names = misc.get_subnetwork_details(scn.subnetwork_name)
        title = 'Subnetwork ' + subnetwork_idx + ': ' + node_names[0] + ' --> {' + ', '.join(node_names[1:]) + '}'
        plt.title(title)

        plt.legend(loc=1)
        plt.tight_layout()
        plt.savefig('./sim_output/' + scn.theater_name + '/empirical_dist/' +
                    fuel_type + '_' + scn.subnetwork_name + '.png')
        plt.close()


def summarize_daily_data(day_idx, data, rgb_color):
    """Summarizes the data by only counting the occurrences of unique values
    and then associating a likelihood of occurrence with each unique value.
    @param day_idx: int
    @param data: list[float]
    @param rgb_color: str
    @rtype x_vals, y_vals, rgba_vals: list[int], list[float], list[tuple(float)]
    """
    alpha_vals = pd.value_counts(data) / float(len(data))
    # noinspection PyUnresolvedReferences
    y_vals = [y / 1000.0 for y in alpha_vals.index]
    # noinspection PyTypeChecker
    rgba_vals = [rgb_color + [alpha] for alpha in alpha_vals]
    x_vals = [day_idx for _ in range(len(y_vals))]

    return x_vals, y_vals, rgba_vals
