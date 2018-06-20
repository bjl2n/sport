from collections import OrderedDict
import copy
import json
import misc
import numpy.random as npr
import stats

CTT_DIST = [['UNIFORM', 2.5, 4],
            ['UNIFORM', 3.75, 7],
            ['UNIFORM', 5, 6]]

CAPACITY_MULTIPLIER = [3, 4]

# OPTEMPO MULTIPLIERS taken from Mark Example Data
LOW_vs_MED = 1.366
LOW_vs_HIGH = 2.240
DIST_TYPES = {1: ['TRIANGULAR', 1200, 2000, 3700]}
# noinspection PyTypeChecker
DIST_TYPES[2] = ['TRIANGULAR'] + [LOW_vs_MED * val for val in DIST_TYPES[1][1:]]
# noinspection PyTypeChecker
DIST_TYPES[3] = ['TRIANGULAR'] + [LOW_vs_HIGH * val for val in DIST_TYPES[1][1:]]

SCN1_DIST = {0: DIST_TYPES[2],
             45: DIST_TYPES[2],
             90: DIST_TYPES[1],
             150: DIST_TYPES[3],
             1000: DIST_TYPES[3]
             }

SCN2_DIST = {'Phoenix': {0: DIST_TYPES[2],
                         35: DIST_TYPES[2],
                         85: DIST_TYPES[1],
                         150: DIST_TYPES[3],
                         1000: DIST_TYPES[3]},
             'Ghazni': {0: DIST_TYPES[1],
                        70: DIST_TYPES[1],
                        130: DIST_TYPES[3],
                        150: DIST_TYPES[1],
                        1000: DIST_TYPES[1]},
             'Salerno': {0: DIST_TYPES[1],
                         45: DIST_TYPES[1],
                         120: DIST_TYPES[3],
                         150: DIST_TYPES[2],
                         1000: DIST_TYPES[2]}
             }


scn2_map = {'Hughie': 'Phoenix',
            'Mustang': 'Phoenix',
            'Monti': 'Phoenix',
            'Connelly': 'Phoenix',
            'Wright': 'Ghazni',
            'Gamberi': 'Ghazni',
            'Bostick': 'Ghazni',
            'Garda': 'Salerno',
            'Shinwar': 'Salerno',
            'Najil': 'Salerno',
            'Falcon': 'Salerno'}


def generate_end_user_node(eu_node_name, SPORT_rndstrm, scn_type):
    """Returns a dictionary containing the necessary information for the
    eu_node_name node.
    @type eu_node_name: str
    @type SPORT_rndstrm: numpy.random.RandomState
    @type scn_type: str
    @rtype: dict
    """
    node_rndstrm = npr.RandomState(misc.bounded_integer_seed(SPORT_rndstrm))

    eu_node = {'node_name': eu_node_name,
               'node_type': 'end user',
               'CTT_dist': CTT_DIST[node_rndstrm.choice(range(len(CTT_DIST)))],
               'TBT_dist': ['UNIFORM', 1.25, 3],
               'POC': 0.975}

    FSP = {'fuel_type': 'MoGas',
           'minimum': 500,
           'maximum': node_rndstrm.choice(CAPACITY_MULTIPLIER)}

    # The FSP['DCR_dist'] declaration must appear AFTER FSP['maximum'] has been
    # set to the multiplier value to make sure that the same capacity is chosen
    # regardless of the scenario.
    if scn_type == 'scn1':
        FSP['DCR_dist'] = SCN1_DIST
    elif scn_type == 'scn2':
        FSP['DCR_dist'] = SCN2_DIST[scn2_map[eu_node_name]]
    elif scn_type == 'scn3':
        FSP['DCR_dist'] = SCN2_DIST[node_rndstrm.choice(['Phoenix', 'Ghazni', 'Salerno'])]
    else:
        assert False, 'Scenario type %s not recognized' % scn_type
    # noinspection PyUnresolvedReferences
    eu_node_DCR_mean = max([stats.mean(dist_spec) for dist_spec in FSP['DCR_dist'].itervalues()])

    FSP['maximum'] = int(FSP['maximum'] * eu_node_DCR_mean)
    FSP['stockage_objective'] = int(0.975 * FSP['maximum'])
    FSP['on_hand'] = FSP['stockage_objective']
    FSP['LPs'] = [{'number_of_points': 4, 'download_rate': 15000, 'upload_rate': 15000}]
    eu_node['FSPs'] = [FSP]
    eu_node['DCR_mean'] = eu_node_DCR_mean

    return eu_node


# noinspection PyDictCreation
def generate_intermediate_node(im_node_name, cNode_names, subnetwork_idx, nodes, SPORT_rndstrm):
    """Returns a dictionary containing the necessary information for the
    im_node_name node.
    @type im_node_name: str
    @type cNode_names: list[str]
    @type subnetwork_idx: int
    @type nodes: dict
    @type SPORT_rndstrm: numpy.random.RandomState
    @rtype: dict
    """
    node_rndstrm = npr.RandomState(misc.bounded_integer_seed(SPORT_rndstrm))

    im_node = {'node_name': im_node_name,
               'node_type': 'intermediate',
               'TTT_dist': ['UNIFORM', 1.5, 2.75],
               'TBT_dist': ['UNIFORM', 1.25, 3],
               'POC': 0.95,
               'DCR_mean': sum(nodes[cNode_name]['DCR_mean'] for cNode_name in cNode_names),
               'CTT_dist': CTT_DIST[node_rndstrm.choice(range(len(CTT_DIST)))],
               'CTTs': {}}
    for child_node_name in cNode_names:
        im_node['CTTs'][child_node_name] = CTT_DIST[node_rndstrm.choice(range(len(CTT_DIST)))]

    FSP = {'fuel_type': 'MoGas',
           'minimum': 500,
           'DCR_dist': str(subnetwork_idx) + '_' + ''.join([im_node_name] + cNode_names),
           'maximum': int(node_rndstrm.choice(CAPACITY_MULTIPLIER) * im_node['DCR_mean'] * 5.5)}
    FSP['stockage_objective'] = int(0.975 * FSP['maximum'])
    FSP['on_hand'] = FSP['stockage_objective']
    FSP['LPs'] = [{'number_of_points': 10, 'download_rate': 15000, 'upload_rate': 15000}]
    im_node['FSPs'] = [FSP]

    return im_node


# noinspection PyDictCreation
def generate_depot_node(depot_node_name, cNode_names, subnetwork_idx, nodes, SPORT_rndstrm):
    """Returns a dictionary containing the necessary information for the
    im_node_name node.
    @type depot_node_name: str
    @type cNode_names: list[str]
    @type subnetwork_idx: int
    @type nodes: dict
    @type SPORT_rndstrm: numpy.random.RandomState
    @rtype: dict
    """
    node_rndstrm = npr.RandomState(misc.bounded_integer_seed(SPORT_rndstrm))

    dpt_node = {'node_name': depot_node_name,
                'node_type': 'depot',
                'TTT_dist': ['UNIFORM', 1.5, 2.75],
                'POC': 0.95,
                'DCR_mean': sum(nodes[cNode_name]['DCR_mean'] for cNode_name in cNode_names),
                'CTTs': {}}
    for child_node_name in cNode_names:
        dpt_node['CTTs'][child_node_name] = CTT_DIST[node_rndstrm.choice(range(len(CTT_DIST)))]

    FSP = {'fuel_type': 'MoGas',
           'minimum': 500,
           'DCR_dist': str(subnetwork_idx) + '_' + ''.join([depot_node_name] + cNode_names),
           'maximum': int(node_rndstrm.choice(CAPACITY_MULTIPLIER) * dpt_node['DCR_mean'] * 2.5)}
    FSP['stockage_objective'] = int(0.975 * FSP['maximum'])
    FSP['on_hand'] = FSP['stockage_objective']
    FSP['LPs'] = [{'number_of_points': 15, 'download_rate': 15000, 'upload_rate': 15000}]
    dpt_node['FSPs'] = [FSP]

    return dpt_node


# noinspection PyPep8Naming
def generate_SPORT_scenario(theater_name, planning_horizon, warm_up, post_ph,
                            num_policies, num_reps, num_TEP_reps,
                            rep_analysis_lvl, pol_analysis_lvl, plot_type, tanker_capacity,
                            lower_bound, demand_proxy, SPORT_rndstrm):
    sim_params = {'theater_name': theater_name,
                  'planning_horizon': planning_horizon,
                  'warm_up': warm_up,
                  'post_ph': post_ph,
                  'num_policies': num_policies,
                  'num_reps': num_reps,
                  'num_TEP_reps': num_TEP_reps,
                  'rep_analysis_lvl': rep_analysis_lvl,
                  'pol_analysis_lvl': pol_analysis_lvl,
                  'lower_bound': lower_bound,
                  'plot_type': plot_type,
                  'demand_proxy': demand_proxy,
                  'fuel_epsilon': 0.1,
                  'eql_epsilon': 0.03,
                  'pns_epsilon': 0.03,
                  'fuel_req_setting': 'delta',
                  'adj_factor_setting': 'omicron',
                  'load_plan_setting': 'non_greedy',
                  'analysis_interval_length': 24.0,
                  'base_safety_stock_days': 5.0,
                  'stale_after': 2.0}

    tanker_types = {'MoGas': [{'num_tankers': 0, 'capacity': tanker_capacity}]}

    eu_node_names = ['Hughie', 'Mustang', 'Monti', 'Connelly',
                     'Wright', 'Gamberi', 'Bostick', 'Garda',
                     'Shinwar', 'Najil', 'Falcon']
    im_node_names = ['Gardez', 'Sharana', 'Fenty', 'Ghazni',
                     'Phoenix', 'Salerno']
    dpt_node_names = ['Marmal']

    sub_network_defs = OrderedDict()
    sub_network_defs['Gardez'] = ['Hughie', 'Mustang']
    sub_network_defs['Sharana'] = ['Monti', 'Connelly']
    sub_network_defs['Fenty'] = ['Garda', 'Shinwar', 'Najil']
    sub_network_defs['Ghazni'] = ['Wright', 'Gamberi', 'Bostick']
    sub_network_defs['Phoenix'] = ['Gardez', 'Sharana']
    sub_network_defs['Salerno'] = ['Fenty', 'Falcon']
    sub_network_defs['Marmal'] = ['Phoenix', 'Ghazni', 'Salerno']

    if 'scn1' in theater_name:
        scn_type = 'scn1'
    elif 'scn2' in theater_name:
        scn_type = 'scn2'
    elif 'scn3' in theater_name:
        scn_type = 'scn3'
    else:
        assert False, 'Scenario type not recognized'

    node_defs = {}
    for eu_node_name in eu_node_names:
        node_defs[eu_node_name] = generate_end_user_node(eu_node_name, SPORT_rndstrm, scn_type)
    subnetwork_idx = 1
    for im_node_name in im_node_names:
        node_defs[im_node_name] = generate_intermediate_node(im_node_name, sub_network_defs[im_node_name],
                                                             subnetwork_idx, node_defs, SPORT_rndstrm)
        subnetwork_idx += 1

    for dpt_node_name in dpt_node_names:
        node_defs[dpt_node_name] = generate_depot_node(dpt_node_name, sub_network_defs[dpt_node_name],
                                                       subnetwork_idx, node_defs, SPORT_rndstrm)

    scn_dict = {}
    subnetwork_idx = 1
    for pNode_name, cNode_names in sub_network_defs.iteritems():
        sim_params['subnetwork_name'] = str(subnetwork_idx) + '_' + ''.join([pNode_name] + cNode_names)
        sim_params['rep_start_index'] = misc.bounded_integer_seed(SPORT_rndstrm)
        sim_params['cNode_lead_times'] = {}
        for cNode_name in cNode_names:
            if node_defs[cNode_name]['node_type'] == 'end user':
                sim_params['cNode_lead_times'][cNode_name + ':' + cNode_name + ':' + 'MoGas'] = 0.0

        scn_dict[subnetwork_idx] = {'SIM_PARAMS': copy.deepcopy(sim_params),
                                    'SUPPLY_CHAIN':
                                        {'CHILD_NODES': [node_defs[cNode_name] for cNode_name in cNode_names],
                                         'PARENT_NODE': node_defs[pNode_name],
                                         'TANKER_TYPES': copy.deepcopy(tanker_types)
                                         }
                                    }
        subnetwork_idx += 1

    with open('./sim_output/' + theater_name + '/' + theater_name + '.spt', 'w') as out_file:
        json.dump(scn_dict, out_file, indent=4, separators=(',', ': '))
