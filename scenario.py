import child_node
import fsp
import itertools as itr
import json
import lp
import parent_node
import tanker as tkr


# noinspection PyPep8Naming
class Scenario:
    """This class holds the various parameters and data that defines the scenario that is being
    simulated (located in the *.tfsp scenario file at tfsp_scenario_file_path). It also contains
    functions related to loading the scenario data from a *.tfsp file and then initializing the
    scenario objects (tankers, parent node, and child nodes).
    @type subnetwork_data: dict
    stop_time: The time (in hours) at which the simulation stops
    num_reps: The number of simulation replications per policy.
    rep_start_index: Integer at which the replication numbers start.
    output_path: The directory where all simulation output is written to.
    plot_type: Used to control what type of file plots are saved as (one of
        {'.png', '.pdf'}), default to '.png'.
    rep_analysis_lvl: Denotes the level at which replication level analysis
        should be carried out (one of {None, 'plots', 'data', or 'all'})
    pol_analysis_lvl: Denotes the level at which policy level analysis
        should be carried out (one of {None, 'plots', or 'all'})
    fuel_epsilon: Tolerance range in gallons for fuel transactions; transactions
        should be within fuel_epsilon gallons of any limits, otherwise an assert should
        trigger.
    pns_epsilon: Tolerance range for the increase or decrease in the proportion not stocked.
    eql_epsilon: Tolerance range for the increase or decrease in the expected queue length.
    stale_after: Amount of time in hours after which a demand event grows stale.
    base_safety_stock_days: Base safety stock level in days.
    adj_factor_setting: Controls which function gets executed, one of {'alpha', 'beta', 'gamma'}.
    fuel_req_setting: Controls which function gets executed, one of {'alpha', 'beta', 'gamma'}.
    load_plan_setting: Controls which function gets executed, one of {'greedy', 'non_greedy'}.
    analysis_interval_length: Length of interval (in hours) when performing analysis.
    """
    def __init__(self, subnetwork_data, rndstrm, TEP_node_subnetwork):
        """This class holds the various parameters and data that defines the scenario that is being
        simulated (located in the *.tfsp scenario file at tfsp_scenario_file_path). It also contains
        functions related to loading the scenario data from a *.tfsp file and then initializing the
        scenario objects (tankers, parent node, and child nodes).
        @type subnetwork_data: dict
        @type rndstrm: numpy.random.RandomState
        @type TEP_node_subnetwork: bool
        stop_time: The time (in hours) at which the simulation stops
        num_reps: The number of simulation replications per policy.
        rep_start_index: Integer at which the replication numbers start.
        output_path: The directory where all simulation output is written to.
        plot_type: Used to control what type of file plots are saved as (one of
            {'.png', '.pdf'}), default to '.png'.
        rep_analysis_lvl: Denotes the level at which replication level analysis
            should be carried out (one of {None, 'plots', 'data', or 'all'})
        pol_analysis_lvl: Denotes the level at which policy level analysis
            should be carried out (one of {None, 'plots', or 'all'})
        fuel_epsilon: Tolerance range in gallons for fuel transactions; transactions
            should be within fuel_epsilon gallons of any limits, otherwise an assert should
            trigger.
        pns_epsilon: Tolerance range for the increase or decrease in the proportion not stocked.
        eql_epsilon: Tolerance range for the increase or decrease in the expected queue length.
        stale_after: Amount of time in hours after which a demand event grows stale.
        base_safety_stock_days: Base safety stock level in days.
        adj_factor_setting: Controls which function gets executed, one of {'alpha', 'beta', 'gamma'}.
        fuel_req_setting: Controls which function gets executed, one of {'alpha', 'beta', 'gamma'}.
        load_plan_setting: Controls which function gets executed, one of {'greedy', 'non_greedy'}.
        analysis_interval_length: Length of interval (in hours) when performing analysis.
        """
        self.scenario_data = subnetwork_data

        self.load_scenario_from_data_dict(rndstrm, TEP_node_subnetwork)

    # noinspection PyAttributeOutsideInit
    def load_scenario_from_data_dict(self, rndstrm, TEP_node_subnetwork):
        """Load the scenario from the data dictionary.
        Also initialize the simulation and experimental parameters.
        @type rndstrm: numpy.random.RandomState
        @type TEP_node_subnetwork: bool
        """
        sim_params = self.scenario_data['SIM_PARAMS']

        self.theater_name = sim_params['theater_name']
        self.planning_horizon = int(sim_params['planning_horizon'])
        self.warm_up = int(sim_params['warm_up'])
        self.post_ph = int(sim_params['post_ph'])
        self.stop_time = (self.warm_up + self.planning_horizon + self.post_ph) * 24.0
        self.cNode_lead_times = self.read_in_cNode_lead_times()
        self.subnetwork_name = sim_params['subnetwork_name']
        self.output_path = './sim_output/' + self.theater_name + '/' + self.subnetwork_name
        self.num_policies = int(sim_params.get('num_policies', 10))
        self.num_reps = int(sim_params.get('num_reps', 25))
        self.num_TEP_reps = int(sim_params.get('num_TEP_reps', 90))
        self.rep_start_index = int(sim_params['rep_start_index'])
        self.demand_proxy = sim_params.get('demand_proxy', 'amount_loaded')
        self.plot_type = sim_params.get('plot_type', '.png')
        self.rep_analysis_lvl = sim_params.get('rep_analysis_lvl', None)
        if sim_params['rep_analysis_lvl'] == 'None':
            self.rep_analysis_lvl = None
        self.pol_analysis_lvl = sim_params.get('pol_analysis_lvl', None)
        if sim_params['pol_analysis_lvl'] == 'None':
            self.rep_analysis_lvl = None
        self.fuel_epsilon = float(sim_params.get('fuel_epsilon', 0.1))
        self.pns_epsilon = float(sim_params.get('pns_epsilon', 0.03))
        self.eql_epsilon = float(sim_params.get('eql_epsilon', 0.03))
        self.stale_after = float(sim_params.get('stale_after', 2.0))
        self.base_safety_stock_days = float(sim_params.get('base_safety_stock_days', 0.0))
        self.adj_factor_setting = sim_params.get('adj_factor_setting', 'delta')
        self.fuel_req_setting = sim_params.get('fuel_req_setting', 'delta')
        self.load_plan_setting = sim_params.get('load_plan_setting', 'beta')
        self.analysis_interval_length = float(sim_params.get('analysis_interval_length', 24.0))
        try:
            if sim_params['lower_bound'] == 'True':
                self.lower_bound = True
            else:
                self.lower_bound = False
        except KeyError:
            self.lower_bound = False
        self.cNode_fuel_info = {}
        self.fuel_cNode_info = {}
        self.pNode_fuel_info = []
        self.cNode_fuel_dist = {}
        self.num_cNode_fuel_type_combos = {}
        self.sample_realization_indices = {}

        self.set_scenario_info(rndstrm, TEP_node_subnetwork)

    def read_in_cNode_lead_times(self):
        """Returns a dictionary that should contain key of the form
        'NodeName_FuelType' for each child node--fuel type combination in the
        subnetwork.  The value is the lead time estimate for that key.
        @rtype: dict[str, float]
        """
        # Read in any End User node lead times from the scenario data
        cNode_lead_times = {key: lead_time for key, lead_time
                            in self.scenario_data['SIM_PARAMS']['cNode_lead_times'].iteritems()}

        # Read in any Intermediate node lead times from the relevant file in
        # the lead_time folder
        for cNode in self.scenario_data['SUPPLY_CHAIN']['CHILD_NODES']:
            if cNode['node_type'] == 'intermediate':
                file_path = './sim_output/' + self.theater_name + '/lead_time/' + cNode['node_name'] + '.json'
                try:
                    with open(file_path, 'r') as lead_time_file:
                        cNode_LT = json.load(lead_time_file)
                        for cNode_euNode_fuel, data in cNode_LT.iteritems():
                            cNode_lead_times[cNode_euNode_fuel] = sum(data) / float(len(data))
                except IOError:
                    assert False, 'There is no lead time file at %s for %s' % (file_path, cNode['node_name'])

        return cNode_lead_times

    # noinspection PyAttributeOutsideInit
    def set_scenario_info(self, rndstrm, TEP_node_subnetwork):
        """Set the scenario information dictionaries where each key is a string
        and each value is a list of strings.
        pNode_fuel_info: maps the parent node name to a list of fuel types stored
            at the parent node
        cNode_fuel_info: maps a child node name to a list of fuel types stored at
            the child node
        fuel_cNode_info: maps a fuel type to list of child node names that store
            that fuel type
        cNode_fuel_dist: maps a child node name--fuel type combination to the demand
            distribution
        num_cNode_fuel_type_combos: maps a fuel type to the number of child nodes
            that store that fuel type
        @type rndstrm: numpy.random.RandomState
        @type TEP_node_subnetwork: bool
        """
        for FSP in self.scenario_data['SUPPLY_CHAIN']['PARENT_NODE']['FSPs']:
            fuel_type = FSP['fuel_type']
            self.pNode_fuel_info.append(fuel_type)

        for cNode in self.scenario_data['SUPPLY_CHAIN']['CHILD_NODES']:
            cNode_name = cNode['node_name']

            for FSP in cNode['FSPs']:
                fuel_type = FSP['fuel_type']

                self.num_cNode_fuel_type_combos[fuel_type] = self.num_cNode_fuel_type_combos.get(fuel_type, 0) + 1
                self.cNode_fuel_info[cNode_name] = self.cNode_fuel_info.get(cNode_name, []) + [fuel_type]
                self.fuel_cNode_info[fuel_type] = self.fuel_cNode_info.get(fuel_type, []) + [cNode_name]
                self.cNode_fuel_dist[(cNode_name, fuel_type)] = self.get_DCR_distributions(FSP, cNode['node_type'])
                self.cNode_fuel_dist[(cNode_name, 'node_type')] = cNode['node_type']

                if cNode['node_type'] == 'intermediate':
                    num_sample_realizations = len(self.cNode_fuel_dist[(cNode_name, fuel_type)]['0'])
                    shuffled_indices = range(num_sample_realizations)
                    rndstrm.shuffle(shuffled_indices)
                    self.sample_realization_indices[cNode_name + ':' + fuel_type] = shuffled_indices

        self.pNode_fuel_info.sort()
        for fuel_list in self.cNode_fuel_info.itervalues():
            fuel_list.sort()
        for cNode_list in self.fuel_cNode_info.itervalues():
            cNode_list.sort()

        if TEP_node_subnetwork:
            for fuel_type, cNode_name_list in self.fuel_cNode_info.iteritems():
                indices_list = []
                num_combinations = 1
                for cNode_name in cNode_name_list:
                    indices_list.append(self.sample_realization_indices[cNode_name + ':' + fuel_type])
                    num_combinations *= len(indices_list[-1])
                    self.sample_realization_indices[cNode_name + ':' + fuel_type] = []

                if num_combinations <= self.num_TEP_reps:
                    combinations = list(itr.product(*indices_list))
                    rndstrm.shuffle(combinations)
                else:
                    combinations = set()
                    while len(combinations) < self.num_TEP_reps:
                        combinations.add(tuple(rndstrm.choice(an_idx_list) for an_idx_list in indices_list))

                for idx_tuple in combinations:
                    for cNode_idx, cNode_name in enumerate(cNode_name_list):
                        self.sample_realization_indices[cNode_name + ':' + fuel_type].append(idx_tuple[cNode_idx])

    def get_DCR_distributions(self, FSP, cNode_type):
        """Returns a dictionary where the keys indicate the last day that the
        particular distribution is in effect, and where the value is a tuple
        with the name and parameters of the distribution if cNode_type = 'end user',
        and the value is a list of floats (empirical distribution) if
        cNode_type = 'intermediate'.
        @type FSP: dict
        @type cNode_type: str
        @return: dict{int: tuple}
        """
        if cNode_type == 'intermediate':
            emp_dist_data_path = './sim_output/' + self.theater_name + '/empirical_dist/' + FSP['DCR_dist'] + '.json'
            try:
                with open(emp_dist_data_path, 'r') as sport_scenario_file:
                    return json.load(sport_scenario_file)[FSP['fuel_type']]
            except IOError:
                assert False, 'File %s does not exist' % emp_dist_data_path
        else:
            return self.get_time_dependent_DCR_distributions(FSP)

    def get_time_dependent_DCR_distributions(self, FSP):
        """Returns a dictionary where the keys indicate the last day that the
        particular distribution is in effect, and where the value is a tuple
        with the name and parameters of the distribution.
        @type FSP: dict
        @return: dict{int: tuple}
        """
        DCR_dist = {}
        for end_time, dist_list in FSP['DCR_dist'].iteritems():
            DCR_dist[int(end_time)] = self.dist_list_to_tuple(dist_list)

        return DCR_dist

    def initialize_scenario_objects(self, rep_rndstrm, rep_idx):
        """Initializes the tanker, parent node, and child node objects
        from the scenario_data dictionary.
        @type rep_rndstrm: numpy.random.RandomState
        @type rep_idx: int
        @return: sim_pNode, sim_cNodes, sim_tankers
        @rtype: parent_node.ParentNode, dict[str, child_node.ChildNode], list[tanker.Tankers]
        """
        sim_tankers = self.initialize_tankers()
        sim_cNodes = self.initialize_child_nodes(rep_rndstrm, rep_idx)
        sim_pNode = self.initialize_parent_node(sim_cNodes, rep_rndstrm)

        return sim_pNode, sim_cNodes, sim_tankers

    def initialize_tankers(self):
        """Initialize all tankers based on the information stored in
        scenario_data under 'TANKER_TYPES'.
        @rtype: list[tanker.Tankers]
        """
        sim_tankers = []
        range_bottom = 0
        for fuel_type, tanker_type_list in self.scenario_data['SUPPLY_CHAIN']['TANKER_TYPES'].iteritems():
            for tanker_type in tanker_type_list:
                num_tankers = int(tanker_type['num_tankers'])
                for i in range(range_bottom, range_bottom + num_tankers):
                    sim_tankers.append(tkr.Tanker('HNTb_%s' % i, 'beta',
                                                  float(tanker_type['capacity']),
                                                  fuel_type))
                range_bottom += num_tankers

        return sim_tankers

    def initialize_child_nodes(self, rep_rndstrm, rep_idx):
        """Initialize all child nodes based on the information stored in
        scenario_data under 'CHILD_NODES'.
        @type rep_rndstrm: numpy.random.RandomState
        @type rep_idx: int
        @rtype: dict[str, child_node.ChildNode]
        """
        pNode_name = self.scenario_data['SUPPLY_CHAIN']['PARENT_NODE']['node_name']

        sim_cNodes = {}
        for cNode in self.scenario_data['SUPPLY_CHAIN']['CHILD_NODES']:
            cNode_name = cNode['node_name']
            cNode_type = cNode['node_type']
            cNode_CTT_dist = {pNode_name: self.dist_list_to_tuple(cNode['CTT_dist'])}
            cNode_TMRs_dist = {'unloaded': self.dist_list_to_tuple(cNode['TBT_dist'])}
            cNode_POC = float(cNode['POC'])

            cNode_FSPs = {}
            for FSP in cNode['FSPs']:
                fuel_type = FSP['fuel_type']
                minimum = float(FSP['minimum'])
                maximum = float(FSP['maximum'])
                stockage_objective = float(FSP['stockage_objective'])
                on_hand = float(FSP['on_hand'])
                DCR_dist = self.get_DCR_distributions(FSP, cNode_type)

                FSP_LPs = []
                range_bottom = 0
                for LP in FSP['LPs']:
                    num_points = int(LP['number_of_points'])
                    for LP_idx in range(range_bottom, range_bottom + num_points):
                        FSP_LPs.append(lp.LoadingPoint('LP%i' % LP_idx, cNode_name, fuel_type,
                                                       float(LP['upload_rate']),
                                                       float(LP['download_rate'])))
                    range_bottom += num_points

                if cNode_type == 'intermediate':
                    sample_realization_idx = self.sample_realization_indices[cNode_name + ':' + fuel_type][rep_idx]
                    cNode_FSPs[fuel_type] = fsp.ChildNodeFSP(cNode_name, cNode_type, fuel_type, FSP_LPs,
                                                             minimum, maximum, stockage_objective,
                                                             on_hand, DCR_dist, sample_realization_idx, rep_rndstrm)
                else:
                    cNode_FSPs[fuel_type] = fsp.ChildNodeFSP(cNode_name, cNode_type, fuel_type, FSP_LPs,
                                                             minimum, maximum, stockage_objective,
                                                             on_hand, DCR_dist, -1, rep_rndstrm)

            sim_cNodes[cNode_name] = child_node.ChildNode(cNode_name, cNode_type, cNode_FSPs, cNode_CTT_dist,
                                                          cNode_TMRs_dist, cNode_POC, rep_rndstrm)
        return sim_cNodes

    def initialize_parent_node(self, sim_cNodes, rep_rndstrm):
        """Initialize the parent node based on the information stored in
        scenario_data under 'PARENT_NODE'.
        @type sim_cNodes: dict[str, child_node.ChildNode]
        @type rep_rndstrm: numpy.random.RandomState
        @rtype: parent_node.ParentNode
        """
        pNode = self.scenario_data['SUPPLY_CHAIN']['PARENT_NODE']
        pNode_name = pNode['node_name']
        pNode_type = pNode['node_type']
        pNode_TMRs_dist = {'loaded': self.dist_list_to_tuple(pNode['TTT_dist'])}
        pNode_POC = float(pNode['POC'])

        pNode_CTTs = {cNode_name: self.dist_list_to_tuple(CTT_dist)
                      for cNode_name, CTT_dist in pNode['CTTs'].iteritems()}

        pNode_FSPs = {}
        for FSP in pNode['FSPs']:
            fuel_type = FSP['fuel_type']
            minimum = float(FSP['minimum'])
            maximum = float(FSP['maximum'])
            stockage_objective = float(FSP['stockage_objective'])
            on_hand = float(FSP['maximum'])

            FSP_LPs = []
            range_bottom = 0
            for LP in FSP['LPs']:
                num_points = int(LP['number_of_points'])
                for LP_idx in range(range_bottom, range_bottom + num_points):
                    FSP_LPs.append(lp.LoadingPoint('LP%i' % LP_idx, pNode_name, fuel_type,
                                                   float(LP['upload_rate']),
                                                   float(LP['download_rate'])))
                range_bottom += num_points

            pNode_FSPs[fuel_type] = fsp.ParentNodeFSP(pNode_name, fuel_type, FSP_LPs,
                                                      minimum, maximum, stockage_objective, on_hand)

        return parent_node.ParentNode(pNode_name, pNode_type, pNode_FSPs, pNode_CTTs,
                                      pNode_TMRs_dist, pNode_POC, rep_rndstrm, sim_cNodes.values())

    @staticmethod
    def dist_list_to_tuple(dist_list):
        """Convert a distribution list from a *.tfsp file into a tuple
        that can be used (the distribution name is converted to lowercase).
        @type dist_list: list
        @type: tuple
        """
        dist_name = dist_list[0].lower()
        dist_params = [float(param) for param in dist_list[1:]]

        return tuple([dist_name] + dist_params)
