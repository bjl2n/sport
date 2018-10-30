import atn_generator
import joblib
import json
import misc
import multiprocessing
import numpy as np
import numpy.random as npr
import p_analysis
import policy
import scenario
import sim_engine
import subnetwork


SIM_TYPE = 'parallel'  # {'series', 'parallel'}


def main():
    scn_seed_combinations = [('scn1', 12394), ('scn1', 90795),
                             ('scn2', 12394), ('scn2', 39372), ('scn2', 90795),
                             ('scn3', 90795)]

    for scn_type, SPORT_seed in scn_seed_combinations:
        SPORT(scn_type, SPORT_seed, 'amount_loaded')


# noinspection PyPep8Naming
def SPORT(scn_type, SPORT_seed, demand_proxy):
    """Performs a simulation of the theater_name scenario using the SPORT simulation engine.
    @type scn_type: str
    @type SPORT_seed: int
    @type demand_proxy: str
    @rtype: None
    """
    # ----------------------------------------------------------------------------------------------
    # THIS CODE GENERATES THE SPORT SCENARIO FILE AS SEEN BELOW
    TEP_node_idx = 7
    SPORT_rndstrm = npr.RandomState(SPORT_seed)
    theater_name = scn_type + '_' + demand_proxy + '_' + str(SPORT_seed)
    planning_horizon = 150  # units of days
    warm_up = 280  # units of days
    post_ph = 30  # units of days
    num_policies = 25
    num_reps = 40
    num_TEP_reps = 40
    rep_analysis_lvl = None  # One of {None, 'plots', 'data', 'all'}
    pol_analysis_lvl = 'plots'  # One of {None, 'plots', 'all'}
    plot_type = '.png'  # One of {'pdf', 'png'}
    tanker_capacity = 5000
    lower_bound = False
    # ----------------------------------------------------------------------------------------------

    misc.create_SPORT_directories(theater_name)
    misc.copy_code('./sim_output/' + theater_name)
    atn_generator.generate_SPORT_scenario(theater_name, planning_horizon, warm_up, post_ph,
                                          num_policies, num_reps, num_TEP_reps,
                                          rep_analysis_lvl, pol_analysis_lvl,
                                          plot_type, tanker_capacity, lower_bound, demand_proxy,
                                          SPORT_rndstrm)

    file_path = './sim_output/' + theater_name + '/' + theater_name + '.spt'
    try:
        with open(file_path, 'r') as sport_scenario_file:
            all_scenario_data = json.load(sport_scenario_file)
    except IOError:
        assert False, 'There is no *.spt file at %s' % file_path

    for subnetwork_idx in range(1, len(all_scenario_data) + 1):
        subnetwork_data = all_scenario_data[str(subnetwork_idx)]
        subnetwork_idx, node_names = misc.get_subnetwork_details(subnetwork_data['SIM_PARAMS']['subnetwork_name'])
        print 'Subnetwork ' + subnetwork_idx + ': ' + node_names[0] + ' --> {' + ', '.join(node_names[1:]) + '}'
        if int(subnetwork_idx) == TEP_node_idx:
            simulate_subnetwork(subnetwork_data, SPORT_rndstrm, TEP_node_subnetwork=True)
        else:
            simulate_subnetwork(subnetwork_data, SPORT_rndstrm, TEP_node_subnetwork=False)
        print


def simulate_subnetwork(subnetwork_scn_data, rndstrm, TEP_node_subnetwork):
    """Simulates the subnetwork of the theater fuel supply chain network and writes
    out the empirical demand distribution for the parent node.
    @type subnetwork_scn_data: dict
    @type rndstrm: numpy.random.RandomState
    @type TEP_node_subnetwork: bool
    """
    scn = scenario.Scenario(subnetwork_scn_data, rndstrm, TEP_node_subnetwork)

    misc.create_clean_output_directory(scn.output_path)
    subnetwork.set_tanker_constraint(scn)
    misc.save_scenario_data(scn, scn.output_path)

    # SIMULATION-OPTIMIZATION
    print 'Simulation-optimization runs'
    pol_history, pol_repo = [], []
    damping_factor = 1.0
    if not TEP_node_subnetwork:
        seed_range = range(scn.rep_start_index, scn.rep_start_index + scn.num_reps)
    else:
        seed_range = range(scn.rep_start_index, scn.rep_start_index + scn.num_TEP_reps)
    adj_factor, pol_lineage = {}, {}

    for policy_idx in range(scn.num_policies):
        policy_path = scn.output_path + "/policy_" + str(policy_idx)
        pol = policy.Policy(policy_idx, policy_path, adj_factor, pol_lineage, damping_factor, pol_history)
        pol.rep_dict = simulate_current_policy(scn, pol, seed_range)
        adj_factor, pol_lineage = p_analysis.policy_analysis(scn, pol, seed_range)

        pol_history.append(minimal_policy_copy(pol))
        pol_repo.append(shallow_policy_copy(pol))
        print pol.score

    # p_analysis.write_policy_data(scn, pol_history)
    p_analysis.policy_score_progression_plot(scn, pol_history)
    misc.save_policy_scores(pol_history, scn.output_path)

    subnetwork.end_of_subnetwork_simulation(scn, pol_repo)


# noinspection PyPep8Naming
def simulate_current_policy(scn, pol, seed_range):
    """Runs the simulation for each seed in seed_range using the current policy.
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type seed_range: list[int]
    @rtype rep_dict: dict[int, (sim_engine.SimulationCalendar, parent_node.ParentNode,
                            dict[str, child_node.ChildNode], list[Tanker])]
    """
    print 'Policy %s' % pol.idx
    print 'Done with replication number:',
    if SIM_TYPE == 'series':
        rep_data = []
        for rep_idx, the_seed in enumerate(seed_range):
            rep_data.append(single_simulation_replication(scn, pol, the_seed, rep_idx))
        print ''

        return dict(rep_data)
    elif SIM_TYPE == 'parallel':
        cores_to_use = max(1, (multiprocessing.cpu_count() - 1))

        parallel_results = []
        for rep_indices, seed_indices in \
                zip(chunks(range(len(seed_range)), cores_to_use), chunks(seed_range, cores_to_use)):
            parallel_results += joblib.Parallel(n_jobs=cores_to_use)(
                joblib.delayed(single_simulation_replication)(scn, pol, the_seed, rep_idx) for rep_idx, the_seed in
                zip(rep_indices, seed_indices))
            print len(parallel_results)

        rep_dict = dict(parallel_results)
        print ''

        return rep_dict
    else:
        assert False, 'SIM_TYPE %s undefined' % SIM_TYPE


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


# noinspection PyPep8Naming
def single_simulation_replication(scn, pol, the_seed, rep_idx):
    """Performs a single simulation replication using the particular seed.
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type the_seed: int
    @type rep_idx: int
    @return: the_seed, (cal, sim_pNode, sim_cNodes, sim_tankers)
    @rtype: int, (sim_engine.SimulationCalendar, parent_node.ParentNode,
             dict[str, child_node.ChildNode], list[Tanker])
    """
    # INITIALIZE THE REPLICATION RANDOM STREAM BASED ON the_seed.
    # Note: rep_rndstrm must be set before data is loaded and objects are initialized.
    rep_rndstrm = npr.RandomState()
    rep_rndstrm.seed(the_seed)

    sim_pNode, sim_cNodes, sim_tankers = scn.initialize_scenario_objects(rep_rndstrm, rep_idx)
    sim_tankers.sort(key=lambda x: x.name)

    # INITIALIZE CALENDAR & ADD INITIAL CONDITIONS
    cal, time = set_initial_conditions(scn, pol, sim_pNode, sim_cNodes, sim_tankers)

    # BEGIN SIMULATION
    while cal.calendar and (cal.calendar[0].occurrence_time < scn.stop_time):
        cal.handle_event(scn, time)

    end_of_simulation_replication_updates(scn, cal, time, sim_pNode, sim_cNodes)

    # print '%s,' % the_seed,

    return the_seed, (cal, sim_pNode, sim_cNodes, sim_tankers)


# noinspection PyPep8Naming
def set_initial_conditions(scn, pol, sim_pNode, sim_cNodes, sim_tankers):
    """Initializes the simulation calendar and simulation time.  Adds all events
    to the calendar that represent the initial conditions of the simulation.
    @type scn: scenario.Scenario
    @type pol: policy.Policy
    @type sim_pNode: parent_node.ParentNode
    @type sim_cNodes: dict[str, child_node.ChildNode]
    @type sim_tankers: list[tanker.Tanker]
    @rtype: (sim_engine.SimulationCalendar, sim_engine.SimulationTime)
    """
    sim_cal = sim_engine.SimulationCalendar()
    sim_time = sim_engine.SimulationTime()

    # Parent Node
    # Initialize the daily generation and submission of loading plans
    # and execution of TMRs
    sim_cal.add_event(sim_pNode, 'daily_plan',
                      [scn, pol, sim_cal, sim_time],
                      8.0)
    # Initialize the daily update of fuel on hand
    sim_cal.add_event(sim_pNode, 'daily_FSP_update',
                      [scn, sim_cal, sim_time],
                      24.0)
    # Child Nodes
    for cNode in sim_cNodes.itervalues():
        # Initialize the daily consumption of fuel
        sim_cal.add_event(cNode, 'child_node_daily_demand',
                          [scn, sim_cal, sim_time],
                          0.0)
        # Initialize the daily processing of TMRs
        sim_cal.add_event(cNode, 'execute_unloaded_tanker_TMR',
                          [scn, sim_cal, sim_time, sim_pNode],
                          9.0)
        # Initialize the daily update of fuel on hand
        sim_cal.add_event(cNode, 'daily_FSP_update',
                          [scn, sim_cal, sim_time],
                          24.0)
        # Set the on_hand amount for each child node--fuel type combination
        for FSP in cNode.FSPs.itervalues():
            FSP.on_hand = FSP.storage_max
            FSP.inv_trace = [(0.0, FSP.on_hand)]

    # Calculate the lead_time for each End User node--fuel type combination
    # that is routed through the cNode.  Lead time is calculated in reference to
    # the start of the planning horizon (which corresponds to the end of the
    #  warm-up period).
    for cNode_euNode_fuel, lead_time in scn.cNode_lead_times.iteritems():
        cNode_name, euNode_name, fuel_type = cNode_euNode_fuel.split(':')
        sim_cal.add_event(sim_cNodes[cNode_name], 'calculate_lead_time',
                          [scn, euNode_name, fuel_type],
                          (scn.warm_up - scn.cNode_lead_times[cNode_euNode_fuel]) * 24.0)

    # Place all type beta tankers on the parent node availableQ
    for tanker in sim_tankers:
        sim_pNode.available_queue[tanker.name] = tanker
        tanker.status_trace.append(
            (sim_time.time, 'on', 'available_Q', sim_pNode.name, np.nan))  # SIM_STATS

    return sim_cal, sim_time


# noinspection PyPep8Naming
def end_of_simulation_replication_updates(scn, sim_cal, sim_time, sim_pNode, sim_cNodes):
    """Add status updates at the very end of the simulation replication
    to those traces that need it (for analysis purposes).
    @type scn: scenario.Scenario
    @type sim_cal: sim_engine.SimulationCalendar
    @type sim_time: sim_engine.SimulationTime
    @type sim_pNode: parent_node.ParentNode
    @type sim_cNodes: dict[str, child_node.ChildNode]
    """
    sim_time.time = scn.stop_time

    # Update all FSPs so that the tanker on_board amounts are up to date
    sim_pNode.daily_FSP_update(scn, sim_cal, sim_time)

    for cNode in sim_cNodes.itervalues():
        cNode.daily_FSP_update(scn, sim_cal, sim_time)

    # Update the on_hand inventory and queue status at all FSPs
    for FSP in sim_pNode.FSPs.values():
        FSP.update_loadingQ_trace(sim_time, 0)  # SIM_STATS
        FSP.update_unloadingQ_trace(sim_time, 0)  # SIM_STATS

    for cNode in sim_cNodes.values():
        for FSP in cNode.FSPs.values():
            FSP.update_loadingQ_trace(sim_time, 0)  # SIM_STATS
            FSP.update_unloadingQ_trace(sim_time, 0)  # SIM_STATS


def minimal_policy_copy(pol):
    """Copies over ONLY the information that is used by subsequent policies
    in order to minimize the overhead when using the parallel simulation
    replication runs.
    @type pol: policy.Policy
    @rtype: policy.Policy
    """
    minimal_pol = policy.Policy(pol.idx, '', pol.adj_factor, pol.lineage, pol.damping_factor, pol.history)
    minimal_pol.cNode_PNS = pol.cNode_PNS
    minimal_pol.cNode_PSSO = pol.cNode_PSSO
    minimal_pol.node_nzQL = pol.node_nzQL
    minimal_pol.score = pol.score

    return minimal_pol


def shallow_policy_copy(pol):
    """Copies over the important statistics from the policy instance
    passed in, but does not copy over the (LARGE) amounts of raw
    simulation data.
    @type pol: policy.Policy
    @rtype: policy.Policy
    """
    shallow_pol = policy.Policy(pol.idx, '', pol.adj_factor, pol.lineage, pol.damping_factor, pol.history)
    shallow_pol.cNode_PNS = pol.cNode_PNS
    shallow_pol.cNode_PSSO = pol.cNode_PSSO
    shallow_pol.node_nzQL = pol.node_nzQL
    shallow_pol.cNode_LT = pol.cNode_LT
    shallow_pol.pNode_PD = pol.pNode_PD
    shallow_pol.cNode_AD = pol.cNode_AD
    shallow_pol.score = pol.score

    return shallow_pol


if __name__ == "__main__":
    main()
