import json
import math
import numpy as np
import os
import re
import shutil
import stats
import time
import yagmail


def bounded_integer_seed(random_stream):
    """Use the random_stream passed in to compute an integer seed that
    is bounded above.
    @type random_stream:
    @rtype: int
    """
    return random_stream.randint(10000000)


def create_clean_output_directory(output_path):
    """Attempts to create the output_path directory, raises an error
    if the error message is anything other than the directory already
    existing.  Cleans the directory if it already exists.
    @type output_path: str
    """
    create_dir(output_path)  # Create the directory if it does not already exist

    # Clean the directory if it is not already empty
    for output in os.listdir(output_path):
        if os.path.isdir(output_path + '/' + output):
            shutil.rmtree(output_path + '/' + output)
        elif os.path.isfile(output_path + '/' + output):
            os.remove(output_path + '/' + output)
        else:
            assert False, 'Output in folder is neither a file nor a folder'


def copy_code(output_path):
    """Copies over the sport code into a folder named 'the_code'
    which is located in the output_path.
    @type output_path: str
    """
    os.makedirs(output_path + '/the_code')
    for file_name in os.listdir('./'):
        if file_name.endswith('.py'):
            shutil.copy('./' + file_name, output_path + '/the_code')


def create_replication_dir(pol_path, rep_path):
    """Create a replication directory.
    @type pol_path: str
    @type rep_path: str
    """
    if not os.path.isdir(pol_path):
        create_dir(pol_path)

    os.makedirs(rep_path)


def create_replication_sub_dirs(pol_path, rep_path, list_of_dirs):
    """Create the replication specific directories for this simulation
    replication inside the current replication directory.
    @type pol_path: str
    @type rep_path: str
    @type list_of_dirs: list[str]
    """
    if not os.path.isdir(rep_path):
        create_replication_dir(pol_path, rep_path)

    for directory in list_of_dirs:
        os.makedirs(rep_path + directory)


def create_dir(dir_path):
    """Create the directory at dir_path if it does not exist.
    @type dir_path: str
    """
    try:
        os.makedirs(dir_path)
    except OSError:
        if not os.path.isdir(dir_path):
            raise


# noinspection PyPep8Naming
def create_SPORT_directories(theater_name):
    """Creates all directories that are absolutely needed for
    the simulation.
    @type theater_name: str
    @rtype: None
    """
    create_clean_output_directory('./sim_output/' + theater_name)
    create_dir('./sim_output/' + theater_name + '/empirical_dist')
    create_dir('./sim_output/' + theater_name + '/historical_dist')
    create_dir('./sim_output/' + theater_name + '/lead_time')


def get_subnetwork_details(subnetwork_name):
    """Returns the index of the subnetwork and a list
    containing and ordered list of the names of the nodes in
    the subnetwork.  The first name is the parent node, while
    the names that follow are of the child nodes.
    @type subnetwork_name: str
    @rtype subnetwork_idx, names: int, list[str]
    """
    subnetwork_idx, node_names_str = subnetwork_name.split('_')
    names = re.findall('[A-Z][^A-Z]*', node_names_str)

    return subnetwork_idx, names


def send_email_alert(scn_file_name):
    """Sends an email about the completion of the simulation runs.
    @type scn_file_name: str
    """
    to = ['bjlobo@gmail.com', 'bjlobo@virginia.edu']
    subject = '%s Sim. Runs\r\n' % scn_file_name
    body = 'All simulation runs using %s completed.' % scn_file_name

    yag = yagmail.SMTP('loboalerts')
    yag.send(to, subject, body)


def save_scenario_data(scn, data_folder, data_file_prefix=''):
    """Updates the scn.scenario_data dictionary with the current values
    of the Scenario attributes (which may have been changed since they were
    loaded from the scenario_data dictionary via scn.load_scenario_from_file()),
    and then saves the dictionary to the specified data_folder + data_file_prefix
    location.
    @type scn: scenario.Scenario
    @type data_folder: str
    @type data_file_prefix: str
    """
    excluded_params = ['scenario_data', 'theater_name', 'subnetwork_name', 'stop_time', 'output_path',
                       'cNode_fuel_info', 'fuel_cNode_info', 'pNode_fuel_info', 'cNode_fuel_dist',
                       'num_cNode_fuel_type_combos']
    sim_params = {}
    for param in scn.__dict__.iterkeys():
        if not (param in excluded_params):
            sim_params[param] = getattr(scn, param)
    scn.scenario_data['SIM_PARAMS'] = sim_params

    if data_file_prefix:
        scenario_data_file_path = './' + data_folder + '/' + data_file_prefix + '_scn.tfm'
    else:
        scenario_data_file_path = './' + data_folder + '/' + time.strftime("%Y_%m_%d_%H_%M") + '_scn.tfm'

    with open(scenario_data_file_path, 'w') as outfile:
        json.dump(scn.scenario_data, outfile)


def save_policy_scores(policy_history, data_folder, data_file_prefix=''):
    """
    @type policy_history: list[policy.Policy]
    @type data_folder: str
    @type data_file_prefix: str
    """
    policy_scores = {}
    for policy_num, the_policy in enumerate(policy_history):
        policy_scores[policy_num] = the_policy.score

    if data_file_prefix:
        file_path = './' + data_folder + '/' + data_file_prefix + '_scores.tfmd'
    else:
        file_path = './' + data_folder + '/' + time.strftime("%Y_%m_%d_%H_%M") + '_scores.tfmd'
    with open(file_path, 'w') as outfile:
        json.dump(policy_scores, outfile)


def write_supply_chain_analysis(scn):
    """
    @type scn: scenario.Scenario
    """
    with open(scn.output_path + '/tanker_days_needed.txt', 'w') as wrtr:
        for fuel_type in scn.pNode_fuel_info:
            wrtr.write('Fuel Type: %s\n' % fuel_type)
            wrtr.write('Tanker-Days available: %s\n' % compute_tanker_days_available(scn, fuel_type))
            wrtr.write('Average Tanker-Days needed: %s\n\n' % compute_avg_tanker_days_needed(scn, fuel_type))


# noinspection PyUnboundLocalVariable
def compute_tanker_days_available(scn, fuel_type):
    """Returns, for the specified fuel type, the number of tanker-days available,
    a measure of the capacity of the supply chain.
    @type scn: scenario.Scenario
    @type fuel_type: str
    """
    # TODO Generalize to handle different tanker types
    assert len(scn.scenario_data['SUPPLY_CHAIN']['TANKER_TYPES'][fuel_type]) == 1, \
        'Too many %s tanker types' % fuel_type

    time_horizon_in_days = scn.stop_time / 24.0
    for tanker_type in scn.scenario_data['SUPPLY_CHAIN']['TANKER_TYPES'][fuel_type]:
        if tanker_type['fuel_type'] == fuel_type:
            num_tankers = int(tanker_type['num_tankers'])

    return num_tankers * time_horizon_in_days


# noinspection PyUnboundLocalVariable,PyPep8Naming
def compute_avg_tanker_days_needed(scn, fuel_type):
    """Returns, for the specified fuel type, the AVERAGE number of tanker-days
    required to satisfy the demand of the supply chain.  Note that this is a estimate
    and even if the number of available tankers days is greater than the average number
    needed, there may still be stock outs.
    @type scn: scenario.Scenario
    @type fuel_type: str
    """
    avg_tanker_days_needed = 0.0

    # TODO Generalize to handle different tanker types
    assert len(scn.scenario_data['SUPPLY_CHAIN']['TANKER_TYPES'][fuel_type]) == 1, \
        'Too many %s tanker types' % fuel_type

    time_horizon_in_days = scn.stop_time / 24.0
    for tanker_type in scn.scenario_data['SUPPLY_CHAIN']['TANKER_TYPES'][fuel_type]:
        tanker_capacity = float(tanker_type['capacity'])

    for cNode in scn.scenario_data['SUPPLY_CHAIN']['CHILD_NODES']:
        cNode_name = cNode['node_name']
        for FSP in cNode['FSPs']:
            if FSP['fuel_type'] == fuel_type:
                FSP_capacity = FSP['maximum']
                FSP_minimum = FSP['minimum']
                on_hand = FSP['on_hand']
                if cNode['node_type'] == 'intermediate':
                    max_avg_DCR = max([np.mean(data) for data in
                                       scn.cNode_fuel_dist[(cNode_name, fuel_type)].itervalues()])
                else:
                    max_avg_DCR = max([stats.mean(dist_info) for dist_info in
                                       scn.cNode_fuel_dist[(cNode_name, fuel_type)].itervalues()])

                avg_horizon_consumption = time_horizon_in_days * max_avg_DCR
                avg_amount_needed = avg_horizon_consumption - (on_hand - FSP_minimum)

                min_convoys_needed = math.ceil(avg_amount_needed / float(FSP_capacity - FSP_minimum))
                num_tankers_per_convoy = math.ceil(float(FSP_capacity - FSP_minimum) / tanker_capacity)

                pNode_CTT_dist = scn.scenario_data['SUPPLY_CHAIN']['PARENT_NODE']['CTTs'][cNode_name]
                avg_days_per_convoy = 1 + stats.mean(cNode['CTT_dist']) + stats.mean(pNode_CTT_dist)

                avg_tanker_days_needed += min_convoys_needed * avg_days_per_convoy * num_tankers_per_convoy

    return avg_tanker_days_needed
