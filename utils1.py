import numpy as np
import random as random
import json
import torch
import warnings
import pandas as pd 
from itertools import product 

inference_stem_mapping = {
            0: {'Camera': 68.3, 'Radar': 329.8,'Lidar': 64.3, 
                'DualCameraFusion':68.3*2, 'CameraLidarFusion':68.3+64.3, 'RadarLidarFusion': 329.8+64.3}, 
            1:{'Camera': 1.761, 'Radar': 2.240,'Lidar': 1.440, 
               'DualCameraFusion':1.761*2, 'CameraLidarFusion':1.761+1.440, 'RadarLidarFusion': 2.240+1.440},
            2: {'Camera': 68.3, 'Radar': 329.8,'Lidar': 64.3, 
                'DualCameraFusion':68.3*2, 'CameraLidarFusion':68.3+64.3, 'RadarLidarFusion': 329.8+64.3}
        }
inference_mapping = {
        2:{'Camera': {'Branch18': 520.4 , 'Branch50': 1893.3, 'Branch101':2319},
            'Radar': {'Branch18': 633.8, 'Branch50': 3439, 'Branch101':10360},
            'Lidar': {'Branch18': 572.2, 'Branch50': 1070, 'Branch101': 2393},
            'DualCameraFusion': {'Branch18': 2077, 'Branch50': 13012, 'Branch101': 21145},
            'CameraLidarFusion': {'Branch18': 23219, 'Branch50': 107798, 'Branch101': 101136},
            'RadarLidarFusion': {'Branch18': 1910, 'Branch50': 11063, 'Branch101': 31824}}
        , # branch only on device
        1:{ # branch only on edge 
            'Camera': {'Branch18': 16.14, 'Branch50': 32.79, 'Branch101': 44.10},
            'Radar': {'Branch18': 17.72, 'Branch50': 35.73, 'Branch101': 47.53},
            'Lidar': {'Branch18': 16.19, 'Branch50': 33.42, 'Branch101': 43.04},
            'DualCameraFusion': {'Branch18': 18.45, 'Branch50': 66.01, 'Branch101': 66.97},
            'CameraLidarFusion': {'Branch18': 19.43, 'Branch50': 65.41, 'Branch101': 66.63},
            'RadarLidarFusion': {'Branch18': 23.46, 'Branch50': 82.30, 'Branch101': 103.7}},
        0:{ # branch only on edge
            'Camera': {'Branch18': 16.14, 'Branch50': 32.79, 'Branch101': 44.10},
            'Radar': {'Branch18': 17.72, 'Branch50': 35.73, 'Branch101': 47.53},
            'Lidar': {'Branch18': 16.19, 'Branch50': 33.42, 'Branch101': 43.04},
            'DualCameraFusion': {'Branch18': 18.45, 'Branch50': 66.01, 'Branch101': 66.97},
            'CameraLidarFusion': {'Branch18': 19.43, 'Branch50': 65.41, 'Branch101': 66.63},
            'RadarLidarFusion': {'Branch18': 23.46, 'Branch50': 82.30, 'Branch101': 103.7}}
        }
    
def decode_action(ai):
    """Decode a single set of actions from network outputs."""
    actions = (ai // 9, (ai % 9) // 3, ai % 3)
    return actions

def calculate_latency(current_channel_states, client_actions, n_rb, bandwidth = 40e6, SNR = 20):

    # dr_rb1 = [0.15*180, 0.23*180, 0.38*180, 0.6*180, 0.88*180, 1.2*180, 1.4*180, 1.9*180, 2.4*180, 2.7*180, 3.3*180, 3.9*180, 4.5*180, 5.1*180, 5.5*180] # kbps
    # dr_rb = []
    # for d in dr_rb1:
    #     d = d/100
    #     dr_rb.append(d)
    # dr_rb = [0.08*180, 0.14*180, 0.22*180, 0.35*180, 0.55*180, 0.75*180, 0.95*180, 
    #  1.2*180, 1.5*180, 1.8*180, 2.2*180, 2.6*180, 3.0*180, 3.4*180, 3.8*180]

    actions = client_actions 
    # data_rate = current_channel_states # change for list of data rate
    # print('current_channel_states = ', current_channel_states)
    # data_rate = dr_rb[current_channel_states-1]*n_rb
    data_rate = current_channel_states*n_rb
    if isinstance(data_rate, torch.Tensor):
        data_rate = data_rate.item()
    
    data_modality = interpret_data_modality(actions[0])
    model_complexity = interpret_model_complexity(actions[1])
    deployment_strategy = actions[2]

    #calcualte latency, look up input size:
    if deployment_strategy == 0: # edge only
        if data_modality == 'Radar':
            input_size = 1*3*1152*1152*8
        elif data_modality == 'Lidar':
            input_size = 1*3*672*376*8 + 7.92e+6
        elif data_modality == 'Camera':
            input_size = 1*3*672*376*8 # 174k bytes
        elif data_modality == 'DualCameraFusion' or 'CameraLidarFusion':
            input_size = 1*3*672*376*8*2
        elif data_modality == 'RadarLidarFusion':
            input_size = 1*3*1152*1152*8 + 1*3*672*376*8
        else:
            raise Exception('wrong data modaltity of latency')
    elif deployment_strategy == 1: # device + edge
        if data_modality == 'Radar':
            input_size = 64*288*288*8
        elif data_modality == 'Lidar' or 'Camera':
            input_size = 1*3*168*94*8
        elif data_modality == 'DualCameraFusion' or 'CameraLidarFusion':
            input_size = 1*3*168*94*8*2
        elif data_modality == 'RadarLidarFusion':
            input_size = 64*288*288*8+1*3*168*94*8
        else:
            raise Exception('wrong data modaltity of latency')
    else: # device = 2
        input_size = 0 
        data_rate = 1

    # if data_rate == 0:
    #     print('current_channel_states', current_channel_states)
    #     print('client_actions', client_actions) 
    #     print('n_rb', n_rb)
    if data_rate == 0:
        channel_latency,total_latency, data_rate = np.inf, np.inf, 0
        return channel_latency,total_latency, data_rate, input_size
    channel_latency = input_size/data_rate/10e3  # data rate is mbps
    stem_inference_time = inference_stem_mapping.get(deployment_strategy).get(data_modality, {})
    
    branch_inference_time = inference_mapping.get(deployment_strategy).get(data_modality, {}).get(model_complexity, "Unknown Latency")
    total_latency = channel_latency + stem_inference_time + branch_inference_time

    return channel_latency,total_latency, data_rate, input_size

def calculate_energy_consumption(actions):
    # Logic to calculate energy consumption
    energy_inference_mapping = {            
            'Camera': {'Branch18': 0.0198, 'Branch50': 0.0775 , 'Branch101': 0.168},
            'Radar': {'Branch18':  0.106, 'Branch50': 0.321 , 'Branch101': 0.522},
            'Lidar': {'Branch18': 0.0209, 'Branch50': 0.0811, 'Branch101': 0.0783},
            'DualCameraFusion': {'Branch18': 0.246, 'Branch50': 0.120, 'Branch101': 1.361},
            'CameraLidarFusion': {'Branch18': 0.261, 'Branch50': 0.987, 'Branch101': 1.421},
            'RadarLidarFusion': {'Branch18': 0.534, 'Branch50': 2.005, 'Branch101': 3.125}
    }
    energy_inference_stem_mapping = {
            0: {'Camera':0.003, 'Radar': 0.0169,'Lidar': 0.003, 
                'DualCameraFusion':0.003*2, 'CameraLidarFusion':0.003*2, 'RadarLidarFusion': 0.0169+0.003}, # 140/153.4/6*11 = 1.6
            # 1:{'Camera':0.003*1.6, 'Radar': 0.0169*1.6,'Lidar': 0.003*1.6,
            #    'DualCameraFusion':0.003*2*1.6, 'CameraLidarFusion':(0.003*2)*1.6, 'RadarLidarFusion': (0.0169+0.003)*1.6}
            1:{'Camera':0.0, 'Radar': 0.0,'Lidar': 0.0,
            'DualCameraFusion':0.0, 'CameraLidarFusion':0.0, 'RadarLidarFusion': 0.0},
            2: {'Camera':0.03, 'Radar': 0.169,'Lidar': 0.03, 
                'DualCameraFusion':0.03*2, 'CameraLidarFusion':0.03*2, 'RadarLidarFusion': 0.169+0.03}
        }
        # Extract the deployment strategy action
        # energy consumption = communication energy + inference energy

    data_modality = interpret_data_modality(actions[0])
    deployment_strategy = actions[2]
    # inference_energy = self.energy_inference_mapping.get(data_modality, {}).get(model_complexity, "Unknown energy")
    # comm_energy = self.energy_comm_mapping.get(deployment_strategy, {}).get(data_modality, {})
    stem_energy = energy_inference_stem_mapping.get(deployment_strategy, {}).get(data_modality, {})
    energy_consumption = stem_energy # + comm_energy inference_energy  +
    return energy_consumption

def interpret_data_modality(action_component):
    modality_map = {
            0: 'Camera',
            1: 'Radar',
            2: 'Lidar',
            3: 'DualCameraFusion',
            4: 'CameraLidarFusion',
            5: 'RadarLidarFusion'
        }
    # Return the corresponding modality string
    return modality_map.get(action_component, 'Unknown')

def interpret_model_complexity(action_component):

    # Define a mapping for model complexities
    complexity_map = {
        0: 'Branch18',
        1: 'Branch50',
        2: 'Branch101'
    }
    # Return the corresponding model complexity string
    return complexity_map.get(action_component, 'UnknownComplexity')

def get_accuracy_mapping(context):
    if context == 'sunny':
        return {
            'Camera': {'Branch18': 59.8, 'Branch50': 68.3, 'Branch101': 69.7},
            'Radar': {'Branch18': 58.1, 'Branch50': 65.1, 'Branch101': 66.2},
            'Lidar': {'Branch18': 57.2, 'Branch50': 67.4, 'Branch101': 67.3},
            'DualCameraFusion': {'Branch18': 60.1, 'Branch50': 74.3, 'Branch101': 80.5},
            'CameraLidarFusion': {'Branch18': 56.4, 'Branch50': 63.0, 'Branch101': 68.2},
            'RadarLidarFusion': {'Branch18': 60.9, 'Branch50': 75.7, 'Branch101': 74.2}
        }
    elif context == 'rain':
        return {
            'Camera': {'Branch18': 19.0, 'Branch50': 21.2, 'Branch101': 22.8},
            'Radar': {'Branch18': 23.2, 'Branch50': 25.1, 'Branch101': 26.2},
            'Lidar': {'Branch18': 21.1, 'Branch50': 22.9, 'Branch101': 23.1},
            'DualCameraFusion': {'Branch18': 20.6, 'Branch50': 20.3, 'Branch101': 21.5},
            'RadarLidarFusion': {'Branch18': 24.7, 'Branch50': 25.1, 'Branch101': 26.8},
            'CameraLidarFusion': {'Branch18': 20.3, 'Branch50': 21.3, 'Branch101': 23.2}
        }
    elif context == 'motorway':
        return {
            'Camera': {'Branch18': 50.5, 'Branch50': 54.2, 'Branch101': 56.7},
            'Radar': {'Branch18': 38.2, 'Branch50': 42.5, 'Branch101': 43.4},
            'Lidar': {'Branch18': 42.8, 'Branch50': 53.9, 'Branch101': 53.6},
            'DualCameraFusion': {'Branch18': 50.7, 'Branch50': 57.9, 'Branch101': 60.5},
            'RadarLidarFusion': {'Branch18': 42.3, 'Branch50': 44.5, 'Branch101': 46.1},
            'CameraLidarFusion': {'Branch18': 57.2, 'Branch50': 62.3, 'Branch101': 63.1}
        }
    elif context == 'snow':
        return {
            'Camera': {'Branch18': 5.0, 'Branch50': 5.1, 'Branch101': 4.9},
            'Radar': {'Branch18': 7.1, 'Branch50': 11.8, 'Branch101': 10.2},
            'Lidar': {'Branch18': 6.5, 'Branch50': 7.4, 'Branch101': 9.4},
            'DualCameraFusion': {'Branch18': 5.8, 'Branch50': 5.3, 'Branch101': 5.5},
            'RadarLidarFusion': {'Branch18': 8.1, 'Branch50': 9.1, 'Branch101': 9.8},
            'CameraLidarFusion': {'Branch18': 6.4, 'Branch50': 6.9, 'Branch101': 7.2}
        }
    elif context == 'fog':
        return {
            'Camera': {'Branch18': 35.3, 'Branch50': 34.4, 'Branch101': 36.2},
            'Radar': {'Branch18': 44.4, 'Branch50': 47.2, 'Branch101': 47.9},
            'Lidar': {'Branch18': 42.8, 'Branch50': 44.1, 'Branch101': 45.4},
            'DualCameraFusion': {'Branch18': 34.3, 'Branch50': 36.2, 'Branch101': 37.3},
            'RadarLidarFusion': {'Branch18': 42.3, 'Branch50': 48.1, 'Branch101': 48.3},
            'CameraLidarFusion': {'Branch18': 37.2, 'Branch50': 42.2, 'Branch101': 43.0}
        }
    elif context == 'night':
        return {
            'Camera': {'Branch18': 32.5, 'Branch50': 44.5, 'Branch101': 46.9},
            'Radar': {'Branch18': 49.0, 'Branch50': 52.2, 'Branch101': 56.4},
            'Lidar': {'Branch18': 51.3, 'Branch50': 53.0, 'Branch101': 53.2},
            'DualCameraFusion': {'Branch18': 30.6, 'Branch50': 32.9, 'Branch101': 36.5},
            'RadarLidarFusion': {'Branch18': 51.8, 'Branch50': 53.4, 'Branch101': 60.2},
            'CameraLidarFusion': {'Branch18': 49.1, 'Branch50': 52.3, 'Branch101': 43.2}
        }
    else:
        return None

def get_accuracy(action, context):
    accuracy_mapping = get_accuracy_mapping(context)
    if accuracy_mapping is None:
        return "Context not recognized"
    
    data_modality = interpret_data_modality(action[0])
    model_complexity = interpret_model_complexity(action[1])
    
    accuracy = accuracy_mapping.get(data_modality, {}).get(model_complexity, 'unknown')
    if accuracy == 'unknown':
        return 'accuracy not recognized'
    return accuracy



def calculate_jains_index(simulation_data, metric='success'):
    """
    Compute Jain's fairness index for a specified metric across each simulation run.

    Args:
        simulation_data (list of dicts): Each entry contains 'R_total', 'L_max', 'method', and the metric.
        metric (str): Column name in simulation_data to compute fairness on.

    Returns:
        pd.DataFrame: Jain's index for each (R_total, L_max, method) group.
    """
    df = pd.DataFrame(simulation_data)
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(['R_total', 'L_max', 'method'])
    jain_results = []

    for (r_total, l_max, method), group in grouped:
        metrics = group[metric].values
        n = len(metrics)
        sum_metrics = metrics.sum()
        sum_squares = (metrics ** 2).sum()

        ji = (sum_metrics ** 2) / (n * sum_squares) if sum_squares != 0 else 0.0

        jain_results.append({
            'R_total': r_total,
            'L_max': l_max,
            'method': method,
            'jains_index': ji
        })

    return pd.DataFrame(jain_results)




# def calculate_reward(client_acc, client_lat,energy_consumed, accuracy_constraint, latency_constraint ):
#     # Extract relevant information from the state
#     MAX_ENERGY_CONSUMPTION = 0.02

#     accuracy_met_count = 0
#     latency_met_count = 0

#     print('client_acc = ', client_acc)
#     accuracy_met = client_acc >= accuracy_constraint
#     latency_met = client_lat  <= latency_constraint
#     # Initialize reward components
#     if accuracy_met:
#         accuracy_met_count += 1
#         accuracy_component = 0.5
#     else:
#         accuracy_component = -0.5

#     if latency_met:
#         latency_met_count += 1
#         latency_component = 0.5 
#     else:
#         latency_component = -0.5

#     # Calculate the base psi value without considering energy yet
#     psi = accuracy_component + latency_component

#     # Adjust the psi value based on specific scenarios:
#     # If accuracy is not met, prioritize minimizing latency and energy
#     if not accuracy_met:
#         if latency_met:
#             psi -= 0.25  # Reduce penalty slightly if latency is met
#         else:
#             psi -= 0.5  # Apply stronger penalty if both accuracy and latency are not met

#     # If latency is not met, prioritize minimizing accuracy and energy
#     if not latency_met:
#         if accuracy_met:
#             psi -= 0.25  # Reduce penalty slightly if accuracy is met
#         else:
#             psi -= 0.5  # Apply stronger penalty if both are not met

#     # Compute the final reward, considering energy efficiency
#     if psi > 0:
#         reward = psi / (1 + energy_consumed)
#     else:
#         reward = psi*(1+energy_consumed/MAX_ENERGY_CONSUMPTION) # Using energy penalty to adjust the reward

#     return accuracy_met, latency_met, psi, reward

def sigmoid(x, k, x0):
    """Sigmoid function with steepness k and midpoint x0"""
    if x == np.inf:
        return -1
    x = np.clip((x - x0), -30000, 30000)
    return 1 / (1 + np.exp(-k*(x - x0)))

def calculate_bid(sigma_A, sigma_L, energy, violation_penalty, R_hat, params):
    # Benefit-to-cost ratio with numerical stability
    benefit_cost_ratio = (params['w1'] * sigma_A + params['w2'] * sigma_L) - params['w3']*energy
    
    # Violation dampener: Penalize violations by forcing higher bids through λ(T)
    # (Server adjusts λ_i(T) externally; client observes past violations)
    lambda_i = params['lambda_0'] * (1 + violation_penalty)
    
    # Bid formula with implicit penalty scaling
    bid = benefit_cost_ratio * R_hat / (1 + lambda_i)

    return bid

# def client_utility(R_hat, A_i, L_i, E_i, b_i, params):
#     """
#     Updated utility function matching theoretical formulation
#     """
#     # Sigmoid transformations
#     sigma_A = sigmoid(A_i, params['k_A'], params['A_min'])
#     sigma_L = sigmoid(L_i, -params['k_L'], params['L_max'])  # Negative k for decreasing sigmoid
    
#     # Constraint indicator
#     constraint_violation = (A_i < params['A_min']) or (L_i > params['L_max'])
    
#     # Utility components
#     accuracy_term = params['w1'] * sigma_A
#     latency_term = params['w2'] * sigma_L
#     energy_penalty = params['w3'] * E_i
#     constraint_penalty = params['w4'] * constraint_violation
#     b_i = calculate_bid(sigma_A, sigma_L, E_i, violation_penalty, R_hat, params) 

#     bidding_cost = params['w5'] * b_i
#     # Overbidding penalty
#     if R_hat == 0:
#         overbidding_penalty = 0
#     else:
#         overbidding_penalty = params['w6'] * max(0, (b_i - R_hat) / R_hat)
#         utility -= overbidding_penalty
#     return accuracy_term + latency_term - energy_penalty - constraint_penalty - bidding_cost - overbidding_penalty




def augmented_lagrangian(R, lambdas, mus, rho, params, clients_data):
    """
    Enhanced ALM implementation for resource allocation optimization
    """
    total_utility = 0
    penalty = 0
    
    for i in range(len(R)):
        # Get client-specific parameters
        client_params = clients_data[i]
        A_i = client_params['achieved_accuracy']
        L_i = client_params['latency']
        E_i = client_params['energy']
        b_i = client_params['bid']
        
        # Calculate utility
        U_i = client_utility(R[i], A_i, L_i, E_i, b_i, params)
        total_utility += U_i
        # Inequality constraints (non-linear penalty)
        g1 = params['A_min'] - A_i
        g2 = L_i - params['L_max']
        
        penalty += mus[0] * max(0, g1) + mus[1] * max(0, g2)
        penalty += (rho/2) * (max(0, g1)**2 + max(0, g2)**2)
    
    # Maximize utility = minimize negative utility
    return -total_utility + penalty

# def bidding(channel_states, context, total_rb, 
#             latency_constraint, accuracy_constraint, n_clients):
    
#     estimate_rb = np.floor(total_rb/n_clients)
#     for action_id in range(6*3*3):
#         action = decode_action(action_id)
#         channel_latency, total_latency, data_rate = calculate_latency(channel_states, action, estimate_rb, bandwidth = 40e6, SNR = 20)
#         acc = get_accuracy(action, context)
#         energy = calculate_energy_consumption(action)

#     # Calculate individual components
#     lat_term = np.clip(-k_L(total_latency - latency_constraint), -100, 100)
#     acc_term = np.clip(-k_A(acc - accuracy_constraint), -100, 100)
#     latency_component = w_L / (1 + lat_term)
#     accuracy_component = w_A / (1 + acc_term)
#     if energy == 0:
#         energy_component = 0
#     else:
#         energy_component = w_E / energy
#     # Calculate the final bid
#     bid = latency_component + accuracy_component - energy_component
#     return bid, latency_component, accuracy_component, energy_component

def calculate_resource_estimate(client_state, params):
    """
    Enhanced resource estimation considering constraint-aware action selection.
    Returns both R_hat and action analysis for bidding.
    """
    action_data = []
    dr_rb = [0.15*180, 0.23*180, 0.38*180, 0.6*180, 0.88*180, 1.2*180, 1.4*180, 
            1.9*180, 2.4*180, 2.7*180, 3.3*180, 3.9*180, 4.5*180, 5.1*180, 5.5*180] # kbps
    # Analyze all possible actions
    
    for action_id in range(6*3*3):
        action = decode_action(action_id)
        # Calculate action-specific metrics
        channel_latency, total_latency, data_rate = calculate_latency(
            client_state['channel_quality'],
            action,
            n_rb=1
        )
        accuracy = get_accuracy(action, client_state['context'])
        energy = calculate_energy_consumption(action)
        
        # Ensure accuracy is a float or integer

        accuracy = float(accuracy)
        if action[2] == 2: # on device
            base_rb = 0
        else:
            # Calculate base resource need (without constraints)
            input_size = channel_latency/data_rate
            inference_lat = total_latency - channel_latency
            if (params['L_max'] - inference_lat) < 0:
                base_rb = 0
            else: 
                base_rb = np.ceil((params['L_max'] - inference_lat)/input_size/dr_rb[client_state['channel_quality']-1])
                updated_channel_lat, total_latency, _ = calculate_latency(
                    client_state['channel_quality'],
                    action,
                    base_rb)

        action_data.append({
            'action_id': action_id,
            'latency': total_latency,
            'accuracy': accuracy,
            'energy': energy,
            'base_rb': np.ceil(base_rb),
            'is_feasible': (
                accuracy >= params['A_min'] and 
                total_latency <= params['L_max']
            )
        })

    # Analyze actions using the utility function
    sorted_actions = sorted(
        action_data,
        key=lambda x: client_utility(
            R_i=x['base_rb'],
            A_i=x['accuracy'],
            L_i=x['latency'],
            E_i=x['energy'],
            b_i =  0, # Placeholder for bid, used in utility calculation
            params=params
        ),
        reverse=True  # Higher utility first
    )
    # Select top 2 candidate actions
    best_action = sorted_actions[0]
    second_best = sorted_actions[1]
    
    # Calculate adaptive resource estimate
    if best_action['is_feasible']:
        # Use best feasible action with 10% margin
        R_hat = best_action['base_rb'] * 1.1
    else:
        # Blend best and second best with penalty
        R_hat = 0.7*best_action['base_rb'] + 0.3*second_best['base_rb']
        R_hat *= 1.5  # Penalty for no feasible solutions
    
    return R_hat, {
        'best_action': best_action,
        'second_best': second_best,
        'all_actions': action_data
    }

def enhanced_bidding(client_state, params):
    """Bidding function with integrated resource estimation"""
    bids = []
    action_analysis = []
    # Get resource estimates and action data
    R_hat, analysis = calculate_resource_estimate(client_state, params)
    action_analysis.append(analysis)
    max_bid_value = 0
    
    # Calculate bids for all actions
    for action in analysis['all_actions']:
        R_hat = action['base_rb']
        if action['is_feasible']:
        # Use best feasible action with 10% margin
            R_hat = action['base_rb'] * 1.1
        else:
            R_hat *= 1.5  # Penalty for no feasible solutions
        try:
            sigma_A = 1 / (1 + np.exp(-params['k_A']*(
                action['accuracy'] - params['A_min'])
            ))
            sigma_L = 1 / (1 + np.exp(-params['k_L']*(
                params['L_max'] - action['latency'])
            ))
        except RuntimeWarning as e:
            print(f"Warning: {e}")
            print(f"Action: {action}")
            print(f"Params: {params}")
            continue
        # Calculate violation penalty
            
        violation_penalty = params['w4'] * int(
            action['accuracy'] < params['A_min'] or 
            action['latency'] > params['L_max']
        )
        action_id = action['action_id']
        client_actions = decode_action(action_id)
        deployment = client_actions[2]
        
        if deployment == 2:
            bid_value = 0 
        else:
            try:
                # bid_value = (
                #     (params['w1']*sigma_A + params['w2']*sigma_L) / R_hat 
                #     - params['w3']*action['energy'] 
                #     - violation_penalty
                # )
                # print('bid_value = ', bid_value)
                bid_value = calculate_bid(sigma_A, sigma_L, action['energy'], violation_penalty, R_hat, params)
                # Raise an error if R_hat is 0
                # if R_hat == 0:
                #     print('client_actions', client_actions)
                #     print('action rb', action['base_rb'])
                #     print(f"params: {params}")
                #     print(f"R_hat: {R_hat}")
                #     print(f"Action: {action}")
                #     print(f"Sigma_A: {sigma_A}")
                #     print(f"Sigma_L: {sigma_L}")
                #     print(f"Violation Penalty: {violation_penalty}")
                #     print(f"Bid Value: {bid_value}")
                #     raise ValueError("R_hat is 0 f, channel condition")
                    
            except RuntimeWarning as e:
                print(f"Warning: {e}")
                print(f"R_hat: {R_hat}")
                print(f"Action: {action}")
                print(f"Sigma_A: {sigma_A}")
                print(f"Sigma_L: {sigma_L}")
                print(f"Violation Penalty: {violation_penalty}")
                print(f"Bid Value: {bid_value}")
                continue

        
        bids.append(bid_value)
        # print('bids', bids)
        if bid_value > max_bid_value:
            # print('bid_value = ', bid_value)
            max_bid_value = bid_value
            best_action = action
    # Return max bid with analysis
    # print('max_bid_value = ', max_bid_value)
    # print('best_action = ', best_action)
    return max_bid_value, action_analysis, best_action

def bidding_data():

    PARAMETERS = {
        'w_L': 10.0,
        'w_A': 5.0,
        'w_E': 0.01,
        'k_L': lambda x: .01 * x,  # Sigmoid scaling factor for latency
        'k_A': lambda x: .05 * x,  # Sigmoid scaling factor for accuracy
        'R_max': 100
    }
    # Define parameter ranges
    actions = range(0, 6*3*3)  
    channel_states = range(1, 16)  
    n_rbs = range(1, 100)  
    accuracy_constraints = [40, 50, 60]  # Example accuracy constraints
    latency_constraints = [2000, 3000, 4000]  # Example latency constraints
    contexts = ['sunny', 'rain', 'motorway', 'snow', 'fog', 'night']  # Example contexts

    # Generate all combinations
    results = []
    for channel_state, action, n_rb, accuracy_constraint, latency_constraint, context in product(
            channel_states, actions, n_rbs, accuracy_constraints, latency_constraints, contexts):
        
        client_state = {
            'channel_quality': channel_state,
            'context': context
        }
        params = {
            'A_min': accuracy_constraint,
            'L_max': latency_constraint,
            'w1': PARAMETERS['w_A'],
            'w2': PARAMETERS['w_L'],
            'w3': PARAMETERS['w_E'],
            'w4': 1.0,  # Example weight for constraint violation penalty
            'w5': 1.0,  # Example weight for bidding cost
            'k_A': 0.05,
            'k_L': 0.01
        }
        bid, action_analysis = enhanced_bidding(client_state, params)
        action = decode_action(action)
        results.append({
            'Channel_State': channel_state,
            'Action': action,
            'N_RB': n_rb,
            'Bid_Value': bid,
            'Action_Analysis': action_analysis,
            'Accuracy_Constraint': accuracy_constraint,
            'Latency_Constraint': latency_constraint,
            'Context': context
        })

    # Create DataFrame and save to Excel
    df = pd.DataFrame(results)
    df.to_excel('../results/bidding_analysis.xlsx', index=False)
    print("Analysis complete. Results saved to bidding_analysis.xlsx")

# bidding_data()