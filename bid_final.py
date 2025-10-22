import numpy as np
from math import floor
import pandas as pd 
import seaborn as sns
import json
import os 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize 
from utils1 import decode_action, sigmoid
from utils1 import  calculate_latency, get_accuracy, calculate_energy_consumption, get_accuracy_mapping
from channel_model import compute_data_rates
import uuid
from plots_final import plot_success_histograms, plot_jains_index, plot_utility_components, create_utility_summary_table, plot_metric_vs_constraints

MIN_BID_PER_BLOCK = 0.001

class Client:
    def __init__(self, params, data_rates_kbps, context):
        """
        Initialize a client with parameters:
        - w1, w2, w5: Utility weights.
        - lambda_0: Base penalty.
        - beta: Overbidding penalty scaling.
        - gamma: Violation penalty scaling.
        - R_hat: Predicted resource need.
        - b_max: Maximum allowed bid.
        """
        self.client_id = uuid.uuid4().hex[:8]  # Add unique identifier
        self.params = params
        self.w1 = params['w1']
        self.w2 = params['w2']
        self.w3 = params['w3']
        self.w4 = params['w4']

        self.lambda_0 = params['lambda_0']

        self.gamma = params['gamma'] # Bid cost coefficient 
        self.R_total = params['R_total']

        self.k_A = params['k_A']
        self.k_L = params['k_L']

        self.client_state = {
            'channel_quality': data_rates_kbps,
            'context': context
        }
        self.A_min = params['context_to_A_min'][context]
        self.L_max = params['L_max']
        self.models = self.get_models()  # Initialize model configurations

    def sigmoid(self, x, k, x0):
        return 1 / (1 + np.exp(-k*(x - x0)))

    def get_models(self):
        models = [] #list of tuples
        for action_id in range(6*3*3):  # Assuming 54 total actions
            action = decode_action(action_id)
            dr = self.client_state['channel_quality']
            channel_latency, total_latency, data_rate, input_size = calculate_latency(
            dr, action, n_rb=1)
            accuracy = float(get_accuracy(action, self.client_state['context']))
            energy = calculate_energy_consumption(action)
            def make_latency_fn(action, input_size, dr, channel_latency):
                def latency_fn(R):
                    if R == 0:
                        if action[2] != 2:  # Not on-device processing
                            return float('inf')
                        else:
                            return total_latency
                    
                    _, total_lat, _, _ = calculate_latency(dr, action, n_rb=R)
            
                    return total_lat
                return latency_fn
            # Create model tuple: (accuracy, latency_fn, energy_fn)
            model = (
                accuracy,               # Accuracy metric
                make_latency_fn(action, input_size, dr, channel_latency),  # Latency function
                energy,              # Energy function
                action
            )
            models.append(model)
        return models

    def get_bid_commitment(self):
        """Phase 1: Submit non-binding bid commitment assuming no competition"""
        # Calculate bid with B_{-i} = 0 (no other bidders)
        bid, R_min, best_bid_dict = self.calculate_bid(B_neg_i=0)
        # Return as non-binding commitment (can be scaled if needed)
        return bid, R_min, best_bid_dict

    def calculate_bid(self, B_neg_i):
        feasible_models = []

        for m in self.models:
            A_m, L_fn, E_m, action = m
            # Skip models that dont meet accuracy requirement
            if A_m < self.A_min:
                continue
            # Find minimal R needed to satisfy latency
            R_min = next((R for R in range(0, self.R_total+1) 
                        if L_fn(R) <= 0.9*self.L_max), None)
            
            if not R_min or R_min > self.R_total:
                continue  #
            # Enhanced bid formula with urgency factor
            urgency = (self.R_total - R_min) / self.R_total
            bid_min = (R_min ** 2) / (self.L_max + 1e-6) * urgency * self.params['num_clients'] / self.R_total
            # Utility with strong success incentive
            utility = (10 if R_min <= self.R_total else 0) - self.gamma * (bid_min ** 2)
            feasible_models.append((utility, bid_min, action, -E_m, R_min))
            # bid_min = (R_min * (B_neg_i + self.lambda_0)) / denominator
            # Core Bid Formula: Directly proportional to R_min
            # bid_min = (R_min / self.R_total) * (self.R_total - R_min)* self.params['num_clients'] / self.R_total/ self.L_max
            # Calculate utility (success reward - bid cost)
            # utility = 1 - self.gamma * (bid_min ** 2)

            # if utility > 0:  # Only consider worthwhile bids
            #     feasible_models.append((utility, bid_min, action, -E_m, R_min))

        if not feasible_models:
            # Fallback: return minimum bid with first model
            first_model = self.models[0]
            min_bid = 0 #(1/self.R_total) * self.params['num_clients'] / self.R_total
            bid_fallback = min_bid
            return bid_fallback, 1, {
                "best_bid": bid_fallback,
                "utility": 0,
                "required_R": 0,
                "optimal_action": first_model[3],
                "E_m": first_model[2]
            }
    
        # Select model with highest net utility
        # print('feasible_models ', feasible_models, action)
        # Prioritize highest utility then lowest R_min
        best_utility, best_bid, optimal_action, E_m, R_min = max(
        feasible_models, 
        key=lambda x: (x[0], -x[4]))

        best_bid_dict = {
            "best_bid": best_bid,
            "utility": best_utility,
            "required_R": R_min,
            "optimal_action": optimal_action,
            "E_m": -E_m
        }
        return best_bid, R_min, best_bid_dict

    def post_allocation_adaptation(self, R_allocated):
        best_model = None
        best_energy = float('inf')
        success = False
        
        for m in self.models:
            A_m, L_fn, E_m, action = m
            L = L_fn(R_allocated)
            # Check success criteria
            if A_m >= self.A_min and L <= self.L_max:
                success = True
                # Track actual usage (could be less than allocated)
                actual_used = next(R for R in range(R_allocated+1) 
                            if L_fn(R) <= self.L_max)
                used_allocation = min(R_allocated, actual_used)
                # Select most energy-efficient successful model
                if E_m < best_energy:
                    best_energy = E_m
                    best_model = m

        if not success:  # Fallback to first model if no success
            best_model = self.models[0]
            used_allocation = 0
            
        utility_dict = {
            "energy": best_model[2] if success else 0,
            "latency": L_fn(used_allocation),
            "accuracy": best_model[0],
            "action": best_model[3]
        }
        return best_model, int(success), success, utility_dict, used_allocation

class Server:
    def __init__(self, clients,  R_total, lambda_0, alpha = 0.1, beta = 0.01):
        """
        Initialize the server with:
        - R_total: Total resource blocks.
        - lambda_0: Base penalty.
        """
        self.clients = clients
        self.R_total = R_total
        self.lambda_0 = lambda_0
        self.alpha = alpha  # Penalty scaling factor
        self.beta = beta    # Fairness coefficient
        self.simulation_data = []  # Store per-client metrics

    def run_auction(self):
        # Collect bids using 
        bids = []
        R_mins = []
        for idx, client in enumerate(self.clients):
            bid, R_min, best_bid_dict = client.get_bid_commitment()
            bids.append(bid)
            R_mins.append(R_min)
        # Calculate proportional allocation with zero-bid protection
        total_bid = max(sum(bids), 1e-6)

        # Phase 1: Priority-based allocation
        allocations = [0] * len(self.clients)
        remaining_rbs = self.R_total
        
        # Create priority queue: (R_min, bid, index)
        priority_queue = sorted(
            [(r, bid, i) for i, (r, bid) in enumerate(zip(R_mins, bids)) if r is not None],
            key=lambda x: (x[0], -x[1])  # Smallest R_min first, then highest bid
        )
        # Allocate to most urgent needs first
        for r_min, _, idx in priority_queue:
            if remaining_rbs >= r_min:
                allocations[idx] = r_min
                remaining_rbs -= r_min
            else:
                break  # Can't satisfy more clients

        # Phase 2: Proportional allocation of remaining resources
        if remaining_rbs > 0:
            total_bid = max(sum(bids), 1e-6)
            for i in range(len(allocations)):
                if R_mins[i] is None or allocations[i] >= R_mins[i]:
                    additional = min(
                        int(bids[i]/total_bid * remaining_rbs),
                        remaining_rbs
                    )
                    allocations[i] += additional
                    remaining_rbs -= additional
                    if remaining_rbs == 0:
                        break


        successes = []
        utility_dicts = []
        utilities = []
        final_allocation = []
        for client, alloc in zip(self.clients, allocations):
            best_model, utility, success, utility_dict, used_allocation = client.post_allocation_adaptation(alloc)
            final_allocation.append(used_allocation)
            successes.append(success)
            A_m, L_fn, E_m, action = best_model
            utility_dicts.append(utility_dict)
            utilities.append(utility)
            # Store per-client data
            self.simulation_data.append({
                "context": client.client_state['context'],
                "channel_quality": client.client_state['channel_quality'],  # in Mbps
                "allocated_rb": used_allocation,
                "method": "bidding",  # Hardcode for now (will vary by loop)
                "success": success,
                "L_max": client.L_max,
                "R_total": self.R_total,

            })
        # After allocations are finalized
        unmet_clients = 0
        for alloc, R_min, client in zip(allocations, R_mins, self.clients):
            if R_min is not None and alloc < R_min:
                unmet_clients += 1
            elif R_min is None:
                unmet_clients += 1
            else: 
                continue
                #print(f"Client met: Alloc={alloc} >= R_min={R_min} (L_max={client.L_max})")
        return allocations, sum(successes)/len(successes), best_model, utilities, utility_dicts, bids, R_mins

def allocation_strategy(clients, param, total_rbs, allocation='average'):
    # Current channel states for each client (1 to 15)
    channel_qualities = []
    # distribution_type = ["poisson", "rayleigh", "exponential"]:
    for client in clients:
        current_channel_state = client.client_state['channel_quality']
        channel_qualities.append(current_channel_state)
    # Data rate per resource block (RB) for each channel state (in kbps)
    
    n_clients = param['num_clients']
    n_rbs = []
    if allocation == 'average':
        # Average allocation: divide RBs equally among clients
        # print('average')
        n_rb = int(total_rbs / n_clients)
        n_rbs = [n_rb] * n_clients

    elif allocation == 'prop fair':
        # Proportional fairness: allocate more RBs to clients with better channel conditions
        # Weighted by channel state
        # rb_allocations = np.ones(n_clients)
        # dr_rates = compute_data_rates(distribution_type, n_clients, rb_allocations, total_bandwidth=20e6, subcarrier_spacing=30e3)
            # Calculate weights inversely proportional to channel quality

        weights = channel_qualities / np.sum(channel_qualities)
        n_rbs = np.round(weights * total_rbs).astype(int)

        # Ensure every client gets at least 1 RB
        # n_rbs = np.maximum(n_rbs, 1)
        # Adjust total RBs if there's a mismatch due to rounding
        while np.sum(n_rbs) > total_rbs:
            # Reduce RBs for the client with the most RBs
            idx = np.argmax(n_rbs)
            n_rbs[idx] -= 1
        while np.sum(n_rbs) < total_rbs:
            # Add RBs to the client with the least RBs
            idx = np.argmin(n_rbs)
            n_rbs[idx] += 1
        # print('prop fair')
        # print('n_rbs', n_rbs)
    elif allocation == 'disprop fair':
        channel_qualities = [c.client_state['channel_quality'] for c in clients]
        # Calculate weights inversely proportional to channel quality
        weights = 1 / np.array(channel_qualities)
        weights /= np.sum(weights)  # Normalize
        # Allocate RBs based on weights
        n_rbs = np.round(weights * total_rbs).astype(int)
        # Adjust total RBs if there's a mismatch due to rounding
        while np.sum(n_rbs) > total_rbs:
            # Reduce RBs for the client with the most RBs
            idx = np.argmax(n_rbs)
            n_rbs[idx] -= 1
        while np.sum(n_rbs) < total_rbs:
            # Add RBs to the client with the least RBs
            idx = np.argmin(n_rbs)
            n_rbs[idx] += 1
        # print('disprop fair')
    else:
        raise NameError('No such allocation method')
    return n_rbs

def optimal_action(clients, params, total_rbs, allocation='average'):
    # params['B'] = total_rbs
    allocations = allocation_strategy(clients, params, total_rbs, allocation)
    success = 0 
    success_list = []
    fail = 0
    optimal_aciton = None  # Initialize with a default value
    utility_dict = None  # Initialize utility_dict to avoid similar issues

    for idx, client in enumerate(clients):
        best_utility = -np.inf
        for action_id in range(6*3*3):  # Assuming 54 total actions
            action = decode_action(action_id)
            data_rates_kbps = client.client_state['channel_quality']
            n_rb = allocations[idx]
            # Calculate base metrics
            channel_latency, total_latency, data_rate, input_size = calculate_latency(
            data_rates_kbps, action, n_rb)
            accuracy = float(get_accuracy(action, client.client_state['context']))
            # scaled_A_im = (accuracy - min_acc) / (max_acc - min_acc)*100 # percent to scale
            energy = calculate_energy_consumption(action)
            sigma_A = sigmoid(accuracy, params['k_A'], client.A_min)# params['A_min'])
            sigma_L = sigmoid(total_latency, -params['k_L'], params['L_max'])  # Negative k for decreasing sigmoid
            benefit = params['w1']*sigma_A + params['w2']*sigma_L 
            energy_penalty = params['w3'] * energy 
            constraint_violation = ((accuracy < client.A_min) or (total_latency > client.L_max))

            if not constraint_violation:
                success_list.append(1)
                success += 1
                utility = 1
                utility_dict = {'accuracy ': params['w1']*sigma_A,
                            'latency': params['w2']*sigma_L, 
                            'energy': energy_penalty,
                            'channel_latency': channel_latency,
                            'total_latency': total_latency, 
                            }
                optimal_aciton = action
                # print(f"OptimalAction:{action} Accuracy violation ({accuracy} > {client.A_min})")
                # print(f" Latency violation ({total_latency} < {client.L_max})")
                break
            
            constraint_penalty = params['w4'] * constraint_violation
            utility =  benefit - energy_penalty - constraint_penalty 

            if utility > best_utility:
                best_utility = utility
                best_action = action
                best_constraint_penalty = constraint_violation
                utility_dict = {'accuracy ': params['w1']*sigma_A,
                            'latency': params['w2']*sigma_L, 
                            'energy': energy_penalty,
                            'constraint_penalty': constraint_penalty,
                            'channeli_latency': channel_latency,
                            'total_latency': total_latency, 
                            }
                optimal_aciton = action
        success_list.append(0)

    success_rate = success/(len(clients))
    return success_rate, allocations, utility_dict, optimal_aciton, success_list

def load_simulation_data(filename):
    """Load simulation data from CSV"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file {filename} not found")
    return pd.read_csv(filename)

def find_rb_for_success_rate(df, context, method, target_success=0.6):
    """
    Find the minimum RBs needed to achieve `target_success` rate for a given context and method.
    """
    # Filter data for the context and method
    context_data = df[(df['context'] == context) & (df['method'] == method)]
    # Sort by allocated RBs
    context_data = context_data.sort_values('allocated_rb')
    
    # Calculate cumulative success rate
    context_data['cumulative_success'] = context_data['success'].expanding().mean()
    # Find the first RB allocation where success rate >= target
    target_row = context_data[context_data['cumulative_success'] >= target_success].iloc[0]
    
    return target_row['allocated_rb'], target_row['channel_quality']

def plot_context_rb_requirements(params):
    # Combine simulation data from all runs
    df = pd.DataFrame(server.simulation_data)
    allocation_methods = ['bidding', 'average', 'prop fair', 'disprop fair']
    # For each context and method, find required RBs at 60% success
    results = []
    for context in params['context_to_A_min'].keys():
        for method in allocation_methods:
            try:
                rb, channel_quality = find_rb_for_success_rate(df, context, method, 0.6)
                results.append({
                    "context": context,
                    "method": method,
                    "required_rb": rb,
                    "channel_quality": channel_quality
                })
            except IndexError:
                print(f"Warning: {method} cannot achieve 60% success for {context}")
    
    results_df = pd.DataFrame(results)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results_df,
        x="channel_quality",
        y="required_rb",
        hue="method",
        style="context",
        markers=True,
        dashes=False
    )
    plt.title("RBs Required to Achieve 60% Success Rate by Channel Quality and Context")
    plt.xlabel("Channel Quality (Mbps)")
    plt.ylabel("Required RBs")
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'plots/success_rate_comparison_n_{num_clients}_rbs_req.png', dpi=300, bbox_inches='tight')

def plot_simulation_results(df, distribution_type):
    """Plot bid distribution and resource allocation for all contexts in one plot, with a clean legend."""
    contexts = df['context'].unique()
    L_max_values = df['L_max'].unique()
    
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']  # Unique markers for each context
    colors = sns.color_palette("tab10", len(L_max_values))  # Different colors for each L_max
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f'Simulation Results - {distribution_type} Distribution', fontsize=14, fontweight='bold')

    # **1. Channel Quality vs Bid (Left Plot)**
    ax1 = axes[0]
    for i, context in enumerate(contexts):
        for j, l_max in enumerate(L_max_values):
            df_filtered = df[(df['context'] == context) & (df['L_max'] == l_max)]
            ax1.scatter(df_filtered['channel_quality']/1000, df_filtered['bid'], 
                        color=colors[j], marker=markers[i],
                        alpha=0.7, edgecolors='k')

    # ax1.set_xlabel('Channel Quality')
    ax1.set_xlabel('Data Rate (Mbps)')
    ax1.set_ylabel('Bid Value')
    ax1.set_title('Bid Distribution vs Channel Quality')
    ax1.grid(True, alpha=0.3)

    # **2. Allocation vs Bid (Right Plot)**
    ax2 = axes[1]
    for i, context in enumerate(contexts):
        for j, l_max in enumerate(L_max_values):
            df_filtered = df[(df['context'] == context) & (df['L_max'] == l_max)]
            ax2.scatter(df_filtered['bid'], df_filtered['allocation'], 
                        color=colors[j], marker=markers[i],
                        alpha=0.7, edgecolors='k')
    ax2.set_xlabel('Bid Value')
    ax2.set_ylabel('Allocated Resources')
    ax2.set_title('Resource Allocation vs Bid')
    ax2.grid(True, alpha=0.3)
    # **Create Single Legend with Separate Entries for L_max and Context**
    legend_handles = []
    
    # **Legend for L_max (Colors)**
    for j, l_max in enumerate(L_max_values):
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[j], markersize=10, label=f"L_max = {l_max}"))

    # **Legend for Context (Markers)**
    for i, context in enumerate(contexts):
        legend_handles.append(plt.Line2D([0], [0], marker=markers[i], color='black', markersize=8, linestyle='None', label=f"{context}"))

    fig.legend(handles=legend_handles, title="Legend", loc="lower right", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f'../results/bid_distribution/bid_vs_channel_condition_{distribution_type}.png'
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

# def plot_successful_rate(params, all_results):
#     """Plot success rates for all distribution types and allocation strategies in subplots"""
#     # Validate input
#     # Create figure with subplots
#     fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#     fig.suptitle('Success Rate Comparison Across Distributions and Allocation Strategies', 
#                 fontsize=16, y=1.02)
#     # Configure plot parameters
#     strategies = ['Bidding', 'Average', 'Proportional', 'Fairness']
#     markers = ['o', 's', '^', 'D']
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
#     handles, labels = None, None
#     Rbs = params['R_total']
#     num_clients = params['num_clients']
#     for idx, (distribution, results) in enumerate(all_results.items()):
#         ax = axs[idx]
#         # Validate results structure
#         required_keys = ['df_bid', 'df_ave', 'df_prop', 'Fairness']
#         if not all(key in results for key in required_keys):
#             raise ValueError(f"Missing data for {distribution} distribution")
#         try:
#             # x = range(len(results['df_bid']['L_max'].unique()))
#             x = results['L_max']
#             orig_rates = sorted(results['df_bid'].iloc[:, 0])
#             ave_rates = sorted(results['df_ave'].iloc[:, 0])
#             prop_rates = sorted(results['df_prop'].iloc[:, 0])
#             disprop_rates = sorted(results['Fairness'].iloc[:, 0])
#         except KeyError as e:
#             raise ValueError(f"Missing column in data: {str(e)}")
        
#         for i, rates in enumerate([orig_rates, ave_rates, prop_rates, disprop_rates]):
#             line, = ax.plot(x, rates, 
#                     marker=markers[i],
#                     color=colors[i],
#                     linestyle='--',
#                     linewidth=1.5,
#                     markersize=8,
#                     label=strategies[i])
#         # Capture legend handles and labels from the first subplot
#         if idx == 0:
#             handles, labels = ax.get_legend_handles_labels()
#         ax.set_xlabel('Constraint L_max', fontsize=10)
#         ax.set_ylabel('Success Rate', fontsize=10)
#         ax.set_title(f'{distribution.capitalize()} Distribution', fontsize=12)
#         ax.grid(True, alpha=0.3)
#         ax.set_ylim(0, 1.1)  # Set consistent y-axis
#         # Add legend to first subplot only
#     Rbs = params['R_total']
#     num_clients = params['num_clients']
#     # Set the global legend at the bottom center spanning across all subplots
#     fig.legend(handles, labels, 
#                loc='lower center', 
#                bbox_to_anchor=(0.5, -0.05), 
#                ncol=len(strategies),
#                fontsize=10)
#     plt.tight_layout(rect=[0, 0.05, 1, 1])
#     plt.savefig(f'plots/success_rate_comparison_n_{num_clients}_rbs_{Rbs}.png', dpi=300, bbox_inches='tight')

def plot_successful_rate(params, all_results):
    """Plot success rates in 3x3 grid (R_total × distributions) with IEEE paper styling"""
    # Set IEEE paper styling
    plt.rcParams.update({
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 2.5,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14
    })

    # Create figure with 3x3 subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.suptitle('Success Rate Analysis by Resource Blocks, Distribution Type, and Allocation Strategy', 
                fontsize=16, y=1.0)

    # Configure plot parameters
    strategies = ['Bidding', 'Average', 'Proportional', 'Fairness']
    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    distributions = list(all_results.keys())
    # Rbs_values = sorted(params['R_total'])  # Assuming R_total is list of 3 values
    Rbs_values = [100, 200, 300]
    # Plot data in grid format
    for row_idx, Rbs in enumerate(Rbs_values):
        for col_idx, distribution in enumerate(all_results.keys()):
            ax = axs[row_idx, col_idx]
            results = all_results[distribution][Rbs]  # Adjusted data structure
            # Extract data
            x = results['L_max']
            bid_rates = sorted(results['df_bid'])
            ave_rates = sorted(results['df_ave'])
            prop_rates = sorted(results['df_prop'])
            fair_rates = sorted(results['Fairness'])
            # bid_rates = results['df_bid'].iloc[:, 0].sort_values()
            # ave_rates = results['df_ave'].iloc[:, 0].sort_values()
            # prop_rates = results['df_prop'].iloc[:, 0].sort_values()
            # fair_rates = results['Fairness'].iloc[:, 0].sort_values()

            # Plot lines
            for i, rates in enumerate([bid_rates, ave_rates, prop_rates, fair_rates]):
                ax.plot(x, rates, 
                       marker=markers[i],
                       color=colors[i],
                       linestyle='--' if i % 2 else '-',
                       markersize=8,
                       label=strategies[i])

            # Axis labels
            ax.set_xlabel(r'$\mathbf{L_{max}}$ (ms)', fontweight='bold')
            ax.set_ylabel(r'$\alpha$', fontweight='bold')
            
            # Titles
            if row_idx == 0:
                ax.set_title(f'{distribution.capitalize()}', pad=15)
            if col_idx == 0:
                ax.text(-0.35, 0.5, f'RBs = {R_total}', 
                       rotation=90, va='center', ha='center',
                       transform=ax.transAxes, fontsize=14)

            # Grid and limits
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis='both', which='major', width=2)

    # Create unified legend
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels,
              loc='lower center',
              bbox_to_anchor=(0.5, 0.02),
              ncol=4,
              frameon=False,
              fontsize=13)

    # Save figure
    plt.savefig(
        f'plots/success_grid_n{params["num_clients"]}_Rbs{min(R_total_values)}-{max(R_total_values)}.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

def compare(clients, params, total_rbs, allocation='average'):
    success_rate, allocations,utility_dict, optimal_aciton, success_list = optimal_action(clients, params, total_rbs, allocation)
    # data = []
    # data['L_max'].append(L_max)
    # data['success_rate'].append(success_rate)
    return success_rate, allocations, utility_dict, optimal_aciton, success_list

def calculate_success_distribution(simulation_data):
    """
    Calculate success rate distribution grouped by context and allocation method.
    Args:
        simulation_data (list of dicts): Each entry contains 'context', 'method', 'success'.
    Returns:
        pd.DataFrame: Success rate per context and method.
    """
    df = pd.DataFrame(simulation_data)
    if df.empty:
        return pd.DataFrame()
    
    # Group by context and method, calculate success rate
    success_rates = df.groupby(['context', 'method'])['success'].mean().reset_index()
    success_rates.rename(columns={'success': 'success_rate'}, inplace=True)
    
    return success_rates

def calculate_jains_index(simulation_data, metric='allocated_rb'):
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
    
    # Group by run parameters and method
    grouped = df.groupby(['R_total', 'L_max', 'method'])
    jain_results = []
    
    for (r_total, l_max, method), group in grouped:
        metrics = group[metric].values
        n = len(metrics)
        if n == 0:
            continue
        print(r_total, l_max, method)
        print('metrics', metrics)
        sum_metrics = sum(metrics)
        sum_squares = sum(x**2 for x in metrics)
        
        if sum_squares == 0:
            ji = 0.0
        else:
            ji = (sum_metrics ** 2) / (n * sum_squares)
        
        jain_results.append({
            'R_total': r_total,
            'L_max': l_max,
            'method': method,
            'jains_index': ji
        })
    print("Jain's index metrics: ", metrics)
    print("Jain's index: ", jain_results)
    return pd.DataFrame(jain_results)

if __name__ == "__main__":
    np.random.seed(2)
    # Initialize clients
    params = {
        'w1': 2,
        'w2': 1.5, # sigma
        'w3': 10,
        # constraint penalty
        'w4': 2, # constraint penalty
        'w5': 1,
        'w6': 0.2,
        'k_A': 0.02,
        'k_L': 0.01,
        # Penalty system
        'lambda_0': 0.0,
        'beta': 0.1,
        'gamma': 0.01,
        # Bid constraints
        'b_max': 4,
        'max_rb':800,
        # total RBs
        'R_total': 300,
        'num_clients': 60,
        'L_max':500,
        'L_max_values': [500+i*100 for i in range(11)],
        'context_to_A_min':{
            'sunny': 50,    
            'rain': 23,     
            'motorway': 56.2, 
            'snow': 11.0,    
            'fog': 44.0,      
            'night': 52.0
            }
    }

    simulation_data = []  # This will store all records for analysis 

    # Define context to A_min mapping using minimum accuracy from each context
    num_clients = params['num_clients']
    contexts_map = {0: 'sunny', 1: 'rain', 2: 'motorway', 3: 'snow', 4: 'fog', 5: 'night'}
    contexts = [contexts_map[np.random.randint(0, 6)] for _ in range(num_clients)]
    # Generate per-client A_min values
    rerun_simulation = True  # Set to False to use existing data
    # distribution_type = "uniform"  # Options: "uniform", "normal", "exponential", "beta"
    all_results = {}
    allocation_methods = ['bidding', 'average', 'prop fair', 'disprop fair']
    R_total_values = [100, 200, 300]  # Example RBs values
    # Run auction
    # for distribution_type in ["poisson", "rayleigh", "exponential"]:
    distribution_type = "poisson"
    filename = f'../results/simulation_data_{distribution_type}.csv'
    all_results[distribution_type] = {}  # Initialize first-level key
    for R_total in R_total_values:
        params['R_total'] = R_total  # Update RBs value
        filename = f'../results/simulation_data_{distribution_type}_R_{R_total}.csv'
        if rerun_simulation:
            # df, df_bid, df_ave, df_prop, df_disprop = save_simulation_data(params, distribution_type)
            success_rates = []
            sr_aves = []
            sr_props = []
            sr_disprops = []
            data_rates_mbps, _ = compute_data_rates(distribution_type, num_clients, scale = 1, rb_allocations = [1 for i in range(num_clients)])
            print('data_rates_mbps', data_rates_mbps)
            
            optimal_actions = []
            for L_max in params['L_max_values']:
                clients = []
                params['L_max'] = L_max

                for i in range(num_clients):
                    client = Client(params, data_rates_mbps[i], contexts[i])
                    clients.append(client)
                # Test all allocation methods
                for method in allocation_methods:
                    if method == 'bidding':
                        # Run bidding auction
                        server = Server(clients, R_total=R_total, lambda_0=1e-6)
                        allocations, _, _, _, _, _, _ = server.run_auction()
                    else:
                        # Run comparison methods
                        _, allocations, _, _, _ = compare(clients, params, R_total, allocation=method)
                    
                    # Collect per-client data
                    for i, client in enumerate(clients):
                        # Get client's adaptation results
                        _, _, success, utility_dict, used_alloc = client.post_allocation_adaptation(allocations[i])
                        
                        simulation_data.append({
                            "context": client.client_state['context'],
                            "method": method,
                            "success": success,  # Directly from post_allocation_adaptation
                            "allocated_rb": used_alloc,
                            "R_total": R_total,
                            "L_max": L_max,
                            "channel_quality": client.client_state['channel_quality'],
                            "energy": utility_dict['energy'],
                            "latency": utility_dict['latency'],
                            "accuracy": utility_dict['accuracy']
                        })
    # with open("n_60_rbs_100-300.json", "w") as file:
    #     json.dump(all_results, file)

    plot_successful_rate(params, all_results)
    
    # plot_context_rb_requirements(params)
        
    # success_rates = calculate_success_distribution(simulation_data)
    # jains_index = calculate_jains_index(simulation_data)

    # plot_success_histograms(success_rates)
    # Plot with R_total as fixed parameter
    
    # plot_jains_index(jains_index)

    # New analysis
    # plot_utility_components(simulation_data)
    # summary_table = create_utility_summary_table(simulation_data)
    # plot_metric_vs_constraints(simulation_data)
    
    # # Print summary table
    # print("\nUtility Component Summary:")
    # print(summary_table.to_markdown(index=False))
