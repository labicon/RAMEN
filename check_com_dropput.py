import matplotlib.pyplot as plt
import json
import networkx as nx 
import argparse
import config
import os 

def plot_edge_weights_over_time(G, node, edge_weight_history):
    """
    Plots the edge weights of a given node's neighbors over time as discrete points
    with subplots for each edge, integer x ticks, y ticks only at 0 and 1, and
    the percentage of times the weight was 1 in the title.

    Args:
      node: The node for which to plot edge weights.
      edge_weight_history: A dictionary containing the history of edge weights.
    """
    num_neighbors = len(G.adj[node])
    fig, axes = plt.subplots(num_neighbors, 1, figsize=(10, 2 * num_neighbors))
    if num_neighbors == 1:  # handle case of single neighbor
        axes = [axes]

    neighbor_index = 0
    for neighbor in G.neighbors(node):
        edge = (node, neighbor) if node < neighbor else (neighbor, node)
        weights = edge_weight_history[str(edge)]
        percent_one = sum(weights) / len(weights) * 100  # Calculate percentage of 1s
        ax = axes[neighbor_index]
        ax.plot(weights, 'o', label=f"Edge {edge}")
        ax.set_title(f"Edge {edge} Communication ({percent_one:.1f}%)", fontsize=18)

        ax.set_ylabel("Com On/Off", fontsize=18)
        ax.set_yticks([0, 1])

        # Only show x label and ticks for the last subplot
        if neighbor_index == num_neighbors - 1:
            ax.set_xlabel("Iteration", fontsize=14)
            num_iterations = len(weights)
            ax.set_xticks(range(0, num_iterations, max(1, num_iterations // 10)))
        else:
            ax.set_xticks([])  # Remove x ticks for other subplots

        ax.tick_params(axis='both', which='major', labelsize=18)  # Set tick label size
        neighbor_index += 1

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--agent', default=0, type=int, help='Which agent to check.')
    args = parser.parse_args()

    cfg = config.load_config(args.config)

    # create graph 
    G = nx.Graph()
    num_agents = cfg['multi_agents']['num_agents']
    G.add_nodes_from([i for i in range(num_agents)]) 
    G.add_edges_from(cfg['multi_agents']['edges_list'], weight=1)

    # Load the saved data from the JSON file
    output_path = os.path.join(cfg['data']['output'], cfg['data']['exp_name'])

    with open(os.path.join(output_path, 'graph_data.json'), 'r') as f:
        loaded_data = json.load(f)

    edge_weight_history = loaded_data['edge_weight_history']

    # Example usage: plot edge weights of node 1 over time
    plot_edge_weights_over_time(G, args.agent, edge_weight_history)