import glob
import os
import networkx as nx
import pandas as pd


def main():
    # Load all graph CSV files into a list of dataframes
    graphs = glob.glob(os.path.join('dataset\edges', "*.csv"))
    # could be replaced with 'dataset\edges_4', get results of networks with the threshold of 0.4

    # Create a dictionary to hold each graph dataframe with its name (e.g., 'airforce', 'army')
    graph_dfs = {}
    for graph in graphs:
        graph_name = os.path.basename(graph).replace('r_', '').replace('_edges.csv', '')
        graph_dfs[graph_name] = pd.read_csv(graph)

    # Load the labels dataframe
    labels_df = pd.read_csv('D:\PycharmProjects\MLRW\project\dataset\label1.csv')

    labels_df['graph_name'] = labels_df['graph_name'].astype(str)
    labels_df['label'] = labels_df['label'].astype(str)

    results = []
    for graph_name, graph_df in graph_dfs.items():
        # Add a 'graph_name' column to the graph dataframe to match with labels dataframe
        graph_df['graph_name'] = graph_name

        # Merge the graph dataframe with the labels dataframe based on the graph name
        merged_df = pd.merge(graph_df, labels_df, on='graph_name')

        # Convert merged dataframe to a NetworkX graph
        G = nx.from_pandas_edgelist(merged_df, source='source', target='target')

        if len(G) == 0:
            continue

        # Calculate metrics
        metrics = calculate_selected_metrics(G)
        if metrics:
            metrics['label'] = merged_df['label'].iloc[0]  # Ensure label is added correctly
            results.append(metrics)

    results_df = pd.DataFrame(results)

    # Display and save the average metrics for each label
    if 'label' in results_df.columns and not results_df.empty:
        summary = results_df.groupby('label').mean()
        print(summary)
        summary.to_csv('D:\PycharmProjects\MLRW\project\dataset\summary_metrics.csv', index=True)
    else:
        print("No data available for grouping by 'label'")


# Define a function to calculate selected graph metrics
def calculate_selected_metrics(G):
    metrics = {}

    if len(G) == 0:
        return metrics

    # Degree Centrality
    degree_centrality = list(nx.degree_centrality(G).values())
    if len(degree_centrality) > 0:
        metrics['degree_centrality'] = sum(degree_centrality) / len(degree_centrality)
    else:
        metrics['degree_centrality'] = float('nan')

    # Betweenness Centrality
    betweenness_centrality = list(nx.betweenness_centrality(G).values())
    if len(betweenness_centrality) > 0:
        metrics['betweenness_centrality'] = sum(betweenness_centrality) / len(betweenness_centrality)
    else:
        metrics['betweenness_centrality'] = float('nan')

    # Clustering Coefficient
    metrics['clustering_coefficient'] = nx.average_clustering(G) if len(G) > 0 else float('nan')

    # Network Density
    metrics['network_density'] = nx.density(G) if len(G) > 0 else float('nan')

    # Average Path Length
    try:
        if nx.is_connected(G):
            metrics['average_path_length'] = nx.average_shortest_path_length(G)
        else:
            metrics['average_path_length'] = float('inf')
    except nx.NetworkXError:
        metrics['average_path_length'] = float('nan')

    # Modularity
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    if len(communities) > 1:
        metrics['modularity'] = nx.algorithms.community.modularity(G, communities)
    else:
        metrics['modularity'] = float('nan')

    return metrics


if __name__ == '__main__':
    main()
