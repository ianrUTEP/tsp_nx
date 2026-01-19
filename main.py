
import argparse
import jointlab as jl
import json
import importlib #for use in interactive

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('json_filepath', type=str)
  parser.add_argument('solution_out', type=str)
  parser.add_argument('-s', '--solve', default=False, action='store_true')
  args = parser.parse_args()
  
  json_filepath = args.json_filepath
  solution_filepath = args.solution_out
  solve_graphs = args.solve
  
    html_set = json.load(settings_file)
  
  graph_list = jl.reset_graph_list(json_filepath)

  if solve_graphs:
    solution_list = jl.solve_graphs_greedy(graph_list)
    jl.save_solutions(solution_list, solution_filepath)
    
    
    # positions = [graph.nodes.data('pos') for graph in graph_list]
  # weights = [graph.edges.data('weight') for graph in graph_list]
  # plt.figure(figsize=(8,6))
  # for graph in graph_list:
  #   nx.draw_networkx_nodes()
  #   nx.draw_networkx_edge_labels()
  #   nx.draw_networkx_labels()
  # make_graph_html(graph_list, 750)
  
  # nx.approximation.asadpour_atsp(graph_list[0])
  # nx.approximation.christofides(graph_list[0])
  # nx.path_weight(graph_list[0],nx.approximation.greedy_tsp(graph_list[0],source=6886),weight='weight')

  # print(graph_list)
  # test_graph = nx.Graph(graph_list[0])
  # print(nx.non_edges(test_graph))
  # print(test_graph.nodes())
  # print(nx.is_weighted(test_graph))
  # print(nx.is_directed(test_graph))
  # print(test_graph[7880][7882])
  # print(test_graph[7882][7880]