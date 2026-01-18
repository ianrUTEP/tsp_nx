import networkx as nx
import pandas as pd
import numpy as np
import json
from pyvis.network import Network
import random
from datetime import datetime
import seaborn as sns
import matplotlib.colors as mcolors

"""
functions to change/remove when not needed anymore (soon)

reset_graph_list:
  doesn't need make_complete_graph anymore

load_graphs_json:
  doesn't need make_complete_graph anymore
  should call the correct graph_from_row()
  
graph_from_row:
  not needed anymore, superceded by the updated format
  
graph_from_row_new:
  should be renamed, like to complete_graph_from_row
  update caller refrerences in load_graphs_json
  
complete_the_graph:
  not needed anymore
  
make_graph_html:
  not needed anymore, superceded by the solution only format
  

"""

def reset_graph_list(json_filepath, make_complete_graph):
  new_graph_list = []
  print("Attempting to load the graphs")
  load_graphs_json(json_filepath, new_graph_list, make_complete_graph)
  print("Loaded", len(new_graph_list), "graphs from the provided file")
  return new_graph_list

def load_graphs_json(file_path: str, graph_list: list, autocomplete: bool) -> list:
  with open(file_path) as json_file:
    # turn into dataframe
    df = pd.read_json(json_file)
    for row in df.itertuples(index=False):
      # each row in dataframe represents a graph, turn into graph object
      graph_list.append(graph_from_row_new(row, autocomplete))
  return graph_list

def graph_from_row(row: tuple, autocomplete: bool) -> nx.Graph:
  G = nx.Graph()
  v = getattr(row, 'V')
  coord_list = getattr(row, 'Vcoords')
  for i, node_id in enumerate(v):
    coords = coord_list[i]
    G.add_node(node_id, pos=(coords[0],coords[1]))
  for edge_str, weight in getattr(row, 'Eweights').items():
    u_str, v_str = edge_str.split(',')
    u, v = int(u_str), int(v_str)
    G.add_edge(u, v, weight=weight)
  if autocomplete:
    #make the graph complete for the asadpour method, giving a default maximum weight as 2x max
    non_edge_value = 2 * getattr(row, 'MaxEWeight')
    G = complete_the_graph(G, non_edge_value)
  return G

def graph_from_row_new(row: tuple, autocomplete: bool = False) -> nx.Graph:
  G = nx.Graph()
  for node_id, node_data in getattr(row, 'nodes').items():
    G.add_node(int(node_id), pos=(node_data['pos'][0], node_data['pos'][1]), group=node_data['streamline'][0]) # streamline number becomes group
  for edge_str, edge_data,  in getattr(row, 'edges').items():
    u_str, v_str = edge_str.split(',')
    G.add_edge(int(u_str), int(v_str), length=edge_data['len'], alignment=edge_data['align']) #no weight given yet
  if autocomplete:
    #make the graph complete for the asadpour method, giving a default maximum weight as 2x max
    non_edge_value = 2 * getattr(row, 'MaxEWeight')
    G = complete_the_graph(G, non_edge_value)
  return G
  

def complete_the_graph(original_graph: nx.Graph, non_edge_weight: int) -> nx.Graph:
  """
  Takes an incomplete graph and makes it complete with a default value for new edges
  
  :param original_graph: An incomplete undirected graph
  :type original_graph: nx.Graph
  :param non_edge_weight: The default weight value for new edges
  :type non_edge_weight: int
  :return: A complete undirected graph
  :rtype: Graph[Any]
  """
  print("Making a graph complete with set value:", non_edge_weight)
  complement: nx.Graph = nx.complement(original_graph)
  #depending on your IDE typing strength, this may show an error. Its fine
  #this sets the 'weight' attribute of all edges in complement to the default, which is non_edge_weight
  nx.set_edge_attributes(complement, non_edge_weight, weight='weight')
  #get the edges and delete the complement graph
  complement_edges = complement.edges(data=True)
  del complement
  print("Made complement graph, attempting to compose graphs")
  # combined = nx.compose(original_graph, complement)
  original_graph.add_edges_from(complement_edges)
  del complement_edges
  print("Composition complete")
  return original_graph

def make_graph_html(graph_list, canvas_height, solution_only: bool = False, sol_list: list = [], vis_opt_dict: dict = {}):
  """
  Make HTML files for the list of graphs provided
  
  :param graph_list: list of graphs to convert
  :param canvas_height: height of the html canvas, 1000 works pretty well
  :param solution_only: If False, WILL modify your graph attributes to add HTML ones. If True, will only generate visuals for the solution of a graph. Requires that a solution list is provided with solutions for each graph
  :type solution_only: bool
  :param sol_list: Solution list as ordered list of nodes
  :type sol_list: list
  :param vis_opt_dict: Python dictionary of valid html options to be converted and used here
  :type vis_opt_dict: dict
  """
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  for i, graph in enumerate(graph_list):
    # to make a complete graph visible, only include edges from the solution
    if solution_only:
      vis_graph: nx.Graph = nx.create_empty_copy(graph)
      edges = [(sol_list[i][j], sol_list[i][j+1]) for j in range(len(sol_list[i])-1)]
      for u,v in edges:
        vis_graph.add_edge(u, v, weight=graph.edges[u,v]['weight'])
      # weighted_edges = [(u, v, ) for u,v in edges]
      # print(weighted_edges)
      # vis_graph.add_edges_from(weighted_edges)
    else:
      vis_graph = graph.copy(as_view=True)
    print("generating graphvis data for graph", i)
    #isolate a list of coords for scaling
    coords = np.array(list(nx.get_node_attributes(vis_graph, 'pos').values()))
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    #scale up the coordinates of drawing to the canvas size (minimul information is pixel, so 1)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_range = (x_max - x_min) if (x_max != x_min) else 1
    y_range = (y_max - y_min) if (y_max != y_min) else 1
    scale = max(x_range, y_range)
    #These two iterations are where individual attributes can be set for nodes and edges
    for u, data in vis_graph.nodes(data=True):
      #apply the scaled x and y values to each coord
      data['x'] = np.float64(((data['pos'][0] - x_min) / scale) * canvas_height)
      data['y'] = np.float64(-((data['pos'][1] - y_min) / scale) * canvas_height)
      data['size'] = 1
      data['title'] = str(u)
    for u, v, data in vis_graph.edges(data=True):
      data['value'] = 1 #data.get('weight', 1)
      data['width'] = 1 #must add width before converting to pyvis.net or weight attribute gets destroyed
      data['color'] = str('#'+hex(random.randrange(0,2**24))[2:])
      data['label'] = str(data.get('weight', 1)) #str(data['value'])
      data['font'] = {"size":1, "strokeWidth":0, "color":"#fffffff"}
    print("generating graphvis html for graph", i)
    net = Network(height=canvas_height, width='100%', notebook=False)#, filter_menu=False, select_menu=False)
    net.from_nx(vis_graph)
    del vis_graph
    net.toggle_drag_nodes(False)
    net.toggle_physics(False)
    net.show_buttons()#filter_=['nodes', 'edges', 'selection', 'renderer', 'interaction', 'phsyics'])
    net.set_options(' '.join(['var','options','=',json.dumps(vis_opt_dict)]))
    net.save_graph(name='/'.join(['./graphvisuals','.'.join(['_'.join([timestamp, 'graph', str(i)]),'html'])]))
    
def make_solution_html(graph_list, canvas_height, sol_list: list, vis_opt_dict: dict = {}, color_scale_attr: str = 'weight'):
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  for i, graph in enumerate(graph_list):
    print("making solution copy of graph", i)
    vis_graph: nx.Graph = nx.create_empty_copy(graph)
    edges = [(sol_list[i][j], sol_list[i][j+1]) for j in range(len(sol_list[i])-1)]
    for u,v in edges:
      vis_graph.add_edge(u, v, weight=graph.edges[u,v]['weight'])
    print("generating graphvis data for graph", i)
    coords = np.array(list(nx.get_node_attributes(vis_graph, 'pos').values()))
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    #scale up the coordinates of drawing to the canvas size (minimul information is pixel, so 1)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_range = (x_max - x_min) if (x_max != x_min) else 1
    y_range = (y_max - y_min) if (y_max != y_min) else 1
    scale = max(x_range, y_range)
    #These two iterations are where individual attributes can be set for nodes and edges
    for u, data in vis_graph.nodes(data=True):
      #apply the scaled x and y values to each coord
      data['x'] = np.float64(((data['pos'][0] - x_min) / scale) * canvas_height)
      data['y'] = np.float64(-((data['pos'][1] - y_min) / scale) * canvas_height)
      data['size'] = 1
      data['title'] = str(u)
    (low_scale, high_scale) = get_attribute_extremes(vis_graph, color_scale_attr)
    normalizer = mcolors.LogNorm(vmin=low_scale, vmax=high_scale)
    palette = sns.cubehelix_palette(as_cmap=True)
    for u, v, data in vis_graph.edges(data=True):
      data['value'] = 1 #data.get('weight', 1)
      data['width'] = 1 #must add width before converting to pyvis.net or weight attribute gets destroyed
      data['color'] = get_color_hex_in_range(data[color_scale_attr], palette, normalizer)      #str(#'+hex(random.randrange(0,2**24))[2:])
      # data['label'] = str(data.get('weight', 1)) #str(data['value'])
      data['font'] = {"size":1, "strokeWidth":0, "color":"#fffffff"}
    print("generating graphvis html for graph", i)
    net = Network(height=canvas_height, width='100%', notebook=False)#, filter_menu=False, select_menu=False)
    net.from_nx(vis_graph)
    del vis_graph
    net.toggle_drag_nodes(False)
    net.toggle_physics(False)
    net.show_buttons()#filter_=['nodes', 'edges', 'selection', 'renderer', 'interaction', 'phsyics'])
    net.set_options(' '.join(['var','options','=',json.dumps(vis_opt_dict)]))
    net.save_graph(name='/'.join(['./graphvisuals','.'.join(['_'.join([timestamp, 'graph', str(i)]),'html'])]))
      
def solve_graphs_greedy(graph_list):
  solution_list = []
  print("Beginning search for solutions")
  for i, graph in enumerate(graph_list):
    print("Solving graph", i)
    sol = nx.approximation.greedy_tsp(graph)
    solution_list.append(sol)
  return solution_list

def solve_graphs_multgreedy(graph_list, n_greedys:int = 10):
  solution_list = []
  print("Beginning search for solutions")
  for i, graph in enumerate(graph_list):
    graph_sols = []
    print("Solving graph", i)
    sources = np.insert(np.random.randint(1, nx.number_of_nodes(graph) + 1, n_greedys-1), 0, 1) #plus one to limit to include, generate n-1 and add 1 as guaranteed source
    for j, source in enumerate(sources):
      print("iteration", j)
      graph_sols.append(nx.approximation.greedy_tsp(graph, source=int(source)))
    solution_list.append(graph_sols)
  return solution_list

def cycle_sol_to_path(sol_list:list):
  return [[cycle[:-1] for cycle in graph] for graph in sol_list]

def save_solutions(solution_list, solution_filepath):
  print("Saving solution sets")
  sol_array = np.array(solution_list, dtype=np.uint16)
  np.savetxt(solution_filepath,sol_array.transpose(),delimiter=',',fmt='%i')

def add_weights(graph_list):
  for graph in graph_list:
    for u, v, data in graph.edges(data=True):
      if data['alignment'] != 0:
        data['weight'] = data['alignment']
      else:
        data['weight'] = 2 + data['length']

def get_attribute_extremes(graph: nx.Graph, attribute: str):
  attrList = nx.get_edge_attributes(graph, attribute) #gets iterable list of specified attribute
  #assume extremes
  minAtt = float('inf')
  maxAtt = float('-inf')
  for edge in attrList:
    if attrList[edge] > maxAtt:
      maxAtt = attrList[edge]
    if attrList[edge] < minAtt:
      minAtt = attrList[edge]
  return (minAtt, maxAtt)

def get_color_hex_in_range(value, colormap: mcolors.ListedColormap, normalizer: mcolors.Normalize):
  return mcolors.to_hex(colormap(normalizer(value)))