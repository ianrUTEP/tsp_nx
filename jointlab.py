import copy
import networkx as nx
import pandas as pd
import numpy as np
import json
from pyvis.network import Network
import random
from datetime import datetime
import seaborn as sns
import matplotlib.colors as mcolors
import pygad
import logging
import time
from os import path, makedirs
import sys
from math import dist
from wolframclient.evaluation import WolframLanguageSession as wls
from wolframclient.language import wl, wlexpr

#region Load Graphs
def read_sol_list(json_filepath):
  with open(json_filepath, 'r') as json_file:
    new_sol_list = json.load(json_file)
  return(new_sol_list)
  

def reset_graph_list(json_filepath):
  new_graph_list = []
  print("Attempting to load the graphs")
  load_graphs_json(json_filepath, new_graph_list)
  print("Loaded", len(new_graph_list), "graphs from the provided file")
  return new_graph_list

def load_graphs_json(file_path: str, graph_list: list) -> list:
  with open(file_path) as json_file:
    # turn into dataframe
    df = pd.read_json(json_file)
    for i, row in enumerate(df.itertuples(index=False)):
      # each row in dataframe represents a graph, turn into graph object
      graph_list.append(complete_graph_from_row(row, i))
      if hasattr(row, 'compnodes'):
        graph_list.append(compressed_complete_from_row(row, i))
  return graph_list

def complete_graph_from_row(row: tuple, num: int) -> nx.Graph:
  G = nx.Graph(compressed=False, number=num)
  for node_id, node_data in getattr(row, 'nodes').items():
    G.add_node(int(node_id), 
              pos=(node_data['pos'][0], node_data['pos'][1]), 
              group=node_data['streamline'][0],     # streamline number becomes group
              groupidx=node_data['streamline'][1],
              pstress=node_data['pStressV'],
              comps=node_data['CompV'],
              internal=bool(node_data['partMember'])) # may have to change partmember behavior to "node identifier" in future and not bool
  for edge_str, edge_data, in getattr(row, 'edges').items():
    u_str, v_str = edge_str.split(',')
    G.add_edge(int(u_str), int(v_str), length=edge_data['len'], alignment=edge_data['align']) #no weight given yet
  return G

def compressed_complete_from_row(row: tuple, num: int) -> nx.Graph:
  G = nx.Graph(compressed=True, number=num)
  for node_id, node_data in getattr(row, 'compnodes').items():
    G.add_node(int(node_id), 
              pos=(node_data['pos'][0], node_data['pos'][1]), 
              group=node_data['streamline'][0],     # streamline number becomes group
              groupidx=node_data['streamline'][1],
              pstress=node_data['pStressV'],
              comps=node_data['CompV'],
              internal=bool(node_data['partMember'])) # may have to change partmember behavior to "node identifier" in future and not bool
  for edge_str, edge_data, in getattr(row, 'compedges').items():
    u_str, v_str = edge_str.split(',')
    G.add_edge(int(u_str), int(v_str), lengths=edge_data['l'], alignments=edge_data['a'], nodes=edge_data['n'])
  return G
#endregion Load Graphs

#region Visualization
def make_solution_html(graph_list, sol_list: list, canvas_height, vis_opt_dict: dict = {}, color_scale_attr: str = 'weight'):
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  for i, graph in enumerate(graph_list):
    print("making solution copy of graph", i)
    vis_graph: nx.Graph = nx.create_empty_copy(graph)
    edges = [(sol_list[i][j], sol_list[i][j+1]) for j in range(len(sol_list[i])-1)]
    for u,v in edges:
      vis_graph.add_edge(u, v, weight=graph.edges[u,v]['weight'])#, length=graph.edges[u,v]['length'], alignment=graph.edges[u,v]['alignment'])
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
      # data['title'] = ';'.join([str(data['length']), str(data['weight']), str(data['alignment'])])
    print("generating graphvis html for graph", i)
    net = Network(height=canvas_height, width='100%', notebook=False)#, filter_menu=False, select_menu=False)
    net.from_nx(vis_graph)
    del vis_graph
    net.toggle_drag_nodes(False)
    net.toggle_physics(False)
    net.show_buttons()#filter_=['nodes', 'edges', 'selection', 'renderer', 'interaction', 'phsyics'])
    net.set_options(' '.join(['var','options','=',json.dumps(vis_opt_dict)]))
    net.save_graph(name='/'.join(['./graphvisuals','.'.join(['_'.join([timestamp, 'graph', str(i)]),'html'])]))
    
def get_color_hex_in_range(value, colormap: mcolors.ListedColormap, normalizer: mcolors.Normalize):
  # print(normalizer.vmax)
  # print(normalizer.vmin)
  try:
    color = mcolors.to_hex(colormap(normalizer(value+1)))
  except ValueError:
    color = mcolors.to_hex(colormap(0.5))
    # print(value)
  # print(color)
  return color
#endregion Visualization
      
#region Solvers
#region Sol.Greedy
def solve_graphs_greedy(graph_list):
  solution_list = []
  print("Beginning search for solutions")
  for i, graph in enumerate(graph_list):
    print("Solving graph", i)
    sol = nx.approximation.greedy_tsp(graph)
    solution_list.append(sol)
  return solution_list

def solve_graphs_multgreedy(graph_list, n_greedys:int = 10, guaranteed:int = -1)->list:
  solution_list = []
  print("Beginning search for solutions")
  for i, graph in enumerate(graph_list):
    graph_sols = []
    if guaranteed != -1:
      sources = np.insert(random.sample(sorted(nx.nodes(graph)), n_greedys-1), 0, guaranteed) #plus one to limit to include, generate n-1 and add 1 as guaranteed source
    else: 
      sources = random.sample(sorted(nx.nodes(graph)), n_greedys)
    for j, source in enumerate(sources):
      print("graph", i, "sol", j)
      graph_sols.append(nx.approximation.greedy_tsp(graph, source=int(source)))
    solution_list.append(graph_sols)
  return solution_list
#endregion Sol.Greedy

#region Sol.DFS
def solve_multdfs(graph_list, n_dfs:int = 1, use_first:bool = True)->list:
  dfs_sols = []
  for i, graph in enumerate(graph_list):
    graph_sols = []
    if use_first:
      sources = np.insert(random.sample(sorted(nx.nodes(graph)), n_dfs-1), 0, list(graph.nodes())[0])
    else:
      sources = random.sample(sorted(nx.nodes(graph)), n_dfs)
    for j, source in enumerate(sources):
      print("graph", i, "sol", j)
      dfs = DDFS(graph, int(source), i, j)
      graph_sols.append(dfs.search())
      dfs.close_log()
    dfs_sols.append(graph_sols)
  return dfs_sols
#endregion Sol.DFS
#endregion Solvers

#region Outputs
def save_solutions(solution_list:list, solution_filepath:str):
  print("Saving solution sets")
  sol_array = np.array(solution_list, dtype=np.uint16)
  np.savetxt(solution_filepath,sol_array.transpose(),delimiter=',',fmt='%i')
  
def select_best_sol(graph_list:list, solution_list:list, cost_attr:str="weight")->list:
  best_sols = []
  for i, graph in enumerate(graph_list):
    best_j = -1
    best_w = float("inf")
    for j, solution in enumerate(solution_list[i]):
      w = nx.path_weight(graph, solution,cost_attr)
      if w < best_w:
        best_j = j
        best_w = w
    best_sols.append(solution_list[i][best_j])
  return best_sols

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

def comps_over_sol_len(graph: nx.Graph, solution) -> list:
  return []

#region Out.LogFileMaker
class LogFileMaker:
  #static "private" values shared between the class as a default for creating logs
  global_lf_lev = logging.DEBUG
  global_c_lev = logging.INFO
  
  #create and set the global variables. Creating a new one is the only way to change it
  def __init__(self, logfile_level:str, console_level:str):
    #set a global level at initalization of class
    self.set_globals(logfile_level,console_level)
  
  @classmethod
  def set_globals(cls, new_g_lf_lev, new_g_c_lev):
    match new_g_lf_lev:
      case 'debug':
        cls.global_lf_lev = logging.DEBUG #filters to debug and above, not recommended for console
      case 'info':
        cls.global_lf_lev = logging.INFO
      case 'none':
        cls.global_lf_lev = None
      case _:
        cls.global_lf_lev = None
    match new_g_c_lev:
      case 'debug':
        cls.global_c_lev = logging.DEBUG #filters to debug and above, not recommended for console
      case 'info':
        cls.global_c_lev = logging.INFO #filters to info and above, good for console
      case 'none':
        cls.global_c_lev = None
      case _:
        cls.global_c_lev = None
  
  #Uses the globals and the input values to return levels
  @classmethod
  def set_levels(cls, new_lf_lev, new_c_lev):
    match new_lf_lev:
      case 'global':
        lf_lev = cls.global_lf_lev #use the global set initially
      case 'debug':
        lf_lev = logging.DEBUG #filters to debug and above, not recommended for console
      case 'info':
        lf_lev = logging.INFO
      case 'none':
        lf_lev = None
      case _:
        lf_lev = None
    match new_c_lev:
      case 'global':
        c_lev = cls.global_c_lev #use the global set initially
      case 'debug':
        c_lev = logging.DEBUG #filters to debug and above, not recommended for console
      case 'info':
        c_lev = logging.INFO #filters to info and above, good for console
      case 'none':
        c_lev = None
      case _:
        c_lev = None
    return lf_lev, c_lev
  
  #Creates a new logger
  @classmethod
  def create_logger(cls, logfile_name:str, logfile_level:str='global', console_level:str='global')-> logging.Logger:
    print("Creating logger:", logfile_name, "with file and console:", logfile_level, console_level)
    #set the levels for the class, but not the global status
    lf_lev, c_lev = cls.set_levels(logfile_level,console_level)
    logger = logging.getLogger(logfile_name)
    logger.setLevel(logging.DEBUG)
    if lf_lev is not None:
      file_handler = logging.FileHandler(logfile_name,'a+','utf-8')
      file_handler.setLevel(lf_lev)
      file_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
      file_handler.setFormatter(file_format)
      logger.addHandler(file_handler)
    if c_lev is not None:
      console_handler = logging.StreamHandler()
      console_handler.setLevel(c_lev)
      console_format = logging.Formatter('%(message)s')
      console_handler.setFormatter(console_format)
      logger.addHandler(console_handler)
    return logger
#endregion Out.LogFileMaker  
#endregion Outputs

#region Modify Graphs
def add_weights(graph_list, travel_threshold:float=0.8, compression_fact:float=0.3):
  for graph in graph_list:
    if graph.graph['compressed'] == False:
      for u, v, data in graph.edges(data=True):
        data['weight'] = compute_weight(data['alignment'], data['length'], travel_threshold, graph.nodes[u], graph.nodes[v])
        # if data['alignment'] != 0:
        #   data['weight'] = data['alignment'] + (data['length'] / travel_threshold)**2# 1 to 2 + d(0,1] = d[2,3] because 1 added already
        # else:
        #   data['weight'] = 3 + (data['length'] / travel_threshold)**2 # 2 + length, minimum 2 + 2*EW 
    if graph.graph['compressed'] == True:
      for u, v, data in graph.edges(data=True):
        subweights = 0
        n_edges = len(data['nodes'])-1
        if n_edges == 1:
          data['weight'] = compute_weight(data['alignments'][0], data['lengths'][0], travel_threshold, graph.nodes[u], graph.nodes[v])
        else:
          for n in range(n_edges):
            subweights += (compute_weight(data['alignments'][n], data['lengths'][n], travel_threshold, graph.nodes[u], graph.nodes[v]))
          data['weight'] = (subweights / n_edges) * compression_fact
        # if data['alignment'] != 0:
        #   data['weight'] = data['alignment'] + (data['length'] / travel_threshold)**2# 1 to 2 + d(0,1] = d[2,3] because 1 added already
        # else:
        #   data['weight'] = 3 + (data['length'] / travel_threshold)**2 # 2 + length, minimum 2 + 2*EW 

def compute_weight(alignment, length, travel_thresh, u, v):
  if (u['internal'] == v['internal'] and u['internal'] == True): #internal->internal
    if alignment == 0:
      return 3 + length_factor(length, travel_thresh)**2 + composition_factor(u['comps'], v['comps'])
    else:
      return alignment + length_factor(length, travel_thresh)**2 + composition_factor(u['comps'], v['comps'])
  elif (u['internal'] == v['internal'] and u['internal'] == False): #perim->perim
    if alignment == 0:
      return 2 + (length_factor(length, travel_thresh)/2) + composition_factor(u['comps'], v['comps'])
    else:
      return alignment + (length_factor(length, travel_thresh)/2) + composition_factor(u['comps'], v['comps'])
  else: # intenal->perim
    # currently same as internal->internal
    if alignment == 0:
      return 3 + length_factor(length, travel_thresh)**2 + composition_factor(u['comps'], v['comps'])
    else:
      return alignment + length_factor(length, travel_thresh)**2 + composition_factor(u['comps'], v['comps'])
  # if alignment != 0:
  #   return alignment + (length / travel_thresh)**2
  # else:
  #   return 3 + (length / travel_thresh)**2

def length_factor(len, thresh):
  return len/thresh

def composition_factor(c1:list, c2:list):
  total = 0
  for i in range(len(c1)):
    total += abs((abs(c2[i])-abs(c1[i])))
  return total

def decompress(graph_list, solution_list):
  uncompressed_graphs = []
  decompressed_sols = []
  for i, g in enumerate(graph_list):
    if g.graph['compressed'] == True:
      uncompressed_graphs.append(graph_list[i-1]) #graph comes before is the uncompressed version
      comp_sol = solution_list[i] #take the solution that matches this compressed one
      full_sol = [comp_sol[0]]  #first node
      for u, v in [comp_sol[n:n+2] for n in range(0, len(comp_sol)-1)]:
        nodes:list = g[u][v]['nodes'] #get the nodes each edge represents
        if u > v:
          full_sol += reversed(nodes[:-1]) #node representation always cannonical order, may need reversed to reflect edge direction and drop the last one
        else:
          full_sol += nodes[1:] #drop the first one
      decompressed_sols.append(full_sol)
  return (uncompressed_graphs, decompressed_sols)

def complete_missing_nodes(graph: nx.Graph, solution: list) -> list:
  log = LogFileMaker.create_logger("/".join(["./logs","_".join([datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),"completer.log"])])) 
  missing_nodes = graph.nodes() - set(solution)
  new_sol = solution.copy()
  for n in missing_nodes: 
    log.info("Adding node %d", n)
    greedy_pos = 0
    greedy_cost = float('inf')
    for i in range(len(new_sol)+1): #try every index from 0 (prepend) to length (append), add 1 because range end exclusive
      test_path = new_sol.copy()
      test_path.insert(i,n)
      if (nx.path_weight(graph, test_path, 'weight') < greedy_cost):
        greedy_pos = i
        greedy_cost = nx.path_weight(graph, test_path, 'weight')
        log.debug("Minweight updated to %f at %d", greedy_cost, i)
    log.debug("Selected position %d", greedy_pos)
    new_sol.insert(greedy_pos, n)
  for handler in log.handlers:
    log.removeHandler(handler)
  return new_sol

def missing_nodes_zag(graph: nx.Graph, solution: list) -> list:
  new_sol = solution.copy()
  log = LogFileMaker.create_logger("/".join(["./logs","_".join([datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),"zag.log"])]))
  # set of nodes missing to add
  missing_nodes = graph.nodes() - set(solution)
  # keeps track of how many new edges are added to existing nodes
  new_connection_counts = dict.fromkeys(solution, 0)
  # keeps track of where the nodes should end up going
  connection_inserts = dict.fromkeys(missing_nodes, [0,0])
  # determine missing node insertions
  log.debug("Length of missing nodes: %d", len(missing_nodes))
  for n in missing_nodes:
    log.debug("Considering missing node %d", n)
    # get edges sorted by weight (consider cheapest options first)
    sorted_edges = sorted(graph.edges(n, data=True), key=lambda x: x[2].get('weight', float('inf')))
    for edge in sorted_edges:
      # a connection for this node has been established, can break this loop
      if connection_inserts.get(n, [0,0]) != [0,0]:
        break
      log.debug("Considering edge to %d", edge[1])
      # check if the edge destination (edges from graph are given as source, dest so this always works) is in the solved set and isn't overcrowded
      if edge[1] in new_connection_counts and new_connection_counts.get(edge[1],2) < 2:
        # don't allow this connection if the destination is right next to the new node (streamline turnarounds mainly)
        if edge[1] == n + 1 or edge[1] == n - 1:
          log.debug("This connection is at a streamline turnaround and will not be considered")
          continue
        # check the nodes immediately before and after the possible connection in the solution
        idx = solution.index(edge[1])
        neighbors = [solution[idx-1] if idx-1>= 0 else None, solution[idx+1] if idx+1 < len(solution) else None]
        neighbors.sort(key=lambda x: graph[n][x]['weight'])
        # sort the neighbors, again greedily
        log.debug("Attempting to connect %d to %d using neighbors %d and %d", n, edge[1], neighbors[0], neighbors[1])
        # check each neighbor
        for neighbor in neighbors:
          # start/end of solution exception
          if neighbor is None:
            continue
          # neighbor isn't overcrowded and is part of the solution
          if neighbor in new_connection_counts and new_connection_counts.get(neighbor,2) < 2:
            log.debug("Will insert %d between %d and %d", n, edge[1], neighbor)
            # save where to put it
            connection_inserts.update({n:[edge[1],neighbor]})
            # CURRENTLY DISABLED - update overcrowding information
            # new_connection_counts[edge[1]] = new_connection_counts[edge[1]] + 1
            # new_connection_counts[neighbor] = new_connection_counts[neighbor] + 1
            # stop searching through neighbors
            break
      else:
        log.debug("Not a solved node or overcrowded")
  # make the instruction set for reversals
  insertions = []
  for new_node, between in connection_inserts.items():
    # keep the max index
    insertions.append([new_node, max(solution.index(between[0]),solution.index(between[1]))])
  # reverse sort by max index
  insertions.sort(key=lambda x:x[1], reverse=True)
  for node, idx in insertions:
    new_sol.insert(idx, node)
  for handler in log.handlers:
    log.removeHandler(handler)
  return new_sol

def reconstruct_ext_path(solution, registry):
  full_path = []
  for gene_id in solution:
    gene = registry[gene_id]
    if gene['type'] in ['compressed', 'standalone']:
      full_path.extend(gene['internal_path'])
    elif gene['type'] == 'new':
       full_path.append(tuple(gene['start_pos']))
  return full_path

def integrate_solution_to_graph(src_graph: nx.Graph, full_path):
  extended_graph:nx.Graph = copy.deepcopy(src_graph)
  new_node_num = max(extended_graph.nodes)
  modified_path = []
  for node in full_path:
    if isinstance(node, tuple) and not extended_graph.has_node(node):
        new_node_num += 1
        extended_graph.add_node(new_node_num, pos=node, internal=False, group=-1)
        modified_path.append(new_node_num)
    else:
       modified_path.append(node)
  for i in range(len(full_path) - 1):
    node_a = modified_path[i]
    node_b = modified_path[i+1]
    if not extended_graph.has_edge(node_a, node_b):
       extended_graph.add_edge(node_a,node_b,weight=1)
  return extended_graph, modified_path

#endregion Modify Graphs

#region DDFS Class
class DDFS:
  def __init__(self, graph:nx.Graph, 
              start_node:int = 0,
              graph_num:int=0,
              search_num:int=0,
              logger=None):
    self.log:logging.Logger = logger if logger is not None else LogFileMaker.create_logger("/".join(["./logs","_".join([datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),"g",str(graph_num),"s",str(search_num),"ddfs.txt"])])) 
    self.g:nx.Graph = graph #graph object
    self.n_count:int = len(self.g.nodes())
    self.start = start_node
    self.setup()
  
  def setup(self):
    self.in_path:set = set()  #nodes existing in the path
    try:
      inital_edges = self.get_edges_from_node(self.start) #get the edges from the start node
    except Exception as e:
      self.log.error("Error at initializing DDFS", exc_info=e)
      self.close_log
      return
    self.branches:list = [inital_edges.copy(), inital_edges.copy()] #stores edges in sorted depth-first by level for each branch
    self.in_path.add(self.start) #tracks nodes included in the path
    self.path:list = [self.start] #the actual solution path. B1 prepends, B2 appends
  
  # closes out the logger to release the file without closing python
  def close_log(self):
    for handler in self.log.handlers:
      self.log.removeHandler(handler)
  
  def get_edges_from_node(self, new_node:int)-> list:
    self.log.debug("Getting edges from node %s", str(new_node))
    #get the graph that remains to be added
    g_rem = self.g.subgraph(self.g.nodes() - self.in_path) #doesn't make a copy, just a subgraph view
    new_edges = []  #list of new edges to add to this branch
    for u, v, data in g_rem.edges(new_node, data=True): #iterate through subgraph edges connected to new node
      #store neighbor node and the weight of the edge
      # self.log.debug("New edge detected: ",u,v,data['weight'])
      new_edges.append((v, data['weight']))
    #sort the new edges based on their weight
    new_edges.sort(key=lambda e: e[1])
    # self.log.debug("New edge collection sorted by weight: %s", str(new_edges))
    #TODO: consider implementing a limit to the number of edges returned to prevent the search tree from growing too large
    #This would allow me to do the original plan and not remove edges when consumed so that they remain when backtracked
    #However, this could lead to cases where all of the cheapest edges are to already explored nodes and the DFS fails
    return new_edges
  
  def add_edges_to_branch(self, branch:int, new_node:int):
    self.log.debug("Updating branch %d from node %d", branch, new_node)
    edges = self.get_edges_from_node(new_node)
    #update information now that it is done
    self.in_path.add(new_node)
    #prepend or append to path based on branch, update branch edges
    if branch == 0:
      self.branches[0] = edges + self.branches[0]
      self.path = [new_node] + self.path
      # self.log.debug("branches[0]: %s", str(self.branches[0]))
    else:
      self.branches[1] = edges + self.branches[1]
      self.path.append(new_node)
      # self.log.debug("branches[1]: %s", str(self.branches[1]))
    self.log.debug("path: %s", str(self.path))

  # removes front nodes from branches until the frontmost is not already in the path
  def clear_dead_limbs(self):
    for i, branch in enumerate(self.branches):
      popped:int = 0
      try:
        while branch[0][0] in self.in_path:
          branch.pop(0)
          popped += 1
      except Exception as e:
        self.log.debug("Failure in clearing branches: ", exc_info=e)
      self.log.debug("Popped %d from branch %d", popped, i)

  # choses the branch with the next lowest cost. This is where exploration could happen
  def select_branch(self)->int:
    cheapest_w = float('inf')
    cheapest_i:int = 0
    for i, branch in enumerate(self.branches):
      if branch[0][1] < cheapest_w:
        cheapest_w = branch[0][1]
        cheapest_i = i
    self.log.debug("Selected branch %d with weight %.4f", cheapest_i, cheapest_w)
    return cheapest_i
  
  # perform other functions automatically until the search is complete
  def search(self)->list:
    start = time.perf_counter()
    while len(self.path) < self.n_count:
      self.clear_dead_limbs()
      next_branch = self.select_branch()
      next_node, next_weight = self.branches[next_branch].pop(0)  #grab the front of the chosen branch
      self.add_edges_to_branch(next_branch, next_node)
    self.log.info("Total search time %.4f", time.perf_counter()-start)
    return self.path
  
  def reset(self):
    self.log.info("RESETTING SEARCH ENTIRELY")
    self.setup()
#endregion DDFS Class

#region GA Class
class GraphGA:
  def __init__(self, graph, path_list, graph_num:int=0, logger=None):
    self.graph = graph
    self.path_list = path_list
    self.gene_range = sorted(nx.nodes(graph)) #range(1, nx.number_of_nodes(self.graph) + 1)
    self.log:logging.Logger = logger if logger is not None else LogFileMaker.create_logger("/".join(["./logs","_".join([datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),"graph",str(graph_num),"ga.txt"])]))

  def close_log(self):
    for handler in self.log.handlers:
      self.log.removeHandler(handler)

  def path_fitness(self, ga_instance: pygad.GA, solution, solution_idx) -> float:
    return (2.0 * len(solution)) / nx.path_weight(self.graph, solution, 'weight')
  
  def run_ga(self):
    self.ga.run()
  
  #region GA.Crossovers
  def order_crossover(self, parents, offspring_size, ga_instance):
    offspring = []

    num_genes = offspring_size[1]
    idx = 0

    while len(offspring) < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0]]
        parent2 = parents[(idx + 1) % parents.shape[0]]
        idx += 1

        # Choose cut points
        c1, c2 = sorted(np.random.choice(range(num_genes), 2, replace=False))

        child = [-1] * num_genes

        # Copy slice from parent1
        child[c1:c2] = parent1[c1:c2]

        # Fill remaining genes from parent2
        p2_idx = 0
        for i in range(num_genes):
            if child[i] == -1:
                while parent2[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
                p2_idx += 1

        offspring.append(child)

    return np.array(offspring)
  
  def fast_edge_recombination_crossover(self, parents, offspring_size, ga_instance):
    offspring = []
    num_genes = offspring_size[1]

    while len(offspring) < offspring_size[0]:
        p1, p2 = random.sample(list(parents), 2)

        # Build adjacency lists
        edge_map = {gene: set() for gene in p1}
        for p in (p1, p2):
            for i in range(num_genes):
                if i > 0:
                    edge_map[p[i]].add(p[i - 1])
                if i < num_genes - 1:
                    edge_map[p[i]].add(p[i + 1])

        unused = set(p1)

        current = random.choice(p1)
        child = [current]
        unused.remove(current)

        while unused:
            neighbors = edge_map[current] & unused

            if neighbors:
                # Choose neighbor with smallest adjacency list
                next_node = min(neighbors, key=lambda x: len(edge_map[x]))
            else:
                next_node = min(unused, key=lambda x: self.graph[current][x]['weight'])

            child.append(next_node)
            unused.remove(next_node)

            # Remove chosen node from its neighbors only
            for n in edge_map[next_node]:
                edge_map[n].discard(next_node)

            current = next_node

        offspring.append(child)

    return np.array(offspring)
  #endregion GA.Crossovers
  
  #region GA.Main
  def reset_ga(self, n_gens: int=5, n_par_mate: int=120,
        parent_keep: int=0, n_elites: int=2, #if n_elites != 0, then parent_keep is ignored in GA
        mut:str='inversion', mut_prob:float=0.4,
        cross_prob:float=0.2, #doesn't really matter with custom crossovers
        cross_type:str='edge_recomb',
        parent_choice:str='tournament', tour_k:int = 3,
        init_pop:bool=True, sol_per_pop:int=30):

    #convert string parameter to class type
    if cross_type == 'edge_recomb':
      crossover = self.fast_edge_recombination_crossover
    elif cross_type == 'order':
      crossover = self.order_crossover
    else:
      crossover = 'single_point'

    #clear the starting population if it isn't to be used
    if init_pop is False:
      start_pop = None
    else:
      start_pop = self.path_list

    self.ga = pygad.GA(num_generations=n_gens,
                      num_parents_mating=n_par_mate,
                      crossover_probability=cross_prob,
                      parent_selection_type=parent_choice,
                      K_tournament=tour_k,
                      mutation_type=mut,
                      mutation_probability=mut_prob,
                      keep_parents=parent_keep,
                      keep_elitism=n_elites,
                      #class values
                      crossover_type=crossover, #type: ignore #locked from the class
                      fitness_func=self.path_fitness, #locked from the class
                      gene_space=list(self.gene_range), #locked from the class
                      initial_population=start_pop,  #locked from the class
                      sol_per_pop=sol_per_pop,
                      num_genes=len(self.gene_range),
                      #non-default values
                      allow_duplicate_genes=False, #non-default, set and forget
                      gene_type=int,   #default, set and forget
                      #default values
                      on_generation=self.on_generation,
                      on_start=self.on_start,
                      on_crossover=self.on_crossover,
                      on_fitness=self.on_fitness,
                      on_parents=self.on_parents,
                      on_mutation=self.on_mutation,
                      on_stop=self.on_stop
                      )
  
  def give_solution(self):
    solution, solution_fitness, solution_idx = self.ga.best_solution()
    self.log.debug(f"Parameters of the best solution : {solution}")
    self.log.info(f"Fitness value of the best solution = {solution_fitness}")
    self.log.info(f"Index of the best solution : {solution_idx}")
    self.log.info(f"Weight of the solution = {nx.path_weight(self.graph, solution, 'weight')}")
    return self.ga.best_solution()
  #endregion GA.Main

  #region GA.On-Functions
  def on_start(self, ga_instance):
      self.log.info("Starting GA search")

  def on_fitness(self, ga_instance, population_fitness):
      self.log.info("Computed fitness")

  def on_parents(self, ga_instance, selected_parents):
      self.log.info("Selected parents")

  def on_crossover(self, ga_instance, offspring_crossover):
      self.log.info("Performed crossovers")

  def on_mutation(self, ga_instance, offspring_mutation):
      self.log.info("Mutated")

  def on_stop(self, ga_instance, last_population_fitness):
      self.log.info("Ending GA search")
      
  def on_generation(self, ga_instance:pygad.GA):
      self.log.info(ga_instance.generations_completed)
      self.log.info(ga_instance.best_solution()[1]) #fitness
      self.log.debug(ga_instance.population)
  #endregion GA.On-Functions
#endregion GA Class

class GraphTraversalManager:
  def __init__(self, nx_graph, hamiltonian_path, new_points, expected_len:float=0.8, comp_weights:list=[1,1,1], fit_len:float=1, fit_comp:float=1, graph_num:int=0, logger=None):
    self.log:logging.Logger = logger if logger is not None else LogFileMaker.create_logger("_".join([datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),"graph",str(graph_num),"ga.txt"]))
    self.graph = nx_graph
    self.original_path = hamiltonian_path
    self.new_points = new_points
    self.expected_len = expected_len
    self.composition_weights = comp_weights
    self.fitness_len_factor = fit_len
    self.fitness_comp_factor = fit_comp
    
    self.registry = {}
    self.next_gene_id = 1

    self._build_registry()
    
    self.gene_range = sorted(self._get_initial_chromosome())

    self.session = wls(kernel="C:\\Program Files\\Wolfram Research\\Wolfram\\14.3\\WolframKernel.exe")
    self.session.evaluate(wlexpr('Get["./compModels/CompositionErrorEstimation.wl"]'))
    self.session.evaluate(wl.CompositionErrorEstimation.InitializeModel("./compModels",28.0156,0.2,0.8))

  def close_log(self):
    for handler in self.log.handlers:
      self.log.removeHandler(handler)

  def close_wls(self, graceful:bool=True):
    if not graceful:
      self.session.terminate()
    self.session.stop_future(gracefully=graceful)

  def run_ga(self):
    self.ga.run()

  def _build_registry(self):
        """Processes the graph, path, and new points to build the 1D dictionary registry."""
        current_chunk = []
        
        for node in self.original_path:
            is_natural_external = not self.graph.nodes[node]['internal']
            current_chunk.append(node)
            
            # We only finalize a chunk if we hit a natural external
            # AND it isn't the very first node of a fresh chunk.
            if is_natural_external and len(current_chunk) > 1:
                
                if len(current_chunk) == 2:
                    # Pattern: [BoundaryNode, NaturalExternal]
                    # Example: [E1, E2] OR the [I2, E3] case you described.
                    # The BoundaryNode stands alone as its own distinct gene.
                    prev_node = current_chunk[0]
                    self._add_to_registry('standalone', [prev_node])
                    
                    # The current natural external starts the next chunk
                    current_chunk = [node] 
                    
                else:
                    # Pattern: [BoundaryNode, Internal(s)..., NaturalExternal]
                    # Example: [E1, I1, E2] OR [I2, I3, E3]
                    # This is a standard compressed sequence.
                    self._add_to_registry('compressed', current_chunk)
                    
                    # CRITICAL STEP: By completely clearing the chunk, the NEXT node in the path 
                    # (even if it's internal like I2) will be forced to act as the BoundaryNode 
                    # for the next sequence. This ensures E2 is consumed only once.
                    current_chunk = [] 

        # Catch any trailing nodes at the end of the Hamiltonian path
        # If the path ended exactly on a compressed block, current_chunk is []
        if len(current_chunk) > 0:
            if len(current_chunk) == 1:
                # Ends on a single standalone node
                self._add_to_registry('standalone', current_chunk)
            else:
                # Edge case: If the path ends with [BoundaryNode, Internal(s)...]
                self._add_to_registry('compressed', current_chunk)

        # 2. Add New External Points to the Registry
        for point in self.new_points:
            self.registry[self.next_gene_id] = {
                'type': 'new',
                'start_pos': point,
                'end_pos': point,
                'start_comp': None, 
                'end_comp': None,
                'internal_path': [],
                'allow_reverse': False
            }
            self.next_gene_id += 1

  def _add_to_registry(self, gene_type, path):
      """Helper to extract node attributes and insert them into the registry."""
      start_node = path[0]
      end_node = path[-1]
      
      self.registry[self.next_gene_id] = {
          'type': gene_type,
          'start_pos': self.graph.nodes[start_node]['pos'],
          'end_pos': self.graph.nodes[end_node]['pos'],
          'start_comp': self.graph.nodes[start_node]['comps'],
          'end_comp': self.graph.nodes[end_node]['comps'],
          'internal_path': path,
          'allow_reverse': True if gene_type == 'compressed' else False
      }
      self.next_gene_id += 1

  def _get_initial_chromosome(self):
      """Returns the list of all gene IDs to seed PyGAD's initial population."""
      return list(self.registry.keys())

  # def get_unassigned_genes(self, chromosome):
  #     """
  #     Returns a list of gene IDs in the chromosome that need composition assignment.
  #     Useful for the fitness function before calling the Wolfram package.
  #     """
  #     return [gene_id for gene_id in chromosome if self.registry[gene_id]['type'] == 'new_point']

  # def calculate_distance(self, chromosome):
  #     """
  #     Calculates the total Euclidean distance of the chromosome sequence.
  #     O(N) time complexity using dictionary lookups.
  #     """
  #     total_distance = 0.0
      
  #     for i in range(len(chromosome) - 1):
  #         current_gene = self.registry[chromosome[i]]
  #         next_gene = self.registry[chromosome[i+1]]
          
  #         # Note: If directionality flipping is enabled in the future, 
  #         # you would check the GA's orientation flag for current_gene/next_gene here
  #         # to decide whether to use 'start_pos' or 'end_pos'.
          
  #         point1 = current_gene['end_pos']
  #         point2 = next_gene['start_pos']
          
  #         total_distance += dist(point1, point2)
          
  #     return total_distance

  def _build_evaluation_paths(self, chromosome, registry):
    """
    Parses a chromosome into the desired and shifted composition paths 
    using a spatial FIFO queue for buffer nodes.
    """
    cumulative_length = 0.0
    slots = []

    # 1. Forward Pass: Calculate cumulative lengths and flatten into 'slots'
    for i in range(len(chromosome)):
        gene_id = chromosome[i]
        gene = registry[gene_id]

        gene_internal_len = dist(gene['start_pos'], gene['end_pos'])

        if i > 0:
            prev_gene = registry[chromosome[i-1]]
            gap_len = dist(prev_gene['end_pos'], gene['start_pos'])
            cumulative_length += gap_len

        start_len = cumulative_length
        end_len = cumulative_length + gene_internal_len

        if gene['type'] == 'new':
            # A blank buffer slot
            slots.append({'len': start_len, 'comp': None, 'is_fixed': False})
        else:
            # A fixed target. All original nodes have a start composition.
            slots.append({'len': start_len, 'comp': gene['start_comp'], 'is_fixed': True})
            
            # Compressed blocks have an end composition at a different length.
            # We treat the end of a compressed block as another fixed target in the queue.
            if gene['type'] == 'compressed' and start_len != end_len:
                slots.append({'len': end_len, 'comp': gene['end_comp'], 'is_fixed': True})

        cumulative_length += gene_internal_len

    # 2. Build the Desired Path (The Physical Target Truth)
    desired_path = []
    for slot in slots:
        if slot['is_fixed'] and slot['comp'] is not None:
            desired_path.append([slot['len'], *slot['comp']])

    # 3. Build the Shifted Path (FIFO Queuing)
    shifted_path = []
    blocks = []
    current_buffer = []
    current_fixed = []
    state = 'buffer'

    # Group the slots into continuous blocks of [Buffers...] -> [Fixed Targets...]
    for slot in slots:
        if not slot['is_fixed']:
            if state == 'fixed':
                # We finished a block, save it and start a new one
                blocks.append({'buffer': current_buffer, 'fixed': current_fixed})
                current_buffer = []
                current_fixed = []
                state = 'buffer'
            current_buffer.append(slot)
        else:
            state = 'fixed'
            current_fixed.append(slot)

    # Append the final block
    if current_buffer or current_fixed:
        blocks.append({'buffer': current_buffer, 'fixed': current_fixed})

    # 4. Process the shifts for each block
    for block in blocks:
        buffers = block['buffer']
        fixeds = block['fixed']

        # Determine how many commands we can actually shift
        num_to_shift = min(len(buffers), len(fixeds))

        # A. Shift the first N commands into the available buffers
        for i in range(num_to_shift):
            shifted_path.append([buffers[i]['len'], *fixeds[i]['comp']])

        # Note: Any remaining buffer nodes (len(buffers) > len(fixeds)) 
        # receive nothing. They just inherit the Wolfram step function's previous state.

        # B. If we ran out of buffers, the remaining commands stay at their original coordinates
        for i in range(num_to_shift, len(fixeds)):
            shifted_path.append([fixeds[i]['len'], *fixeds[i]['comp']])

    return shifted_path, desired_path, cumulative_length
  
  def _fast_edge_recombination_crossover(self, parents, offspring_size, ga_instance):
    offspring = []
    num_genes = offspring_size[1]

    while len(offspring) < offspring_size[0]:
        p1, p2 = random.sample(list(parents), 2)

        # Build adjacency lists
        edge_map = {gene: set() for gene in p1}
        for p in (p1, p2):
            for i in range(num_genes):
                if i > 0:
                    edge_map[p[i]].add(p[i - 1])
                if i < num_genes - 1:
                    edge_map[p[i]].add(p[i + 1])

        unused = set(p1)

        current = random.choice(p1)
        child = [current]
        unused.remove(current)

        while unused:
            neighbors = edge_map[current] & unused

            if neighbors:
                # Choose neighbor with smallest adjacency list
                next_node = min(neighbors, key=lambda x: len(edge_map[x]))
            else:
                next_node = min(unused, key=lambda x: dist(self.registry[current]['end_pos'],self.registry[x]['start_pos']))

            child.append(next_node)
            unused.remove(next_node)

            # Remove chosen node from its neighbors only
            for n in edge_map[next_node]:
                edge_map[n].discard(next_node)

            current = next_node

        offspring.append(child)

    return np.array(offspring)
  
  def _path_fitness(self, ga_instance: pygad.GA, solution, solution_idx) -> float:
    shifted_path, desired_path, cumulative_len = self._build_evaluation_paths(solution, self.registry)
    errorByComp = list(self.session.evaluate(wl.CompositionErrorEstimation.EvaluatePathErrorWithEmpty(shifted_path,desired_path)))
    return self._total_fitness(cumulative_len, errorByComp, len(solution))
  
  def _total_fitness(self, cum_len, ind_errors, num_nodes):
    total_comp_error = 0
    for i, error in enumerate(ind_errors):
       total_comp_error += error * self.composition_weights[i]
    combined_cost = (self.fitness_len_factor * (cum_len / (num_nodes * self.expected_len))) + (self.fitness_comp_factor * total_comp_error)
    return (1 / combined_cost)
  
  def _batch_fitness(self, ga_instance: pygad.GA, solutions, solution_indices):
    batch_shifted = []
    batch_desired = []
    batch_lengths = []
    
    # 1. Parse all chromosomes in the generation
    for solution in solutions:
        shifted, desired, total_len = self._build_evaluation_paths(solution, self.registry)
        batch_shifted.append(shifted)
        batch_desired.append(desired)
        batch_lengths.append(total_len)
        
    # 2. Call the Wolfram Model ONCE for the entire batch
    # Assumes 'wl' is your active Wolfram Language session
    # and the WL function is modified to return a list of scalar error sums.
    try:
        population_errors = self.session.evaluate(
            wl.CompositionErrorEstimation.BatchEvalPathWEmpty(batch_shifted, batch_desired)
        )
    except Exception as e:
        # Handle WL errors by penalizing the batch
        print(f"Wolfram Evaluation Failed: {e}")
        return [-9999] * len(solutions)

    # 3. Calculate final combined fitness for each solution
    fitness_scores = []
    num_genes = len(solutions[0])
    
    for i in range(len(solutions)):
        fitness_scores.append(self._total_fitness(batch_lengths[i],population_errors[i],num_genes))
        
    return fitness_scores
  
  def reset_ga(self, n_gens: int=5, n_par_mate: int=120,
        parent_keep: int=0, n_elites: int=2, #if n_elites != 0, then parent_keep is ignored in GA
        mut:str='inversion', mut_prob:float=0.4,
        cross_prob:float=0.2, #doesn't really matter with custom crossovers
        cross_type:str='edge_recomb',
        parent_choice:str='tournament', tour_k:int = 3,
        sol_per_pop:int=30):

    #convert string parameter to class type
    if cross_type == 'edge_recomb':
      crossover = self._fast_edge_recombination_crossover
    else:
      crossover = 'single_point'

    self.ga = pygad.GA(num_generations=n_gens,
                      num_parents_mating=n_par_mate,
                      crossover_probability=cross_prob,
                      parent_selection_type=parent_choice,
                      K_tournament=tour_k,
                      mutation_type=mut,
                      mutation_probability=mut_prob,
                      keep_parents=parent_keep,
                      keep_elitism=n_elites,
                      #class values
                      crossover_type=crossover, #type: ignore #locked from the class
                      fitness_func=self._batch_fitness, #locked from the class
                      fitness_batch_size=sol_per_pop, #locked from the class
                      gene_space=list(self.gene_range), #locked from the class
                      initial_population=[sorted(self._get_initial_chromosome())]*sol_per_pop,  #locked from the class
                      sol_per_pop=sol_per_pop,
                      num_genes=len(self.gene_range),
                      #non-default values
                      allow_duplicate_genes=False, #non-default, set and forget
                      gene_type=int,   #default, set and forget
                      #default values
                      on_generation=self.on_generation,
                      on_start=self.on_start,
                      on_crossover=self.on_crossover,
                      on_fitness=self.on_fitness,
                      on_parents=self.on_parents,
                      on_mutation=self.on_mutation,
                      on_stop=self.on_stop
                      )
  
  def give_solution(self):
    solution, solution_fitness, solution_idx = self.ga.best_solution()
    self.log.debug(f"Parameters of the best solution : {solution}")
    self.log.info(f"Fitness value of the best solution = {solution_fitness}")
    self.log.info(f"Index of the best solution : {solution_idx}")
    # self.log.info(f"Length of the solution = {}")
    return self.ga.best_solution()
  #endregion GA.Main

  #region GA.On-Functions
  def on_start(self, ga_instance):
      self.log.info("Starting GA search")

  def on_fitness(self, ga_instance, population_fitness):
      self.log.info("Computed fitness")

  def on_parents(self, ga_instance, selected_parents):
      self.log.info("Selected parents")

  def on_crossover(self, ga_instance, offspring_crossover):
      self.log.info("Performed crossovers")

  def on_mutation(self, ga_instance, offspring_mutation):
      self.log.info("Mutated")

  def on_stop(self, ga_instance, last_population_fitness):
      self.log.info("Ending GA search")
      
  def on_generation(self, ga_instance:pygad.GA):
      self.log.info(ga_instance.generations_completed)
      self.log.info(ga_instance.best_solution()[1]) #fitness
      self.log.debug(ga_instance.population)
  #endregion GA.On-Functions


class LightGTM:
  def __init__(self, nx_graph, hamiltonian_path, new_points, expected_len:float=0.8, comp_weights:list=[1,1,1], fit_len:float=1, fit_comp:float=1, graph_num:int=0, logger=None):
    self.log:logging.Logger = logger if logger is not None else LogFileMaker.create_logger("_".join([datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),"graph",str(graph_num),"ga.txt"]))
    self.graph = nx_graph
    self.original_path = hamiltonian_path
    self.new_points = new_points
    self.expected_len = expected_len
    self.composition_weights = comp_weights
    self.fitness_len_factor = fit_len
    self.fitness_comp_factor = fit_comp
    
    self.registry = {}
    self.next_gene_id = 1

    self._build_registry()
    
    self.gene_range = sorted(self._get_initial_chromosome())

    self.constraints = self._build_constraint_rules()

    self.session = wls(kernel="C:\\Program Files\\Wolfram Research\\Wolfram\\14.3\\WolframKernel.exe")
    self.session.evaluate(wlexpr('Get["./compModels/CompositionErrorEstimation.wl"]'))
    self.session.evaluate(wl.CompositionErrorEstimation.InitializeModel("./compModels",28.0156,0.2,0.8))

  def close_log(self):
    for handler in self.log.handlers:
      self.log.removeHandler(handler)

  def close_wls(self, graceful:bool=True):
    if not graceful:
      self.session.terminate()
    self.session.stop_future(gracefully=graceful)

  def run_ga(self):
    self.ga.run()

  def _build_constraint_rules(self):
    rules:list = [None]
    for i in range(len(self.gene_range)-1):
      rules.append(lambda sol,values: [val for val in values if dist(self.registry[val]['start_pos'], self.registry[sol[i]]['end_pos'])<5])
    return rules

  def _build_registry(self):
        """Processes the graph, path, and new points to build the 1D dictionary registry."""
        current_chunk = []
        
        for node in self.original_path:
            is_natural_external = not self.graph.nodes[node]['internal']
            current_chunk.append(node)
            
            # We only finalize a chunk if we hit a natural external
            # AND it isn't the very first node of a fresh chunk.
            if is_natural_external and len(current_chunk) > 1:
                
                if len(current_chunk) == 2:
                    # Pattern: [BoundaryNode, NaturalExternal]
                    # Example: [E1, E2] OR the [I2, E3] case you described.
                    # The BoundaryNode stands alone as its own distinct gene.
                    prev_node = current_chunk[0]
                    self._add_to_registry('standalone', [prev_node])
                    
                    # The current natural external starts the next chunk
                    current_chunk = [node] 
                    
                else:
                    # Pattern: [BoundaryNode, Internal(s)..., NaturalExternal]
                    # Example: [E1, I1, E2] OR [I2, I3, E3]
                    # This is a standard compressed sequence.
                    self._add_to_registry('compressed', current_chunk)
                    
                    # CRITICAL STEP: By completely clearing the chunk, the NEXT node in the path 
                    # (even if it's internal like I2) will be forced to act as the BoundaryNode 
                    # for the next sequence. This ensures E2 is consumed only once.
                    current_chunk = [] 

        # Catch any trailing nodes at the end of the Hamiltonian path
        # If the path ended exactly on a compressed block, current_chunk is []
        if len(current_chunk) > 0:
            if len(current_chunk) == 1:
                # Ends on a single standalone node
                self._add_to_registry('standalone', current_chunk)
            else:
                # Edge case: If the path ends with [BoundaryNode, Internal(s)...]
                self._add_to_registry('compressed', current_chunk)

        # 2. Add New External Points to the Registry
        for point in self.new_points:
            self.registry[self.next_gene_id] = {
                'type': 'new',
                'start_pos': point,
                'end_pos': point,
                'start_comp': None, 
                'end_comp': None,
                'internal_path': [],
                'allow_reverse': False
            }
            self.next_gene_id += 1

  def _add_to_registry(self, gene_type, path):
      """Helper to extract node attributes and insert them into the registry."""
      start_node = path[0]
      end_node = path[-1]
      
      self.registry[self.next_gene_id] = {
          'type': gene_type,
          'start_pos': self.graph.nodes[start_node]['pos'],
          'end_pos': self.graph.nodes[end_node]['pos'],
          'start_comp': self.graph.nodes[start_node]['comps'],
          'end_comp': self.graph.nodes[end_node]['comps'],
          'internal_path': path,
          'allow_reverse': True if gene_type == 'compressed' else False
      }
      self.next_gene_id += 1

  def _get_initial_chromosome(self):
      """Returns the list of all gene IDs to seed PyGAD's initial population."""
      return list(self.registry.keys())

  # def get_unassigned_genes(self, chromosome):
  #     """
  #     Returns a list of gene IDs in the chromosome that need composition assignment.
  #     Useful for the fitness function before calling the Wolfram package.
  #     """
  #     return [gene_id for gene_id in chromosome if self.registry[gene_id]['type'] == 'new_point']

  # def calculate_distance(self, chromosome):
  #     """
  #     Calculates the total Euclidean distance of the chromosome sequence.
  #     O(N) time complexity using dictionary lookups.
  #     """
  #     total_distance = 0.0
      
  #     for i in range(len(chromosome) - 1):
  #         current_gene = self.registry[chromosome[i]]
  #         next_gene = self.registry[chromosome[i+1]]
          
  #         # Note: If directionality flipping is enabled in the future, 
  #         # you would check the GA's orientation flag for current_gene/next_gene here
  #         # to decide whether to use 'start_pos' or 'end_pos'.
          
  #         point1 = current_gene['end_pos']
  #         point2 = next_gene['start_pos']
          
  #         total_distance += dist(point1, point2)
          
  #     return total_distance

  def _build_evaluation_paths(self, chromosome, registry):
    """
    Parses a chromosome into the desired and shifted composition paths 
    using a spatial FIFO queue for buffer nodes.
    """
    cumulative_length = 0.0
    slots = []

    # 1. Forward Pass: Calculate cumulative lengths and flatten into 'slots'
    for i in range(len(chromosome)):
        gene_id = chromosome[i]
        gene = registry[gene_id]

        gene_internal_len = dist(gene['start_pos'], gene['end_pos'])

        if i > 0:
            prev_gene = registry[chromosome[i-1]]
            gap_len = dist(prev_gene['end_pos'], gene['start_pos'])
            cumulative_length += gap_len

        start_len = cumulative_length
        end_len = cumulative_length + gene_internal_len

        # if gene['type'] == 'new':
        #     # A blank buffer slot
        #     slots.append({'len': start_len, 'comp': None, 'is_fixed': False})
        # else:
        #     # A fixed target. All original nodes have a start composition.
        #     slots.append({'len': start_len, 'comp': gene['start_comp'], 'is_fixed': True})
            
        #     # Compressed blocks have an end composition at a different length.
        #     # We treat the end of a compressed block as another fixed target in the queue.
        #     if gene['type'] == 'compressed' and start_len != end_len:
        #         slots.append({'len': end_len, 'comp': gene['end_comp'], 'is_fixed': True})

        cumulative_length += gene_internal_len

    # 2. Build the Desired Path (The Physical Target Truth)
    # desired_path = []
    # for slot in slots:
    #     if slot['is_fixed'] and slot['comp'] is not None:
    #         desired_path.append([slot['len'], *slot['comp']])

    # 3. Build the Shifted Path (FIFO Queuing)
    # shifted_path = []
    # blocks = []
    # current_buffer = []
    # current_fixed = []
    # state = 'buffer'

    # Group the slots into continuous blocks of [Buffers...] -> [Fixed Targets...]
    # for slot in slots:
    #     if not slot['is_fixed']:
    #         if state == 'fixed':
    #             # We finished a block, save it and start a new one
    #             blocks.append({'buffer': current_buffer, 'fixed': current_fixed})
    #             current_buffer = []
    #             current_fixed = []
    #             state = 'buffer'
    #         current_buffer.append(slot)
    #     else:
    #         state = 'fixed'
    #         current_fixed.append(slot)

    # Append the final block
    # if current_buffer or current_fixed:
    #     blocks.append({'buffer': current_buffer, 'fixed': current_fixed})

    # 4. Process the shifts for each block
    # for block in blocks:
    #     buffers = block['buffer']
    #     fixeds = block['fixed']

    #     # Determine how many commands we can actually shift
    #     num_to_shift = min(len(buffers), len(fixeds))

    #     # A. Shift the first N commands into the available buffers
    #     for i in range(num_to_shift):
    #         shifted_path.append([buffers[i]['len'], *fixeds[i]['comp']])

    #     # Note: Any remaining buffer nodes (len(buffers) > len(fixeds)) 
    #     # receive nothing. They just inherit the Wolfram step function's previous state.

    #     # B. If we ran out of buffers, the remaining commands stay at their original coordinates
    #     for i in range(num_to_shift, len(fixeds)):
    #         shifted_path.append([fixeds[i]['len'], *fixeds[i]['comp']])

    return cumulative_length #shifted_path, desired_path, cumulative_length
  
  def _fast_edge_recombination_crossover(self, parents, offspring_size, ga_instance):
    offspring = []
    num_genes = offspring_size[1]

    while len(offspring) < offspring_size[0]:
        p1, p2 = random.sample(list(parents), 2)

        # Build adjacency lists
        edge_map = {gene: set() for gene in p1}
        for p in (p1, p2):
            for i in range(num_genes):
                if i > 0:
                    edge_map[p[i]].add(p[i - 1])
                if i < num_genes - 1:
                    edge_map[p[i]].add(p[i + 1])

        unused = set(p1)

        current = random.choice(p1)
        child = [current]
        unused.remove(current)

        while unused:
            neighbors = edge_map[current] & unused

            if neighbors:
                # Choose neighbor with smallest adjacency list
                next_node = min(neighbors, key=lambda x: len(edge_map[x]))
            else:
                next_node = min(unused, key=lambda x: dist(self.registry[current]['end_pos'],self.registry[x]['start_pos']))

            child.append(next_node)
            unused.remove(next_node)

            # Remove chosen node from its neighbors only
            for n in edge_map[next_node]:
                edge_map[n].discard(next_node)

            current = next_node

        offspring.append(child)

    return np.array(offspring)
  
  def _path_fitness(self, ga_instance: pygad.GA, solution, solution_idx) -> float:
    cumulative_len = self._build_evaluation_paths(solution, self.registry)
    # errorByComp = list(self.session.evaluate(wl.CompositionErrorEstimation.EvaluatePathErrorWithEmpty(shifted_path,desired_path)))
    return self._total_fitness(cumulative_len, len(solution))
  
  def _total_fitness(self, cum_len, num_nodes):
    total_comp_error = 0
    # for i, error in enumerate(ind_errors):
      #  total_comp_error += error * self.composition_weights[i]
    combined_cost = (self.fitness_len_factor * (cum_len**2 / (num_nodes * self.expected_len))) + (self.fitness_comp_factor * total_comp_error)
    return (1 / combined_cost)
  
  def _batch_fitness(self, ga_instance: pygad.GA, solutions, solution_indices):
    # batch_shifted = []
    # batch_desired = []
    batch_lengths = []
    
    # 1. Parse all chromosomes in the generation
    for solution in solutions:
        total_len = self._build_evaluation_paths(solution, self.registry)
        # batch_shifted.append(shifted)
        # batch_desired.append(desired)
        batch_lengths.append(total_len)
        
    # 2. Call the Wolfram Model ONCE for the entire batch
    # Assumes 'wl' is your active Wolfram Language session
    # and the WL function is modified to return a list of scalar error sums.
    # try:
    #     population_errors = self.session.evaluate(
    #         wl.CompositionErrorEstimation.BatchEvalPathWEmpty(batch_shifted, batch_desired)
    #     )
    # except Exception as e:
    #     # Handle WL errors by penalizing the batch
    #     print(f"Wolfram Evaluation Failed: {e}")
    #     return [-9999] * len(solutions)

    # 3. Calculate final combined fitness for each solution
    fitness_scores = []
    num_genes = len(solutions[0])
    
    for i in range(len(solutions)):
        fitness_scores.append(self._total_fitness(batch_lengths[i],num_genes))
        
    return fitness_scores
  
  def reset_ga(self, n_gens: int=5, n_par_mate: int=20,
        parent_keep: int=0, n_elites: int=5, #if n_elites != 0, then parent_keep is ignored in GA
        mut:str='inversion', mut_prob:float=0.4,
        cross_prob:float=0.2, #doesn't really matter with custom crossovers
        cross_type:str='edge_recomb',
        parent_choice:str='tournament', tour_k:int = 3,
        sol_per_pop:int=30):

    #convert string parameter to class type
    if cross_type == 'edge_recomb':
      crossover = self._fast_edge_recombination_crossover
    else:
      crossover = 'single_point'

    self.ga = pygad.GA(num_generations=n_gens,
                      num_parents_mating=n_par_mate,
                      crossover_probability=cross_prob,
                      parent_selection_type=parent_choice,
                      K_tournament=tour_k,
                      mutation_type=mut,
                      mutation_probability=mut_prob,
                      keep_parents=parent_keep,
                      keep_elitism=n_elites,
                      #class values
                      crossover_type=crossover, #type: ignore #locked from the class
                      fitness_func=self._batch_fitness, #locked from the class
                      fitness_batch_size=sol_per_pop, #locked from the class
                      gene_space=list(self.gene_range), #locked from the class
                      initial_population=[self.gene_range]*sol_per_pop,  #locked from the class
                      sol_per_pop=sol_per_pop,
                      num_genes=len(self.gene_range),
                      #non-default values
                      allow_duplicate_genes=False, #non-default, set and forget
                      gene_type=int,   #default, set and forget
                      #default values
                      on_generation=self.on_generation,
                      on_start=self.on_start,
                      on_crossover=self.on_crossover,
                      on_fitness=self.on_fitness,
                      on_parents=self.on_parents,
                      on_mutation=self.on_mutation,
                      on_stop=self.on_stop
                      )

  def give_solution(self):
    solution, solution_fitness, solution_idx = self.ga.best_solution()
    self.log.debug(f"Parameters of the best solution : {solution}")
    self.log.info(f"Fitness value of the best solution = {solution_fitness}")
    self.log.info(f"Index of the best solution : {solution_idx}")
    # self.log.info(f"Length of the solution = {}")
    return self.ga.best_solution()
  #endregion GA.Main

  #region GA.On-Functions
  def on_start(self, ga_instance):
      self.log.info("Starting GA search")

  def on_fitness(self, ga_instance, population_fitness):
      self.log.info("Computed fitness")

  def on_parents(self, ga_instance, selected_parents):
      self.log.info("Selected parents")

  def on_crossover(self, ga_instance, offspring_crossover):
      self.log.info("Performed crossovers")

  def on_mutation(self, ga_instance, offspring_mutation):
      self.log.info("Mutated")

  def on_stop(self, ga_instance, last_population_fitness):
      self.log.info("Ending GA search")
      
  def on_generation(self, ga_instance:pygad.GA):
      self.log.info(ga_instance.generations_completed)
      self.log.info(ga_instance.best_solution()[1]) #fitness
      self.log.debug(ga_instance.population)
  #endregion GA.On-Functions