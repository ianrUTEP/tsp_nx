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

#region Load Graphs
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
    for row in df.itertuples(index=False):
      # each row in dataframe represents a graph, turn into graph object
      graph_list.append(complete_graph_from_row(row))
  return graph_list

def complete_graph_from_row(row: tuple) -> nx.Graph:
  G = nx.Graph()
  for node_id, node_data in getattr(row, 'nodes').items():
    G.add_node(int(node_id), pos=(node_data['pos'][0], node_data['pos'][1]), group=node_data['streamline'][0]) # streamline number becomes group
  for edge_str, edge_data,  in getattr(row, 'edges').items():
    u_str, v_str = edge_str.split(',')
    G.add_edge(int(u_str), int(v_str), length=edge_data['len'], alignment=edge_data['align']) #no weight given yet
  return G
#endregion Load Graphs

#region Visualization
def make_solution_html(graph_list, canvas_height, sol_list: list, vis_opt_dict: dict = {}, color_scale_attr: str = 'weight'):
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  for i, graph in enumerate(graph_list):
    print("making solution copy of graph", i)
    vis_graph: nx.Graph = nx.create_empty_copy(graph)
    edges = [(sol_list[i][j], sol_list[i][j+1]) for j in range(len(sol_list[i])-1)]
    for u,v in edges:
      vis_graph.add_edge(u, v, weight=graph.edges[u,v]['weight'], length=graph.edges[u,v]['length'], alignment=graph.edges[u,v]['alignment'])
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
  return mcolors.to_hex(colormap(normalizer(value)))
#endregion Visualization
      
#region Greedy Solvers
def solve_graphs_greedy(graph_list):
  solution_list = []
  print("Beginning search for solutions")
  for i, graph in enumerate(graph_list):
    print("Solving graph", i)
    sol = nx.approximation.greedy_tsp(graph)
    solution_list.append(sol)
  return solution_list

def solve_graphs_multgreedy(graph_list, n_greedys:int = 10, guaranteed:int = 1)->list:
  solution_list = []
  print("Beginning search for solutions")
  for i, graph in enumerate(graph_list):
    graph_sols = []
    sources = np.insert(random.sample(sorted(nx.nodes(graph)), n_greedys-1), 0, guaranteed) #plus one to limit to include, generate n-1 and add 1 as guaranteed source
    for j, source in enumerate(sources):
      print("graph", i, "sol", j)
      graph_sols.append(nx.approximation.greedy_tsp(graph, source=int(source)))
    solution_list.append(graph_sols)
  return solution_list

def cycle_sol_to_path_simple(sol_list:list)->list:
  return [[cycle[:-1] if cycle else cycle for cycle in graph] for graph in sol_list]

def cycle_to_path(sol_list:list, graph_list:list)->list:
  all_path_sols = []
  for i, solutions in enumerate(sol_list):
    graph_path_sols = []
    graph:nx.Graph = graph_list[i]
    for cycle in solutions: #for each given cycle, identify the largest edge position by weight value
      largest_edge_pos = 0
      largest_edge_val = 0
      for j in range(len(cycle) - 1):
        if graph[cycle[j]][cycle[j+1]]['weight'] > largest_edge_val:
          largest_edge_pos = j
          largest_edge_val = graph[cycle[j]][cycle[j+1]]['weight']
      graph_path_sols.append(cycle[largest_edge_pos+1:-1] + cycle[:largest_edge_pos+1]) #everything that comes after the bad edge, not including the last (duplicated) node from the cycle + everything that comes before the bad edge
    all_path_sols.append(graph_path_sols)
  return all_path_sols
#endregion Greedy Solvers

#region Outputs
def save_solutions(solution_list:list, solution_filepath:str):
  print("Saving solution sets")
  sol_array = np.array(solution_list, dtype=np.uint16)
  np.savetxt(solution_filepath,sol_array.transpose(),delimiter=',',fmt='%i')

def create_logger(logfile_name:str, logfile_level:str, console_level:str)-> logging.Logger:
  print('Creating logger')
  match logfile_level:
    case 'debug':
      lf_lev = logging.DEBUG #filters to debug and above, not recommended for console
    case 'info':
      lf_lev = logging.INFO
    case 'none':
      lf_lev = None
    case _:
      lf_lev = None
  match console_level:
    case 'debug':
      c_lev = logging.DEBUG #filters to debug and above, not recommended for console
    case 'info':
      c_lev = logging.INFO #filters to info and above, good for console
    case 'none':
      c_lev = None
    case _:
      c_lev = None
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
  
#endregion Outputs

#region Modify Graphs
def add_weights(graph_list, travel_threshold:float=1.6):
  for graph in graph_list:
    for u, v, data in graph.edges(data=True):
      if data['alignment'] != 0:
        data['weight'] = data['alignment'] + (data['length'] / travel_threshold) # 1 to 2 + d(0,1] = d[2,3] because 1 added already
      else:
        data['weight'] = 3 + (data['length'] / travel_threshold) # 2 + length, minimum 2 + 2*EW 

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
#endregion Modify Graphs

#region GA Class
class GraphGA:
  def __init__(self, graph, path_list, logger):
    self.graph = graph
    self.path_list = path_list
    self.gene_range = sorted(nx.nodes(graph)) #range(1, nx.number_of_nodes(self.graph) + 1)
    self.logger:logging.Logger = logger

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
  
  def reset_ga(self, n_gens: int=5, n_par_mate: int=120,
              parent_keep: int=0, n_elites: int=2, #if n_elites != 0, then parent_keep is ignored in GA
              mut:str='inversion', mut_prob:float=0.4,
              cross_prob:float=0.2, #doesn't really matter with custom crossovers
              cross_type:str='edge_recomb',
              parent_choice:str='tournament', tour_k:int = 3):

    if cross_type == 'edge_recomb':
      crossover = self.fast_edge_recombination_crossover
    elif cross_type == 'order':
      crossover = self.order_crossover
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
                      fitness_func=self.path_fitness, #locked from the class
                      gene_space=list(self.gene_range), #locked from the class
                      initial_population=self.path_list,  #locked from the class
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
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
    print(f"Weight of the solution = {nx.path_weight(self.graph, solution, 'weight')}")
    return self.ga.best_solution()

  #region GA.On-Functions
  def on_start(self, ga_instance):
      self.logger.info("Starting GA search")

  def on_fitness(self, ga_instance, population_fitness):
      self.logger.info("Computed fitness")

  def on_parents(self, ga_instance, selected_parents):
      self.logger.info("Selected parents")

  def on_crossover(self, ga_instance, offspring_crossover):
      self.logger.info("Performed crossovers")

  def on_mutation(self, ga_instance, offspring_mutation):
      self.logger.info("Mutated")

  def on_stop(self, ga_instance, last_population_fitness):
      self.logger.info("Ending GA search")
      
  def on_generation(self, ga_instance:pygad.GA):
      self.logger.info(ga_instance.generations_completed)
      self.logger.info(ga_instance.best_solution()[1])
      self.logger.debug(ga_instance.population)
  #endregion GA.On-Functions
#endregion GA Class
