import argparse
import jointlab as jl
import json

parser = argparse.ArgumentParser()
parser.add_argument('graph_list_jsonfile', type=str)
parser.add_argument('sol_list_jsonfile', type=str)
parser.add_argument('solution_out_file', type=str)
parser.add_argument('log_file_level', type=str)
parser.add_argument('console_level', type=str)
parser.add_argument('--dir', type=str, nargs='?')
args = parser.parse_args()

graph_list_filepath = args.graph_list_jsonfile
sol_list_filepath = args.sol_list_jsonfile
if args.dir:
  log_base_directory = args.dir
else:
  log_base_directory = 'logs'
solution_filepath = args.solution_out_file
file_level_str = args.log_file_level
console_level_str = args.console_level

jl.LogFileMaker(file_level_str, console_level_str, log_base_directory)

sol_list = jl.read_sol_list(sol_list_filepath)
graph_list = jl.reset_graph_list(graph_list_filepath)
jl.add_weights(graph_list)

uncompressed_data = jl.decompress(graph_list, sol_list) #0 = complete graphs, 1 = decompressed paths (may be incomplete)
recompleted_paths = []
for i, solution in enumerate(uncompressed_data[1]):
  recompleted_paths.append(jl.missing_nodes_zag(uncompressed_data[0][i], solution))
comps_by_node = jl.comps_by_node(uncompressed_data[0])

with open(solution_filepath, 'w') as json_sol_out:
  json.dump([recompleted_paths, comps_by_node], json_sol_out)