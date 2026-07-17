import argparse
import internalSolverOR_borderBreaks as isor
import internalSolverExact as ise
import jointlab as jl
from json import load as jsonload

parser = argparse.ArgumentParser()
parser.add_argument('internal_weight_adj_file', type=str)
parser.add_argument('original_node_count', type=int)
parser.add_argument('search_time_sec', type=int)
parser.add_argument('solution_out_file', type=str)
parser.add_argument('pref_breaks', type=str)
parser.add_argument('sec_breaks',type=str)
parser.add_argument('break_penalty', type=int)
parser.add_argument('log_file_level', type=str)
parser.add_argument('console_level', type=str)
parser.add_argument('--dir', type=str, nargs='?')
args = parser.parse_args()

internal_weight_adj_file = args.internal_weight_adj_file
original_node_count = args.original_node_count
search_time_sec = args.search_time_sec
solution_filepath = args.solution_out_file
file_level_str = args.log_file_level
console_level_str = args.console_level
pref_breaks_file = args.pref_breaks
sec_breaks_file = args.sec_breaks
break_penalty = args.break_penalty
if args.dir:
  log_base_directory = args.dir
else:
  log_base_directory = 'logs'

jl.LogFileMaker(file_level_str, console_level_str, log_base_directory)

with open(pref_breaks_file,'r') as pref_file:
  preferred_breaks = jsonload(pref_file)
with open(sec_breaks_file,'r') as sec_file:
  secondary_breaks = jsonload(sec_file)

isor.solve_open_tsp_ortools(internal_weight_adj_file, original_node_count, solution_filepath, preferred_breaks, secondary_breaks, break_penalty, search_time_sec)
# ise.solve_tsp_exact(internal_weight_adj_file, original_node_count, solution_filepath, search_time_sec)
# with open(solution_filepath, 'wb') as sol_out:
#   for node in solution:
#     sol_out.write(struct.packbytes(solution))