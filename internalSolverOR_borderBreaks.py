import numpy as np
from datetime import datetime
from jointlab import LogFileMaker
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import os
import sys
from contextlib import contextmanager
from logging import DEBUG

@contextmanager
def redirect_native_output_to_file(path, mode="a"):
  """
  Redirect process-level stdout and stderr to a file.

  This catches:
    - Python print(...)
    - C/C++ writes to fd 1 / fd 2
    - many native library logs

  It does not route through Python logging, which avoids pipe deadlocks.
  """
  sys.stdout.flush()
  sys.stderr.flush()

  stdout_fd = sys.stdout.fileno()
  stderr_fd = sys.stderr.fileno()

  saved_stdout_fd = os.dup(stdout_fd)
  saved_stderr_fd = os.dup(stderr_fd)

  with open(path, mode, buffering=1) as f:
    try:
      os.dup2(f.fileno(), stdout_fd)
      os.dup2(f.fileno(), stderr_fd)
      yield
    finally:
      sys.stdout.flush()
      sys.stderr.flush()

      os.dup2(saved_stdout_fd, stdout_fd)
      os.dup2(saved_stderr_fd, stderr_fd)

      os.close(saved_stdout_fd)
      os.close(saved_stderr_fd)

# @contextmanager
# def log_solver_to_python(logger, level=DEBUG):
#   """Safely intercepts C++ stdout and routes it to a Python logger."""
#   stdout_fd = sys.stdout.fileno()

#   # 1. Save original stdout descriptor so we can restore it later
#   saved_stdout_fd = os.dup(stdout_fd)

#   # 2. Create the pipe and explicitly keep track of both ends
#   pipe_r, pipe_w = os.pipe()

#   # 3. Redirect stdout to the write end of our pipe
#   os.dup2(pipe_w, stdout_fd)

#   # 4. We must close our local copy of pipe_w now so that the only 
#   # active write descriptor belongs to the redirected stdout_fd.
#   os.close(pipe_w)

#   try:
#     yield
#     # Flush any remaining buffers before tearing down
#     sys.stdout.flush()
#   finally:
#     # 5. Restore the original stdout first. This closes the redirected 
#     # wrapper and safely cuts off the writing end of the pipe.
#     os.dup2(saved_stdout_fd, stdout_fd)
#     os.close(saved_stdout_fd)

#     # 6. Read the captured stream from the read end (it will no longer block)
#     with os.fdopen(pipe_r, "r") as f:
#       captured_text = f.read()
#       for line in captured_text.splitlines():
#         if line.strip():
#           logger.log(level, line)

def solve_open_tsp_ortools(in_bin_file_path, n_original, solution_path, preferred_breaks, secondary_breaks, break_penalty_int, time_limit_seconds=5*60):
  break_penalty = np.int64(break_penalty_int)
  secondary_penalty = np.int64(break_penalty*0.5)
  log_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  log = LogFileMaker.create_logger("_".join([log_name,"internal_solver.log"]))
  dist_matrix_int = np.fromfile(in_bin_file_path,dtype=np.int64)
  dist_matrix_int = dist_matrix_int.reshape((n_original, n_original))

  native_log_path = os.path.join(LogFileMaker.global_base_path, "_".join([log_name,"or_search.log"]))
  
  dist_matrix_int = np.pad(dist_matrix_int, ((0,1),(0,1)), mode='constant', constant_values=0)
  for i in range(n_original):
    cost = 0 if i in preferred_breaks else secondary_penalty if i in secondary_breaks else break_penalty
    dist_matrix_int[i, n_original] = cost
    dist_matrix_int[n_original, i] = cost
  
  n_total = n_original + 1

  # dist_matrix_int = np.round(dist_matrix_float * scale_factor).astype(np.int64)

  zero_count = np.count_nonzero(dist_matrix_int==0) - n_total- (n_total * 2)
  log.info(f"number of explicit 0-weight edges seen by OR-Tools:{zero_count}")

  #                                       length, vehicles, start/stop at free
  manager = pywrapcp.RoutingIndexManager(n_total, 1, n_original)
  routing = pywrapcp.RoutingModel(manager)

  # def distance_callback(from_index, to_index):
  #   from_node = manager.IndexToNode(from_index)
  #   to_node = manager.IndexToNode(to_index)
  #   return dist_matrix_int[from_node][to_node]
  
  routing.RegisterCumulDependentTransitCallback
  dist_matrix_list_int = dist_matrix_int.astype(int).tolist()
  transit_matrix = routing.RegisterTransitMatrix(dist_matrix_list_int)
  routing.SetArcCostEvaluatorOfAllVehicles(transit_matrix)
  # transit_callback_index = routing.RegisterTransitCallback(distance_callback)  
  # routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

  search_parameters = pywrapcp.DefaultRoutingSearchParameters()
  search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_SAVINGS
  search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH #SIMULATED_ANNEALING = allow some bad, TABU_SEARCH = forbid recent swaps
  search_parameters.local_search_operators.use_lin_kernighan = pywrapcp.BOOL_TRUE
  # search_parameters.local_search_operators.use_two_opt = True
  # search_parameters.local_search_operators.use_or_opt = True #3opt variant
  # search_parameters.local_search_operators.use_relocate = True
  # search_parameters.local_search_operators.use_exchange = True
  # search_parameters.solution_limit = 10000
  search_parameters.log_search = True
  search_parameters.time_limit.seconds = time_limit_seconds

  log.info("Starting search")
  with redirect_native_output_to_file(native_log_path):
    solution = routing.SolveWithParameters(search_parameters)
  log.info("Search finished")

  if not solution:
    raise Exception("OR-Tools could not find a solution.")
  
  index = routing.Start(0)
  route = []
  while not routing.IsEnd(index):
    route.append(manager.IndexToNode(index))
    index = solution.Value(routing.NextVar(index))
  
  route.remove(n_original)

  for handler in log.handlers:
    log.removeHandler(handler)
  
  np.array([int(node+1) for node in route],dtype=np.int64).tofile(solution_path)