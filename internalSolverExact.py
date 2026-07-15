import numpy as np
from datetime import datetime
from jointlab import LogFileMaker
from python_tsp.exact import solve_tsp_dynamic_programming

def solve_tsp_exact(bin_file_path, n_original, solution_path, time_limit_seconds=5*60):
  log = LogFileMaker.create_logger("_".join([datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),"completer.log"]))
  dist_matrix_float = np.fromfile(bin_file_path,dtype=np.int64)
  dist_matrix_float = dist_matrix_float.reshape((n_original, n_original))

  dist_matrix_float = np.pad(dist_matrix_float, ((0,1),(0,1)), mode='constant', constant_values=0)
  n_total = n_original + 1

  # dist_matrix_int = np.round(dist_matrix_float * scale_factor).astype(np.int64)

  zero_count = np.count_nonzero(dist_matrix_float==0) - n_total- (n_total * 2)
  log.info(f"number of explicit 0-weight edges seen by OR-Tools:{zero_count}")

  solution, dist = solve_tsp_dynamic_programming(dist_matrix_float,maxsize=None)

  if not solution:
    raise Exception("Exact tools could not find a solution.")

  for handler in log.handlers:
    log.removeHandler(handler)
  
  np.array([int(node+1) for node in solution],dtype=np.int64).tofile(solution_path)