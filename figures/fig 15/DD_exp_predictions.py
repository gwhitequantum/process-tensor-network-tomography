import os

os.environ["QUIMB_NUM_THREAD_WORKERS"] = "1"
import sys

sys.path.append("../../src")


from pathlib import Path

import numpy as np
import quimb as qu
import quimb.tensor as qtn

from DD_optimize import TNOptimizer as TNOptimizer_circ
from dynamical_decoupling import *
from process_tensor_networks import produce_LPDO
from utilities import *

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]

# Construct the path to the data directory
data_dir = project_root / "data"
est_dir = data_dir / "DD_opt" / "PT_estimates"


file_name = str(est_dir) + "/auckland_33_step_DD_jobs_25_05_optimizer.pickle"
fit_optimized = pickle.load(open(file_name, "rb"))
best_val_mpo = fit_optimized  # .best_val_mpo


DD_optmzr = TNOptimizer_circ(
    xy8_seed,  # our initial input, the tensors of which to optimize
    loss_fn=evaluate_DD_sequence,
    loss_constants={
        "PT_est": produce_LPDO(best_val_mpo)
    },  # this is a constant TN to supply to loss_fn
    autodiff_backend="jax",  # {'jax', 'tensorflow', 'autograd'}
    optimizer="L-BFGS-B",  # supplied to scipy.minimize
)

print("Starting fit")
test_opt = DD_optmzr.optimize_basinhopping(5000, 5)
DD_optmzr.optimizer = "adam"
test_opt = DD_optmzr.optimize(5000)
opt_params = [test_opt.tensors[i].params for i in range(3 * NSTEPS)]
opt_params_ordered = [opt_params[3 * i : 3 * (i + 1)] for i in range(NSTEPS)]
for i in range(NSTEPS):
    opt_params_ordered[i].reverse()
opt_network = params_to_rz_PTN(np.array(opt_params_ordered))


val_TNs = [params_to_rz_tensor(u3_to_rz_params(P), shift=0) for P in v_params]
val_TNs = [unparametrise_TN(P) for P in val_TNs]
val_Us = [np.array(make_unitary(*V)) for V in v_params]
ideal_val_outs = [U @ ideal_input @ U.conj().T for U in val_Us]


def test_DD_sequence(param_seq, PT_est):
    # PT_est in LPDO form
    trace_dists = []
    for i in range(len(val_TNs)):
        tmp_op = param_seq & val_TNs[i] & meas_TN
        output_DM = (
            (tmp_op & PT_est)
            .contract(output_inds=(f"bo{NSTEPS+1}_q0", f"ko{NSTEPS+1}_q0"))
            .data
        )
        output_DM = output_DM / np.trace(output_DM)
        tmp_dist = 0.5 * np.linalg.norm(output_DM - ideal_val_outs[i], "nuc")
        trace_dists.append(tmp_dist)

    return trace_dists


opt_difs = test_DD_sequence(opt_network, produce_LPDO(best_val_mpo))  # states
XY_difs = test_DD_sequence(xy8_seed, produce_LPDO(best_val_mpo))
ID_difs = test_DD_sequence(id_seed, produce_LPDO(best_val_mpo))
import pandas as pd

columns = {"identity": ID_difs, "XY": XY_difs, "optimised": opt_difs}
DD_data_df = pd.DataFrame(data=columns)

# with open('DD_predictions/' + job_string[:-7] + f'_predicted_DD_data.pickle','wb') as handle:
#     pickle.dump(DD_data_df, handle)
