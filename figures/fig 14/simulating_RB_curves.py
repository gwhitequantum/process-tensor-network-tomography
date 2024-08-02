"""
Script to load in a PT estimate for SU4 decomposition (found in data/PT_estimates) and compute optimal RB parameters
Then simulate the RB curves.
Since this is done for each datapoint, the whole script is quite slow. Results used in arXiv:2312.08454 can be found in the data directory in the folder RB_predictions
"""

import os

os.environ["QUIMB_NUM_THREAD_WORKERS"] = "1"

import sys

import jax.numpy as jnp

sys.path.append("../../src")
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import quimb as qu
import quimb.tensor as qtn
from scipy.stats import unitary_group

from optimize import TNOptimizer_circ
from preprocess import (
    op_arrays_to_single_vector_TN_padded_X_decomp,
    pure_measurement,
    shadow_seqs_to_op_array_rz,
    val_rz_unitaries_vT,
)
from process_tensor_networks import create_PEPO_X_decomp, create_PT_PEPO_guess
from two_qubit_gate_opt import *
from utilities import *

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[0]

# Construct the path to the data directory
data_dir = project_root / "data"
su4_dir = data_dir / "SU4_opt"
est_dir = su4_dir / "PT_estimates"

device_string = "cairo_SU4_job_dict_cnot_03_10.pickle"
# or pick any of the other devices


best_val_mpo = pickle.load(
    open(est_dir + device_string + "_best_fit_TN_SC.pickle", "rb")
)

timesteps = 4
NSTEPS = 3
nQ = 2
####### INITIAL TOOLS FOR SU4 OPT ########

qiskit_output_inds = [
    "ko0_q0",
    "ko0_q1",
    "ki4_q0",
    "ki4_q1",
    "bo0_q0",
    "bo0_q1",
    "bi4_q0",
    "bi4_q1",
]

CNOT = qu.controlled("not")
nS = NSTEPS


K_lists = [[1] + [1 for j in range(nS - 1)] + [1] for i in range(nQ)]
vertical_bonds = [[1] + [1 for i in range(nQ - 3)] + [1] for j in range(nS)] + [
    [1 for i in range(nQ - 1)]
]
horizontal_bonds = [1 for i in range(nS)]
initial_guess = create_PT_PEPO_guess(nS, nQ, horizontal_bonds, vertical_bonds, K_lists)
initial_guess = qu.tensor.tensor_2d.TensorNetwork2DFlat.from_TN(
    initial_guess,
    site_tag_id="q{}_I{}",
    Ly=nS + 1,
    Lx=nQ,
    row_tag_id="ROWq{}",
    col_tag_id="COL{}",
)

ideal_PT = create_PT_PEPO_guess(nS, nQ, horizontal_bonds, vertical_bonds, K_lists)
ideal_PT = qu.tensor.tensor_2d.TensorNetwork2DFlat.from_TN(
    ideal_PT,
    site_tag_id="q{}_I{}",
    Ly=nS + 1,
    Lx=nQ,
    row_tag_id="ROWq{}",
    col_tag_id="COL{}",
)


ideal_PT.gate_inds_(CNOT, inds=("ko3_q1", "ko3_q0"), contract="split")
ideal_PT.gate_inds_(CNOT, inds=("ko2_q1", "ko2_q0"), contract="split")
ideal_PT.gate_inds_(CNOT, inds=("ko1_q1", "ko1_q0"), contract="split")

ideal_X = create_PEPO_X_decomp(nS, nQ, rand_strength=0.0)

ideal_PT.squeeze_()

ideal_PT = prepare_PT_for_opt(ideal_PT & ideal_X)


tmp_seq = [[[61, 52, 4, 53, 72]]]
tmp_keys = [["0", "1"]]
tmp_sequence_v_rz = shadow_seqs_to_op_array_rz(
    tmp_seq, tmp_keys, pure_measurement, val_rz_unitaries_vT
)


example_op = op_arrays_to_single_vector_TN_padded_X_decomp(tmp_sequence_v_rz[0]).select(
    ["RZ", "OP KET"]
)

processed_PT = prepare_PT_for_opt(best_val_mpo)
test_P = params_to_PTN_SU4(jnp.array(np.random.rand(24)), example_op)
bell = jnp.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])


def params_to_labelled_SU4(params, PT, step):
    total_map = (PTN_ket_to_full_SU4(params) & PT).contract().data
    tmp_trace = jnp.trace(total_map.reshape(16, 16))
    total_map = total_map / tmp_trace

    map_inds = (f"ko{step}_q0", f"ki{step}_q0", f"ko{step}_q1", f"ki{step}_q1")
    map_inds += (f"bo{step}_q0", f"bi{step}_q0", f"bo{step}_q1", f"bi{step}_q1")
    total_map_T = qtn.Tensor(total_map, inds=map_inds, tags=("SU4", f"SU4_{step}"))

    return total_map_T


def create_identity_prop(step):
    tmp_inds_q0 = (f"ki{step}_q0", f"ko{step-1}_q0", f"bi{step}_q0", f"bo{step-1}_q0")
    tmp_inds_q1 = (f"ki{step}_q1", f"ko{step-1}_q1", f"bi{step}_q1", f"bo{step-1}_q1")

    bell_q0 = qtn.Tensor(4 * bell.reshape(2, 2, 2, 2), inds=tmp_inds_q0)
    bell_q1 = qtn.Tensor(4 * bell.reshape(2, 2, 2, 2), inds=tmp_inds_q1)
    return bell_q0 & bell_q1


def create_SU4_sequence(length):
    # note in reversing order of implementation
    # length includes the inverse of the sequence
    su4_list = []
    for i in range(length - 1):
        su4_list.append(unitary_group.rvs(4))

    if len(su4_list) >= 2:
        net_unitary = jnp.linalg.multi_dot(su4_list)
    else:
        net_unitary = su4_list[0]
    inverse_unitary = net_unitary.conj().T
    su4_list.reverse()
    su4_list = [inverse_unitary] + su4_list

    return su4_list


def unitary_to_opt_params(unitary, PT):
    su4_target = generate_2q_unitary_choi(unitary, "i4_", "o0_")
    su4_bra = su4_target.H
    su4_bra.reindex_({ind: "b" + ind[1:] for ind in su4_target.outer_inds()})
    su4_kraus = su4_target & su4_bra

    test_P = params_to_PTN_SU4(np.random.rand(24), example_op)
    PT_op_trace = jnp.trace(
        (PTN_ket_to_full_SU4(test_P) & PT).contract().data.reshape(16, 16)
    )

    SU4_optmzr = TNOptimizer_circ(
        test_P,
        loss_fn=params_to_su4_overlap,
        loss_constants={
            "process_tensor": (16 / PT_op_trace) * PT,
            "example_op": example_op,
            "target_su4": su4_kraus,
        },
        autodiff_backend="jax",
        optimizer="L-BFGS-B",
        progbar=False,
    )
    if PT is ideal_PT:
        param_opt = SU4_optmzr.optimize_basinhopping(5000, 10)
    else:
        param_opt = SU4_optmzr.optimize_basinhopping(5000, 100)

    return param_opt


def SU4_sequence_to_PT_params(unitary_sequence, PT):
    param_list = []
    for U in unitary_sequence:
        param_list.append(unitary_to_opt_params(U, PT))

    return param_list


def simulate_PT_sequence_outcome(PT_params, PT):

    final_step = len(PT_params)
    initial_state = jnp.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ).reshape(2, 2, 2, 2)
    rho_T = qtn.Tensor(initial_state, inds=("ki0_q0", "ki0_q1", "bi0_q0", "bi0_q1"))

    SU4_tensors = qtn.TensorNetwork(
        [params_to_labelled_SU4(P, PT, i) for i, P in enumerate(reversed(PT_params))]
    )
    identity_tensors = qtn.TensorNetwork(
        [create_identity_prop(i) for i in range(1, final_step)]
    )

    return (rho_T & SU4_tensors & identity_tensors).contract(
        output_inds=(
            f"ko{final_step-1}_q0",
            f"ko{final_step-1}_q1",
            f"bo{final_step-1}_q0",
            f"bo{final_step-1}_q1",
        )
    )


def simulate_RB_datapoint(length, PT):

    RB_sequence = create_SU4_sequence(length)
    ideal_params = SU4_sequence_to_PT_params(RB_sequence, ideal_PT)
    opt_params = SU4_sequence_to_PT_params(RB_sequence, PT)

    naive_outcome = simulate_PT_sequence_outcome(ideal_params, PT).data.reshape(4, 4)
    optimised_outcome = simulate_PT_sequence_outcome(opt_params, PT).data.reshape(4, 4)

    naive_outcome = jnp.real(naive_outcome[0, 0] / jnp.trace(naive_outcome))
    optimised_outcome = jnp.real(optimised_outcome[0, 0] / jnp.trace(optimised_outcome))

    return naive_outcome, optimised_outcome


def simulate_RB_experiment(lengths, samples, PT):
    results_dict = {}
    try:
        for L in lengths:
            print(f"At length {L}")
            tmp_dict = {"naive": [], "opt": []}
            for i in range(samples):
                print(i)
                outcome = simulate_RB_datapoint(L, PT)
                tmp_dict["naive"].append(outcome[0])
                tmp_dict["opt"].append(outcome[1])
            results_dict[L] = tmp_dict
        return results_dict
    except:
        return results_dict


X = [2, 3, 4, 8, 16, 32]
res_set = simulate_RB_experiment(X, 25, processed_PT)

with open(device_string + "_RB_predictions.pickle", "wb") as handle:
    pickle.dump(res_set, handle)
