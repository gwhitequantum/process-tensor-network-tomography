"""
Script to load in a PT estimate for SU4 decomposition (found in data/PT_estimates) and compute optimal parameters for a set of random two-qubit unitaries, then compute the diamond distance between ideal and optimised.
Results used in arXiv:2312.08454 can be found in the data directory in the file SU4_results_diamond.pickle. Data is interpreted and plots are constructed in the notebook diamond_dist_results.ipynb
"""

import os

os.environ["QUIMB_NUM_THREAD_WORKERS"] = "1"

import sys

sys.path.append("../../src")
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import qiskit
import quimb as qu
import quimb.tensor as qtn
from scipy.stats import describe, unitary_group

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

best_val_mpo = pickle.load(
    open(est_dir + device_string + "_best_fit_TN_SC.pickle", "rb")
)


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

no_epochs = 600
batch_size = 1000
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


def check_SU4_improvement():
    su4_target = generate_2q_unitary_choi(unitary_group.rvs(4), "i4_", "o0_")
    su4_bra = su4_target.H
    su4_bra.reindex_({ind: "b" + ind[1:] for ind in su4_target.outer_inds()})
    su4_kraus = su4_target & su4_bra

    PT_op_trace = jnp.trace(
        (PTN_ket_to_full_SU4(test_P) & processed_PT).contract().data.reshape(16, 16)
    )

    noisy_SU4_optmzr = TNOptimizer_circ(
        test_P,  # our initial input, the tensors of which to optimize
        loss_fn=params_to_su4_overlap,
        loss_constants={
            "process_tensor": (16 / PT_op_trace) * processed_PT,
            "example_op": example_op,
            "target_su4": su4_kraus,
        },  # this is a constant TN to supply to loss_fn
        autodiff_backend="jax",  # {'jax', 'tensorflow', 'autograd'}
        optimizer="L-BFGS-B",  # supplied to scipy.minimize
        progbar=False,
    )

    ideal_SU4_optmzr = TNOptimizer_circ(
        test_P,  # our initial input, the tensors of which to optimize
        loss_fn=params_to_su4_overlap,
        loss_constants={
            "process_tensor": ideal_PT,
            "example_op": example_op,
            "target_su4": (su4_target & su4_bra),
        },  # this is a constant TN to supply to loss_fn
        autodiff_backend="jax",  # {'jax', 'tensorflow', 'autograd'}
        optimizer="L-BFGS-B",  # supplied to scipy.minimize
        progbar=False,
    )
    param_opt = noisy_SU4_optmzr.optimize_basinhopping(5000, 100)
    ideal_param_opt = ideal_SU4_optmzr.optimize_basinhopping(5000, 10)

    dense_target = su4_kraus.contract(
        output_inds=(
            "ki4_q0",
            "ki4_q1",
            "ko0_q0",
            "ko0_q1",
            "bi4_q0",
            "bi4_q1",
            "bo0_q0",
            "bo0_q1",
        )
    ).data.reshape(16, 16)

    SU4_target_qis = su4_kraus.contract(output_inds=qiskit_output_inds).data.reshape(
        16, 16
    )
    SU4_target_qis = 4 * SU4_target_qis / np.trace(SU4_target_qis)
    SU4_target_qis = qiskit.quantum_info.Choi(np.array(SU4_target_qis), 4, 4)

    PT_opt = (
        (PTN_ket_to_full_SU4(param_opt) & processed_PT)
        .contract(output_inds=qiskit_output_inds)
        .data.reshape(16, 16)
    )
    PT_opt = 4 * PT_opt / jnp.trace(PT_opt)
    PT_opt = qiskit.quantum_info.Choi(np.array(PT_opt), 4, 4)

    new_error = qiskit.quantum_info.diamond_norm(PT_opt - SU4_target_qis)
    new_fid = qiskit.quantum_info.process_fidelity(PT_opt, SU4_target_qis)

    PT_naive = (
        (PTN_ket_to_full_SU4(ideal_param_opt) & processed_PT)
        .contract(output_inds=qiskit_output_inds)
        .data.reshape(16, 16)
    )
    PT_naive = 4 * PT_naive / jnp.trace(PT_naive)
    PT_naive = qiskit.quantum_info.Choi(jnp.array(PT_naive), 4, 4)

    old_error = qiskit.quantum_info.diamond_norm(PT_naive - SU4_target_qis)
    old_fid = qiskit.quantum_info.process_fidelity(PT_naive, SU4_target_qis)

    print(f"Device DN : {old_error}, Optimised DN : {new_error}")
    print(f"Device fid : {old_fid}, Optimised fid : {new_fid}")
    # print(f'Ratio of infidelities is {float(np.real(old_fid / new_fid))}')
    return old_fid, new_fid, old_error, new_error


improvement_stat_collector = []
for i in range(100):
    tmp = check_SU4_improvement()
    improvement_stat_collector.append(tmp)

SU4_results_dict = {device_string: {"single_SU4_stats": improvement_stat_collector}}

old_errors = [err[0] for err in improvement_stat_collector]
new_errors = [err[1] for err in improvement_stat_collector]
print(describe(old_errors))
print(describe(new_errors))


# with open("SU4_results_diamond.pickle", "wb") as handle:
#     pickle.dump(SU4_results_dict, handle)
