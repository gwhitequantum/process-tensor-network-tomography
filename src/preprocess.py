"""Misc. functions to preprocess the data and create relevant objects etc."""

import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import quimb.tensor as qtn

from utilities import *

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[0]

# Construct the path to the data directory
data_dir = project_root / "data"

val_file_path = data_dir / "validation_params.pickle"
data_file_path = data_dir / "clifford_params.npy"


# some initial objects
validation_param_dict = pickle.load(open(val_file_path, "rb"))
v_params = [validation_param_dict[i] for i in range(100)]


clifford_params = jnp.load(data_file_path)
unique_cliffords = jnp.array([jnp.array(make_unitary(*C)) for C in clifford_params])
clifford_dict = {i: unique_cliffords[i] for i in range(24)}
clifford_param_dict = {i: clifford_params[i] for i in range(24)}

clifford_rz_params = [u3_to_rz_params(clifford_param_dict[i]) for i in range(24)]
val_rz_params = [u3_to_rz_params(validation_param_dict[i]) for i in range(100)]
clifford_rz_unitaries = [
    [rz_unitary(P) for P in clifford] for clifford in clifford_rz_params
]
val_rz_unitaries = [[rz_unitary(P) for P in val] for val in val_rz_params]
unique_v_unitaries = jnp.array([jnp.array(make_unitary(*C)) for C in v_params])


clifford_meas_vec = jnp.array(
    [
        jnp.array([u.conj().T @ zero_vec for u in unique_cliffords]),
        jnp.array([u.conj().T @ one_vec for u in unique_cliffords]),
    ]
)
validation_meas_vec = jnp.array(
    [
        jnp.array([u.conj().T @ zero_vec for u in unique_v_unitaries]),
        jnp.array([u.conj().T @ one_vec for u in unique_v_unitaries]),
    ]
)

clifford_unitaries_vT = {
    i: 2 * TN_choi_vec(unique_cliffords[i]).conj().reshape(1, 2, 2) for i in range(24)
}
val_unitaries_vT = {
    i: 2 * TN_choi_vec(unique_v_unitaries[i]).conj().reshape(1, 2, 2)
    for i in range(100)
}

clifford_rz_unitaries_vT = {
    i: [2 * TN_choi_vec(RZ).conj().reshape(1, 2, 2) for RZ in reversed(clifford)]
    for i, clifford in enumerate(clifford_rz_unitaries)
}
val_rz_unitaries_vT = {
    i: [2 * TN_choi_vec(RZ).conj().reshape(1, 2, 2) for RZ in reversed(val)]
    for i, val in enumerate(val_rz_unitaries)
}

clifford_measurements_vT = {
    0: [
        jnp.kron(2 * zero_vec, clifford_meas_vec[0][i]).reshape(1, 2, 2)
        for i in range(24)
    ],
    1: [
        jnp.kron(2 * zero_vec, clifford_meas_vec[1][i]).reshape(1, 2, 2)
        for i in range(24)
    ],
}
val_measurements_vT = {
    0: [
        jnp.kron(2 * zero_vec, validation_meas_vec[0][i]).reshape(1, 2, 2)
        for i in range(100)
    ],
    1: [
        jnp.kron(2 * zero_vec, validation_meas_vec[1][i]).reshape(1, 2, 2)
        for i in range(100)
    ],
}

pure_measurement = {
    0: jnp.kron(2 * zero_vec, zero_vec).reshape(1, 2, 2),
    1: jnp.kron(2 * one_vec, one_vec).reshape(1, 2, 2),
}


def shadow_results_to_data_vec(results, shots, nQ):
    data_vec = []
    data_keys = []

    for res in results:
        tmp = []
        for i in range(2**nQ):
            key = np.binary_repr(i, nQ)
            key = key[::-1]

            p = res.get(key)
            if p is not None:
                data_vec.append(p / shots)
                tmp.append(key)
        data_keys.append(tmp)
    return data_vec, data_keys


def shadow_seqs_to_op_array(sequences, keys, measurements, unitaries):
    nsteps = len(sequences[0][0]) - 1
    nQ = len(sequences[0])
    nUnique = sum([len(K) for K in keys])

    seq_of_seqs = []
    for i, S in enumerate(sequences):
        for key in keys[i]:
            tmp_nQ_seq = []
            for j in range(nQ):
                tmp_seq = []
                tmp_seq.append(measurements[int(key[j])][S[j][0]])
                for k in range(nsteps):
                    tmp_seq.append(unitaries[S[j][k + 1]])
                tmp_seq = np.concatenate(tmp_seq)
                tmp_nQ_seq.append(tmp_seq)
            tmp_nQ_seq = np.vstack(tmp_nQ_seq)
            seq_of_seqs.append(tmp_nQ_seq)

    final_shape = (nUnique, nQ, nsteps + 1, 2, 2)

    return jnp.array(np.vstack(seq_of_seqs).reshape(*final_shape))


def shadow_seqs_to_op_array_rz(sequences, keys, measurements, unitaries):
    nseqs = len(sequences)
    nsteps = len(sequences[0][0]) - 1
    nQ = len(sequences[0])
    nUnique = sum([len(K) for K in keys])

    seq_of_seqs = []
    for i, S in enumerate(sequences):
        for key in keys[i]:
            tmp_nQ_seq = []
            for j in range(nQ):
                tmp_seq = []
                tmp_seq.append(measurements[int(key[j])])
                for k in range(nsteps + 1):
                    for n in range(3):
                        tmp_seq.append(unitaries[S[j][k]][n])
                tmp_seq = np.concatenate(tmp_seq)
                tmp_nQ_seq.append(tmp_seq)
            tmp_nQ_seq = np.vstack(tmp_nQ_seq)
            seq_of_seqs.append(tmp_nQ_seq)

    final_shape = (nUnique, nQ, 3 * (nsteps + 1) + 1, 2, 2)

    return jnp.array(np.vstack(seq_of_seqs).reshape(*final_shape))


def op_arrays_to_single_vector_TN_padded(op_seq):
    k = len(op_seq[0]) - 1
    nQ = op_seq.shape[0]
    # input in order measure, U_k-1, ... , U_0

    TN_list = []

    for i in range(nQ):
        initial = qtn.Tensor(
            op_seq[i][0],
            inds=(f"kP_q{i}", f"ko{k}_q{i}"),
            tags=["U3", f"q{i}_U{k}", f"ROW{i}", f"COL{k}"],
        )
        for j, O in enumerate(op_seq[i][1:]):
            initial = initial & qtn.Tensor(
                O,
                inds=(f"ki{k-j}_q{i}", f"ko{k-j-1}_q{i}"),
                tags=["U3", f"q{i}_U{k-j-1}", f"ROW{i}", f"COL{k-j-1}"],
            )
        TN_list.append(initial)
    TN_list = qtn.TensorNetwork(TN_list)
    OTN_ket = qtn.tensor_2d.TensorNetwork2DFlat.from_TN(
        TN_list,
        site_tag_id="q{}_U{}",
        Ly=k + 1,
        Lx=nQ,
        row_tag_id="ROWq{}",
        col_tag_id="COL{}",
    )
    OTN_bra = OTN_ket.H.copy()
    OTN_bra.reindex_(
        {f"ko{i}_q{j}": f"bo{i}_q{j}" for i in range(k + 1) for j in range(nQ)}
    )
    OTN_bra.reindex_(
        {f"ki{i}_q{j}": f"bi{i}_q{j}" for i in range(1, k + 1) for j in range(nQ)}
    )

    OTN_ket.add_tag("OP KET")
    OTN_bra.add_tag("OP BRA")
    return OTN_ket & OTN_bra


def op_arrays_to_single_vector_TN_padded_X_decomp(op_seq):
    k = len(op_seq[0]) - 1
    tsteps = int(k / 3)
    nQ = op_seq.shape[0]
    # input in order | measure -- U_k-1 -- ... -- U_0 |

    TN_list = []

    for i in range(nQ):
        cStep = tsteps
        initial = qtn.Tensor(
            op_seq[i][0], inds=(f"kP_q{i}", f"ko{tsteps}_q{i}"), tags=["MEAS"]
        )
        for j, O in enumerate(op_seq[i][1:]):

            if j % 3 == 0:
                initial = initial & qtn.Tensor(
                    O, inds=(f"ki{cStep}_q{i}", f"kXmo{cStep-1}_q{i}"), tags=["RZ"]
                )

            if j % 3 == 1:
                initial = initial & qtn.Tensor(
                    O, inds=(f"kXmi{cStep-1}_q{i}", f"kXpo{cStep-1}_q{i}"), tags=["RZ"]
                )

            if j % 3 == 2:
                cStep -= 1
                initial = initial & qtn.Tensor(
                    O, inds=(f"kXpi{cStep}_q{i}", f"ko{cStep}_q{i}"), tags=["RZ"]
                )

        TN_list.append(initial)

    OTN_ket = qtn.TensorNetwork(TN_list)
    # OTN_ket = qtn.tensor_2d.TensorNetwork2DFlat.from_TN(TN_list, site_tag_id='q{}_U{}', Ly=k+1, Lx = nQ, row_tag_id='ROWq{}',col_tag_id='COL{}')
    OTN_bra = OTN_ket.H.copy()
    OTN_bra.reindex_(
        {f"ko{i}_q{j}": f"bo{i}_q{j}" for i in range(k + 1) for j in range(nQ)}
    )
    OTN_bra.reindex_(
        {f"ki{i}_q{j}": f"bi{i}_q{j}" for i in range(1, k + 1) for j in range(nQ)}
    )
    OTN_bra.reindex_(
        {f"kXpo{i}_q{j}": f"bXpo{i}_q{j}" for i in range(k + 1) for j in range(nQ)}
    )
    OTN_bra.reindex_(
        {f"kXpi{i}_q{j}": f"bXpi{i}_q{j}" for i in range(k + 1) for j in range(nQ)}
    )
    OTN_bra.reindex_(
        {f"kXmo{i}_q{j}": f"bXmo{i}_q{j}" for i in range(k + 1) for j in range(nQ)}
    )
    OTN_bra.reindex_(
        {f"kXmi{i}_q{j}": f"bXmi{i}_q{j}" for i in range(k + 1) for j in range(nQ)}
    )

    OTN_ket.add_tag("OP KET")
    OTN_bra.add_tag("OP BRA")
    return OTN_ket & OTN_bra
