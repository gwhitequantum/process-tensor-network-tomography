"""Helper functions for performing tensor network estimation of non-Markovian processes"""

import jax
import jax.numpy as jnp
import numpy as np
import quimb.tensor as qtn

from preprocess import (
    op_arrays_to_single_vector_TN_padded,
    op_arrays_to_single_vector_TN_padded_X_decomp,
)
from process_tensor_networks import produce_LPDO
from utilities import *

two_qubit_pauli_dict = {
    (i, j): jnp.kron(qubit_pauli_set[i], qubit_pauli_set[j]).reshape(2, 2, 2, 2)
    for i in range(4)
    for j in range(4)
}


def generate_random_causality_keys(nSteps, nQ, nSamples):
    # try for nQ = 1 for now
    nqubits = 2 * nSteps + 1
    rand_keys = []
    for i in range(nSamples):
        tmp = []
        for n in range(nQ):
            system = 2 * np.random.randint(nSteps) + 1
            n_left_qubits = system
            n_right_qubits = nqubits - n_left_qubits - 1
            lk = [0 for j in range(n_left_qubits)]
            sk = [np.random.randint(1, 4)]
            rk = list(np.random.randint(4, size=n_right_qubits))
            tmp.append(tuple(lk + sk + rk))
        rand_keys.append(tmp)

    return rand_keys


def causality_key_to_pauli_tn(cKey):
    k = int((len(cKey[0]) - 1) / 2)
    nQ = len(cKey)
    TN_list = []
    for i in range(nQ):
        initial_key = (0, cKey[i][0])
        initial = qtn.Tensor(
            two_qubit_pauli_dict[initial_key],
            inds=(f"kP_q{i}", f"ko{k}_q{i}", f"kP_q{i}", f"bo{k}_q{i}"),
            tags=["U3", f"q{i}_U{k}", f"ROW{i}", f"COL{k}"],
        )
        for j in range(k):
            current_key = cKey[i][2 * j + 1 : 2 * j + 3]
            initial = initial & qtn.Tensor(
                two_qubit_pauli_dict[current_key],
                inds=(
                    f"ki{k-j}_q{i}",
                    f"ko{k-j-1}_q{i}",
                    f"bi{k-j}_q{i}",
                    f"bo{k-j-1}_q{i}",
                ),
                tags=["U3", f"q{i}_U{k-j-1}", f"ROW{i}", f"COL{k-j-1}"],
            )
        TN_list.append(initial)
    return qtn.TensorNetwork(TN_list)


def causality_keys_to_op_arrays(cKeys):
    nsteps = int((len(cKeys[0][0]) - 1) / 2)
    nQ = len(cKeys[0])

    seq_of_seqs = []
    for i, key in enumerate(cKeys):
        tmp_nQ_seq = []
        for j in range(nQ):
            tmp_seq = []
            initial_key = (0, key[j][0])
            tmp_seq.append(two_qubit_pauli_dict[initial_key])
            for k in range(nsteps):
                current_key = key[j][2 * k + 1 : 2 * k + 3]
                tmp_seq.append(two_qubit_pauli_dict[current_key])
            tmp_seq = np.concatenate(tmp_seq)
            tmp_nQ_seq.append(tmp_seq)
        tmp_nQ_seq = np.vstack(tmp_nQ_seq)
        seq_of_seqs.append(tmp_nQ_seq)
    final_shape = (len(cKeys), nQ, nsteps + 1, 2, 2, 2, 2)
    return jnp.array(np.vstack(seq_of_seqs).reshape(*final_shape))


def causality_ops_to_pauli_tn(op_seq):
    k = len(op_seq[0]) - 1
    nQ = op_seq.shape[0]
    TN_list = []
    for i in range(nQ):
        initial = qtn.Tensor(
            op_seq[i][0],
            inds=(f"kP_q{i}", f"ko{k}_q{i}", f"kP_q{i}", f"bo{k}_q{i}"),
            tags=["U3", f"q{i}_U{k}", f"ROW{i}", f"COL{k}"],
        )
        for j, O in enumerate(op_seq[i][1:]):
            initial = initial & qtn.Tensor(
                O,
                inds=(
                    f"ki{k-j}_q{i}",
                    f"ko{k-j-1}_q{i}",
                    f"bi{k-j}_q{i}",
                    f"bo{k-j-1}_q{i}",
                ),
                tags=["U3", f"q{i}_U{k-j-1}", f"ROW{i}", f"COL{k-j-1}"],
            )
        TN_list.append(initial)
    return qtn.TensorNetwork(TN_list)


def causality_term_k(mpo_traced, k, nQ):
    trace_op_i = qtn.TensorNetwork(
        [
            qtn.Tensor(
                0.5 * jnp.eye(2),
                inds=(f"bi{k}_q{j}", f"ki{k}_q{j}"),
                tags=[f"q{j}_I{k}"],
            )
            for j in range(nQ)
        ]
    )
    mpo_traced_double = ((2**nQ) * trace_op_i & mpo_traced).contract_tags(
        [f"q{j}_I{k}" for j in range(nQ)]
    )
    c1 = mpo_traced
    c2 = trace_op_i & mpo_traced_double
    v1 = jnp.real((c1 & c2.H).contract(optimize="greedy"))
    v2 = jnp.real((c1 & c1.H).contract(optimize="greedy"))

    print(type(c1))
    print(type(c2))
    return jnp.abs(1 - v1 / v2), mpo_traced_double


def causality_regularisation(mpo_half):
    ntimes = mpo_half.Ly
    nQ = mpo_half.Lx
    mpo_traced = produce_LPDO(mpo_half).copy()

    summed_causalities = []
    for k in reversed(range(1, ntimes)):
        trace_op_o = qtn.TensorNetwork(
            [
                qtn.Tensor(
                    jnp.eye(2),
                    inds=(f"bo{k}_q{j}", f"ko{k}_q{j}"),
                    tags=[f"q{j}_I{k+1}"],
                )
                for j in range(nQ)
            ]
        )
        tmp_tags = [
            f"q{j}_I{i}" for i in reversed(range(k, ntimes + 1)) for j in range(nQ)
        ]
        mpo_traced = (trace_op_o & mpo_traced).contract_tags(tmp_tags)
        c_term, mpo_traced = causality_term_k(mpo_traced, k, nQ)
        summed_causalities.append(c_term)
    return sum(summed_causalities)


def randomised_causality_regularisation(
    mpo_half, op_arrays, T_decomp=False, opt="auto-hq"
):
    #     ntimes = mpo_half.Ly
    #     nQ = mpo_half.Lx
    if T_decomp:
        first_half = mpo_half.select("PT")
        second_half = mpo_half.select("Tester")
        mpo_model = produce_LPDO(first_half & second_half)
    else:
        mpo_model = produce_LPDO(mpo_half.select("PT"))

    def evaluate_single_expectation(op):
        pauli_tn = causality_ops_to_pauli_tn(op)

        return (mpo_model & pauli_tn).contract(optimize=opt)

    p_list = jax.vmap(evaluate_single_expectation)(op_arrays)
    p_list = jnp.abs(jnp.real(p_list))

    return sum(p_list)


def trace_PT(mpo_half):
    # mpo_model = produce_LPDO(mpo_half.select('PT'))
    trace_keys = [[tuple([0 for i in range(7)]) for j in range(2)]]
    trace_arrays = causality_keys_to_op_arrays(trace_keys)

    return randomised_causality_regularisation(mpo_half, trace_arrays)


def compare_POVMs_to_identity(mpo_half):
    POVM0 = mpo_half.select("POVM_q0")
    POVM1 = mpo_half.select("POVM_q1")

    POVM0b = POVM0.H
    POVM0b.reindex_({"ko4_q0": "bo4_q0", "ki4_q0": "bi4_q0"})
    POVM1b = POVM1.H
    POVM1b.reindex_({"ko4_q1": "bo4_q1", "ki4_q1": "bi4_q1"})

    povm0_q0 = (POVM0 & POVM0b).contract().data[:, 0, :, 0]
    povm1_q0 = (POVM0 & POVM0b).contract().data[:, 1, :, 1]

    povm0_q1 = (POVM1 & POVM1b).contract().data[:, 0, :, 0]
    povm1_q1 = (POVM1 & POVM1b).contract().data[:, 1, :, 1]

    return jnp.linalg.norm(jnp.eye(2) - (povm0_q0 + povm1_q0)) + jnp.linalg.norm(
        jnp.eye(2) - (povm0_q1 + povm1_q1)
    )


def compute_likelihood(
    mpo_half,
    sequence_list,
    data,
    kappa,
    cArrays,
    X_decomp=False,
    T_decomp=False,
    opt="auto-hq",
):
    nD = len(data)
    # nQ = mpo_half.Lx
    #     nQ = sequence_list.shape[1]
    data_sum = sum(data)
    mpo_model = produce_LPDO(mpo_half)

    if X_decomp:

        def evaluate_single_prob(sequence):
            sequence_TN = op_arrays_to_single_vector_TN_padded_X_decomp(sequence)
            return (mpo_model & sequence_TN).contract(optimize=opt)

    else:

        def evaluate_single_prob(sequence):
            sequence_TN = op_arrays_to_single_vector_TN_padded(sequence)
            return (mpo_model & sequence_TN).contract(optimize=opt)

    p_list = jax.vmap(evaluate_single_prob)(sequence_list)
    p_list = jnp.real(p_list)

    p_list = jnp.abs(p_list) + 1e-10
    p_list = p_list / sum(p_list)
    p_list = data_sum * p_list
    p_list = jnp.log(p_list)
    data = jnp.array(data)

    c1 = -(1 / nD) * sum(data * p_list)  # data fit
    c2 = kappa * randomised_causality_regularisation(
        mpo_half, cArrays, T_decomp
    )  # causal bias
    #     c3 = np.abs(1 - trace_PT(mpo_half)) #unit trace
    # c4 = compare_POVMs_to_identity(mpo_half) #povm1 + povm2 = identity

    return c1 + c2  # + c4


def compute_probabilities(
    mpo_half, op_seqs, X_decomp=False, T_decomp=False, opt="auto-hq"
):
    mpo_model = produce_LPDO(mpo_half)
    #     nQ = mpo_half.Lx
    # pad_tensor = qtn.TensorNetwork([qtn.Tensor(0.5*I, inds = (f'kP_q{j}',f'bP_q{j}')) for j in range(nQ)])
    # mpo_model = pad_tensor & mpo_model

    if X_decomp:
        p_list = [
            (op_arrays_to_single_vector_TN_padded_X_decomp(S) & mpo_model).contract(
                optimize=opt
            )
            for S in op_seqs
        ]
    else:
        p_list = [
            (op_arrays_to_single_vector_TN_padded(S) & mpo_model).contract(optimize=opt)
            for S in op_seqs
        ]
    p_list = jnp.array(p_list)
    p_list = 0.5 * jnp.real(p_list)
    return p_list


def randomly_check_causality(mpo_half, nSamples, opt="auto-hq"):
    mpo_model = produce_LPDO(mpo_half.select("PT"))
    nQ = mpo_half.Lx
    nSteps = mpo_half.Ly - 1
    cKeys = generate_random_causality_keys(nSteps, nQ, nSamples)

    expectation_list = [
        (causality_key_to_pauli_tn(C) & mpo_model).contract(optimize=opt) for C in cKeys
    ]

    return sum(jnp.abs(jnp.array(expectation_list)))
