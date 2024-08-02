"""some helper tools for performing SU(4) optimisations
on process tensors which are in the KAK form from Fig 14 of arXiv:2312.08454"""

import quimb as qu
import quimb.tensor as qtn

from process_tensor_networks import produce_LPDO
from utilities import *


def generate_2q_unitary_choi(unitary, i_name, o_name):
    T1 = qtn.Tensor(
        bell_vec.reshape(2, 2, 1),
        inds=("k" + i_name + "q0", "k" + o_name + "q0", "B"),
        tags=["C0"],
    )
    T2 = qtn.Tensor(
        bell_vec.reshape(2, 2, 1),
        inds=("k" + i_name + "q1", "k" + o_name + "q1", "B"),
        tags=["C1"],
    )

    tmp_choi = T1 & T2

    tmp_choi.gate_inds_(
        unitary, inds=("k" + o_name + "q0", "k" + o_name + "q1"), contract="split"
    )

    return tmp_choi


def prepare_PT_for_opt(input_TN):
    input_LPDO = produce_LPDO(input_TN, X_decomp=True)
    PT_and_pulses_LPDO = input_LPDO.select(
        ["PT", "sqrtX"], "any"
    )  # leave out the POVMs

    trace_tensor = qtn.Tensor(
        np.eye(4).reshape(2, 2, 2, 2),
        inds=("ko0_q0", "ko0_q1", "bo0_q0", "bo0_q1"),
        tags=["TRACE"],
    )

    PT_traced = trace_tensor & PT_and_pulses_LPDO
    PT_traced.contract_ind("ko0_q0")
    PT_traced.contract_ind(ind="ko0_q1")

    PT_traced.contract_ind(ind="bo0_q0")
    PT_traced.contract_ind(ind="bo0_q1")

    return PT_traced


def params_to_operator_SU4(param):
    return 2 * TN_choi_vec(make_rz_unitary(param)).conj().reshape(2, 2)


def params_to_PTN_SU4(param_list, example_op):

    nQ = 2
    k = 4

    TN_list = []
    for i, P in enumerate(param_list):
        TN_list.append(
            qtn.tensor_core.PTensor(
                params_to_operator_SU4,
                P,
                inds=example_op.tensors[i].inds,
                tags=example_op.tensors[i].tags,
            )
        )

    OTN_ket = qtn.TensorNetwork(TN_list)
    return OTN_ket


def PTN_ket_to_full_SU4(OTN_ket):
    nQ = 2
    k = 4

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
    OTN_bra.reindex_({"kP": "bP"})

    return qtn.TensorNetwork(OTN_ket & OTN_bra)


def params_to_su4_overlap(PTN, process_tensor, example_op, target_su4, opt="auto-hq"):
    full_op = PTN_ket_to_full_SU4(PTN)

    return -(1 / 16) * np.abs(
        (full_op & process_tensor & target_su4.H).contract(optimize=opt)
    )
