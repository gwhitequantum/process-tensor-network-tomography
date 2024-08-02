"""Helper functions for optimising dynamical decoupling sequences in a process"""

import jax.numpy as jnp
import numpy as np
import quimb as qu
import quimb.tensor as qtn

from preprocess import *
from utilities import *


def make_rz_unitary(theta):
    return jnp.array([[1, 0], [0, jnp.exp(1.0j * theta)]])


def params_to_rz_operator(param):
    return 4 * TN_choi_mat(make_rz_unitary(param)).T.reshape(2, 2, 2, 2)


def TN_choi_mat(unitary):
    left_subsystem = jnp.einsum("ij,lmjk->lmik", unitary, choi_bases)
    left_subsystem = jnp.einsum("ijkl,lm", left_subsystem, unitary.conj().T)

    combined_choi = jnp.einsum("ijkl,ijmn->kmln", left_subsystem, choi_bases)
    combined_choi = jnp.reshape(combined_choi, (4, 4))

    return 0.5 * combined_choi


def params_to_rz_tensor(initial_params, shift):
    # initial_params is just three rz params that make up a single unitary
    # initial_params should be structured as rz_to_u3_params(u3_params) for u3_params:=[theta,phi,lambda]

    # input in order measure, U_k-1, ... , U_0

    TN_list = []

    cStep = shift + 1

    TN_list.append(
        qtn.tensor_core.PTensor(
            params_to_rz_operator,
            initial_params[2],
            inds=(
                f"ki{cStep}_q{0}",
                f"kXmo{cStep-1}_q{0}",
                f"bi{cStep}_q{0}",
                f"bXmo{cStep-1}_q{0}",
            ),
            tags=["RZ"],
        )
    )
    TN_list.append(
        qtn.tensor_core.PTensor(
            params_to_rz_operator,
            initial_params[1],
            inds=(
                f"kXmi{cStep-1}_q{0}",
                f"kXpo{cStep-1}_q{0}",
                f"bXmi{cStep-1}_q{0}",
                f"bXpo{cStep-1}_q{0}",
            ),
            tags=["RZ"],
        )
    )
    cStep -= 1
    TN_list.append(
        qtn.tensor_core.PTensor(
            params_to_rz_operator,
            initial_params[0],
            inds=(
                f"kXpi{cStep}_q{0}",
                f"ko{cStep}_q{0}",
                f"bXpi{cStep}_q{0}",
                f"bo{cStep}_q{0}",
            ),
            tags=["RZ"],
        )
    )

    return qtn.TensorNetwork(TN_list)


def unparametrise_TN(PTN):
    un_TN = []
    for T in PTN.tensors:
        un_TN.append(T.unparametrize())
    un_TN = qtn.TensorNetwork(un_TN)
    return un_TN


def params_to_rz_PTN(param_list):
    TN_list = []
    for i, P in enumerate(param_list):
        TN_list.append(params_to_rz_tensor(P, i + 1))

    return qtn.TensorNetwork(TN_list)


NSTEPS = 8

ideal_input = jnp.array([[1, 0], [0, 0]])
prep_params = [clifford_param_dict[i] for i in range(NSTEPS)]
prep_Us = [jnp.array(make_unitary(*prep_params[i])) for i in range(NSTEPS)]
ideal_outs = [U @ ideal_input @ U.conj().T for U in prep_Us]

prep_TNs = [params_to_rz_tensor(u3_to_rz_params(P), shift=0) for P in prep_params]
prep_TNs = [unparametrise_TN(P) for P in prep_TNs]


meas_TN = params_to_rz_tensor(u3_to_rz_params([0, 0, 0]), shift=NSTEPS + 1)

X_params = [jnp.pi, 0, jnp.pi]
Y_params = [jnp.pi, 0, jnp.pi / 2]
XY8 = [X_params, Y_params, X_params, Y_params, X_params, Y_params, X_params, Y_params]
XY8_RZ = [u3_to_rz_params(P) for P in XY8]
id_params = [[0, 0, 0] for i in range(8)]
id_params_RZ = [u3_to_rz_params(P) for P in id_params]

xy8_seed = params_to_rz_PTN(jnp.array(XY8_RZ))
id_seed = params_to_rz_PTN(jnp.array(id_params_RZ))


def evaluate_DD_sequence(param_seq, PT_est):
    # PT_est in LPDO form
    ideal_dist = 0
    for i in range(len(prep_TNs)):
        tmp_op = param_seq & prep_TNs[i] & meas_TN
        output_DM = (
            (tmp_op & PT_est)
            .contract(output_inds=[f"bo{NSTEPS+1}_q0", f"ko{NSTEPS+1}_q0"])
            .data
        )
        output_DM = output_DM / jnp.trace(output_DM)

        ideal_dist += jnp.linalg.norm(ideal_outs[i] - output_DM, "nuc")

    return ideal_dist
