"""Creation and manipulation of tensor network code for process tensors"""

import jax.numpy as jnp
import numpy as np
import quimb.tensor as qtn
import scipy

from utilities import *


def create_PT_MPO_guess_full_kraus(nsteps, bond_dim, K_list, q_str, pos):
    # creates one half of a PT MPO 'ring'
    # Each site is a rank 4 tensor (except \rho_0, which is 3): middle ones have two bond and two site
    # Final step has two site, one bond, one kraus (K1)
    # initial state has one site, one bond, one kraus (K0)
    # q_str will be an addition to each bond to label it for each qubit
    # pos will be 'u', 'd', or 'ud' to indicate the position of each qubit

    extra_shape = tuple(1 for i in range(len(pos)))
    extra_label = tuple(f"k{P}" + "_t{}" + f"_{q_str}" for P in pos)

    final_shape = (K_list[-1], 2, 2, min(4, bond_dim)) + extra_shape
    final_step = 0.999 * (2 * np.random.rand(*final_shape) - 1) + 0.999 * (
        1.0j * (2 * np.random.rand(*final_shape) - 1)
    )
    final_step = qtn.Tensor(
        jnp.array(final_step),
        inds=(
            f"K{nsteps}_{q_str}",
            f"ko{nsteps}_{q_str}",
            f"ki{nsteps}_{q_str}",
            f"bond-{nsteps}_{q_str}",
        )
        + tuple(label.format(nsteps) for label in extra_label),
        tags=[f"{q_str}_I{nsteps}", "PT", f"ROW{q_str}", f"COL{nsteps}"],
    )

    initial_shape = (min(2, bond_dim), 2, K_list[0]) + extra_shape
    initial_state = 0.999 * (2 * np.random.rand(*initial_shape) - 1) + 0.999 * 1.0j * (
        2 * np.random.rand(*initial_shape) - 1
    )
    initial_state = qtn.Tensor(
        jnp.array(initial_state, dtype=complex),
        inds=(f"bond-0_{q_str}", f"ko0_{q_str}", f"K{0}_{q_str}")
        + tuple(label.format(0) for label in extra_label),
        tags=[f"{q_str}_I0", "PT", f"ROW{q_str}", f"COL{0}"],
    )

    middle_sites = []

    for i in reversed(range(1, nsteps)):
        dim_left_L = 4 ** (nsteps - i)
        dim_right_L = 4 ** (i) * 2
        dim_left_R = 4 ** (nsteps - i + 1)
        dim_right_R = 4 ** (i - 1) * 2

        bond_size_L = min(dim_left_L, dim_right_L, bond_dim)
        bond_size_R = min(dim_left_R, dim_right_R, bond_dim)

        tmp_shape = (bond_size_L, 2, 2, bond_size_R, K_list[i]) + extra_shape
        tmp_middle = 0.999 * (2 * np.random.rand(*tmp_shape) - 1) + 0.999 * (
            1.0j * (2 * np.random.rand(*tmp_shape) - 1)
        )
        tmp_middle = qtn.Tensor(
            jnp.array(tmp_middle),
            inds=(
                f"bond-{i}-L_{q_str}",
                f"ko{i}_{q_str}",
                f"ki{i}_{q_str}",
                f"bond-{i}-R_{q_str}",
                f"K{i}_{q_str}",
            )
            + tuple(label.format(i) for label in extra_label),
            tags=[f"{q_str}_I{i}", "PT", f"ROW{q_str}", f"COL{i}"],
        )

        middle_sites.append(tmp_middle)

    total_tensors = [final_step] + middle_sites + [initial_state]

    for i in range(nsteps):
        qtn.tensor_core.connect(total_tensors[i], total_tensors[i + 1], 3, 0)

    return qtn.TensorNetwork(total_tensors)


def create_PT_PEPO_guess(nsteps, nqubits, bond_dims_t, bond_dims_q, K_lists):
    # First will create nqubits worth of just temporal PT MPOs
    # Each bond collection will be a list of lists (with |bond_dims_t| = nqubits and |bond_dims_q| = nqubits-1)
    # Will create these lines with their own function first and then expand and connect the bonds
    assert len(K_lists) == nqubits, "Mismatch of bond allocations!"
    individual_PTs = [
        create_PT_MPO_guess_full_kraus(nsteps, bond_dims_t[0], K_lists[0], f"q{0}", "u")
    ]
    individual_PTs += [
        create_PT_MPO_guess_full_kraus(
            nsteps, bond_dims_t[i + 1], K_lists[i + 1], f"q{i+1}", "ud"
        )
        for i, K in enumerate(K_lists[1:-1])
    ]
    if nqubits > 1:
        individual_PTs += [
            create_PT_MPO_guess_full_kraus(
                nsteps, bond_dims_t[-1], K_lists[-1], f"q{nqubits-1}", "d"
            )
        ]

    connect_dict = {j: 5 for j in range(nsteps)}
    connect_dict[0] = 4
    connect_dict[nsteps] = 3

    for i in range(nsteps + 1):
        for j in range(nqubits - 1):
            connect_0 = connect_dict[i]
            connect_1 = connect_0 + 1
            if j == nqubits - 2:
                connect_1 = connect_0

            T1 = individual_PTs[j].tensors[i]
            T2 = individual_PTs[j + 1].tensors[i]

            T1.expand_ind(T1.inds[connect_0], bond_dims_q[i][j])
            T2.expand_ind(T2.inds[connect_1], bond_dims_q[i][j])

            qtn.tensor_core.connect(
                individual_PTs[j].tensors[i],
                individual_PTs[j + 1].tensors[i],
                connect_0,
                connect_1,
            )

    PTTN = qtn.TensorNetwork(individual_PTs)
    PTTN = qtn.tensor_2d.TensorNetwork2DFlat.from_TN(
        PTTN,
        site_tag_id="q{}_I{}",
        Ly=nsteps + 1,
        Lx=nqubits,
        row_tag_id="ROWq{}",
        col_tag_id="COL{}",
    )
    return PTTN


# def create_X_decomp(nsteps, q_str, rand_strength=0.01):
#     sqrtX = scipy.linalg.sqrtm(X)

#     tmp_data = TN_choi_vec(sqrtX).reshape(2, 2).conj()
#     tmp_data = jnp.array(
#         [
#             tmp_data,
#             rand_strength * np.random.rand(2, 2)
#             + rand_strength * 1.0j * np.random.rand(2, 2),
#         ]
#     )

#     meas_data_0 = jnp.array(
#         [
#             jnp.array([1.0, 0.0]),
#             rand_strength * np.random.rand(2)
#             + 1.0j * rand_strength * np.random.rand(2),
#         ]
#     )
#     meas_data_1 = jnp.array(
#         [
#             jnp.array([0.0, 1.0]),
#             rand_strength * np.random.rand(2)
#             + 1.0j * rand_strength * np.random.rand(2),
#         ]
#     )
#     measure_data = jnp.array([meas_data_0, meas_data_1])

#     X_TN_k = []
#     for i in range(nsteps + 1):
#         tmp_Tp = qtn.Tensor(
#             tmp_data,
#             inds=(f"KXp{i}_" + q_str, f"kXpo{i}_" + q_str, f"kXpi{i}_" + q_str),
#             tags=["sqrtX"],
#         )
#         X_TN_k.append(tmp_Tp)
#         tmp_Tm = qtn.Tensor(
#             tmp_data,
#             inds=(f"KXm{i}_" + q_str, f"kXmo{i}_" + q_str, f"kXmi{i}_" + q_str),
#             tags=["sqrtX"],
#         )
#         X_TN_k.append(tmp_Tm)

#     X_TN_k.append(
#         qtn.Tensor(
#             measure_data,
#             inds=(f"ko{nsteps+1}_" + q_str, "KM_" + q_str, f"ki{nsteps+1}_" + q_str),
#             tags=["POVM_" + q_str],
#         )
#     )
#     return qtn.TensorNetwork(X_TN_k)


def create_X_decomp(nsteps, q_str, rand_strength=0.01):
    sqrtX = scipy.linalg.sqrtm(X)
    tmp_data = TN_choi_vec(sqrtX).reshape(2, 2).conj()
    tmp_data = jnp.array(
        [
            tmp_data,
            rand_strength * np.random.rand(2, 2)
            + rand_strength * 1.0j * np.random.rand(2, 2),
        ]
    )

    meas_data_0 = jnp.array(
        [
            np.array([1.0, 0.0]),
            rand_strength * np.random.rand(2)
            + 1.0j * rand_strength * np.random.rand(2),
        ]
    )
    meas_data_1 = jnp.array(
        [
            np.array([0.0, 1.0]),
            rand_strength * np.random.rand(2)
            + 1.0j * rand_strength * np.random.rand(2),
        ]
    )
    measure_data = jnp.array([meas_data_0, meas_data_1])

    X_TN_k = []
    for i in range(nsteps + 1):
        tmp_Tp = qtn.Tensor(
            tmp_data,
            inds=(f"KXp{i}_" + q_str, f"kXpo{i}_" + q_str, f"kXpi{i}_" + q_str),
            tags=["sqrtX"],
        )
        X_TN_k.append(tmp_Tp)
        tmp_Tm = qtn.Tensor(
            tmp_data,
            inds=(f"KXm{i}_" + q_str, f"kXmo{i}_" + q_str, f"kXmi{i}_" + q_str),
            tags=["sqrtX"],
        )
        X_TN_k.append(tmp_Tm)

    X_TN_k.append(
        qtn.Tensor(
            measure_data,
            inds=(f"ko{nsteps+1}_" + q_str, "KM_" + q_str, f"ki{nsteps+1}_" + q_str),
            tags=["POVM_" + q_str],
        )
    )
    return qtn.TensorNetwork(X_TN_k)


# def create_PEPO_X_decomp(nsteps, nQ, rand_strength=0.01, bond_dim=1):
#     individual_decomps = [
#         create_X_decomp(nsteps, f"q{i}", rand_strength) for i in range(nQ)
#     ]
#     individual_decomps = qtn.TensorNetwork(individual_decomps)

#     if bond_dim > 1:
#         for i in range(2 * (nsteps + 1)):
#             if i == 0:
#                 individual_decomps.tensors[i].new_ind(f"left_bond_{i}")
#                 individual_decomps.tensors[i].expand_ind(f"left_bond_{i}", bond_dim)

#             elif i == 2 * (nsteps + 1) - 1:
#                 individual_decomps.tensors[i].new_ind(f"right_bond_{i}")
#                 individual_decomps.tensors[i].expand_ind(f"right_bond_{i}", bond_dim)
#             else:
#                 individual_decomps.tensors[i].new_ind(f"left_bond_{i}")
#                 individual_decomps.tensors[i].expand_ind(f"left_bond_{i}", bond_dim)
#                 individual_decomps.tensors[i].new_ind(f"right_bond_{i}")
#                 individual_decomps.tensors[i].expand_ind(f"right_bond_{i}", bond_dim)

#         for i in range(2 * (nsteps + 1) - 1):
#             if i == 0:
#                 qtn.tensor_core.connect(
#                     individual_decomps.tensors[i],
#                     individual_decomps.tensors[i + 1],
#                     0,
#                     0,
#                 )
#             if i > 0:
#                 qtn.tensor_core.connect(
#                     individual_decomps.tensors[i],
#                     individual_decomps.tensors[i + 1],
#                     1,
#                     0,
#                 )
#         rand_strength = 0.01

#         for i in range(2 * (nsteps + 1)):
#             tmp_D = individual_decomps.tensors[i].data
#             individual_decomps.tensors[i].modify(
#                 data=tmp_D
#                 + rand_strength * np.random.rand(*tmp_D.shape)
#                 + rand_strength * 1.0j * np.random.rand(*tmp_D.shape)
#             )
#         individual_decomps.mangle_inner_()

#     return individual_decomps


def create_PEPO_X_decomp(nsteps, nQ, rand_strength=0.01):
    individual_decomps = [
        create_X_decomp(nsteps, f"q{i}", rand_strength) for i in range(nQ)
    ]

    return qtn.TensorNetwork(individual_decomps)


def produce_LPDO(first_half):

    bra_tn = first_half.copy().H
    bra_tn.reindex_(
        {
            ind: "b" + ind[1:]
            for ind in first_half.outer_inds()
            if ind[0] == "k" and ind[1] != "M"
        }
    )

    return first_half & bra_tn


def expand_initial_guess_(
    guess,
    kraus_bond_list,
    horizontal_bond_list,
    vertical_bond_list,
    rand_strength=0.05,
    squeeze=True,
):
    nS = guess.Ly - 1
    nQ = guess.Lx
    # replace data
    for i in range(nQ):
        tmp_shape = guess[i, 0].shape
        tmp_zero = zero_vec.reshape(*tmp_shape)
        guess[i, 0].modify(data=tmp_zero)

    for i in range(nQ):
        for j in range(1, nS + 1):
            tmp_shape = guess[i, j].shape
            tmp_bell = bell_mid.reshape(*tmp_shape)
            guess[i, j].modify(data=tmp_bell)
    # expand Kraus indices
    for i in range(nQ):
        for j in range(nS + 1):
            T = guess[i, j]
            if "K{}_q{}".format(j, i) in T.inds:
                T.expand_ind("K{}_q{}".format(j, i), kraus_bond_list[i][j])

    for i in range(nQ):
        for j in range(nS + 1):
            T = guess[i, j]
            if j == 0:
                T.expand_ind(T.inds[0], horizontal_bond_list[i][j])
            if j > 0 and j < nS:
                T.expand_ind(T.inds[0], horizontal_bond_list[i][j])
                T.expand_ind(T.inds[3], horizontal_bond_list[i][j - 1])
            if j == nS:
                T.expand_ind(T.inds[3], horizontal_bond_list[i][j - 1])

    for i in range(nQ - 1):
        for j in range(nS + 1):
            T1 = guess[i, j]
            T2 = guess[i + 1, j]
            tmp_bonds = T1.filter_bonds(T2)[0][0]
            T1.expand_ind(tmp_bonds, vertical_bond_list[j][i])
            T2.expand_ind(tmp_bonds, vertical_bond_list[j][i])

    for T in guess.tensors:
        tmp = T.data.copy()
        tmp += rand_strength * (
            np.random.rand(*tmp.shape) + 1.0j * np.random.rand(*tmp.shape)
        )
        T.modify(data=tmp)
    if squeeze:
        guess.squeeze_()
    return None


def create_single_X(nsteps, q_str, rank=2, rand_strength=0.01):
    tmp_data = TN_choi_vec(sqrtX).reshape(2, 2).conj()
    X_data = [tmp_data]
    for i in range(rank - 1):
        X_data.append(
            rand_strength * np.random.rand(2, 2)
            + rand_strength * 1.0j * np.random.rand(2, 2)
        )

    X_data = jnp.array(X_data)

    meas_data_0 = jnp.array(
        [
            jnp.array(
                [
                    1.0,
                    rand_strength * np.random.rand()
                    + 1.0j * rand_strength * np.random.rand(),
                ]
            ),
            rand_strength * np.random.rand(2)
            + 1.0j * rand_strength * np.random.rand(2),
        ]
    )
    # meas_data_1 = np.array([np.array([0.0,1.0]), rand_strength * np.random.rand(2) + 1.j*rand_strength*np.random.rand(2)])
    meas_data_1 = jnp.array(
        [
            rand_strength * np.random.rand(2)
            + 1.0j * rand_strength * np.random.rand(2),
            jnp.array(
                [
                    rand_strength * np.random.rand()
                    + 1.0j * rand_strength * np.random.rand(),
                    1.0,
                ]
            ),
        ]
    )
    meas_data_2 = jnp.array(
        rand_strength * np.random.rand(2, 2)
        + 1.0j * rand_strength * np.random.rand(2, 2)
    )
    meas_data_3 = jnp.array(
        rand_strength * np.random.rand(2, 2)
        + 1.0j * rand_strength * np.random.rand(2, 2)
    )

    measure_data = jnp.array([meas_data_0, meas_data_1, meas_data_2, meas_data_3])

    X_TN = [
        qtn.Tensor(
            X_data,
            inds=(f"KXp{0}_" + q_str, f"kXpo{0}_" + q_str, f"kXpi{0}_" + q_str),
            tags=["sqrtX", "decomp"],
        )
    ]

    X_TN.append(
        qtn.Tensor(
            measure_data,
            inds=("KM_" + q_str, f"ko{nsteps+1}_" + q_str, f"ki{nsteps+1}_" + q_str),
            tags=["POVM_" + q_str, "decomp"],
        )
    )

    return qtn.TensorNetwork(X_TN)


def create_X_set(nsteps, nQ, rank=2, rand_strength=0.01):
    individual_decomps = [
        create_single_X(nsteps, f"q{i}", rank, rand_strength) for i in range(nQ)
    ]

    return qtn.TensorNetwork(individual_decomps)
