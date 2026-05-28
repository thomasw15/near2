"""
Section experiments for Schatten p-norm with p = 3/2.

This script generates NPZ data for forward-error (accuracy) and runtime (speed)
under four constraint settings:
- unconstrained
- product-constrained
- rank-constrained
- eigenvalue-constrained
"""
from __future__ import annotations

import argparse
import timeit

import numpy as np
import numpy.linalg as la

from experiment_3norm import generate_random, prod_Frobenius
from experiment_eig import eig_Frobenius, rank_r_approximation
from experiment_rank import rank_Frobenius
from product_experiment_utils import generate_product_trial
from schatten_prox import PNORM, min_phalfnorm

ADMM_ITERS = 300
RHO_ACC = 0.01
RHO_SPEED = 1e-7
SPEED_TOL = 1e-8

ACC_UNC = "accuracy_data_p_unconstrained_0527.npz"
ACC_PROD = "accuracy_data_p_prod_0527.npz"
ACC_RANK = "accuracy_data_p_rank_0527.npz"
ACC_EIG = "accuracy_data_p_eig_0527.npz"
SPD_UNC = "speed_data_p_unconstrained_0527.npz"
SPD_PROD = "speed_data_p_prod_0527.npz"
SPD_RANK = "speed_data_p_rank_0527.npz"
SPD_EIG = "speed_data_p_eig_0527.npz"


def solve_phalf_unconstrained(A, B, C, X_true, rho, iteration=ADMM_ITERS):
    Bpinv = la.pinv(B)
    Cpinv = la.pinv(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros((B.shape[1], C.shape[0]))
    forward_error_list = []
    for _ in range(iteration):
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_phalfnorm(Q, U, VT, sq, rho)
        X = Bpinv @ (A + D - Y) @ Cpinv
        D = D - Y + A - B @ X @ C
        forward_error_list.append(
            la.norm(X - X_true, "fro") / la.norm(X_true, "fro")
        )
    return X, forward_error_list


def tol_phalf_unconstrained(A, B, C, X_true, rho, tol=SPEED_TOL):
    Bpinv = la.pinv(B)
    Cpinv = la.pinv(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros((B.shape[1], C.shape[0]))
    forward_error = 1.0
    while forward_error > tol:
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_phalfnorm(Q, U, VT, sq, rho)
        X = Bpinv @ (A + D - Y) @ Cpinv
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro") / la.norm(X_true, "fro")
    return X


def solve_phalf_prod(A, B, C, F, G, H, X_true, rho, iteration=ADMM_ITERS):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros((B.shape[1], C.shape[0]))
    forward_error_list = []
    for _ in range(iteration):
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_phalfnorm(Q, U, VT, sq, rho)
        X = prod_Frobenius(
            A + D - Y, B, C, F, G, H, UB, sB, VBT, rB, UC, sC, VCT, rC
        )
        D = D - Y + A - B @ X @ C
        forward_error_list.append(
            la.norm(X - X_true, "fro") / la.norm(X_true, "fro")
        )
    return X, forward_error_list


def tol_phalf_prod(A, B, C, F, G, H, X_true, rho, tol=SPEED_TOL):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros((B.shape[1], C.shape[0]))
    forward_error = 1.0
    while forward_error > tol:
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_phalfnorm(Q, U, VT, sq, rho)
        X = prod_Frobenius(
            A + D - Y, B, C, F, G, H, UB, sB, VBT, rB, UC, sC, VCT, rC
        )
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro") / la.norm(X_true, "fro")
    return X


def solve_phalf_rank(A, B, C, X_true, r, rho, iteration=ADMM_ITERS):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros((B.shape[1], C.shape[0]))
    forward_error_list = []
    for _ in range(iteration):
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_phalfnorm(Q, U, VT, sq, rho)
        X = rank_Frobenius(
            A + D - Y, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, r
        )
        D = D - Y + A - B @ X @ C
        forward_error_list.append(
            la.norm(X - X_true, "fro") / la.norm(X_true, "fro")
        )
    return X, forward_error_list


def tol_phalf_rank(A, B, C, X_true, r, rho, tol=SPEED_TOL):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros((B.shape[1], C.shape[0]))
    forward_error = 1.0
    while forward_error > tol:
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_phalfnorm(Q, U, VT, sq, rho)
        X = rank_Frobenius(
            A + D - Y, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, r
        )
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro") / la.norm(X_true, "fro")
    return X


def solve_phalf_eig(A, B, C, X_true, eigenvalue, rho, iteration=ADMM_ITERS):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros((B.shape[1], C.shape[0]))
    forward_error_list = []
    for _ in range(iteration):
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_phalfnorm(Q, U, VT, sq, rho)
        X = eig_Frobenius(
            A + D - Y, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, eigenvalue
        )
        D = D - Y + A - B @ X @ C
        forward_error_list.append(
            la.norm(X - X_true, "fro") / la.norm(X_true, "fro")
        )
    return X, forward_error_list


def tol_phalf_eig(A, B, C, X_true, eigenvalue, rho, tol=SPEED_TOL):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros((B.shape[1], C.shape[0]))
    forward_error = 1.0
    while forward_error > tol:
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_phalfnorm(Q, U, VT, sq, rho)
        X = eig_Frobenius(
            A + D - Y, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, eigenvalue
        )
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro") / la.norm(X_true, "fro")
    return X


def run_accuracy_unconstrained(case=20, output_file=ACC_UNC):
    phalf_list = []
    n_list = list(range(1, case + 1))
    for i in range(case):
        print(f"[unconstrained] case {i + 1}/{case}", flush=True)
        n = 32
        np.random.seed(2000 + i)
        B = generate_random(n, max(1.0, np.random.rand() * n))
        C = generate_random(n, max(1.0, np.random.rand() * n))
        X_true = generate_random(n, max(1.0, np.random.rand() * n * n))
        A = B @ X_true @ C
        _, errs = solve_phalf_unconstrained(A, B, C, X_true, rho=RHO_ACC)
        phalf_list.append(errs[ADMM_ITERS - 1])
    np.savez(
        output_file,
        n_list=np.array(n_list),
        phalf_list=np.array(phalf_list),
        p=PNORM,
    )
    print(f"Saved {output_file}")
    return n_list, phalf_list


def run_accuracy_prod(
    case=20,
    output_file=ACC_PROD,
    trial_seed_base=1000,
    n=32,
):
    phalf_list = []
    n_list = list(range(1, case + 1))
    for i in range(case):
        print(f"[product] case {i + 1}/{case}", flush=True)
        trial = generate_product_trial(n, seed=trial_seed_base + i)
        _, errs = solve_phalf_prod(
            trial["A"],
            trial["B"],
            trial["C"],
            trial["F"],
            trial["G"],
            trial["H"],
            trial["X_true"],
            rho=RHO_ACC,
        )
        phalf_list.append(errs[ADMM_ITERS - 1])
    np.savez(
        output_file,
        n_list=np.array(n_list),
        phalf_list=np.array(phalf_list),
        p=PNORM,
    )
    print(f"Saved {output_file}")
    return n_list, phalf_list


def run_accuracy_rank(case=20, output_file=ACC_RANK, rank_r=10):
    phalf_list = []
    n_list = list(range(1, case + 1))
    for i in range(case):
        print(f"[rank] case {i + 1}/{case}", flush=True)
        n = 32
        np.random.seed(3000 + i)
        B = generate_random(n, max(1.0, np.random.rand() * n))
        C = generate_random(n, max(1.0, np.random.rand() * n))
        X_true = generate_random(n, max(1.0, np.random.rand() * n * n))
        X_true = rank_r_approximation(X_true, rank_r)
        A = B @ X_true @ C
        _, errs = solve_phalf_rank(A, B, C, X_true, r=rank_r, rho=RHO_ACC)
        phalf_list.append(errs[ADMM_ITERS - 1])
    np.savez(
        output_file,
        n_list=np.array(n_list),
        phalf_list=np.array(phalf_list),
        p=PNORM,
        rank_r=rank_r,
    )
    print(f"Saved {output_file}")
    return n_list, phalf_list


def run_accuracy_eig(case=20, output_file=ACC_EIG, eigenvalue=10):
    phalf_list = []
    n_list = list(range(1, case + 1))
    for i in range(case):
        print(f"[eig] case {i + 1}/{case}", flush=True)
        n = 32
        np.random.seed(4000 + i)
        B = generate_random(n, max(1.0, np.random.rand() * n))
        C = generate_random(n, max(1.0, np.random.rand() * n))
        X_true = generate_random(n, max(1.0, np.random.rand() * n * n))
        X_true = rank_r_approximation(
            X_true - eigenvalue * np.eye(n), n - 1
        ) + eigenvalue * np.eye(n)
        A = B @ X_true @ C
        _, errs = solve_phalf_eig(
            A, B, C, X_true, eigenvalue, rho=RHO_ACC
        )
        phalf_list.append(errs[ADMM_ITERS - 1])
    np.savez(
        output_file,
        n_list=np.array(n_list),
        phalf_list=np.array(phalf_list),
        p=PNORM,
        eigenvalue=eigenvalue,
    )
    print(f"Saved {output_file}")
    return n_list, phalf_list


def _run_speed_loop(step_fn, build_instance, n_trials=10):
    dimensions_ours = [2 ** (i + 1) for i in range(10)]
    phalf_times = []
    for n in dimensions_ours:
        print(f"  dimension n={n}", flush=True)
        total = 0.0
        for j in range(n_trials):
            t0 = timeit.default_timer()
            step_fn(*build_instance(n, j))
            total += timeit.default_timer() - t0
        phalf_times.append(total / n_trials)
    return dimensions_ours, phalf_times


def run_speed_unconstrained(output_file=SPD_UNC):
    print("[speed unconstrained]", flush=True)

    def build(n, j):
        np.random.seed(50_000 * n + j)
        B = generate_random(n, max(1.0, np.random.rand() * n))
        C = generate_random(n, max(1.0, np.random.rand() * n))
        X_true = generate_random(n, max(1.0, np.random.rand() * n * n))
        A = B @ X_true @ C
        return (A, B, C, X_true)

    def step(A, B, C, X_true):
        tol_phalf_unconstrained(A, B, C, X_true, rho=RHO_SPEED, tol=SPEED_TOL)

    dims, times = _run_speed_loop(step, build)
    np.savez(
        output_file,
        dimensions_ours=np.array(dims),
        phalf_times=np.array(times),
        p=PNORM,
    )
    print(f"Saved {output_file}")


def run_speed_prod(output_file=SPD_PROD):
    print("[speed product]", flush=True)

    def build(n, j):
        trial = generate_product_trial(n, seed=60_000 * n + j)
        return (
            trial["A"],
            trial["B"],
            trial["C"],
            trial["F"],
            trial["G"],
            trial["H"],
            trial["X_true"],
        )

    def step(A, B, C, F, G, H, X_true):
        tol_phalf_prod(A, B, C, F, G, H, X_true, rho=RHO_SPEED, tol=SPEED_TOL)

    dims, times = _run_speed_loop(step, build)
    np.savez(
        output_file,
        dimensions_ours=np.array(dims),
        phalf_times=np.array(times),
        p=PNORM,
    )
    print(f"Saved {output_file}")


def run_speed_rank(output_file=SPD_RANK, rank_r=10):
    print("[speed rank]", flush=True)

    def build(n, j):
        np.random.seed(70_000 * n + j)
        B = generate_random(n, max(1.0, np.random.rand() * n))
        C = generate_random(n, max(1.0, np.random.rand() * n))
        X_true = rank_r_approximation(
            generate_random(n, max(1.0, np.random.rand() * n * n)), rank_r
        )
        A = B @ X_true @ C
        return (A, B, C, X_true, rank_r)

    def step(A, B, C, X_true, r):
        tol_phalf_rank(A, B, C, X_true, r, rho=RHO_SPEED, tol=SPEED_TOL)

    dims, times = _run_speed_loop(step, build)
    np.savez(
        output_file,
        dimensions_ours=np.array(dims),
        phalf_times=np.array(times),
        p=PNORM,
        rank_r=rank_r,
    )
    print(f"Saved {output_file}")


def run_speed_eig(output_file=SPD_EIG, eigenvalue=10):
    print("[speed eig]", flush=True)

    def build(n, j):
        np.random.seed(80_000 * n + j)
        B = generate_random(n, max(1.0, np.random.rand() * n))
        C = generate_random(n, max(1.0, np.random.rand() * n))
        X_true = generate_random(n, max(1.0, np.random.rand() * n * n))
        X_true = rank_r_approximation(
            X_true - eigenvalue * np.eye(n), n - 1
        ) + eigenvalue * np.eye(n)
        A = B @ X_true @ C
        return (A, B, C, X_true, eigenvalue)

    def step(A, B, C, X_true, ev):
        tol_phalf_eig(A, B, C, X_true, ev, rho=RHO_SPEED, tol=SPEED_TOL)

    dims, times = _run_speed_loop(step, build)
    np.savez(
        output_file,
        dimensions_ours=np.array(dims),
        phalf_times=np.array(times),
        p=PNORM,
        eigenvalue=eigenvalue,
    )
    print(f"Saved {output_file}")


def run_all_accuracy(case=20):
    run_accuracy_unconstrained(case=case)
    run_accuracy_prod(case=case)
    run_accuracy_rank(case=case)
    run_accuracy_eig(case=case)


def run_all_speed():
    run_speed_unconstrained()
    run_speed_prod()
    run_speed_rank()
    run_speed_eig()


def main():
    parser = argparse.ArgumentParser(
        description="Run Schatten p=3/2 accuracy/speed experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["accuracy", "speed", "all"],
        default="all",
        help="Which experiment set to run (default: all).",
    )
    parser.add_argument(
        "--case",
        type=int,
        default=20,
        help="Number of trials for accuracy runs (default: 20).",
    )
    args = parser.parse_args()

    if args.mode in ("accuracy", "all"):
        run_all_accuracy(case=args.case)
    if args.mode in ("speed", "all"):
        run_all_speed()


if __name__ == "__main__":
    main()
