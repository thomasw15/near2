"""Benchmark Section 6.3 (CFAR, spectral norm on PSD cone).

    min_{X >= 0}  ||A - B X B^T||_{sigma,infty}

Compares Algorithm 3 against CVX/SCS. Writes an NPZ archive and figure PDFs.

Usage:
    python experiment_63_cfar.py
    python experiment_63_cfar.py --n-values 20 30 40 50
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import timeit

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy.linalg import polar


def PSD_projection(a: np.ndarray) -> np.ndarray:
    ah = (a + a.T) / 2
    _, h = polar(ah)
    return (h + ah) / 2


def find_k(sigmas: np.ndarray, rho: float) -> int:
    """Pick l in Lemma (spectral prox), Section 6 / experiment_prod."""
    ell = 1
    for i in range(1, len(sigmas) + 1):
        if sigmas[i - 1] >= (np.sum(sigmas[:i]) - 1.0 / rho) / i:
            ell = i
    return ell


def min_spectralnorm(Q: np.ndarray, U: np.ndarray, VT: np.ndarray, sigmas: np.ndarray, rho: float) -> np.ndarray:
    """Y-update for spectral norm (Lemma spec in neartwo5j)."""
    ell = find_k(sigmas, rho)
    min_dim = min(Q.shape)
    S = np.zeros(Q.shape)
    if np.sum(sigmas) > 1.0 / rho:
        S[:min_dim, :min_dim] = np.diag(sigmas)
        new_sigma = (np.sum(sigmas[:ell]) - 1.0 / rho) / ell
        np.fill_diagonal(S[:ell, :ell], new_sigma)
    return U @ S @ VT


def rank(vec: np.ndarray) -> int:
    for i in range(len(vec)):
        if vec[i] < 1e-12:
            return i
    return len(vec)


def _inv_mask_from_sb_sc(
    sb: np.ndarray, sc: np.ndarray, rb: int, rc: int, p: int, q: int
) -> np.ndarray:
    sigmaB = np.zeros(p)
    sigmaC = np.zeros(q)
    sigmaB[:rb] = sb[:rb]
    sigmaC[:rc] = sc[:rc]
    s = np.outer(sigmaB, sigmaC)
    lamda = sb[0] * sb[rb - 1] * sc[0] * sc[rc - 1]
    safe_s = np.where(s > 0, s, 1.0)
    return np.where(s > 0, 1.0 / (safe_s + lamda / safe_s), 0.0)


def _projection_rel_error(Y_prev: np.ndarray, Y: np.ndarray, denom_eps: float = 1e-12) -> float:
    return la.norm(Y_prev - Y, "fro") / max(la.norm(Y, "fro"), denom_eps)


def fro_PSD(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    UB: np.ndarray,
    sb: np.ndarray,
    VBT: np.ndarray,
    rb: int,
    UC: np.ndarray,
    sc: np.ndarray,
    VCT: np.ndarray,
    rc: int,
    state=None,
    tol: float = 1e-8,
    max_iter: int = 500,
):
    p = B.shape[1]
    q = C.shape[0]
    mask = _inv_mask_from_sb_sc(sb, sc, rb, rc, p, q)
    if state is None:
        Y = np.zeros((p, q))
        X = np.zeros((p, q))
        Z = np.zeros((p, q))
    else:
        Y, X, Z = (M.copy() for M in state)

    rel_error = float("inf")
    iter_count = 0
    while rel_error > tol and iter_count < max_iter:
        Ym = Y.copy()
        Y = PSD_projection(X - Z)
        W = Y + Z
        Ap = A - B @ W @ C
        temp = UB.T @ Ap @ VCT.T
        X = np.zeros((p, q))
        X[:rb, :rc] = temp[:rb, :rc]
        X = X * mask
        X = VBT.T @ X @ UC.T + W
        Z = Z - X + Y
        rel_error = _projection_rel_error(Ym, Y)
        iter_count += 1
    return X, (Y, X, Z)


def spectral_PSD(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    rho: float,
    iteration: int = 300,
    outer_tol: float = 1e-8,
    inner_tol: float = 1e-8,
) -> np.ndarray:
    UB, sb, VBT = la.svd(B, full_matrices=True)
    UC, sc, VCT = la.svd(C, full_matrices=True)
    rb = rank(sb)
    rc = rank(sc)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1], C.shape[0]])
    inner_state = None
    iter_count = 0
    rel_error = float("inf")

    while rel_error > outer_tol and iter_count < iteration:
        Ym = Y.copy()
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q, full_matrices=False)
        Y = min_spectralnorm(Q, U, VT, sq, rho)
        X, inner_state = fro_PSD(
            A + D - Y, B, C, UB, sb, VBT, rb, UC, sc, VCT, rc,
            state=inner_state, tol=inner_tol,
        )
        D = D - Y + A - B @ X @ C
        rel_error = _projection_rel_error(Ym, Y)
        iter_count += 1

    # Ensure returned X is exactly PSD.
    X = PSD_projection(X)
    return X


def solve_cvx_cfar(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> dict:
    t0 = timeit.default_timer()
    p = B.shape[1]
    X = cp.Variable((p, p), PSD=True)
    objective = cp.Minimize(cp.norm(A - B @ X @ B.T, 2))
    prob = cp.Problem(objective)

    prob.solve(solver=cp.SCS, eps_abs=eps, eps_rel=eps, verbose=False)
    elapsed = timeit.default_timer() - t0

    if X.value is None:
        return {"X": None, "time": elapsed, "status": str(prob.status), "obj": np.nan}

    Xv = np.asarray(X.value, dtype=float)
    obj = la.norm(A - B @ Xv @ B.T, 2)
    return {"X": Xv, "time": elapsed, "status": str(prob.status), "obj": float(obj)}


def _cvx_worker(queue: mp.Queue, A: np.ndarray, B: np.ndarray, eps: float):
    try:
        queue.put(("ok", solve_cvx_cfar(A, B, eps=eps)))
    except MemoryError as exc:
        queue.put(("oom", str(exc)))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def solve_cvx_cfar_with_timeout(
    A: np.ndarray, B: np.ndarray, eps: float = 1e-8, timeout_sec: float = 1800.0
) -> dict:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_cvx_worker, args=(queue, A, B, eps), daemon=True)
    t0 = timeit.default_timer()
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        elapsed = timeit.default_timer() - t0
        return {"X": None, "time": elapsed, "status": "timeout", "obj": np.nan}

    elapsed = timeit.default_timer() - t0

    if queue.empty():
        return {"X": None, "time": elapsed, "status": "failed_empty_result", "obj": np.nan}

    status, payload = queue.get()
    if status == "ok":
        payload["time"] = elapsed
        return payload
    if status == "oom":
        return {"X": None, "time": elapsed, "status": "oom", "obj": np.nan}
    return {"X": None, "time": elapsed, "status": f"error:{payload}", "obj": np.nan}


def run_custom_cfar(A: np.ndarray, B: np.ndarray, rho: float = 0.01) -> dict:
    # Spectral-norm custom solver for Section 6.3 CFAR.
    C = B.T
    t0 = timeit.default_timer()
    X = spectral_PSD(A, B, C, rho, iteration=300)
    elapsed = timeit.default_timer() - t0
    obj = la.norm(A - B @ X @ B.T, 2)
    return {"X": X, "time": elapsed, "obj": float(obj)}


def compute_detection_potential(A: np.ndarray, B: np.ndarray, X: np.ndarray) -> float:
    """Match experiment_dp.py: DP via largest/minus smallest singular values."""
    sA = la.svd(A, full_matrices=False, compute_uv=False)
    dp_A = float(sA[0] - sA[-1])

    BXBt = B @ X @ B.T
    sY = la.svd(BXBt, full_matrices=False, compute_uv=False)
    denom = float(sY[0] - sY[-1])
    if abs(denom) < 1e-15:
        return np.nan
    return dp_A / denom


def generate_cfar_instance(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    A = G @ G.T

    B = rng.standard_normal((n, p))
    s = rng.random(p) + 1.0
    UB, _, VBT = la.svd(B, full_matrices=True)
    B = UB[:, :p] @ np.diag(s) @ VBT[:p, :]
    return A, B


def run_benchmark(
    n_values: list[int], base_seed: int = 20260527, cvx_timeout_sec: float = 1800.0
) -> dict:
    payload: dict[str, np.ndarray] = {"n_values": np.array(n_values, dtype=int)}

    for n in n_values:
        p_values = np.arange(1, n + 1, dtype=int)
        t_custom = np.full(n, np.nan, dtype=float)
        t_cvx = np.full(n, np.nan, dtype=float)
        dp_custom = np.full(n, np.nan, dtype=float)
        dp_cvx = np.full(n, np.nan, dtype=float)
        obj_custom = np.full(n, np.nan, dtype=float)
        obj_cvx = np.full(n, np.nan, dtype=float)
        cvx_status = np.empty(n, dtype=object)

        print(f"\n=== n={n} ===")
        for idx, p in enumerate(p_values):
            print(f"  p={p}/{n}")
            A, B = generate_cfar_instance(n=n, p=p, seed=base_seed + 1000 * n + p)

            custom = run_custom_cfar(A, B)
            t_custom[idx] = custom["time"]
            obj_custom[idx] = custom["obj"]
            if custom["X"] is not None:
                dp_custom[idx] = compute_detection_potential(A, B, custom["X"])

            cvx = solve_cvx_cfar_with_timeout(A, B, timeout_sec=cvx_timeout_sec)
            t_cvx[idx] = cvx["time"]
            obj_cvx[idx] = cvx["obj"]
            cvx_status[idx] = cvx["status"]
            if cvx["X"] is not None:
                dp_cvx[idx] = compute_detection_potential(A, B, cvx["X"])

        payload[f"p_values_n{n}"] = p_values
        payload[f"time_custom_n{n}"] = t_custom
        payload[f"time_cvx_n{n}"] = t_cvx
        payload[f"dp_custom_n{n}"] = dp_custom
        payload[f"dp_cvx_n{n}"] = dp_cvx
        payload[f"obj_custom_n{n}"] = obj_custom
        payload[f"obj_cvx_n{n}"] = obj_cvx
        payload[f"cvx_status_n{n}"] = cvx_status

    return payload


def save_npz(payload: dict, out_file: str):
    np.savez(out_file, **payload)
    print(f"Saved {out_file}")


def _mask_p_upto_n_minus_1(p: np.ndarray, n: int) -> np.ndarray:
    """Keep p = 1, ..., n-1 (exclude p = n)."""
    return p <= (n - 1)


def plot_cfar_speed_objective_combined(payload: dict, out_pdf: str):
    """Speed (top) and spectral objective (bottom) in one 4x2 figure."""
    n_values = list(payload["n_values"])
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharey=False)

    for i, n in enumerate(n_values):
        col = i % 2
        row = i // 2
        p = payload[f"p_values_n{n}"]
        mask = _mask_p_upto_n_minus_1(p, n)

        ax_t = axes[row, col]
        t_custom = payload[f"time_custom_n{n}"]
        t_cvx = payload[f"time_cvx_n{n}"]
        ax_t.plot(p[mask], t_custom[mask], "b-o", label="Algorithm 3")
        ax_t.plot(p[mask], t_cvx[mask], "r-o", label="CVX")
        ax_t.set_yscale("log")
        ax_t.set_title(f"n = {n}")
        ax_t.set_ylabel("time (s)")
        ax_t.grid(True)

        ax_o = axes[row + 2, col]
        obj_custom = payload[f"obj_custom_n{n}"]
        obj_cvx = payload[f"obj_cvx_n{n}"]
        ax_o.plot(p[mask], obj_custom[mask], "b-o", label="Algorithm 3")
        ax_o.plot(p[mask], obj_cvx[mask], "r-o", label="CVX")
        ax_o.set_yscale("log")
        ax_o.set_ylabel(r"$\|A-BXB^\top\|_{\sigma,\infty}$")
        ax_o.grid(True)

    for ax in axes[2:, :].ravel():
        ax.set_xlabel("p")
    for ax in axes[:2, :].ravel():
        ax.set_xlabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


def plot_objective_difference_grid(payload: dict, out_pdf: str):
    """Absolute objective gap |f_alg3 - f_CVX| on log scale."""
    n_values = list(payload["n_values"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.ravel()

    for ax, n in zip(axes, n_values):
        p = payload[f"p_values_n{n}"]
        obj_custom = payload[f"obj_custom_n{n}"]
        obj_cvx = payload[f"obj_cvx_n{n}"]
        mask = _mask_p_upto_n_minus_1(p, n) & np.isfinite(obj_custom) & np.isfinite(obj_cvx)
        if np.any(mask):
            gap = np.abs(obj_custom[mask] - obj_cvx[mask])
            ax.plot(p[mask], np.maximum(gap, 1e-16), "k-o")
        ax.set_yscale("log")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("p")
        ax.set_ylabel("objective difference")
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


def plot_detection_potential_grid(payload: dict, out_pdf: str):
    n_values = list(payload["n_values"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.ravel()

    for ax, n in zip(axes, n_values):
        p = payload[f"p_values_n{n}"]
        dp_custom = payload[f"dp_custom_n{n}"]
        dp_cvx = payload[f"dp_cvx_n{n}"]
        mask = _mask_p_upto_n_minus_1(p, n)

        ax.plot(p[mask], dp_custom[mask], "b-o", label="Algorithm 3")
        ax.plot(p[mask], dp_cvx[mask], "r-o", label="CVX")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("p")
        ax.set_ylabel("detection potential")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Section 6.3 CFAR benchmark: Algorithm 3 vs CVX/SCS.")
    parser.add_argument(
        "--n-values",
        nargs="*",
        type=int,
        default=[20, 30, 40, 50],
        help="Values of n (default: 20 30 40 50)",
    )
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--cvx-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--out-npz", type=str, default="benchmark_63_cfar.npz")
    parser.add_argument("--out-combined-fig", type=str, default="cfar_comparison_50.pdf")
    parser.add_argument(
        "--extra-plots",
        action="store_true",
        help="Also write objective-difference and detection-potential grids",
    )
    args = parser.parse_args()

    payload = run_benchmark(
        args.n_values, base_seed=args.seed, cvx_timeout_sec=args.cvx_timeout_sec
    )
    save_npz(payload, args.out_npz)
    plot_cfar_speed_objective_combined(payload, args.out_combined_fig)
    if args.extra_plots:
        plot_objective_difference_grid(payload, "objective_difference_cfar.pdf")
        plot_detection_potential_grid(payload, "detection_potential_cfar.pdf")


if __name__ == "__main__":
    main()

