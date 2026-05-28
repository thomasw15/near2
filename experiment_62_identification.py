"""Benchmark Section 6.2 (Hankel system identification, nuclear norm).

    min_X  ||X C||_*   s.t.  ||X_hat - X||_F <= delta,  X Hankel.

Compares Algorithm 3 against CVX/SCS.
Writes an NPZ archive and two PDFs (runtime, feasibility violation).

Usage:
    python experiment_62_fixed.py
    python experiment_62_fixed.py --n-values 10 20 30 40 50
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import timeit
from dataclasses import dataclass

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


@dataclass
class ResultRow:
    n: int
    r: int
    custom_time: float
    cvx_time: float
    custom_obj: float
    cvx_obj: float
    custom_violation: float
    cvx_violation: float
    custom_status: str
    cvx_status: str


def hankel_from_vector(h: np.ndarray, n: int) -> np.ndarray:
    m = n + 1
    X = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            X[i, j] = h[i + j]
    return X


def hankel_expr_from_variable(h: cp.Variable, n: int) -> cp.Expression:
    m = n + 1
    rows = [[h[i + j] for j in range(m)] for i in range(m)]
    return cp.bmat(rows)


def generate_instance(
    n: int, r: int, delta: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = n + 1
    h_true = rng.standard_normal(2 * n + 1)
    X_true = hankel_from_vector(h_true, n)
    C = rng.standard_normal((m, r))
    noise = rng.standard_normal((m, m))
    noise = delta * noise / max(la.norm(noise, "fro"), 1e-12)
    X_hat = X_true + noise
    return C, X_hat, X_true


# ---------------------------------------------------------------------------
# Inlined solver utilities (formerly imported from helper_fixed.py)
# ---------------------------------------------------------------------------


def hankel_projection(a: np.ndarray) -> np.ndarray:
    """Project a square matrix onto the subspace of Hankel matrices."""
    n = a.shape[0]
    idx_i, idx_j = np.indices((n, n))
    k = idx_i + idx_j
    sums = np.zeros(2 * n - 1, dtype=a.dtype)
    np.add.at(sums, k.ravel(), a.ravel())
    lengths = np.array([min(s + 1, 2 * n - 1 - s, n) for s in range(2 * n - 1)], dtype=a.dtype)
    means = sums / lengths
    return means[k]


def min_1norm(Q: np.ndarray, U: np.ndarray, VT: np.ndarray, sigmas: np.ndarray, rho: float) -> np.ndarray:
    s_thr = np.maximum(sigmas - 1.0 / rho, 0.0)
    return (U * s_thr) @ VT


def rank(vec: np.ndarray, tol: float = 1e-12) -> int:
    for i in range(len(vec)):
        if vec[i] < tol:
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


def fro_hankel_distance(
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
    Ex: np.ndarray,
    delta: float,
    state=None,
    tol: float = 1e-9,
    max_iter: int = 5000,
):
    p = B.shape[1]
    q = C.shape[0]
    mask = _inv_mask_from_sb_sc(sb, sc, rb, rc, p, q)

    if state is None:
        X = np.zeros((p, q))
        Y = np.zeros((p, q))
        Z1 = np.zeros((p, q))
        W = Ex.copy() if Ex.shape == (p, q) else np.zeros((p, q))
        Z2 = np.zeros((p, q))
    else:
        X, Y, Z1, W, Z2 = (M.copy() for M in state)

    rel_error = float("inf")
    iter_count = 0
    while rel_error > tol and iter_count < max_iter:
        X_prev = X
        Y = hankel_projection(W - Z2)
        T = 0.5 * (X + Y + Z1 + Z2)
        direction = T - Ex
        distance = la.norm(direction, "fro")
        if distance <= delta:
            W = T
        else:
            W = Ex + (delta / distance) * direction

        P = W - Z1
        Ap = A - B @ P @ C
        temp = UB.T @ Ap @ VCT.T
        X_temp = np.zeros((p, q))
        X_temp[:rb, :rc] = temp[:rb, :rc]
        X_temp = X_temp * mask
        X = VBT.T @ X_temp @ UC.T + P

        Z1 = Z1 + X - W
        Z2 = Z2 + Y - W
        rel_error = la.norm(X - X_prev, "fro") / max(la.norm(X, "fro"), 1e-12)
        iter_count += 1

    return X, (X, Y, Z1, W, Z2)


def nuclear_hankel_distance(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    rho: float,
    Ex: np.ndarray,
    delta: float,
    iteration: int = 300,
    inner_tol: float = 1e-9,
    inner_max_iter: int = 5000,
    outer_tol: float = 1e-10,
) -> np.ndarray:
    UB, sb, VBT = la.svd(B, full_matrices=True)
    UC, sc, VCT = la.svd(C, full_matrices=True)
    rb = rank(sb)
    rc = rank(sc)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros((B.shape[1], C.shape[0]))
    inner_state = None

    for i in range(iteration):
        Y_prev = Y
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q, full_matrices=False)
        Y = min_1norm(Q, U, VT, sq, rho)
        X, inner_state = fro_hankel_distance(
            A + D - Y,
            B,
            C,
            UB,
            sb,
            VBT,
            rb,
            UC,
            sc,
            VCT,
            rc,
            Ex,
            delta,
            state=inner_state,
            tol=inner_tol,
            max_iter=inner_max_iter,
        )
        D = D - Y + A - B @ X @ C
        if i > 0:
            change = la.norm(Y - Y_prev, "fro") / max(la.norm(Y, "fro"), 1e-12)
            if change < outer_tol:
                break

    X = hankel_projection(X)
    direction = X - Ex
    dist = la.norm(direction, "fro")
    if dist > delta:
        X = Ex + (delta / dist) * direction
    return X


# ---------------------------------------------------------------------------
# Custom solver
# ---------------------------------------------------------------------------

def run_custom(
    C: np.ndarray,
    X_hat: np.ndarray,
    delta: float,
    rho: float = 0.1,
    iteration: int = 300,
    inner_tol: float = 1e-10,
    inner_max_iter: int = 5000,
) -> dict:
    """Solve Section 6.2 via Algorithm 2 / Algorithm 3 of neartwo."""
    m, r = C.shape
    A = np.zeros((m, r))
    B = np.eye(m)

    t0 = timeit.default_timer()
    X = nuclear_hankel_distance(
        A, B, C, rho=rho, Ex=X_hat, delta=delta,
        iteration=iteration,
        inner_tol=inner_tol,
        inner_max_iter=inner_max_iter,
    )
    elapsed = timeit.default_timer() - t0

    obj = float(la.norm(X @ C, "nuc"))
    violation = float(max(0.0, la.norm(X_hat - X, "fro") - delta))
    return {"X": X, "time": elapsed, "obj": obj, "violation": violation}


def _custom_worker(queue, C, X_hat, delta, rho, iteration, inner_tol, inner_max_iter):
    try:
        queue.put(("ok", run_custom(C, X_hat, delta, rho=rho, iteration=iteration,
                                     inner_tol=inner_tol, inner_max_iter=inner_max_iter)))
    except MemoryError as exc:
        queue.put(("oom", str(exc)))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def solve_custom_with_timeout(
    C, X_hat, delta, timeout_sec=1800.0,
    rho=0.1, iteration=300, inner_tol=1e-10, inner_max_iter=5000,
):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_custom_worker,
        args=(queue, C, X_hat, delta, rho, iteration, inner_tol, inner_max_iter),
        daemon=True,
    )
    t0 = timeit.default_timer()
    proc.start()
    proc.join(timeout=timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        return {"X": None, "time": float(timeout_sec),
                "obj": np.nan, "violation": np.nan, "status": "timeout"}
    elapsed = timeit.default_timer() - t0
    if queue.empty():
        return {"X": None, "time": elapsed,
                "obj": np.nan, "violation": np.nan, "status": "failed_empty_result"}
    state, payload = queue.get()
    if state == "ok":
        payload.setdefault("status", "ok")
        payload["time"] = elapsed
        return payload
    if state == "oom":
        return {"X": None, "time": float(timeout_sec),
                "obj": np.nan, "violation": np.nan, "status": "timeout"}
    return {"X": None, "time": elapsed,
            "obj": np.nan, "violation": np.nan, "status": f"error:{payload}"}


# ---------------------------------------------------------------------------
# CVX reference solver
# ---------------------------------------------------------------------------

def solve_cvx(C, X_hat, delta, eps=1e-8):
    t0 = timeit.default_timer()
    n = C.shape[0] - 1
    h = cp.Variable(2 * n + 1)
    X = hankel_expr_from_variable(h, n)
    prob = cp.Problem(
        cp.Minimize(cp.normNuc(X @ C)),
        [cp.norm(X_hat - X, "fro") <= delta],
    )
    prob.solve(solver=cp.SCS, eps_abs=eps, eps_rel=eps, verbose=False)
    elapsed = timeit.default_timer() - t0
    if h.value is None:
        return {"X": None, "time": elapsed,
                "obj": np.nan, "violation": np.nan, "status": str(prob.status)}
    X_val = hankel_from_vector(np.asarray(h.value).ravel(), n)
    obj = la.norm(X_val @ C, "nuc")
    violation = max(0.0, la.norm(X_hat - X_val, "fro") - delta)
    return {"X": X_val, "time": elapsed,
            "obj": float(obj), "violation": float(violation),
            "status": str(prob.status)}


def _cvx_worker(queue, C, X_hat, delta, eps):
    try:
        queue.put(("ok", solve_cvx(C, X_hat, delta, eps=eps)))
    except MemoryError as exc:
        queue.put(("oom", str(exc)))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def solve_cvx_with_timeout(C, X_hat, delta, eps=1e-8, timeout_sec=1800.0):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_cvx_worker, args=(queue, C, X_hat, delta, eps), daemon=True)
    t0 = timeit.default_timer()
    proc.start()
    proc.join(timeout=timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        return {"X": None, "time": float(timeout_sec),
                "obj": np.nan, "violation": np.nan, "status": "timeout"}
    elapsed = timeit.default_timer() - t0
    if queue.empty():
        return {"X": None, "time": elapsed,
                "obj": np.nan, "violation": np.nan, "status": "failed_empty_result"}
    state, payload = queue.get()
    if state == "ok":
        payload["time"] = elapsed
        return payload
    if state == "oom":
        return {"X": None, "time": float(timeout_sec),
                "obj": np.nan, "violation": np.nan, "status": "timeout"}
    return {"X": None, "time": elapsed,
            "obj": np.nan, "violation": np.nan, "status": f"error:{payload}"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_benchmark(
    n_values: list[int] | None = None,
    delta: float = 0.5,
    timeout_sec: float = 1800.0,
    base_seed: int = 20260527,
) -> list[ResultRow]:
    if n_values is None:
        n_values = list(range(10, 101, 10))
    rows: list[ResultRow] = []
    custom_skip = False
    cvx_skip = False

    for n in n_values:
        r = max(2, int(round(0.78 * (n + 1))))
        print(f"\n=== n={n}, r={r} ===")
        C, X_hat, _ = generate_instance(n=n, r=r, delta=delta, seed=base_seed + n)

        if custom_skip:
            custom = {"X": None, "time": timeout_sec,
                      "obj": np.nan, "violation": np.nan, "status": "timeout"}
        else:
            custom = solve_custom_with_timeout(
                C, X_hat, delta, timeout_sec=timeout_sec,
                rho=0.1, iteration=300, inner_tol=1e-10, inner_max_iter=5000,
            )
            if custom["status"] == "timeout":
                custom_skip = True

        if cvx_skip:
            cvx = {"X": None, "time": timeout_sec,
                   "obj": np.nan, "violation": np.nan, "status": "timeout"}
        else:
            cvx = solve_cvx_with_timeout(C, X_hat, delta, eps=1e-8, timeout_sec=timeout_sec)
            if cvx["status"] == "timeout":
                cvx_skip = True

        rows.append(ResultRow(
            n=n, r=r,
            custom_time=float(custom["time"]),
            cvx_time=float(cvx["time"]),
            custom_obj=float(custom["obj"]),
            cvx_obj=float(cvx["obj"]),
            custom_violation=float(custom["violation"]),
            cvx_violation=float(cvx["violation"]),
            custom_status=str(custom["status"]),
            cvx_status=str(cvx["status"]),
        ))

    return rows


def save_npz(rows, out_file="benchmark_62_fixed.npz"):
    np.savez(
        out_file,
        n=np.array([r.n for r in rows], dtype=int),
        r=np.array([r.r for r in rows], dtype=int),
        custom_time=np.array([r.custom_time for r in rows], dtype=float),
        cvx_time=np.array([r.cvx_time for r in rows], dtype=float),
        custom_obj=np.array([r.custom_obj for r in rows], dtype=float),
        cvx_obj=np.array([r.cvx_obj for r in rows], dtype=float),
        custom_violation=np.array([r.custom_violation for r in rows], dtype=float),
        cvx_violation=np.array([r.cvx_violation for r in rows], dtype=float),
        custom_status=np.array([r.custom_status for r in rows], dtype=object),
        cvx_status=np.array([r.cvx_status for r in rows], dtype=object),
    )
    print(f"Saved {out_file}")


def plot_speed(rows: list[ResultRow], out_file: str, timeout_sec: float = 1800.0):
    n = np.array([r.n for r in rows], dtype=int)
    ct = np.array([r.custom_time for r in rows], dtype=float)
    xt = np.array([r.cvx_time for r in rows], dtype=float)
    xs = np.array([r.cvx_status for r in rows], dtype=object)
    xtm = xs == "timeout"

    fig, ax = plt.subplots(figsize=(7.0, 5))
    ax.plot(n, ct, "b-o", label="Algorithm 3")
    ax.plot(n, xt, "r-o", label="CVX")
    if np.any(xtm):
        ax.plot(
            n[xtm],
            np.full(np.sum(xtm), timeout_sec),
            marker="^",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="red",
            markersize=8,
            label="CVX (timeout)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("time (s)")
    ax.set_title("speed")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def plot_violation(rows: list[ResultRow], out_file: str):
    n = np.array([r.n for r in rows], dtype=int)
    cv = np.array([r.custom_violation for r in rows], dtype=float)
    xv = np.array([r.cvx_violation for r in rows], dtype=float)
    mask = np.isfinite(cv) & np.isfinite(xv)

    fig, ax = plt.subplots(figsize=(7.0, 5))
    if np.any(mask):
        ax.plot(n[mask], np.maximum(cv[mask], 1e-16), "b-o", label="Algorithm 3 violation")
        ax.plot(n[mask], np.maximum(xv[mask], 1e-16), "r-o", label="CVX violation")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("violation")
    ax.set_title("violation")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Section 6.2 benchmark: Algorithm 3 vs CVX/SCS.")
    parser.add_argument(
        "--n-values",
        nargs="*",
        type=int,
        default=list(range(10, 81, 10)),
        help="Problem sizes n (default: 10 20 ... 80)",
    )
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--out-npz", type=str, default="benchmark_62_fixed.npz")
    parser.add_argument("--out-speed", type=str, default="62_speed.pdf")
    parser.add_argument("--out-violation", type=str, default="62_diff.pdf")
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    args = parser.parse_args()

    rows = run_benchmark(
        n_values=args.n_values,
        delta=args.delta,
        timeout_sec=args.timeout_sec,
        base_seed=args.seed,
    )
    save_npz(rows, out_file=args.out_npz)
    plot_speed(rows, out_file=args.out_speed, timeout_sec=args.timeout_sec)
    plot_violation(rows, out_file=args.out_violation)


if __name__ == "__main__":
    main()
