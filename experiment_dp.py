import multiprocessing as mp
import time

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from helper import nuclear_PSD, PSD_projection, fro_PSD

# Match acc_eig.pdf-style figures (plots.plot_accuracy_eig): axis labels + title.
LABEL_FS = 20

# Skip/retry a trial if nuclear_PSD runs longer than this (seconds).
TRIAL_TIMEOUT_SEC = 90
MAX_TRIAL_RETRIES = 4

def generate_random_PSD(n):
    """Generate a random PSD matrix"""
    A = np.random.randn(n, n)
    return A @ A.T

def detection_potential(A, BXBt):
    """Compute detection potential between A and BXB^T"""
    sA = la.svd(A, full_matrices=False, compute_uv=False)
    dp_A = sA[0] - sA[-1]

    sBXBt = la.svd(BXBt, full_matrices=False, compute_uv=False)
    dp_BXBt = sBXBt[0] - sBXBt[-1]

    return dp_A / dp_BXBt

def _trial_worker(n, s, rho, seed, result_queue):
    """Run one trial in a subprocess so the parent can enforce a timeout."""
    np.random.seed(seed)
    try:
        A = generate_random_PSD(n)
        sb = np.random.rand(s) + 1
        B = np.random.randn(n, s)
        UB, _, VBT = la.svd(B, full_matrices=True)
        B = UB[:, :s] @ np.diag(sb) @ VBT[:s, :]
        C = B.T

        X = nuclear_PSD(A, B, C, rho)
        BXBt = B @ X @ B.T
        dp = detection_potential(A, BXBt)
        result_queue.put(('ok', dp))
    except Exception as exc:
        result_queue.put(('error', repr(exc)))

def run_single_trial(n, s, rho, timeout_sec=TRIAL_TIMEOUT_SEC, max_retries=MAX_TRIAL_RETRIES):
    """Run one random trial; retry on timeout or worker error. Returns dp or None."""
    ctx = mp.get_context('spawn')
    for attempt in range(1, max_retries + 1):
        seed = int(np.random.randint(0, 2**31))
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_trial_worker,
            args=(n, s, rho, seed, queue),
            daemon=True,
        )
        t0 = time.perf_counter()
        proc.start()
        proc.join(timeout_sec)

        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            elapsed = time.perf_counter() - t0
            print(
                f"  timeout after {elapsed:.1f}s "
                f"(attempt {attempt}/{max_retries}), retrying..."
            )
            continue

        if queue.empty():
            print(f"  no result from worker (attempt {attempt}/{max_retries}), retrying...")
            continue

        status, payload = queue.get()
        if status == 'ok':
            return payload

        print(f"  worker error: {payload} (attempt {attempt}/{max_retries}), retrying...")

    print(f"  skipped after {max_retries} failed attempts")
    return None

def run_experiment_for_n(n, rho=0.01, num_iterations=10, timeout_sec=TRIAL_TIMEOUT_SEC):
    """Run detection-potential experiment for fixed n, s = 1..n."""
    s_values = range(1, n + 1)
    detection_potentials = []

    for s in s_values:
        print(f"\n[n={n}] Running experiment for s = {s}")
        detection_potentials_temp = []
        for k in range(num_iterations):
            print(f"iteration {k}")
            dp = run_single_trial(n, s, rho, timeout_sec=timeout_sec)
            if dp is not None:
                detection_potentials_temp.append(dp)

        if not detection_potentials_temp:
            print(f"Warning: no successful trials for s={s}, recording NaN")
            mean_dp = np.nan
        else:
            n_ok = len(detection_potentials_temp)
            if n_ok < num_iterations:
                print(f"  used {n_ok}/{num_iterations} successful trials")
            mean_dp = np.mean(detection_potentials_temp)

        detection_potentials.append(mean_dp)
        print(f"Detection Potential: {mean_dp:.6f}")

    return list(s_values), detection_potentials

def run_experiment():
    np.random.seed(48)

    n_values = [10, 15, 20, 25]
    rho = 0.01
    all_results = {}

    for n in n_values:
        print(f"\n{'=' * 40}\nExperiment set: n = {n}\n{'=' * 40}")
        s_values, detection_potentials = run_experiment_for_n(n, rho=rho)
        all_results[n] = {
            's_values': np.array(s_values),
            'detection_potentials': np.array(detection_potentials),
        }

        plt.figure(figsize=(10, 6))
        plt.plot(s_values, detection_potentials, 'b-o')
        plt.xlabel('$p$', fontsize=LABEL_FS)
        plt.ylabel('detection potential', fontsize=LABEL_FS)
        plt.title(f'$n = {n}$', fontsize=LABEL_FS)
        plt.grid(True)
        plt.xticks(s_values)
        out_path = f'detection_potential_n{n}.pdf'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {out_path}")

        finite = [dp for dp in detection_potentials if np.isfinite(dp)]
        if finite:
            print("\nSummary Statistics:")
            print(f"Minimum DP: {min(finite):.6f} (s = {s_values[np.argmin(finite)]})")
            print(f"Maximum DP: {max(finite):.6f} (s = {s_values[np.argmax(finite)]})")
            print(f"Mean DP: {np.mean(finite):.6f}")
            print(f"Std DP: {np.std(finite):.6f}")
        else:
            print("\nSummary Statistics: no finite results")

    save_dict = {'n_values': np.array(n_values)}
    for n in n_values:
        save_dict[f's_values_n{n}'] = all_results[n]['s_values']
        save_dict[f'detection_potentials_n{n}'] = all_results[n]['detection_potentials']
    np.savez('experiment_results.npz', **save_dict)

if __name__ == "__main__":
    run_experiment()
