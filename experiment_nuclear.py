import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from helper import nuclear_hankel, hankel_projection, fro_hankel, min_1norm, rank, fro_hankel_distance

def generate_random_hankel(n):
    """Generate a random Hankel matrix for testing using hankel_projection"""
    R = 10 * np.random.rand(n, n)
    return hankel_projection(R)

def run_single_trial(n, r, Ex, delta, rho=0.01, max_iter=40, tag=None):
    """Run one ADMM trial for fixed n, r, and target Ex. Returns metric arrays."""
    tag = tag or f"n{n}_r{r}"
    print(f"\n{'=' * 60}\nTrial: {tag} (n={n}, r={r})\n{'=' * 60}")

    A = np.zeros((n, r))
    B = np.eye(n)
    sc = np.random.rand(r) + 1
    C = np.random.randn(n, r)
    UC, _, VCT = la.svd(C, full_matrices=True)
    print(f"dimensions are UC {UC[:, :r].shape}, s {np.diag(sc).shape}, VCT {VCT[:r, :].shape}")
    C = UC[:, :r] @ np.diag(sc) @ VCT[:r, :]

    print("\nInitial Matrix Properties:")
    print(f"C shape: {C.shape}, rank: {np.linalg.matrix_rank(C)}")
    print(f"Ex shape: {Ex.shape}, is Hankel: {np.allclose(Ex[1:, :-1], Ex[:-1, 1:])}")
    print(f"Ex Frobenius norm: {np.linalg.norm(Ex, 'fro')}")
    print(f"C Frobenius norm: {np.linalg.norm(C, 'fro')}")

    nuclear_norms = []
    distances = []
    hankel_errors = []

    UB, sb, VBT = la.svd(B, full_matrices=True)
    UC, sc, VCT = la.svd(C, full_matrices=True)
    rb = rank(sb)
    rc = rank(sc)
    Y = np.random.rand(n, r)
    D = np.random.rand(n, r)
    X = np.random.rand(n, n)

    for i in range(max_iter):
        print(f"iteration {i}")
        curr_nuclear_norm = la.norm(X @ C, 'nuc')
        curr_distance = la.norm(X - Ex, 'fro')
        hankel_error = la.norm(X - hankel_projection(X), 'fro') / la.norm(X, 'fro')
        nuclear_norms.append(curr_nuclear_norm)
        distances.append(curr_distance)
        hankel_errors.append(hankel_error)

        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_1norm(Q, U, VT, sq, rho)

        X_prev = X.copy() if i > 0 else None
        X = fro_hankel_distance(A + D - Y, B, C, UB, sb, VBT, rb, UC, sc, VCT, rc, Ex, delta)
        D = D - Y + A - X @ C

        print(f"\nIteration {i}:")
        print(f"Nuclear Norm = {curr_nuclear_norm:.6f}")
        print(f"Distance from Ex = {curr_distance:.6f}")
        print(f"Relative Hankel Error = {hankel_error:.6f}")
        if X_prev is not None:
            x_change = la.norm(X - X_prev, 'fro') / la.norm(X_prev, 'fro')
            print(f"Relative X change = {x_change:.6e}")
        print(f"Y Frobenius norm = {la.norm(Y, 'fro'):.6e}")
        print(f"D Frobenius norm = {la.norm(D, 'fro'):.6e}")

    x_iters = range(1, max_iter)
    plt.figure(figsize=(10, 6))
    plt.plot(x_iters, nuclear_norms[1:], 'b-o')
    plt.xlabel('iteration')
    plt.ylabel('nuclear norm')
    plt.grid(True)
    plt.xticks(x_iters)
    plt.savefig(f'nuclear_norm_{tag}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x_iters, distances[1:], 'b-o')
    plt.xlabel('iteration')
    plt.ylabel('distance from $E_x$')
    plt.grid(True)
    plt.xticks(x_iters)
    plt.axhline(y=delta, color='g', linestyle='--')
    plt.savefig(f'distance_{tag}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x_iters, hankel_errors[1:], 'b-o')
    plt.xlabel('iteration')
    plt.ylabel('relative hankel error')
    plt.grid(True)
    plt.xticks(x_iters)
    plt.yscale('log')
    plt.savefig(f'hankel_error_{tag}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nFinal Statistics:")
    print(f"Final nuclear norm: {nuclear_norms[-1]:.6f}")
    print(f"Minimum nuclear norm: {min(nuclear_norms):.6f}")
    print(f"Maximum nuclear norm: {max(nuclear_norms):.6f}")
    print(f"Final distance from Ex: {distances[-1]:.6f}")
    print(f"Final Hankel error: {hankel_errors[-1]:.6f}")
    print(f"Distance constraint (delta): {delta}")

    np.savez(
        f'experiment_data_{tag}.npz',
        nuclear_norms=np.array(nuclear_norms),
        distances=np.array(distances),
        hankel_errors=np.array(hankel_errors),
        n=n,
        r=r,
        delta=delta,
    )
    return nuclear_norms, distances, hankel_errors


def run_experiment():
    np.random.seed(42)

    # (n, r) pairs: r ≈ n/1.28 in each case
    size_pairs = [(32, 25), (64, 50), (128, 100), (256, 200)]
    delta = 0.5
    rho = 0.01
    max_iter = 40

    all_results = {}

    for n, r in size_pairs:
        tag = f"n{n}_r{r}"
        np.random.seed(42 + n)  # reproducible Ex, C, Y, D per (n, r)
        Ex = generate_random_hankel(n)
        nuclear_norms, distances, hankel_errors = run_single_trial(
            n, r, Ex, delta, rho=rho, max_iter=max_iter, tag=tag
        )
        all_results[tag] = {
            'nuclear_norms': np.array(nuclear_norms),
            'distances': np.array(distances),
            'hankel_errors': np.array(hankel_errors),
        }

    np.savez(
        'experiment_data_all.npz',
        size_pairs=np.array(size_pairs),
        delta=delta,
        **{
            f'{tag}_{key}': all_results[tag][key]
            for tag in all_results
            for key in ('nuclear_norms', 'distances', 'hankel_errors')
        },
    )


if __name__ == "__main__":
    run_experiment() 