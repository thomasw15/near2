import timeit
import cvxpy as cp
import numpy as np
from scipy.linalg import polar
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle

# ============= Helper Functions =============
def find_k(sigmas, rho):
    """Find k for spectral norm minimization"""
    for i in range(1, len(sigmas) + 1):
        if sigmas[i - 1] >= (np.sum(sigmas[:i])-1/rho)/i:
            k = i
    return k

def generate_random_orthogonal(n, rng):
    U, _ = la.qr((rng.rand(n, n) - 5.0) * 200)
    return U

def min_1norm(Q, U, VT, sigmas, rho):
    """Minimize nuclear norm"""
    S = np.zeros(Q.shape)
    min_dim = min(Q.shape)
    
    for i in range(1, min_dim+1):
        new_sigma = sigmas[i-1] - 1/rho
        if new_sigma > 0:
            S[i-1,i-1] = new_sigma
        else:
            break
    
    return U @ S @ VT

def min_infnorm(Q, U, VT, sigmas, rho, k):
    """Minimize spectral norm"""
    S = np.zeros(Q.shape)
    min_dim = min(Q.shape)
    total = np.sum(sigmas)
    
    if total > 1/rho:
        S[:min_dim, :min_dim] = np.diag(sigmas)
        new_sigma = (np.sum(sigmas[:k])-1/rho)/k
        np.fill_diagonal(S[:k,:k], new_sigma)
    
    return U @ S @ VT

def prod_Frobenius(A, B, C, F, G, H, UB, sB, VBT, rB, UC, sC, VCT, rC):
    FB = F @ VBT.transpose() @ np.diag(np.reciprocal(sB[0:rB]))
    GC = np.diag(np.reciprocal(sC[0:rC])) @ UC.transpose() @ G
    Ahat = UB.transpose() @ A @ VCT.transpose()
    middle = (
        FB.transpose()
        @ la.pinv(FB @ FB.transpose())
        @ (H - FB @ Ahat @ GC)
        @ la.pinv(GC.transpose() @ GC)
        @ GC.transpose()
    )
    middle = middle + Ahat 
    X = VBT.transpose() @ np.diag(np.reciprocal(sB[0:rB])) @ middle @ np.diag(np.reciprocal(sC[0:rC])) @ UC.transpose()
    return X

def generate_random(n, cond_P, rng):
    """Random n×n matrix with prescribed condition number (typically invertible)."""
    log_cond_P = np.log(cond_P)
    s = np.exp((rng.rand(n) - 0.5) * log_cond_P)
    S = np.diag(s)
    U = generate_random_orthogonal(n, rng)
    V = generate_random_orthogonal(n, rng)
    return U @ S @ V.T


def generate_product_constraint_FG(n, rng):
    """Rank-deficient F, G so {X : F X G = H} is not a singleton (neither factor invertible)."""
    rank_F = int(rng.randint(1, n))
    rank_G = int(rng.randint(1, n))

    def rank_r_matrix(rows, r):
        U = rng.randn(rows, r)
        V = rng.randn(rows, r)
        return U @ V.T

    F = rank_r_matrix(n, rank_F)
    G = rank_r_matrix(n, rank_G)
    return F, G

def generate_product_trial(n, seed):
    """One reproducible product-constraint instance (B, C, F, G, X_true, H, A)."""
    rng = np.random.RandomState(seed)
    B = generate_random(n, max(1.0, rng.rand() * n), rng)
    C = generate_random(n, max(1.0, rng.rand() * n), rng)
    F, G = generate_product_constraint_FG(n, rng)
    X_true = generate_random(n, max(1.0, rng.rand() * n * n), rng)
    H = F @ X_true @ G
    A = B @ X_true @ C
    return {
        "n": n,
        "B": B,
        "C": C,
        "F": F,
        "G": G,
        "H": H,
        "A": A,
        "X_true": X_true,
        "rank_F": int(la.matrix_rank(F)),
        "rank_G": int(la.matrix_rank(G)),
    }

def relative_forward_error(X, X_true):
    """Relative Frobenius forward error (same metric for ADMM and CVX)."""
    return la.norm(X - X_true, "fro") / la.norm(X_true, "fro")
# ============= Solver Functions =============
# solve two-sided product-constrained approximation

def solve_1prod(A, B, C, F, G, H, X_true, rho, iteration = 300):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    
    forward_error_list = []
    for i in range(iteration):
        # Y-update
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_1norm(Q,U,VT,sq,rho)
        # X-update
        X = prod_Frobenius(A + D - Y, B, C, F, G, H, UB, sB, VBT, rB, UC, sC, VCT, rC)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        forward_error_list.append(forward_error)
    return X, forward_error_list
        
def solve_infprod(A, B, C, F, G, H, X_true, rho, iteration = 300):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    
    forward_error_list = []
    for i in range(iteration):
        # Y-update
        Q = A + D - B @ X @ C 
        U, sq, VT = la.svd(Q)
        k = find_k(sq, rho)
        Y = min_infnorm(Q,U,VT,sq,rho,k)
        # X-update
        X = prod_Frobenius(A + D - Y, B, C, F, G, H, UB, sB, VBT, rB, UC, sC, VCT, rC)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        forward_error_list.append(forward_error)
    return X, forward_error_list

def cp_prod_1(A,B,C,F,G,H,e=1e-8,a=10):
    row=B.shape[1]
    column=C.shape[0]
    X=cp.Variable((row,column))
    obj=cp.Minimize(cp.norm(A-B@X@C, 'nuc'))
    constraints = [F @ X @ G == H]
    prob=cp.Problem(obj, constraints)
    start = timeit.default_timer()
    prob.solve(solver=cp.SCS,eps_rel=e,eps_abs=e, acceleration_interval=a)
    stop = timeit.default_timer()
    return X.value, stop-start

def cp_prod_inf(A,B,C,F,G,H,e=1e-8,a=10):
    row=B.shape[1]
    column=C.shape[0]
    X=cp.Variable((row,column))
    obj=cp.Minimize(cp.norm(A-B@X@C, 2))
    constraints = [F @ X @ G == H]
    prob=cp.Problem(obj, constraints)
    start = timeit.default_timer()
    prob.solve(solver=cp.SCS,eps_rel=e,eps_abs=e, acceleration_interval=a)
    stop = timeit.default_timer()
    return X.value, stop-start

def tol_1prod(A, B, C, F, G, H, X_true, n, rho, tol=1e-8):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    forward_error = 1
    while forward_error > tol:
        # Y-update
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_1norm(Q,U,VT,sq,rho)
        # X-update
        X = prod_Frobenius(A + D - Y, B, C, F, G, H, UB, sB, VBT, rB, UC, sC, VCT, rC)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        
    return X
        
def tol_infprod(A, B, C, F, G, H, X_true, n, rho, tol=1e-8):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    forward_error = 1
    while forward_error > tol:
        # Y-update
        Q = A + D - B @ X @ C 
        U, sq, VT = la.svd(Q)
        k = find_k(sq, rho)
        Y = min_infnorm(Q,U,VT,sq,rho,k)
        # X-update
        X = prod_Frobenius(A + D - Y, B, C, F, G, H, UB, sB, VBT, rB, UC, sC, VCT, rC)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
    
    return X

# ============= Experiment Functions =============
def run_product_accuracy_experiments(
    case=20,
    prod_output_file='accuracy_data_prod_0526.npz',
    three_output_file='accuracy_data_prod_3_0526.npz',
    cvx_eps=1e-16,
    admm_iters=300,
    n=32,
    trial_seed_base=1000,
):
    """All product-constraint accuracy methods on the same random trials per case."""
    from experiment_3norm import solve_3prod

    inf_list = []
    one_list = []
    cvxinf_list = []
    cvxone_list = []
    three_list = []
    n_list = list(range(1, case + 1))

    for i in range(case):
        print(f"Running case {i+1}/{case}", flush=True)
        trial = generate_product_trial(n, seed=trial_seed_base + i)
        A = trial["A"]
        B = trial["B"]
        C = trial["C"]
        F = trial["F"]
        G = trial["G"]
        H = trial["H"]
        X_true = trial["X_true"]
        print(
            f"  ranks: F={trial['rank_F']}, G={trial['rank_G']} (n={n})",
            flush=True,
        )

        print(f"  spectral ({admm_iters} iters)...", flush=True)
        _, inf_list_temp = solve_infprod(
            A, B, C, F, G, H, X_true, rho=0.01, iteration=admm_iters
        )
        print(f"  nuclear ({admm_iters} iters)...", flush=True)
        _, one_list_temp = solve_1prod(
            A, B, C, F, G, H, X_true, rho=0.01, iteration=admm_iters
        )
        print(f"  Schatten-3 ({admm_iters} iters)...", flush=True)
        _, three_list_temp = solve_3prod(
            A, B, C, F, G, H, X_true, rho=0.01, iteration=admm_iters
        )
        print(f"  CVX spectral (eps={cvx_eps})...", flush=True)
        Xcpinf, cp_timeinf = cp_prod_inf(A, B, C, F, G, H, e=cvx_eps, a=10)
        print(f"    CVX spectral done in {cp_timeinf:.1f}s", flush=True)
        print(f"  CVX nuclear (eps={cvx_eps})...", flush=True)
        Xcp1, cp_time1 = cp_prod_1(A, B, C, F, G, H, e=cvx_eps, a=10)
        print(f"    CVX nuclear done in {cp_time1:.1f}s", flush=True)

        inf_list.append(inf_list_temp[admm_iters - 1])
        one_list.append(one_list_temp[admm_iters - 1])
        three_list.append(three_list_temp[admm_iters - 1])
        if Xcpinf is None or Xcp1 is None:
            raise RuntimeError(f"CVX failed on case {i+1} (status not optimal)")
        cvxinf_list.append(relative_forward_error(Xcpinf, X_true))
        cvxone_list.append(relative_forward_error(Xcp1, X_true))

    prod_payload = dict(
        n_list=np.array(n_list),
        inf_list=np.array(inf_list),
        one_list=np.array(one_list),
        cvxinf_list=np.array(cvxinf_list),
        cvxone_list=np.array(cvxone_list),
        three_list=np.array(three_list),
    )
    np.savez(prod_output_file, **prod_payload)
    np.savez(
        three_output_file,
        n_list=np.array(n_list),
        three_list=np.array(three_list),
    )
    print(f"Saved {prod_output_file} and {three_output_file}")

    return n_list, inf_list, one_list, cvxinf_list, cvxone_list, three_list


def run_accuracy_experiment(
    case=20,
    output_file='accuracy_data_prod_0526.npz',
    three_output_file='accuracy_data_prod_3_0526.npz',
    cvx_eps=1e-8,
    admm_iters=300,
    **kwargs,
):
    """Backward-compatible wrapper; runs combined product accuracy experiment."""
    return run_product_accuracy_experiments(
        case=case,
        prod_output_file=output_file,
        three_output_file=three_output_file,
        cvx_eps=cvx_eps,
        admm_iters=admm_iters,
        **kwargs,
    )

def plot_accuracy_results_nuc(data_file='accuracy_data_prod.npz'):
    """Plot accuracy results from saved data"""
    data = np.load(data_file)
    n_list = data['n_list']
    one_list = data['one_list']
    cvxone_list = data['cvxone_list']
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_list, one_list, 'b-o', label='nuclear norm')
    plt.plot(n_list, cvxone_list, 'r-o', label='cvx')
    plt.xlabel('trial')
    plt.ylabel('forward error')
    plt.grid(True)
    plt.xticks(n_list)
    plt.yscale('log')
    plt.legend()
    plt.savefig('prod_acc_nuc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_results_inf(data_file='accuracy_data_prod.npz'):
    """Plot accuracy results from saved data"""
    data = np.load(data_file)
    n_list = data['n_list']
    inf_list = data['inf_list']
    cvxinf_list = data['cvxinf_list']
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_list, inf_list, 'b-o', label='spectral norm')
    plt.plot(n_list, cvxinf_list, 'r-o', label='cvx')
    plt.xlabel('trial')
    plt.ylabel('forward error')
    plt.grid(True)
    plt.xticks(n_list)
    plt.yscale('log')
    plt.legend()
    plt.savefig('prod_acc_inf.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def run_speed_experiment(
    output_file='speed_data_prod_0526.npz',
    dimensions_cvx=None,
):
    """Run speed comparison experiment"""
    # Dimensions for our algorithms (up to 2^10)
    dimensions_ours = [2**(i+1) for i in range(10)]
    # CVX product constraint: 2, 2^3, 2^4 only 
    if dimensions_cvx is None:
        dimensions_cvx = [2, 2**3, 2**4]
    
    cvx_times_nuc = []
    cvx_times_inf = []
    nuclear_times = []
    spectral_times = []
    # run CVX 
    for n in dimensions_cvx:
        print(f"Testing CVX for dimension {n}")
        cvx_temp_nuc = 0
        cvx_temp_inf = 0
        for j in range(10):
            print(f"Testing CVX for dimension {n}, trial {j+1}")
            trial = generate_product_trial(n, seed=10_000 * n + j)
            B, C, F, G, H, A, X_true = (
                trial["B"],
                trial["C"],
                trial["F"],
                trial["G"],
                trial["H"],
                trial["A"],
                trial["X_true"],
            )

            # CVX timing
            _, cvx_time_nuc = cp_prod_1(A, B, C, F, G, H, e=1e-8, a=10)
            _, cvx_time_inf = cp_prod_inf(A, B, C, F, G, H, e=1e-8, a=10)
            print(f"nuc time: {cvx_time_nuc}, inf time: {cvx_time_inf}")
            cvx_temp_nuc += cvx_time_nuc
            cvx_temp_inf += cvx_time_inf
        cvx_times_nuc.append(cvx_temp_nuc/10)
        cvx_times_inf.append(cvx_temp_inf/10)

    #run our algorithms for full range
    for n in dimensions_ours:
        print(f"Testing our algorithms for dimension {n}")
        nuclear_temp = 0
        spectral_temp = 0
        
        for j in range(10):
            trial = generate_product_trial(n, seed=20_000 * n + j)
            B, C, F, G, H, A, X_true = (
                trial["B"],
                trial["C"],
                trial["F"],
                trial["G"],
                trial["H"],
                trial["A"],
                trial["X_true"],
            )

            # Nuclear norm timing
            start = timeit.default_timer()
            _ = tol_1prod(A, B, C, F, G, H, X_true, n, rho=0.0000001, tol=1e-8)
            nuclear_temp += timeit.default_timer() - start
            
            # Spectral norm timing
            start = timeit.default_timer()
            _ = tol_infprod(A, B, C, F, G, H, X_true, n, rho=0.0000001, tol=1e-8)
            spectral_temp += timeit.default_timer() - start
        
        nuclear_times.append(nuclear_temp/10)
        spectral_times.append(spectral_temp/10)
    
    np.savez(
        output_file,
        dimensions_ours=np.array(dimensions_ours),
        dimensions_cvx=np.array(dimensions_cvx),
        cvx_times_nuc=np.array(cvx_times_nuc),
        cvx_times_inf=np.array(cvx_times_inf),
        nuclear_times=np.array(nuclear_times),
        spectral_times=np.array(spectral_times),
    )
    print(f"Saved speed data to {output_file}")

    return dimensions_ours, dimensions_cvx, cvx_times_nuc, cvx_times_inf, nuclear_times, spectral_times

def plot_speed_results(data_file='speed_data_prod.npz'):
    """Plot speed results from saved data"""
    data = np.load(data_file)
    dimensions_ours = data['dimensions_ours']
    dimensions_cvx = data['dimensions_cvx']
    cvx_times_nuc = data['cvx_times_nuc']
    cvx_times_inf = data['cvx_times_inf']
    nuclear_times = data['nuclear_times']
    spectral_times = data['spectral_times']
    
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions_ours, nuclear_times, 'b-o', label='nuclear norm')
    plt.plot(dimensions_cvx, cvx_times_nuc, 'r-o', label='cvx nuclear norm')
    plt.plot(dimensions_ours, spectral_times, 'g-o', label='spectral norm')
    plt.plot(dimensions_cvx, cvx_times_inf, 'y-o', label='cvx spectral norm')
    plt.xlabel('dimension')
    plt.ylabel('time (s)')
    plt.grid(True)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend()
    plt.savefig('speed_comparison_prod.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Run experiments and save data
    print("Running accuracy experiment...") 
    run_accuracy_experiment(case=20)
    print("\nGenerating accuracy plots...")
    plot_accuracy_results_nuc()
    plot_accuracy_results_inf()
    
    print("\nRunning speed experiment...")
    run_speed_experiment()
    
    # Generate plots from saved data
    print("\nGenerating speed plots...")
    plot_speed_results()
