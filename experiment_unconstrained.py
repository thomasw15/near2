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

def generate_random_orthogonal(n):
    U, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    return U

def generate_random(n, cond_P):
    """Generate random matrix with given condition number"""
    log_cond_P = np.log(cond_P)
    s = np.exp((np.random.rand(n)-0.5) * log_cond_P)
    S = np.diag(s)
    U = generate_random_orthogonal(n)
    V = generate_random_orthogonal(n)
    return U @ S @ V.T

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

# ============= Solver Functions =============
def solve_1unconstrained(A, B, C, X_true, rho, iteration=300):
    """Solve unconstrained problem with nuclear norm"""
    Bpinv = la.pinv(B)
    Cpinv = la.pinv(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    
    forward_error_list = []
    for i in range(iteration):
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_1norm(Q, U, VT, sq, rho)
        X = Bpinv @ (A+D-Y) @ Cpinv
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        forward_error_list.append(forward_error)
    
    return X, forward_error_list

def solve_infunconstrained(A, B, C, X_true, rho, iteration = 300):
    Bpinv = la.pinv(B)
    Cpinv = la.pinv(C)
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
        X = Bpinv @ (A+D-Y) @Cpinv
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        forward_error_list.append(forward_error)
    return X, forward_error_list

def cp_unconstrained(A, B, C, e=1e-8, a=10):
    """Solve using CVX with Frobenius norm"""
    row = B.shape[1]
    column = C.shape[0]
    X = cp.Variable((row,column))
    obj = cp.Minimize(cp.norm(A-B@X@C, 'fro'))
    prob = cp.Problem(obj)
    start = timeit.default_timer()
    prob.solve(solver=cp.SCS, eps_rel=e, eps_abs=e, acceleration_interval=a)
    stop = timeit.default_timer()
    return X.value, stop-start

def cp_unconstrained_1(A, B, C, e=1e-8, a=10):
    """Solve using CVX with nuclear norm"""
    row = B.shape[1]
    column = C.shape[0]
    X = cp.Variable((row,column))
    obj = cp.Minimize(cp.norm(A-B@X@C, 'nuc'))
    prob = cp.Problem(obj)
    start = timeit.default_timer()
    prob.solve(solver=cp.SCS, eps_rel=e, eps_abs=e, acceleration_interval=a)
    stop = timeit.default_timer()
    return X.value, stop-start

def cp_unconstrained_inf(A,B,C,e=1e-8,a=10):
    row=B.shape[1]
    column=C.shape[0]
    X=cp.Variable((row,column))
    obj=cp.Minimize(cp.norm(A-B@X@C, 2))
    prob=cp.Problem(obj)
    start = timeit.default_timer()
    prob.solve(solver=cp.SCS,eps_rel=e,eps_abs=e, acceleration_interval=a)
    stop = timeit.default_timer()
    return X.value, stop-start

def tol_1unconstrained(A, B, C, X_true, rho, tol=1e-8):
    Bpinv = la.pinv(B)
    Cpinv = la.pinv(C)
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
        X = Bpinv @ (A+D-Y) @Cpinv
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
    return X
        
def tol_infunconstrained(A, B, C, X_true, rho, tol=1e-8):
    Bpinv = la.pinv(B)
    Cpinv = la.pinv(C)
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
        X = Bpinv @ (A+D-Y) @Cpinv
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
    return X

# ============= Experiment Functions =============
def run_accuracy_experiment(case=20):
    """Run accuracy comparison experiment"""
    inf_list = []
    one_list = []
    cvxinf_list = []
    cvxone_list = []
    n_list = list(range(1, case+1))
    
    for i in range(case):
        print(f"Running case {i+1}/{case}")
        n = 32
        B = generate_random(n, max(1,np.random.rand()*n))
        C = generate_random(n, max(1,np.random.rand()*n))
        X_true = generate_random(n, max(1,np.random.rand()*n*n))
        A = B @ X_true @ C
        
        Xinf, inf_list_temp = solve_infunconstrained(A, B, C, X_true, rho=0.01)
        X1, one_list_temp = solve_1unconstrained(A, B, C, X_true, rho=0.01)
        
        Xcpinf, cp_timeinf = cp_unconstrained_inf(A, B, C, e=1e-16, a=10)
        Xcp1, cp_time1 = cp_unconstrained_1(A, B, C, e=1e-16, a=10)
        cvxerrorinf = la.norm(Xcpinf - X_true, 2)/la.norm(X_true, 2)
        cvxerror1 = la.norm(Xcp1 - X_true, "nuc")/la.norm(X_true, "nuc")

        inf_list.append(inf_list_temp[299])
        one_list.append(one_list_temp[299])
        cvxinf_list.append(cvxerrorinf)
        cvxone_list.append(cvxerror1)
    
    # Save data
    np.savez('accuracy_data.npz',
             n_list=np.array(n_list),
             inf_list=np.array(inf_list),
             one_list=np.array(one_list),
             cvxinf_list=np.array(cvxinf_list),
             cvxone_list=np.array(cvxone_list))
    
    return n_list, inf_list, one_list, cvxinf_list, cvxone_list

def plot_accuracy_results_nuc(data_file='accuracy_data.npz'):
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
    plt.savefig('un_acc_nuc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_results_inf(data_file='accuracy_data.npz'):
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
    plt.savefig('un_acc_inf.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def run_speed_experiment():
    """Run speed comparison experiment"""
    # Dimensions for our algorithms (up to 2^10)
    dimensions_ours = [2**(i+1) for i in range(10)]
    # Dimensions for CVX (only up to 2^7)
    dimensions_cvx = [2**(i+1) for i in range(7)]
    
    cvx_times_inf = []
    cvx_times_nuc = []
    nuclear_times = []
    spectral_times = []
    
    #run our algorithms for full range
    for n in dimensions_ours:
        print(f"Testing our algorithms for dimension {n}")
        nuclear_temp = 0
        spectral_temp = 0
        
        for _ in range(10):
            B = generate_random(n, max(1,np.random.rand()*n))
            C = generate_random(n, max(1,np.random.rand()*n))
            X_true = generate_random(n, max(1,np.random.rand()*n*n))
            A = B @ X_true @ C
            
            # Nuclear norm timing
            start = timeit.default_timer()
            _ = tol_1unconstrained(A, B, C, X_true, rho=0.0000001, tol=1e-8)
            nuclear_temp += timeit.default_timer() - start
            
            # Spectral norm timing
            start = timeit.default_timer()
            _ = tol_infunconstrained(A, B, C, X_true, rho=0.0000001, tol=1e-8)
            spectral_temp += timeit.default_timer() - start
        
        nuclear_times.append(nuclear_temp/10)
        spectral_times.append(spectral_temp/10)

    # run CVX 
    for n in dimensions_cvx:
        print(f"Testing CVX for dimension {n}")
        cvx_temp_inf = 0
        cvx_temp_nuc = 0
        for _ in range(10):
            B = generate_random(n, max(1,np.random.rand()*n))
            C = generate_random(n, max(1,np.random.rand()*n))
            X_true = generate_random(n, max(1,np.random.rand()*n*n))
            A = B @ X_true @ C
            
            # CVX timing
            _, cvx_time_inf = cp_unconstrained_inf(A, B, C, e=1e-8, a=10)
            _, cvx_time_nuc = cp_unconstrained_1(A, B, C, e=1e-8, a=10)
            cvx_temp_inf += cvx_time_inf
            cvx_temp_nuc += cvx_time_nuc
        cvx_times_inf.append(cvx_temp_inf/10)
        cvx_times_nuc.append(cvx_temp_nuc/10)
    
    # Save data
    np.savez('speed_data.npz',
             dimensions_ours=np.array(dimensions_ours),
             dimensions_cvx=np.array(dimensions_cvx),
             cvx_times_inf=np.array(cvx_times_inf),
             cvx_times_nuc=np.array(cvx_times_nuc),
             nuclear_times=np.array(nuclear_times),
             spectral_times=np.array(spectral_times))
    
    return dimensions_ours, dimensions_cvx, cvx_times_inf, cvx_times_nuc, nuclear_times, spectral_times

def plot_speed_results(data_file='speed_data.npz'):
    """Plot speed results from saved data"""
    data = np.load(data_file)
    dimensions_ours = data['dimensions_ours']
    dimensions_cvx = data['dimensions_cvx']
    cvx_times_inf = data['cvx_times_inf']
    cvx_times_nuc = data['cvx_times_nuc']
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
    plt.savefig('speed_comparison.pdf', dpi=300, bbox_inches='tight')
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
