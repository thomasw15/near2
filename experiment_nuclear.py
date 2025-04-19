import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from helper import nuclear_hankel, hankel_projection, fro_hankel, min_1norm, rank, fro_hankel_distance

def generate_random_hankel(n):
    """Generate a random Hankel matrix for testing using hankel_projection"""
    R = 10 * np.random.rand(n, n)
    return hankel_projection(R)

def run_experiment():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n = 256
    r = 200  # rank of C
    delta = .5  # Distance constraint
    
    # Generate matrices
    A = np.zeros((n, r))  # A = 0 with shape n x r
    B = np.eye(n)  # Identity matrix n x n
    sc = np.random.rand(r)+1
    C = np.random.randn(n, r)  # Full rank matrix of size n x r
    UC, s, VCT = la.svd(C, full_matrices=True)
    print(f"dimensions are UC {UC[:,:r].shape}, s {np.diag(sc).shape}, VCT {VCT[:r,:].shape}")
    C = UC[:,:r] @ np.diag(sc) @ VCT[:r,:]
    Ex = generate_random_hankel(n)  # Random Hankel matrix
    
    # Print initial matrix properties
    print("\nInitial Matrix Properties:")
    print(f"C shape: {C.shape}, rank: {np.linalg.matrix_rank(C)}")
    print(f"Ex shape: {Ex.shape}, is Hankel: {np.allclose(Ex[1:,:-1], Ex[:-1,1:])}")
    print(f"Ex Frobenius norm: {np.linalg.norm(Ex, 'fro')}")
    print(f"C Frobenius norm: {np.linalg.norm(C, 'fro')}")
    
    # Parameters for nuclear_hankel_distance
    rho = 0.01
    max_iter = 40
    
    # Lists to store metrics
    nuclear_norms = []
    distances = []
    hankel_errors = []  # Track how well Hankel property is preserved
    
    # Modified nuclear_hankel_distance function to track nuclear norm
    UB, sb, VBT = la.svd(B, full_matrices=True)
    UC, sc, VCT = la.svd(C, full_matrices=True)
    rb = rank(sb)
    rc = rank(sc)
    Y = np.random.rand(n,r)
    D = np.random.rand(n,r)
    X = np.random.rand(n,n)
    
    for i in range(max_iter):
        print(f"iteration {i}")
        # Calculate metrics
        curr_nuclear_norm = la.norm(X @ C, 'nuc')
        curr_distance = la.norm(X - Ex, 'fro')
        hankel_error = la.norm(X - hankel_projection(X), 'fro') / la.norm(X, 'fro')
        nuclear_norms.append(curr_nuclear_norm)
        distances.append(curr_distance)
        hankel_errors.append(hankel_error)
        # Y-update
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_1norm(Q, U, VT, sq, rho)
        
        # X-update
        X_prev = X.copy() if i > 0 else None
        X = fro_hankel_distance(A + D - Y, B, C, UB, sb, VBT, rb, UC, sc, VCT, rc, Ex, delta)
        
        # Update D
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
    
    # Plot nuclear norm
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter), nuclear_norms[1:], 'b-o')
    plt.xlabel('iteration')
    plt.ylabel('nuclear norm')
    plt.grid(True)
    plt.xticks(range(1, max_iter))
    plt.savefig('nuclear_norm.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot distance from Ex
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter), distances[1:], 'b-o')
    plt.xlabel('iteration')
    plt.ylabel('distance from $E_x$')
    plt.grid(True)
    plt.xticks(range(1, max_iter))
    plt.axhline(y=delta, color='g', linestyle='--')
    plt.savefig('distance.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Hankel errors
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter), hankel_errors[1:], 'b-o')
    plt.xlabel('iteration')
    plt.ylabel('relative hankel error')
    plt.grid(True)
    plt.xticks(range(1, max_iter))
    plt.yscale('log')
    plt.savefig('hankel_error.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Final nuclear norm: {nuclear_norms[-1]:.6f}")
    print(f"Minimum nuclear norm: {min(nuclear_norms):.6f}")
    print(f"Maximum nuclear norm: {max(nuclear_norms):.6f}")
    print(f"Final distance from Ex: {distances[-1]:.6f}")
    print(f"Final Hankel error: {hankel_errors[-1]:.6f}")
    print(f"Distance constraint (delta): {delta}")
    
    # Save the data
    np.savez('experiment_data.npz', 
             nuclear_norms=np.array(nuclear_norms),
             distances=np.array(distances),
             hankel_errors=np.array(hankel_errors))

if __name__ == "__main__":
    run_experiment() 