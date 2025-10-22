#necessary imports
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
#random seed setup
seed_rand = 42
np.random.seed(seed_rand)

#function that will be used for the 2pz hydrogen orbital in atomic units
def psi_2p_z(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r = np.sqrt(x * x + y * y + z * z)
    safe = np.where(r == 0.0, 1.0, r)
    cos_theta = z / safe
    pref = 1.0 / (4.0 * np.sqrt(2.0 * np.pi))
    psi = pref * r * np.exp(-0.5 * r) * cos_theta
    psi = np.where(r == 0.0, 0.0, psi)
    return psi

#the definition for the overlap where our 2 2pz orbitals are apart by +/- R/2
def overlap_integrand(x: np.ndarray, y: np.ndarray, z: np.ndarray, R: float) -> np.ndarray:
    psi_plus = psi_2p_z(x, y, z + R / 2.0)
    psi_minus = psi_2p_z(x, y, z - R / 2.0)
    return psi_plus * psi_minus

#Monte Carlo estimator of overlap, sampled across cube defined by L(from - L to L)
def mc_overlap_uniform(R: float, L: float, N: int, seed: int | None = None, batch: int = 2_000_000) -> float:
    rng = np.random.default_rng(seed)
    V = (2.0 * L) ** 3
    total = 0.0
    remaining = int(N)
    #calculate in chunks
    while remaining > 0:
        n = min(remaining, batch)
        xyz = rng.uniform(-L, L, size=(n, 3))
        vals = overlap_integrand(xyz[:, 0], xyz[:, 1], xyz[:, 2], R)
        total += float(np.sum(vals))
        remaining -= n
    mean_val = total / N
    return V * mean_val
#Function for importance method of sampling using exponential function combined with symmetry of our functions to make sampling more efficient.
def mc_overlap_importance(R: float, N: int, scale: float =2.0, seed: int | None = None, batch: int = 2_000_000)-> tuple[float, float]:
    rng = np.random.default_rng(seed)
    beta = 1.0 / scale
    #exponential function for each coordinate
    def sample_pos_oct(n: int):
        x = rng.exponential(scale=scale, size = n)
        y = rng.exponential(scale= scale, size =n)
        z= rng.exponential(scale= scale, size =n)
        return x, y, z
    # product of the exponential functions
    def pdf_pos_oct(x, y, z):
        return(beta *np.exp(-beta*x)) * (beta* np.exp(-beta*y))*(beta * np.exp(-beta*z))
    total_w = 0.0
    total_w2 = 0.0
    remaining = int(N)
    while remaining > 0:
        n = min(remaining, batch)
        x, y, z = sample_pos_oct(n)
        
        vals = overlap_integrand(x, y, z, R)
        g = pdf_pos_oct(x, y, z)
        w = vals/ g
        total_w += float(np.sum(w))
        total_w2 += float(np.sum(w*w))
        remaining -= n 
    #final calculation
    mean_w = total_w / N
    var_w = max(total_w2 / N - mean_w**2, 0.0)
    S_hat = 8.0 * mean_w
    stderr= 8.0 * np.sqrt(var_w / N)
    return S_hat, stderr







def run_convergence_plots():
    R_demo = 2.0
    L_demo = 20.0
    Ns_uniform = np.array([10**3, 3*10**3, 10**4, 3 * 10**4, 10**5,3*10**5, 10**6, 3*10**6, 10**7], dtype=int)
    Ns_import = np.array([10**3, 3*10**3, 10**4, 3 * 10**4, 10**5, 3*10**5, 10**6, 3*10**6, 10**7], dtype=int)

    # compute estimates for each N
    S_uniform = [mc_overlap_uniform(R_demo, L_demo, int(N), seed=123) for N in Ns_uniform]
    S_import = []
    SE_import = []
    for N in Ns_import:
        est, se = mc_overlap_importance(R_demo, int(N), scale = 2.0)
        S_import.append(est)
        SE_import.append(se)

    plt.figure()
    plt.loglog(Ns_uniform, S_uniform, marker='o')
    plt.xlabel('N(samples)')
    plt.ylabel('(S(R=2)) (uniform MC)')
    plt.title('Conv of S(R=2) with Uniform Monte Carlo')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig('uniform_convergence.png', dpi=150)

    plt.figure()
    plt.loglog(Ns_import, S_import, marker='o')
    plt.xlabel('N (samples)')
    plt.ylabel('(S(R=2)) (importance sampling)')
    plt.title('Convergence of S(R=2) with Importance Sampling')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig('importance_convergence.png', dpi =150)
    
    plt.show()
    print("Uniform MC (R=2):")
    for N, est in zip(Ns_uniform, S_uniform):
        print(f" N={N:>8,d}  S={est:.6e}")

    print("\nImportance sampling (R=2):")
    for N, est in zip(Ns_import, S_import):
        print(f" N={N:>8,d}  S={est:.6e}")







