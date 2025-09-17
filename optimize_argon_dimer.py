# all necessary imports and constants are below
import os
import numpy as np
outdir = "homework-2-1"
os.makedirs(outdir, exist_ok=True)
epsilon = 0.01
sigma = 3.4
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys, os
import contextlib
with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
    from Prob2 import compute_bond_length, compute_bond_angle
#defining the potential function
def lennard_jones(r, epsilon=.01,sigma=3.4):
    r= np.asarray(r,dtype=float)
    if np.any(r <= 0):
        return np.where(r <=0, np.inf, 0.0)
    sr6= (sigma/r)**6
    return 4*epsilon*(sr6**2 - sr6) 
# for xyz output coordinates
def write_xyz(path, symbols, coords):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"{len(symbols)}\n")
        f.write("Argon dimer optimized with Lennard-Jones potential\n")
        for s, (x,y,z) in zip(symbols, coords):
            f.write(f"{s} {x:.6f} {y:.5F} {z:.5f}\n")
            #section below for the optimization of the lennard-jones potential using the selected method
            # and the imported minimize functionality.
def optimize_ar2(epsilon=.01, sigma=3.4, r0=4.0):
    def obj(r_scalar):
        return lennard_jones(r_scalar[0], epsilon, sigma)
    bounds = [(.4,20)]
    res = minimize(obj, x0=[r0], method="L-BFGS-B", bounds=bounds)
    r_eq = float(res.x[0])
    Vmin = float(res.fun)
    return r_eq, Vmin, res
# all necessary outputs for the main notebook to reference are defined in the following main function
# (not all of these outputs may be necesary anymore, but it includes all)
def main():
    outdir="homework-2-1"
    os.makedirs(outdir, exist_ok=True)
    epsilon=0.01
    sigma=3.4
    r_eq, Vmin, res= optimize_ar2(epsilon=epsilon, sigma=sigma, r0=4.0)
    r_eq_analytic= 2**(1/6)*sigma
    ar1=[0.0, 0.0, 0.0]
    ar2=[r_eq, 0.0, 0.0]
    r12= compute_bond_length(ar1, ar2)
    print("=== Ar2 Optimization (Lennard-Jones) ===")
    print(f"Numerical r_eq = {r_eq:.6f} A")
    print(f"Analytic r_eq = {r_eq_analytic:.6f} A")
    print(f"V(r_eq) = {Vmin:.6f} eV")
    print(f"epsilon = {epsilon} eV")
    xyz_path= os.path.join(outdir, "ar2_equilibrium.xyz")
    write_xyz(xyz_path, ["Ar", "Ar"], [ar1, ar2])
    print(f"Wrote {xyz_path}")
    return r_eq, Vmin, outdir
r_eq, Vmin, outdir = main()
if __name__ == "__main__":
    main()