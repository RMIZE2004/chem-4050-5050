#all necessary imports and constants
import os
import numpy as np
outdir = "homework-2-1"
os.makedirs(outdir, exist_ok=True)
EPSILON = 0.01
SIGMA = 3.4
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys, os
import contextlib
with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
    from Prob2 import compute_bond_length, compute_bond_angle
# the lennard-jones potential is once again defined for use below
def lennard_jones(r, epsilon=.01,sigma=3.4):
    r= np.asarray(r,dtype=float)
    if np.any(r <= 0):
        return np.where(r <=0, np.inf, 0.0)
    sr6= (sigma/r)**6
    return 4*epsilon*(sr6**2 - sr6) 
#this is the xyz coordinate setup for later use in the notebook output
def write_xyz(path, symbols, coords):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"{len(symbols)}\n")
        f.write("Argon dimer optimized with Lennard-Jones potential\n")
        for s, (x,y,z) in zip(symbols, coords):
            f.write(f"{s} {x:.6f} {y:.5F} {z:.5f}\n")
# function to start calculation of minimizing potential arrangement of trimer.
def total_energy(vars_vec, epsilon=EPSILON, sigma=SIGMA):
    r12, x3, y3 = vars_vec
    if r12<=0:
     return np.inf
    p1=np.array([0.0, 0.0, 0.0])
    p2=np.array([r12, 0.0, 0.0])
    p3=np.array([x3, y3, 0.0])
    r12_val=compute_bond_length(p1, p2)
    r13_val=compute_bond_length(p1, p3)
    r23_val=compute_bond_length(p2, p3)
    return float(
       lennard_jones(r12_val, epsilon, sigma) +
       lennard_jones(r13_val, epsilon, sigma) +
       lennard_jones(r23_val, epsilon, sigma)
    )
# actual optimization mnimmization of our setup once we have defined a way to reference total potenial, and 
# now we can use a similar setup to dimer to calculate the minima here.
def optimize_ar3(epsilon=.01, sigma=3.4):
    req= (2.0**(1/6))*sigma
    x0= np.array([req, req/2, (np.sqrt(3)/2)*req])
    bounds= [(0.4, 20), (-20, 20), (-20,20)]
    res= minimize(total_energy, x0=x0, method="L-BFGS-B", bounds=bounds,
                  args=(epsilon, sigma),options=dict(maxiter=1000, ftol=1e-6))
    return res
# main function here uses the old compute bond angle and length functions to get easily readable 
#outputs in the main notebook.
def main():
    outdir="homework-2-1"
    os.makedirs(outdir, exist_ok=True)
    epsilon=0.01
    sigma=3.4
    res= optimize_ar3()
    r12_opt, x3_opt,y3_opt =map(float, res.x)
    r_eq_analytic= (2.0**(1/6))*sigma
    bond_angle_analytic= 60.0
    p1=np.array([0.0, 0.0, 0.0])
    p2=np.array([r12_opt, 0.0, 0.0])
    p3=np.array([res.x[1], res.x[2], 0.0])
    r12= compute_bond_length(p1, p2)
    r13= compute_bond_length(p1, p3)
    r23= compute_bond_length(p2, p3)
    angle_123= compute_bond_angle(p1, p2, p3)[0]
    angle_132= compute_bond_angle(p1, p3, p2)[0]
    angle_231= compute_bond_angle(p2, p3, p1)[0]
    print("=== Ar3 Optimization (Lennard-Jones) ===")
    print(f"Numerical r_eq = {r12:.6f} A")
    print(f"Analytic r_eq = {r_eq_analytic:.6f} A")
    print(f"V(r_eq) = {res.fun:.6f} eV")
    print(f"epsilon = {epsilon} eV")
    print()
    print(f"Computed bond lengths (A):")
    print(f"r12: {r12:.6f}")
    print(f"r13: {r13:.6f}")
    print(f"r23: {r23:.6f}")
    print()
    print(f"Computed bond angles (degrees):")
    print(f"angle 123: {angle_123:.6f}")
    print(f"angle 132: {angle_132:.6f}")
    print(f"angle 231: {angle_231:.6f}")
    print(f"Analytic bond angle: {bond_angle_analytic:.6f}")
    print()
  #check to see if this is close to an equilateral triangle geometry  
    lengths_close= np.allclose([r12, r13, r23], np.mean([r12, r13, r23]), rtol=1e-3, atol=1e-3)
    angles_close= np.allclose([angle_123, angle_132, angle_231], [60.0, 60.0, 60.0], atol=0.5)
    nearly_equilateral= lengths_close and angles_close
    if nearly_equilateral:
            print("The triangle is nearly equilateral.")
    else:
            print("The triangle is not equilateral.")

#final outputs that will be called by main notebook
    xyz_path= os.path.join(outdir, "ar3_equilibrium.xyz")
    write_xyz(xyz_path, ["Ar","Ar","Ar"], coords=[p1, p2, p3])
    print(f"Wrote {xyz_path}")
    distances= dict(r12=r12, r13=r13, r23=r23)
    angles= dict(a231=angle_231, a123=angle_123,a132= angle_132)
    coords= dict(p1=p1, p2=p2, p3=p3)
    return  distances, angles, res.fun, outdir, coords
if __name__ == "__main__":
    main()