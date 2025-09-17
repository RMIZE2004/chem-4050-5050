# this section includes necessary imports and constants
import os 
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
Na= 6.02214076e23
kB=8.617333262145e-5
Epsilon=.01
Sigma=3.4
LAMBDA= 1.5
OUTDIR = "homework-2-2"
from optimize_argon_dimer import lennard_jones as lj_HWQ1
#this section includes all necessary potentials that will be used throughout
def hard_sphere_pot(r, sigma=Sigma):
    r=np.asarray(r, dtype=float)
    u= np.zeros_like(r)
    u[r<sigma] = np.inf
    return u
def square_well_pot(r, epsilon=Epsilon, sigma=Sigma, lamb=LAMBDA):
    r=np.asarray(r, dtype=float)
    u= np.zeros_like(r)
    u[r<sigma] = np.inf
    mask_well = (r>= sigma) & (r < lamb *sigma)
    u[mask_well] = -float(epsilon)
    return u
# this section involves the calculation for the energies at different temperatures 
def boltzmann_factor(u_eV, T_K):
    return np.exp(-u_eV / (kB * T_K))
def b2v_integrand(r, T, u_func, **p):
    u= u_func(r, **p)
    bf = np.exp(-np.minimum(u, 1e9) / (kB * T))
    return (bf-1.0) * (r**2)
# this function to do the integration using the trapezoid rule
def compute_b2v(T, r_grid, u_func, **p):
    integrand_vals = b2v_integrand(r_grid, T, u_func, **p)
    integral = trapezoid(integrand_vals, r_grid)
    B2V_A3_per_mol = -2.0 * np.pi*Na*integral
    B2V_cm3_per_mol =B2V_A3_per_mol *1e-24
    return B2V_A3_per_mol, B2V_cm3_per_mol
# the main function will make our grid of values, loop over all the temperatures in the set
# so that the output can be used to generate our graphs in the main notebook.
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rmin= 1e-3 
    rmax= 5.0*Sigma
    Nr = 1000
    r= np.linspace(rmin, rmax, Nr)
    T_min, T_max, N_T = 100.0, 800.0, 100
    T_vals = np.linspace(T_min, T_max, N_T)
    rows= []
    B2V_LJ_cm3, B2V_SW_cm3, B2V_HS_cm3= [], [], []
    #T loop within the main function that actually allows us to obtain values for each potential
    for T in T_vals:
        b2v_lj_A3, b2v_lj_cm3 = compute_b2v(T, r, lj_HWQ1, epsilon= Epsilon, sigma= Sigma)
        b2v_sw_A3, b2v_sw_cm3 = compute_b2v(T, r, square_well_pot, epsilon=Epsilon, sigma=Sigma, lamb=LAMBDA)
        b2v_hs_A3, b2v_hs_cm3 = compute_b2v(T, r, hard_sphere_pot, sigma=Sigma)
        rows.append([T, b2v_lj_A3, b2v_sw_A3, b2v_hs_A3, 
                    b2v_lj_cm3, b2v_sw_cm3, b2v_hs_cm3])
        B2V_LJ_cm3.append(b2v_lj_cm3)
        B2V_SW_cm3.append(b2v_sw_cm3)
        B2V_HS_cm3.append(b2v_hs_cm3)
    rows = np.array(rows)
    B2V_LJ_cm3 = np.array(B2V_LJ_cm3)
    B2V_SW_cm3 = np.array(B2V_SW_cm3)
    B2V_HS_cm3 = np.array(B2V_HS_cm3)

    csv_path= os.path.join(OUTDIR, "b2v_vs_T.csv")
    header = "T_K,B2V_LJ_A3mol, B2V_SW_A3mol, B2V_HS_A3mol,B2V_LJ_cm3mol, B2V_SW_cm3mol, B2V_HS_cm3mol"
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")
    print(f"Wrote {csv_path}")
    # plotting for each of the virial coefficients with respect to temperature 
    plt.figure(figsize=(7,5))
    plt.plot(T_vals, B2V_LJ_cm3, label="Lennard-Jones")
    plt.plot(T_vals, B2V_SW_cm3, label="Square well (lambda=1.5)")
    plt.plot(T_vals, B2V_HS_cm3, label="Hard sphere")
    plt.axhline(0, ls= "--", lw=1, alpha= .7, label="B2V=0")
    plt.xlabel("Temperature (K)")
    plt.ylabel (r"$B_2^V$ (cm$^3$/mol)")
    plt.title(r"Second Virial Coefficient $B_2^V(T)$")
    plt.legend(title="Potential", loc="best")
    plt.tight_layout()
    fig_path= os.path.join(OUTDIR, "b2v_vs_T.png")
    plt.savefig(fig_path, dpi=300)
    plt.show
    print(f"Wrote {fig_path}")
    return {"T": T_vals, "B2_cm3": {"LJ": B2V_LJ_cm3,"SW": B2V_SW_cm3, "HS": B2V_HS_cm3}, "csv_path": csv_path, "fig_path": fig_path, "outdir": OUTDIR}
if __name__ == "__main__":
    main()
    