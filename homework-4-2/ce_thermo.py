#functionalities
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.constants import k as kB_JperK, eV as J_PER_eV
#unit conversion
kB_eV_per_K= kB_JperK / J_PER_eV
#function that takes energy levels, degeneracies and temperatures to return
#partition functions, internal energy, helmholtz free energy, and entropy.
def thermo_from_levels(E_eV: np.ndarray, g: np.ndarray, T_k:np.ndarray):
    E= np.asarray(E_eV, dtype=float).reshape((-1, 1))
    g= np.asarray(g, dtype=float).reshape((-1, 1))
    beta = 1.0 / (kB_eV_per_K * T_k.reshape((1, -1)))
    weights = g * np.exp(-E * beta)
    #define the relevant items
    Z= np.sum(weights, axis=0)
    p = weights / Z
    #thermal averages
    U = np.sum(p * E, axis=0)
    F = -kB_eV_per_K * T_k * np.log(Z)
    S = (U - F) / T_k
    return Z, U, F, S
#main function
def main():
    # levels of energy that will be used
    E_iso_eV = np.array([0.0])
    g_iso = np.array([14])
   
    E_soc_eV = np.array([0.0, 0.28])
    g_soc = np.array([6, 8])
    
    E_cfs_eV = np.array([0.00, 0.07, 0.12, 0.13, 0.14])
    g_cfs = np.array([ 4, 2, 2, 4, 2])
   
    T= np.linspace(300.0, 2000.0, 200)
    #all necessary thermodynamics
    Z_iso, U_iso_eV, F_iso_eV, S_iso_eV_perK = thermo_from_levels(E_iso_eV, g_iso, T)
    Z_soc, U_soc_eV, F_soc_eV, S_soc_eV_perK = thermo_from_levels(E_soc_eV, g_soc, T)
    Z_cfs, U_cfs_eV, F_cfs_eV, S_cfs_eV_perK = thermo_from_levels(E_cfs_eV, g_cfs, T)
    #conversion function
    def eV_to_J(X): return X*J_PER_eV
    U_iso_J, F_iso_J, S_iso_J_perK = map(eV_to_J, (U_iso_eV, F_iso_eV, S_iso_eV_perK))
    U_soc_J, F_soc_J, S_soc_J_perK= map(eV_to_J, (U_soc_eV, F_soc_eV, S_soc_eV_perK))
    U_cfs_J, F_cfs_J, S_cfs_J_perK = map(eV_to_J, (U_cfs_eV, F_cfs_eV, S_cfs_eV_perK))
    #write a csv
    here = Path(__file__).resolve().parent
    out = here / "ce_thermo.csv"
    df = pd.DataFrame({"T_K": T,"Z_iso":Z_iso, "Z_soc": Z_soc, "Z_cfs":Z_cfs, "U_iso_eV": U_iso_eV, "F_iso_eV": F_iso_eV, "S_iso_eV_perK":S_iso_eV_perK, "U_soc_eV": U_soc_eV, "F_soc_eV": F_soc_eV, "S_soc_eV_perK": S_soc_eV_perK,
                    "U_cfs_eV": U_cfs_eV, "F_cfs_eV": F_cfs_eV, "S_cfs_eV_perK":S_cfs_eV_perK,
                    "U_iso_J": U_iso_J, "F_iso_J": F_iso_J, "S_iso_J_perK": S_iso_J_perK,
                    "U_soc_J": U_soc_J, "F_soc_J": F_soc_J, "S_soc_J_perK":S_soc_J_perK, 
                    "U_cfs_J": U_cfs_J, "F_cfs_J": F_cfs_J,"S_cfs_J_perK": S_cfs_J_perK})
    df.to_csv(out, index=False)
    print("Wrote", out)
if __name__ == "__main__":
        main()
