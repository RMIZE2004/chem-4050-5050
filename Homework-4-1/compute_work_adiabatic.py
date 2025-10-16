#functionalities
from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid
import csv
from pathlib import Path
#constants
n= 1.0
R = 8.314
T = 300.0
V_initial = .1
gamma = 1.4
Vf_min = V_initial
Vf_max = 3.0*V_initial
num_Vf =101
points_per_integral = 1000
Pi = n*R*T/V_initial
K = Pi * (V_initial**gamma)
#The following functions:
# find the pressure for the adiabatic system, find the corresponding work, and then go to the main() 
# to find final volumes and write the corresponding works to the csv. 
def P_adiabatic(V:np.ndarray) -> np.ndarray:
        return K / (V**gamma)
def work_adiabatic(V_initial:float, Vf:float, num_points:int=1000) -> float:
    V_grid = np.linspace(V_initial, Vf, num_points, endpoint = True)
    P_grid = P_adiabatic(V_grid)
    return -trapezoid(P_grid, V_grid)
def main() -> None:
    outdir = Path(__file__).parent
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir/ "work_adiabatic.csv"
    print("cwd:", Path.cwd())
    print("saved to:", outpath)
    Vf_values = np.linspace(Vf_min, Vf_max, num_Vf)
    work_values = [work_adiabatic(V_initial, Vf, points_per_integral) for Vf in Vf_values]
    #export and print check
    with outpath.open( 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Vf_m3', 'W_J'])
        writer.writerows(zip(Vf_values, work_values))
    print(f"Wrote {outpath.resolve()} with {len(Vf_values)} rows.")
if __name__ == "__main__":
        main()  