#all functionalities
from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid
import csv
from pathlib import Path
#all constants
n= 1.0
R = 8.314  
T = 300.0
V_initial = 0.1
gamma = 1.4
Vf_min = V_initial
Vf_max = 3.0*V_initial
num_Vf =101
points_per_integral = 101
# the following functions:
#  find pressure on an isothermal path, compute work for an isothermal change, 
# builds a volume grid, and then Main just goes through the final volumes
# and gets the corresponding work and sends to the csv.
def P_isothermal(V:np.ndarray) -> np.ndarray:
        return n*R*T/V
def work_isothermal(V_initial:float, Vf:float, num_points:int=1000) -> float:
    V_grid = np.linspace(V_initial, Vf, num_points)
    P_grid = P_isothermal(V_grid)
    return -trapezoid(P_grid, V_grid)
def main() -> None:
    Vf_values = np.linspace(Vf_min, Vf_max, num_Vf)
    work_values = np.array([work_isothermal(V_initial, Vf, points_per_integral) for Vf in Vf_values])
    outpath = Path("work_isothermal.csv")
    #export and print check
    with outpath.open( 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Vf_ms', 'W_J'])
        writer.writerows(zip(Vf_values, work_values))
    print(f"Wrote {outpath.resolve()} with {len(Vf_values)} rows.")
if __name__ == "__main__":
        main()    