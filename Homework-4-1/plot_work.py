# all functions
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
# get the columns from the dataframe, first trying for the exact match,
#then looking at a fallback.
def _get_col(df: pd.DataFrame, *colnames: str) -> pd.Series:
    """Return the first column found in df from the provided colnames."""
    lower_map = {col.lower(): col for col in df.columns}
    for name in colnames:
        key = name.lower()
        if name in lower_map:
            return df[lower_map[key]]
    for name in colnames:
        key = name.lower()
        for c in df.columns:
            if key in c.lower():
                return df[c]
    raise KeyError(f"None of the columns {colnames} found in DataFrame.")
# look at csv and get arrays
def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    vf = _get_col(df, 'Vf_m3', 'vf', 'volume', "final_volume").to_numpy()
    w = _get_col(df, 'W_J', 'w_J', 'work', 'w').to_numpy()
    return vf, w
def main() -> None:
    # take the local CSVs and use them for the plot.
    here = Path(__file__).resolve().parent
    
    iso = here / "work_isothermal.csv"
    adi = here / "work_adiabatic.csv"
    
    v_iso, w_iso = load_csv(iso)
    v_adi, w_adi = load_csv(adi)
    #all necessary figures
    plt.figure(figsize=(8,6))
    plt.plot(v_iso, w_iso, label="Isothermal(T=300)", color="blue")
    plt.plot(v_adi, w_adi, label=r"Adiabatic($\gamma=1.4$)", color="red")
    plt.xlabel("Final Volume $V_f$ (m$^3$)", fontsize=14)
    plt.ylabel("Work $W$ (J)", fontsize=14)
    plt.title("Work Done by Expanding Gas, Adiabatic vs Isothermal", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    plt.tight_layout()
    out = Path("work_vs_volume.png")
    plt.savefig(out, dpi=200)
    print(f"Saved plot to", out)
if __name__ == "__main__":
    main()