#functions from L7 that were imported
def ols_slope(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator
def ols_intercept(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = ols_slope(x, y)
    return y_mean - slope * x_mean
def ols(x, y):
    slope = ols_slope(x, y)
    intercept = ols_intercept(x, y)
    return slope, intercept
#all other necessary imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#confidence interval function 
def _ci95_from_se(est, se, n):
    try:
        from scipy.stats import t
        tcrit = t.ppf(.975, df=n-2)
    except Exception:
        df= max(n-2, 1)
        lookup= {1: 12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365,
                 8:2.306, 9:2.262, 10:2.228, 12:2.179, 15:2.131, 20:2.086, 30:2.042, 60:2.000 }
        keys= sorted(lookup.keys())
        nearest=min(keys, key=lambda k: abs(k-df))
        tcrit=lookup[nearest]
    #gives you the estimate with plus minus attached
    return est- tcrit * se, est + tcrit * se
# function with std errors
def _ols_core(x, y):
    x= np.asarray(x, float)
    y= np.asarray(y, float)
    n= x.size
    # necessary means and sums of squares for our calculations
    xbar= x.mean()
    ybar= y.mean()
    Sxx= np.sum((x-xbar)**2)
    Sxy= np.sum((x-xbar)*(y-ybar))
    # basic fit parameters necessary for our final plot.
    a= Sxy / Sxx
    b = ybar - a * xbar
    yhat = a * x + b
    resid= y - yhat
    s2 = np.sum(resid**2) / (n-2)
    # std errors for the slope and intercept that will be plotted.
    Sa = np.sqrt(s2 / Sxx)
    Sb = np.sqrt(s2 * (1.0/n + xbar**2/ Sxx))
    return a, b, Sa, Sb, n
# main function for making final plot with all annotations and saving it to the notebook. It will also return necessary results.
def run_troutons_rule(csv_path="trouton.csv", outdir = "homework-3-1", save_png = True, show_plot=True):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")
    df = pd.read_csv(csv_path)


    
    T = df["T_B(K)"].to_numpy(float)
    Hv_kJ = (df["H_v(kcal/mol)"]. to_numpy(float))*4.184
    a, b = ols(T, Hv_kJ)
    # Std errors calculated with function for confidence intervals
    a_cf, b_cf, Sa, Sb, n = _ols_core(T, Hv_kJ)
    lo_a, hi_a = _ci95_from_se(a, Sa, n)
    lo_b, hi_b = _ci95_from_se(b, Sb, n)
    
    a_J = 1000.0 * a
    lo_a_J = 1000.0 * lo_a
    hi_a_J =1000.0 *hi_a
    
    os.makedirs(outdir, exist_ok= True)
    # plot setup 
    fig, ax = plt.subplots(figsize=(7, 5), dpi= 150)
    
    for cls in df["Class"].unique():
        m= (df["Class"] == cls)
        ax.scatter(df.loc[m, "T_B(K)"],
                df.loc[m, "H_v(kcal/mol)"]*4.184, label = str(cls), alpha =0.9)
    # line over the range of desired T values
    xfit = np.linspace(.98*T.min(), 1.02*T.max(), 300)
    yfit = a *xfit + b
    ax.plot(xfit, yfit, linewidth =2)
    
    ax.set_xlabel("Boiling Temperature, $T_B$ (K)")
    ax.set_ylabel("Enthalpy of Vaporization, $H_v$ (kJ/mol)")
    ax.set_title("Trouton's Rule")
    # equation that will be displayed in plot
    eqn = (r"$H_v = a\,T_B + b$" +"\n" 
           + rf"$a = {a_J:0.1f}$ J/mol K"
           rf" [95% CI: {lo_a_J: .1f}, {hi_a_J:0.1f}]"
           +rf"$b ={b:0.2f}$ kJ/mol "
           rf"[95% CI: {lo_b:0.2f}, {hi_b:0.2f}]")
    
    ax.text(0.02, 0.98, eqn, transform=ax.transAxes, va = "top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray"))
    #legend setup 
    ax.legend(title="Class", frameon=True)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    outpath= os.path.join(outdir, "troutons_rule.png")
    if save_png:
        fig.savefig(outpath)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    # all results needed for final plot.
    return {"slope_kJ_per_molK": a, "slope_J_per_molK": a_J,
            "slope_CI95_J_per_molK": (lo_a_J, hi_a_J),
            "intercept_kJ_per_mol": b, "intercept_CI95_kJ_per_mol": (lo_b, hi_b),
             "n": int(n), "figure_path": outpath if save_png else None, }



