import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def theoretical_model(t, L1, L2):
    # y(t)^2 from the linearised SEIR solution with ICs y(0)=0, ydot(0)=1
    return ((np.exp(L1 * t) - np.exp(L2 * t)) / (L1 - L2)) ** 2


def exp_model(t, a, b, c):
    # phenomenological exponential g(t) = a + b*e^(ct)
    return a + b * np.exp(c * t)


def main():
    # load data
    df = pd.read_csv('data.csv', header=None, names=['t', 'y2'])
    t  = df['t'].values
    y2 = df['y2'].values

    # fit theoretical SEIR model (nonlinear least squares)
    # expect L1 > 0 (growing mode), L2 < 0 (decaying mode)
    popt_theory, pcov_theory = curve_fit(
        theoretical_model, t, y2, p0=[0.5, -0.5], maxfev=10000
    )
    L1, L2 = popt_theory
    L1_err, L2_err = np.sqrt(np.diag(pcov_theory))

    print("=== Theoretical SEIR Fit ===")
    print(f"Lambda_1 = {L1:.4f}  (+/- {L1_err:.4f})")
    print(f"Lambda_2 = {L2:.4f}  (+/- {L2_err:.4f})")
    print(f"sigma + gamma      = {-(L1+L2):.4f}")
    print(f"sigma*(gamma-beta) = {L1*L2:.4f}")
    print(f"Doubling time      = {np.log(2)/L1:.3f} time units")
    print()

    # fit exponential model
    popt_exp, _ = curve_fit(exp_model, t, y2, p0=[0, 1, 0.4], maxfev=10000)
    a_e, b_e, c_e = popt_exp

    print("=== Exponential Fit: g(t) = a + b*exp(c*t) ===")
    print(f"a = {a_e:.4f},  b = {b_e:.4f},  c = {c_e:.4f}")
    print(f"c/2 = {c_e/2:.4f}  (should be close to Lambda_1 = {L1:.4f})")
    print()

    # polynomial fit via normal equations A^T A x = A^T b
    degree = 3
    A = np.column_stack([t**k for k in range(degree, -1, -1)])
    ATA    = A.T @ A
    ATb    = A.T @ y2
    coeffs = np.linalg.solve(ATA, ATb)

    print(f"=== Polynomial Fit (degree {degree}) ===")
    for i, val in enumerate(coeffs):
        print(f"  c_{degree-i} = {val:.6f}")
    print()

    poly_vals = A @ coeffs

    # RMSE for each model
    def rmse(pred):
        return np.sqrt(np.mean((pred - y2)**2))

    print("=== RMSE ===")
    print(f"Theoretical : {rmse(theoretical_model(t, L1, L2)):.4f}")
    print(f"Exponential : {rmse(exp_model(t, a_e, b_e, c_e)):.4f}")
    print(f"Polynomial  : {rmse(poly_vals):.4f}")
    print()

    # plot
    t_fine = np.linspace(t.min(), t.max(), 500)
    A_fine = np.column_stack([t_fine**k for k in range(degree, -1, -1)])

    plt.figure(figsize=(10, 6))
    plt.scatter(t, y2, color='steelblue', alpha=0.4, s=15,
                label=r'Data (noisy $y^2$)')
    plt.plot(t_fine, theoretical_model(t_fine, L1, L2), 'r-', lw=2.5,
             label=rf'Theoretical: $\Lambda_1$={L1:.3f}, $\Lambda_2$={L2:.3f}')
    plt.plot(t_fine, exp_model(t_fine, a_e, b_e, c_e), 'g--', lw=2,
             label=rf'Exp: {a_e:.2f} + {b_e:.2f}$e^{{{c_e:.2f}t}}$')
    plt.plot(t_fine, A_fine @ coeffs, 'k:', lw=2,
             label=f'Poly deg {degree}')

    plt.xlabel('Time (t)', fontsize=13)
    plt.ylabel(r'$(y(t))^2$', fontsize=13)
    plt.title('Data Fits for Question 4', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q4_model_fits.png', dpi=300)
    print("saved q4_model_fits.png")


if __name__ == "__main__":
    main()
