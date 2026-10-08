import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DELTA_KICK = 1e-3
N_CYC = 8
RT, AT = 1e-12, 1e-13

I_LIF = 1.5
PERIOD_LIF = np.log(I_LIF / (I_LIF - 1.0))          


def lif_T_next(phi):
    v0 = I_LIF - I_LIF * np.exp(-phi * PERIOD_LIF) + DELTA_KICK
    ev = lambda t, y: y[0] - 1.0
    ev.terminal, ev.direction = True, 1
    s = solve_ivp(lambda t, y: [-y[0] + I_LIF], [0, 10], [v0], events=ev, rtol=RT, atol=AT)
    return phi * PERIOD_LIF + s.t_events[0][0]


def fhn(t, y):
    v, w = y
    return [v - v**3 / 3 - w + 0.5, 0.08 * (v + 0.7 - 0.8 * w)]


def up(t, y):
    return y[0]
up.direction = 1

s = solve_ivp(fhn, [0, 400], [-1.2, -0.6], events=up, rtol=RT, atol=AT, method="DOP853")
te = s.t_events[0]
P_FHN = te[-1] - te[-2]                              
Y_REF = s.y_events[0][-1]                           


def R_fhn(phi):
    y = Y_REF.copy()
    if phi > 0:
        y = solve_ivp(fhn, [0, phi * P_FHN], y, rtol=RT, atol=AT, method="DOP853").y[:, -1]
    y[0] += DELTA_KICK
    horizon = (N_CYC + 0.5) * P_FHN - phi * P_FHN
    sol = solve_ivp(fhn, [0, horizon], y, events=up, rtol=RT, atol=AT, method="DOP853")
    times = phi * P_FHN + sol.t_events[0]          
    T_cross = times[np.argmin(np.abs(times - N_CYC * P_FHN))]
    return (N_CYC * P_FHN - T_cross) / (P_FHN * DELTA_KICK)


ph = np.linspace(0.01, 0.99, 50)
R_num = np.array([(PERIOD_LIF - lif_T_next(p)) / (PERIOD_LIF * DELTA_KICK) for p in ph])
R_ana = np.exp(ph * PERIOD_LIF) / (I_LIF * PERIOD_LIF)

phs = np.linspace(0.005, 0.995, 100)
R_f = np.array([R_fhn(p) for p in phs])
mn = minimize_scalar(R_fhn, bracket=(0.28, 0.305, 0.33), tol=1e-6)
mx = minimize_scalar(lambda p: -R_fhn(p), bracket=(0.86, 0.895, 0.93), tol=1e-6)
z1 = brentq(R_fhn, 0.2, 0.25)
z2 = brentq(R_fhn, 0.77, 0.83)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ph, R_num, "o", ms=3, label="numerical (kick 1e-3)")
ax[0].plot(ph, R_ana, "-", label=r"$e^{\phi\Delta}/(I\Delta)$")
ax[0].set_xlabel(r"phase $\phi$ (cycles)")
ax[0].set_ylabel(r"$R(\phi)$ (phase per unit voltage)")
ax[0].set_title(r"LIF: $\tau=1,v_r=0,v_{th}=1,I=1.5$, $\Delta=%.4f$" % PERIOD_LIF, fontsize=9)
ax[0].legend()
ax[1].plot(phs, R_f, "g-")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel(r"phase $\phi$ (cycles; 0 = upward crossing of $v=0$)")
ax[1].set_ylabel(r"$R(\phi)$ (phase per unit voltage)")
ax[1].set_title("FitzHugh-Nagumo, I=0.5, period %.3f (time units)" % P_FHN, fontsize=9)
