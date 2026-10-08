import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 200.0
PHI0 = 0.4


def simulate(eps, t_end):
    v = np.array([-L, -1.0 / np.tan(np.pi * (1 - PHI0))])   
    f = lambda t, y: [1 + y[0]**2 + eps * (y[1] - y[0]),
                      1 + y[1]**2 + eps * (y[0] - y[1])]
    events = []
    for i in range(2):
        e = (lambda i: (lambda t, y: y[i] - L))(i)
        e.terminal, e.direction = True, 1
        events.append(e)
    t, spikes = 0.0, [[], []]
    while t < t_end:
        s = solve_ivp(f, [t, t_end], v, events=events, rtol=1e-10, atol=1e-10, method="DOP853")
        t, v = s.t[-1], s.y[:, -1].copy()
        if s.status != 1:
            break
        for i in range(2):
            if len(s.t_events[i]):
                spikes[i].append(s.t_events[i][0])
                v[i] = -L                                  
    return np.array(spikes[0]), np.array(spikes[1])


def lags(s1, s2):
    t_k, phi_k = [], []
    for k in range(len(s1) - 1):
        nxt = s2[s2 >= s1[k]]
        if len(nxt) == 0:
            break
        if nxt[0] >= s1[k + 1]:
            continue
        phi_k.append((nxt[0] - s1[k]) / (s1[k + 1] - s1[k]))
        t_k.append(s1[k])
    return np.array(t_k), np.array(phi_k)


results = {}
for eps in (0.005, 0.02, 0.05):
    t_end = np.log(np.tan(np.pi * PHI0) / 0.02) / (2 * eps)   # run until tan(pi phi) ~ 0.02
    s1, s2 = simulate(eps, t_end)
    t_k, phi_k = lags(s1, s2)
    m = phi_k > 0.01
    slope = np.polyfit(t_k[m], np.log(np.tan(np.pi * phi_k[m])), 1)[0]
    results[eps] = (t_k, phi_k, slope)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
eps = 0.02
t_k, phi_k, _ = results[eps]
tt = np.linspace(0, t_k.max(), 300)
ax[0].plot(tt, np.arctan(np.tan(np.pi * PHI0) * np.exp(-2 * eps * tt)) / np.pi, "r-", label="prediction")
ax[0].plot(t_k, phi_k, "ko", ms=3, label="simulation (spike times)")
ax[0].set_xlabel("time t (units where uncoupled period = pi)")
ax[0].set_ylabel(r"lag $\phi_k$ (cycles)")
ax[0].set_title(r"$\epsilon=0.02$, L=200, $\phi_0=0.4$", fontsize=9)
ax[0].legend()
for e, (t_k, phi_k, slope) in results.items():
    ax[1].semilogy(t_k, np.tan(np.pi * phi_k), ".", label=r"$\epsilon$=%g, fit slope %.4f (-2$\epsilon$=%.4f)" % (e, slope, -2 * e))
ax[1].set_xlabel("time t")
ax[1].set_ylabel(r"$\tan(\pi\phi_k)$")
ax[1].legend(fontsize=8)
