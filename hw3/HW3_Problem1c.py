import os
from fractions import Fraction
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 0.5
W = 2 * np.pi
D = 1 + W**2
I_STAR = 1 / (1 - np.exp(-1))     
AMP = EPS / np.sqrt(D)            
T_END, T_TRANS = 300.0, 100.0     
H_STEP = 0.02                     


def G(t, I):
    return I + EPS * (np.sin(W * t) - W * np.cos(W * t)) / D


def next_spike(T, I, h=H_STEP, window=60.0):
    gT = G(T, I)
    f = lambda t: G(t, I) - gT * np.exp(-(t - T)) - 1.0
    t = T
    while t - T < window:
        if f(t + h) >= 0:
            return brentq(f, t, t + h, xtol=1e-13)
        t += h
    return None


def simulate(I):
    if I + AMP < 1:
        return 0.0, np.array([])
    T, spikes = 0.0, []
    while T < T_END:
        Tn = next_spike(T, I)
        if Tn is None:
            return 0.0, np.array(spikes)
        spikes.append(Tn)
        T = Tn
    spikes = np.array(spikes)
    return np.sum(spikes > T_TRANS) / (T_END - T_TRANS), spikes


_, sp = simulate(1.6)

Is = np.linspace(0.9, 3.0, 1051)
rates = np.array([simulate(I)[0] for I in Is])

s = 0
for k in range(1, len(Is) + 1):
    if k == len(Is) or abs(rates[k] - rates[s]) > 1e-6:
        s = k

Iu = np.linspace(1.0005, 3.0, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Is, rates, "k.", ms=2, label=r"forced, $\epsilon=0.5$")
ax.plot(Iu, 1 / np.log(Iu / (Iu - 1)), "r-", lw=1, label=r"unforced $1/\ln[I/(I-1)]$")
ax.plot([0.9, 1.0], [0, 0], "r-", lw=1)
ax.axvline(I_STAR - AMP, color="b", ls="--", lw=1, label="1:1 tongue edges (theory)")
ax.axvline(I_STAR + AMP, color="b", ls="--", lw=1)
ax.set_xlabel("constant input I (dimensionless)")
ax.set_ylabel("firing number (spikes per forcing period)")
ax.set_title(r"Forced LIF: $\dot v=-v+I+\epsilon\sin 2\pi t$, $\epsilon=0.5$, "
             "measured over t in (100, 300)", fontsize=9)
ax.legend(loc="upper left")
