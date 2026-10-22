import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import ndtr

N = 500
alphas = np.arange(0.02, 0.63, 0.04)

fu = []
mm = []
ms = []

for al in alphas:
    P = int(round(al * N))
    X = np.random.choice([-1, 1], size=(N, P))
    W = X @ X.T / N
    np.fill_diagonal(W,0)
    
    H = W @ X
    fu.append(np.mean(X * H < 0))
    
    ov = []
    for c in range(10):
        mu = np.random.randint(P)
        xi = X[:,mu].copy()
        s = xi.copy()
        fl = np.random.choice(N, N//10, replace=False)
        s[fl] *= -1
        while True:
            ch = 0
            for i in np.random.permutation(N):
                new = 1 if W[i] @ s >= 0 else -1
                if new != s[i]:
                    s[i] = new
                    ch += 1
            if ch == 0:
                break
        ov.append(xi @ s / N)
    mm.append(np.mean(ov))
    ms.append(np.std(ov))
    
    print(f"alpha={al:.2f} P={P} unstable={fu[-1]:.4f} theory={ndtr(-1/np.sqrt(al)):.4f} <m>={mm[-1]:.4f} sd={ms[-1]:.3f}")
    
fig, ax = plt.subplots(1, 2, figsize=(10,4))
aa = np.linspace(0.02, 0.62, 200)
ax[0].plot(alphas, fu, "o", label="simulation")
ax[0].plot(aa, ndtr(-1/np.sqrt(aa)), label="Φ(-1/√α)")
ax[0].set_xlabel("α=P/N")
ax[0].set_ylabel("fraction unstable bits")
ax[0].legend()
ax[1].errorbar(alphas, mm, ms, fmt="o-")
ax[1].set_xlabel("α=P/N")
ax[1].set_ylabel("<m> (N=500, 10% cue noise)")
ax[1].axvline(0.138, ls=":", c="gray")
plt.tight_layout()
