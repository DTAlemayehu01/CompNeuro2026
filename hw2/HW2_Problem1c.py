import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

c = lambda a : np.sqrt(a)*(1-2*a)/np.sqrt(a*(1-a))
A = np.linspace(0.0001, 0.99, 100)
C = c(A)
plt.plot(A, C, label=r'$c(a)$', color='r')
plt.axhline(2, label=r'Passive apparent speed', color='orange')
plt.axhline(1/2, label=r'Crossing $a$')
plt.xlabel('$a$')
plt.ylabel('$c$')
plt.title("$D=1$ and Apparent Speed")
plt.legend()
