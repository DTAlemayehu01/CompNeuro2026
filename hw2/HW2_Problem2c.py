import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
B = lambda V: (1 + (1/3.57)*np.e**(-V/16.13))**(-1)
f = lambda V: 1 + V*(1 - B(V))/16.13
res = fsolve(f, [-30, 10])[0]
print(res)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

meta_B = lambda Mg, V: (1 + Mg/3.57*np.e**(-V/16.13))**(-1)
B1 = lambda V: meta_B(0, V)
B2 = lambda V: meta_B(0.1, V)
B3 = lambda V: meta_B(1, V)

g_bar = 1
I1 = lambda V: g_bar*B1(V)*(V - 0)
I2 = lambda V: g_bar*B2(V)*(V - 0)
I3 = lambda V: g_bar*B3(V)*(V - 0)

X = np.linspace(-90, 0, 1000)
Y1 = I1(X)
Y2 = I2(X)
Y3 = I3(X)

plt.plot(X, Y1, label='0 mM', color='red')
plt.plot(X, Y2, label='0.1 mM', color='orange')
plt.plot(X, Y3, label='1 mM', color='blue')
plt.xlabel("Voltage")
plt.ylabel("Current")
plt.title(r"Voltage v Current given $\bar g = 1$")
plt.legend()
