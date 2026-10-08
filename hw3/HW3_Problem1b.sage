I = 1.6
Is = e/(e-1)
eps = 0.5
D = sqrt(4*pi^2 + 1)
A = D*(Is - I)/eps
ans = arcsin(A)
f(x) = (arctan(2*pi) - x)/2/pi
s1 = f(ans)
s2 = f(ans - pi)
print(N(s1))
print(N(s2))

I(t) = 1.6 + 0.5*sin(2*pi*t)
L(t) = (1/e)*(I(t))/(I(t)-1)
print(f"{N(s1)} => {N(L(s1))}")
print(f"{N(s2)} => {N(L(s2))}")
