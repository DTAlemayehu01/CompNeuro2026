f(x) = abs(x - e/(e-1))
g(x) = x/(4*pi^2 + 1)
h(x,y) = f(x) - g(y)
implicit_plot(h, (1, 2), (0,2), fill=True, axes_labels=['$\epsilon$', '$I$'], incolor='red', title='Locking Region')
