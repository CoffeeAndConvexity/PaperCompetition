import numpy as np
from matplotlib import pyplot as plt

d1, d2 = 1.5, 0.8
c1, c2 = (1-d1) * 2, (1-d2) * 2

a = np.linspace(0, 1, num = 1000)

u1 = c1 * a**2 / 2 + d1 * a
u2 = c2 * (1 - a**2) / 2 + d2 * (1 - a)

plt.plot(u1, u2, linewidth=2.33)
plt.xlabel(r'$u_1$')
plt.ylabel(r'$u_2$')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.fill_between(u1, u2, color=(0.2, 0.2, 0.2, 0.5))
plt.savefig('U-set-two-buyers.pdf')