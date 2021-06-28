import numpy as np
from matplotlib import pyplot as plt

d1, d2 = 1.5, 0.8
c1, c2 = (1-d1) * 2, (1-d2) * 2

a = np.linspace(0, 1, num = 1000)

s = c1/2 * a**2 + d1 * a
w = -c2/2 * a**2 - d2 * a

plt.plot(s, w, linewidth=2.33)
plt.plot(s, -s, linewidth=2.33)
plt.xlabel(r'$s_i$')
plt.ylabel(r'$w_i$')
plt.xlim([0, 1])
plt.ylim([-1, 0])
plt.fill_between(s, -s, w, color=(0.2, 0.2, 0.2, 0.5))
plt.savefig('S-set-two-buyers.pdf')