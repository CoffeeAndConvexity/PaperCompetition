import numpy as np
from matplotlib import pyplot as plt

d1, d2 = 1.5, 0.6
c1, c2 = (1-d1) * 2, (1-d2) * 2

s = np.linspace(0, 1, num = 1000)
t = s ** 2

z = (c1/2) * t + d1 * s
w = -(c2/2) * t - d2 * s

# let t > s_fix ** 2
s_fix = 0.8
t = np.linspace(s_fix, 5, num = 1000) ** 2

zt = (c1/2) * t + d1 * s_fix
wt = -(c2/2) * t - d2 * s_fix

plt.plot(z, w, linewidth=2.33, color = 'blue', label = r'$(z_i, w_i)$ as $s_i \in [0,1]$ and $t_i = s_i^2$')
plt.plot(z, -z, linewidth=2.33, color = 'red', label = r'$z_i + w_i = 0$')
plt.plot(zt, wt, linewidth=2.33, color = 'brown', linestyle='dotted', label = r'Fixed $s_0 = 0.5$, let $t_i \in [s_0^2, 1]$')
plt.xlabel(r'$z_i$')
plt.ylabel(r'$w_i$')
plt.xlim([0, 1])
plt.ylim([-1, 0])
plt.legend()
plt.title(r'$d_i = {}$, $d_{{i+1}} = {}$'.format(d1, d2))
w_lin = -w
plt.fill_between(z, -z, w, color=(0.2, 0.2, 0.2, 0.5))