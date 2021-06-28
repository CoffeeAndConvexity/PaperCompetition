''' Fair division of [0,1] given n buyers with linear valuation functions '''

import numpy as np
import cvxpy as cp
np.random.seed(888)

n = 4

# first consider uniform budgets
# B = np.random.exponential(n) + 1
# B = np.ones(n)
B = np.array([1,3,2,4])
B = B/np.sum(B)
# generate n linear valuation functions, nonnegative and normalized on [0,1]
# d = np.random.uniform(low=0.0, high=2.0 ,size = n)
d = np.array([1.2, 0.6, 0.3, 1.9])
c = (1 - d) * 2 
# sort them and construct inverse permutation given sorting
sorted_indices = np.argsort(d)[::-1]
place_of_buyer = [-1] * n 
for ii, jj in enumerate(sorted_indices):
    place_of_buyer[jj] = ii

# sort budgest and linear coefficients
B, c, d = B[sorted_indices], c[sorted_indices], d[sorted_indices]

# build EG using CVXPY
u = cp.Variable(n, nonneg=True)
z, w, s, t = cp.Variable(n-1), cp.Variable(n-1), cp.Variable(n-1), cp.Variable(n-1)
G = [np.array([ [d[i], c[i]/2], [-d[i+1], -c[i+1]/2] ]) for i in range(n-1)]
objective = cp.Maximize(cp.sum(B * cp.log(u)))
# linear constraints for slack variables z, w
constraints = [z>=0, z<=1, w<=0, w>=-1] 

# linear constraints u together with z, w
constraints.append(u[0] <= z[0])
for i in range(1, n-1):
    constraints.append(u[i] <= z[i] + w[i-1])
constraints.append(u[n-1] <= 1+w[n-2])
# second order cone constraints
for i in range(n-1):
    constraints.append(z[i]+w[i] >= 0)
    constraints.append(G[i][0,0]*s[i]+G[i][0,1]*t[i] == z[i])
    constraints.append(G[i][1,0]*s[i]+G[i][1,1]*t[i] == w[i])
    constraints.append(s[i]**2 <= t[i])

print('Solve the conic program (after normalization & sorting)')
prob = cp.Problem(objective, constraints)
prob.solve()
u, z, w, s, t = u.value, z.value, w.value, s.value, t.value

print('buyers sorted by decreasing d[i]: {}'.format(sorted_indices))
print('EG opt. obj. = {:.5f}'.format(prob.value))
# print('EG opt. u = {}'.format(u))

def eval(i,ll,rr):
    ''' evaluate [ll, rr] under  v[i] '''
    return 0.5 * c[i] * (rr**2 - ll**2) + d[i] * (rr-ll)

def move_knife(i, ui, l):
    ''' given v[i], utility value and left endpoint, find right endpoint '''
    aa, bb, cc = c[i]/2, d[i], - (c[i]/2 * l**2 + d[i] * l + ui)
    return (-bb + np.sqrt(bb**2 - 4 * aa * cc))/(2*aa)

# reconstruct original c and d
B, c, d = B[place_of_buyer], c[place_of_buyer], d[place_of_buyer]

# reconstruct u
u = u[place_of_buyer]

# find all breakpoints, including 0 and 1
bpts = [0]
for j in range(n-1):
    i = sorted_indices[j] # get the j-th buyer i
    bpts.append(move_knife(i, u[i], bpts[-1]))
bpts.append(1)

# construct allocation of all buyers
allocation = []
for i in range(n):
    jj = place_of_buyer[i]
    allocation.append((bpts[jj], bpts[jj+1])) # left and right endpoints of buyer i
    print('buyer {} gets interval ({:.4f}, {:.4f}) with utility {:.4f}, its intercept is ranked {}'.format(i, bpts[jj], bpts[jj+1], eval(i, bpts[jj], bpts[jj+1]), place_of_buyer[i]))

eval(1, allocation[1][0], allocation[1][1])
# eval(sorted_indices[-1], bpts[-2], bpts[-1])

# check inf-dim optimality by computing the dual objective value
beta = B/u
first_term = np.sum([beta[i] * eval(i, bpts[place_of_buyer[i]], bpts[place_of_buyer[i]+1]) for i in range(n)])
# first_term = np.sum(beta[i] * u[i] for i in range(n))
second_term = - np.sum(B * np.log(beta))
constant = np.sum(B) - np.sum(B*np.log(B))
primal_obj = prob.value
dual_obj = first_term + second_term - constant
print('dual obj. (from beta := B/u) = {:.5f}'.format(primal_obj, dual_obj))

####################################################################
# try to plot the buyers
from matplotlib import pyplot as plt
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
thetas = np.linspace(0, 1, 1000)
p_discrete_approx = np.max(np.array([beta[i] * (c[i]*thetas + d[i]) for i in range(n)]), axis=0)
# plot them
for i in range(n):
    plt.plot(thetas, beta[i] * (c[i]*thetas + d[i]), linestyle = linestyles[i], label = r'$\beta^*_{}v_{}$'.format(i+1,i+1))
    # plt.plot(x, beta_ave[i] * (c[i]*x + d[i]), label =  r"$\beta_{} v_{}: [{:,.2f}, {:.2f}]$".format(i, i, *division[i]), linestyle = linestyles[i])
    # plt.plot(x, beta_ave[i] * (c[i]*x + d[i]), label =  r"$\beta_{} v_{}$", linestyle = linestyles[i])
p_line_indices = np.arange(0, 1000, 20)
plt.scatter(thetas[p_line_indices], p_discrete_approx[p_line_indices], label = r"$ p^* := \max_i \beta^*_i v_i$", marker = '*', s = 50, color = 'brown', alpha = 0.368)

i = sorted_indices[0]
bpvals = [beta[i] * (c[i] * allocation[i][0] + d[i])]
for j in range(n):
    i = sorted_indices[j] # serial number of the j-th largest buyer
    lep = beta[i] * (c[i] * allocation[i][1] + d[i]) # left endpoint of the line segment of j-th buyer i
    # print(j, i, lep)
    bpvals.append(lep)

# plt.scatter(bpts, bpvals, marker = '|', color = 'brown')
plt.legend(loc = 'lower left')
plt.xlabel(r"$\theta$")
# plt.show()
plt.savefig("n-linear.pdf")
