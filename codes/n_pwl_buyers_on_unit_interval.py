''' Fair division of [0,1] given n buyers with linear valuation functions '''

import numpy as np
import cvxpy as cp
np.random.seed(38)

n = 4
K = 3 # every buyer has (at most) linear pieces
print("n = {} buyers and K = {} pieces per buyer...".format(n, K))
bpts = np.random.exponential(size=(K+1,))
bpts = np.cumsum(bpts)
bpts /= bpts[-1]
bpts[0] = 0
# permutations[1]
c, d = np.random.normal(size=(n, K)), np.random.normal(size=(n, K)) # coefficients of linear pieces of each buyer
# make sure all of them are >= 0
for i in range(n): 
    for k in range(K):
        min_val = min(c[i,k]*bpts[k] + d[i,k], c[i,k]*bpts[k+1] + d[i,k])
        d[i,k] += max(0, -min_val) + 1e-10 # print(min(c[i,k]*bpts[k] + d[i,k], c[i,k]*bpts[k] + d[i,k]))

# buyers' budgest (sum(B) == 1 w.l.o.g.)
B = np.random.uniform(0, 1, n) + 5
B = B/np.sum(B)

# for every i, k, find constants
func = lambda i, k: c[i,k]/2 * (bpts[k+1]-bpts[k])**2 + (bpts[k+1]-bpts[k])*(c[i,k]*bpts[k]+d[i,k])
M = np.array([[func(i,k) for k in range(K)] for i in range(n)])
func = lambda i, k: (bpts[k+1]-bpts[k])**2 * c[i,k]/M[i,k]
c_hat = np.array([[ func(i,k) for k in range(K) ] for i in range(n)])
func = lambda i, k: (bpts[k+1]-bpts[k]) * (c[i,k]*bpts[k] + d[i,k])/M[i,k]
d_hat = np.array([[func(i,k) for k in range(K)] for i in range(n)]) # d_hat[:, 1] M[:,1]
permutations = []
for k in range(K): 
    print(d_hat[:, k])
    permutations.append(np.argsort(d_hat[:,k])[::-1]) # the top j-th buyer is i = permutation[k][j] 
    print('permutation {} = {}'.format(k, permutations[-1]))

for k in range(K):
    c_hat[:, k], d_hat[:, k] = c_hat[permutations[k], k], d_hat[permutations[k], k]

def get_inverse(perm): # get the inverse permutation
    res = [-1] * len(perm)
    for ii, jj in enumerate(perm):
        res[jj] = ii
    return np.array(res)

inverse_permutations = [get_inverse(perm) for perm in permutations] # in k-th interval, buyer i's d[i,k] is ranked in the inverse_permutations[k][i]-th place

u = cp.Variable((n,K), nonneg=True)
u_hat = cp.Variable((n,K), nonneg=True) # U_set = DP(U_hat_set)
# auxiliary variables, k copies each
z, w, s, t = cp.Variable((n-1, K)), cp.Variable((n-1, K)), cp.Variable((n-1, K)), cp.Variable((n-1, K))
func = lambda i, k: np.array([[d_hat[i, k], c_hat[i, k]/2], [-d_hat[i+1, k], -c_hat[i+1, k]/2]])
G = np.array([[func(i,k) for k in range(K)] for i in range(n-1)]) # G[i,k] is a 2-by-2 matrix

# linear transformlations from u_hat to u (via diagonal and permutation matrices)
func = lambda i, k: 0.5 * c[i,k] * (bpts[k+1]**2 - bpts[k]**2) + d[i,k] *(bpts[k+1] - bpts[k])
constraints = [u[permutations[k][j],k] == func(permutations[k][j],k) * u_hat[j, k] for j in range(n) for k in range(K)]
constraints.extend([z>=0, z<=1, w<=0, w>=-1]) # true for all z[i,k], w[i,k]
constraints.append(z+w >= 0)

for k in range(K): # linear constraints on u_hat, z, w
    constraints.append(u_hat[0,k] <= z[0,k])     
    for i in range(1, n-1):
        constraints.append(u_hat[i,k] <= z[i,k] + w[i-1,k])
    constraints.append(u_hat[n-1,k] <= 1 + w[n-2, k])

for k in range(K): # second-order cone constraints
    for i in range(n-1):
        constraints.append(G[i,k][0,0]*s[i, k] + G[i,k][0,1]*t[i,k] == z[i,k])
        constraints.append(G[i,k][1,0]*s[i,k]+G[i,k][1,1]*t[i,k] == w[i,k])
        constraints.append(s[i,k]**2 <= t[i,k])

objective = cp.Maximize(cp.sum(B * cp.log(cp.sum(u, axis=1)))) # set EG maximization objective

print('Solve the conic program (after normalization & sorting)')
prob = cp.Problem(objective, constraints)

prob.solve(solver='MOSEK', verbose=False)
print('solving status = {}'.format(prob.status))
u, z, w, s, t = u.value, z.value, w.value, s.value, t.value

print('partitioning each predefined subinterval k into (at most) n smaller intervals')
def eval_sub(i, k, ll, rr): # given (ll, rr) within k-th subinterval
    assert(bpts[k] <= ll <= rr <= bpts[k+1])
    return 0.5 * c[i,k] * (rr**2 - ll**2) + d[i,k] * (rr - ll)

def move_knife_sub(i, k, ui, ll): # given ll within k-th subinterval, find rr (also within) s.t. utility = ui
    if ui <= 1e-8: # buyer i gets nothing in k-th interval
        return ll 
    assert(bpts[k] <= ll <= bpts[k+1])
    assert(eval_sub(i, k, ll, bpts[k+1]) >= ui) # otherwise cannot attain ui
    aa, bb, cc = c[i,k]/2, d[i,k], - (c[i,k]/2 * ll**2 + d[i,k] * ll + ui)
    return (-bb + np.sqrt(bb**2 - 4 * aa * cc))/(2*aa) # this should <= bpts[k+1], otherwise something is WRONG

allocations = [[None] * K for i in range(n)] # record the interval of buyer i in k-th predefined interval 
bpts_all = [] # for record...
for k in range(K):
    print("predefined interval {} is ({:.4f}, {:.4f})".format(k, bpts[k], bpts[k+1]))
    bpts_sub = [bpts[k]]
    for j in range(n):
        i = permutations[k][j]
        if j == n-1:
            bpts_sub.append(bpts[k+1])
        else:
            bpts_sub.append(move_knife_sub(i, k, u[i,k], bpts_sub[-1]))
        print('{}-th buyer is {}, u[{},{}] = {:.4f}, its interval is ({:.4f}, {:.4f}), valuated utility = {:.4f}'.format(j, i, i, k, u[i,k], bpts_sub[-2], bpts_sub[-1], eval_sub(i, k, bpts_sub[-2], bpts_sub[-1])))
        allocations[i][k] = (bpts_sub[-2], bpts_sub[-1])
    bpts_all.append(bpts_sub)

u_buyers = np.sum(u, axis=1)

for i in range(n):
    [print("{:.4f}".format(d[i,k]), end=' & ' if k < K-1 else ' \\\\') for k in range(K)]
    print('')

u_computed = np.array([sum(eval_sub(i, k, allocations[i][k][0], allocations[i][k][1]) for k in range(K)) for i in range(n)])
print('indicator of numerical error: ||u_buyers - u_computed|| = {:.4f}'.format(np.linalg.norm(u_buyers - u_computed)))

# construct a dual (feasible) solution and compute its dual objective
beta = B/u_buyers
first_term = sum(beta[i] * eval_sub(i, k, allocations[i][k][0], allocations[i][k][1]) for i in range(n) for k in range(K))
second_term = - np.sum(B * np.log(beta))
constant = np.sum(B) - np.sum(B*np.log(B))
primal_obj = prob.value
dual_obj = first_term + second_term - constant
print('EG primal obj. = {:.4f}, dual obj. = {:.4f}'.format(primal_obj, dual_obj))

####################################################################
# try to plot the buyers
from matplotlib import pyplot as plt
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
colors = ['orange' ,'blue', 'red', 'green', 'brown']
thetas = np.linspace(0, 1, 1000)

# plot each buyer on each interval
for k in range(K):
    p_line_indices = np.arange(0, 1000, 100)
    thetas = np.linspace(bpts[k], bpts[k+1], 1000)
    prices = np.zeros(1000)
    for i in range(n):
        yvals = beta[i] * (c[i,k] * thetas + d[i,k])
        prices = np.maximum(prices, yvals)  
        plt.plot(thetas, yvals, linestyle = linestyles[i], color = colors[i], label = r'$\beta^*_{} v_{}$'.format(i+1, i+1) if k == 0 else None)
    plt.scatter(thetas[p_line_indices], prices[p_line_indices], label = r"$ p^* := \max_i \beta^*_i v_i$" if k == 0 else None, marker = '*', s = 50,  color = 'brown', alpha = 0.368)

plt.legend(loc='upper left', ncol = 3)
plt.xlabel(r"$\theta$")
# plt.show()

plt.savefig("n-pwl.pdf")