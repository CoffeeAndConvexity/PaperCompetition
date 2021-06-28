# consider n buyers with linear valuations on [0,1]
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(2021)

def integrate(ci, di, li=0, ui=1):
    ''' integral of ci*theta + di over theta in [li,ui]; assuming >= 0 on [0,1] and ui >= li'''
    assert(ui>=li)
    if li == ui:
        return 0
    return 0.5 * ci * (ui ** 2 - li ** 2) + di * (ui - li)

n = 5 # number of buyers
c, d = np.random.normal(0, 1, (n, ))/3, 1 + np.random.normal(2, 1, (n, )) # make the plot prettier
# c, d = np.array([-3, -1, 2, 4])/5, np.array([11, 8, 13, 15])
# c, d = np.array([-3, -2, 1, 2, 4])/2, np.array([5, 7, 9, 16, 20])
min_val = np.minimum(d, c + d)
d = d + 1e-10 + 0.5 * np.random.exponential(1, (n,)) + np.maximum(-min_val, 0) # make sure all are >= 0 on [0,1]
assert(np.prod(d>=0) == 1 and np.prod(c+d>=0) == 1) 
# normalize all vi
integrals = [integrate(cc, dd) for (cc,dd) in zip(c, d)]
c, d = c/integrals, d/integrals
B = np.abs(np.random.normal(0, 1, (n,))) + 5 # budgets
# B = np.ones((n,)) # everyone has the same budget
B = B/np.sum(B) # normalize to sum(B) == 1

def find_breakpoints_winners(beta_ave):
    ''' n == len(beta_ave) == len(c) == len(d); find the breakpoint of max beta[i] * v[i], v[i] has coeff. c[i], d[i] '''
    breakpoints = [0] # 0 is always defined as the first b.p.
    bpvals = [np.max(beta_ave * d)] # value of p at 0
    for i in range(n):
        for j in range(i+1, n):
            thetaij = -(beta_ave[j] * d[j] - beta_ave[i] * d[i]) / (beta_ave[j] * c[j] - beta_ave[i] * c[i])
            if thetaij >= 1 or thetaij <= 0: # intersecting outside [0,1]
                continue
            # check if function value of thetaij is above all others
            valij = beta_ave[i] *(c[i] * thetaij + d[i]) # = ...using j
            # assert(valij == beta_ave[j] *(c[j] * thetaij + d[j]))
            isbp = True
            for k in range(n):
                if k!=i and k!=j:
                    if beta_ave[k] * (c[k] * thetaij + d[k]) > valij: # (thetaij, vij) is not a breakpoint if another buyer is strictly above it
                        isbp = False
                        break
            if isbp and (thetaij not in breakpoints):
                breakpoints.append(thetaij)
                bpvals.append(valij)
    # sort them and add the point at theta = 1
    temp_argsort = np.argsort(breakpoints)
    breakpoints = np.array(breakpoints)[temp_argsort]
    bpvals = np.array(bpvals)[temp_argsort]
    breakpoints = np.append(breakpoints, 1)
    bpvals = np.append(bpvals, np.max(beta_ave*(c+d)))
    winners = [np.argmax(beta_ave * d)] # list of winners corr. to the line segments following the b.p.'s
    for k in range(1, len(breakpoints)-1): # there are exactly n breakpoints (including 0 excluding 1)
        theta_middle = (breakpoints[k] + breakpoints[k+1])/2
        # find the winner at theta_middle, which is the winner on segment k as well
        ik = np.argmax(beta_ave * (c * theta_middle + d))
        winners.append(ik)
        # if len(breakpoints) - 1 < n:
        #     print('number of linear pieces = {}'.format(len(breakpoints)-1))
    return breakpoints, bpvals, winners

def subgrad_first_term(beta):
    breakpoints, _, winners = find_breakpoints_winners(beta)
    gg = np.zeros((n,))
    for k in range(len(breakpoints)-1): # in the beginning, the number of breakpoints may be less than n
        ll, rr = breakpoints[k], breakpoints[k+1] # left and right endpoints of current piece
        ii = winners[k] # winner of current piece, contribute to ii-th component of gg
        gg[ii] += beta[ii] * ( 0.5 * c[ii] * (rr**2 - ll**2) + d[ii] * (rr - ll))
    return gg

def subgrad_full(beta):
    return subgrad_first_term(beta) - B/beta

# # Deterministic DA on beta
# # beta_min, beta_max = np.ones((n,)), 1/B
# beta = (B+1)/2
# beta_ave = np.zeros((n,))
# g_ave = 0.01*np.ones((n,))/n
# T = 50000
# for t in range(1, T+1):
#     # compute full subgrad
#     gg = subgrad_first_term(beta)
#     g_ave = (t-1)/t * g_ave + gg/t
#     # update beta and beta_ave
#     beta = np.minimum(np.maximum(B/g_ave, B), 1)
#     beta_ave = (t-1)/t * beta_ave + beta/t
#     if ( t%(T//20) ==0):
#     # if True:
#         print("t = {}, beta_ave = {}, B/beta_ave = {}".format(t, beta_ave, B/beta_ave))

# SDA
beta, beta_ave, g_ave = (B+1)/2, np.zeros((n,)), 0.01*np.ones((n,))/n
T = 500000
for t in range(1, T+1):
    # sample theta and find winner
    theta = np.random.uniform()
    v_theta = c * theta + d
    winner = np.argmax(beta * v_theta)
    # update dual average
    g_ave *= (t-1)/t
    g_ave[winner] += v_theta[winner] / t
    # update beta and beta_ave
    beta = np.minimum(np.maximum(B/g_ave, B), 1)
    beta_ave = (t-1)/t * beta_ave + beta/t
    if ( t%(T//20) ==0):
    # if True:
        print("t = {}, beta_ave = {}, B/beta_ave = {}".format(t, beta_ave, B/beta_ave))

linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), ]
x = np.linspace(0, 1, 1000)

# find breakpoints and winners
# beta_ave = np.array([1,1,1])
breakpoints, bpvals, winners = find_breakpoints_winners(beta_ave)
# final division, i.e., division[i] = the interval buyer i gets
division = [None] * n
for k in range(n):
    i = winners[k]
    division[i] = (breakpoints[k], breakpoints[k+1])

# find discrete approx. p = max of beta[i]*v[i]
p_discrete_approx = np.max(np.array([beta_ave[i] * (c[i]*x + d[i]) for i in range(n)]), axis=0)
# plot them
for i in range(n):
    plt.plot(x, beta_ave[i] * (c[i]*x + d[i]), label =  r"$\beta_{} v_{}: [{:.2f}, {:.2f}]$".format(i+1, i+1, *division[i]), linestyle = linestyles[i])
    # plt.plot(x, beta_ave[i] * (c[i]*x + d[i]), label =  r"$\beta_{} v_{}$", linestyle = linestyles[i])
plt.plot(x, p_discrete_approx, label = r"$ p := \max_i \beta_i v_i$", linewidth=2, color = 'brown')
plt.scatter(breakpoints, bpvals, marker = '|', color = 'brown')
plt.legend()
plt.xlabel(r"$\theta$")
plt.savefig('n-linear-sda.pdf')

# compute envy of each buyer
umat = np.zeros((n,n)) # umat[i,j] is vi of buyer j's interval
for i in range(n):
    for j in range(n):
        umat[i,j] = integrate(c[i], d[i], *division[j])

envy = [np.max(umat[i]) - umat[i,i] for i in range(n)]
for i in range(n):
    print("envy of buyer {} = {}".format(i, envy[i]))

# compute duality gap
udual = B / beta_ave
uprimal = np.array([integrate(c[i], d[i], *division[i]) for i in range(n)])
obj_eg_primal = np.sum(B * np.log(uprimal))
obj_dual = np.sum(beta_ave * uprimal) - np.sum(B * np.log(beta_ave))
C = sum(B) - np.sum(B * np.log(B))
print("dgap = {}".format(obj_dual - C - obj_eg_primal))