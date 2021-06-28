# consider n buyers with linear valuations on [0,1]

import numpy as np
from matplotlib import pyplot as plt
np.random.seed(5)

def integrate(ci, di, li=0, ui=1):
    ''' integral of ci*theta + di over theta in [li,ui]; assuming >= 0 on [0,1] and ui >= li'''
    assert(ui>=li)
    if li == ui:
        return 0
    return 0.5 * ci * (ui ** 2 - li ** 2) + di * (ui - li)

n = 4 # number of buyers
# c, d = np.random.normal(0, 1, (n, ))/2, np.random.normal(5, 1, (n, )) # generate some buyers each buyer vi is characterized by its (c[i], d[i])
# c, d = np.array([-3, -2, 0, 4, 7])/5, np.array([11, 12, 13, 14, 15])
c, d = np.array([1, -2, -3, 1])/8, np.array([2, 6, 7, 5])*2
min_val = np.minimum(d, c + d)
d = d + 1e-10 + 0.5 * np.random.exponential(1, (n,)) + np.maximum(-min_val, 0) # make sure all are >= 0 on [0,1]
assert(np.prod(d>=0) == 1 and np.prod(c+d>=0) == 1) 
# normalize all vi
integrals = [integrate(cc, dd) for (cc,dd) in zip(c, d)]
c, d = c/integrals, d/integrals
B = np.abs(np.random.normal(0, 1, (n,))) # budgets
# B = np.ones((n,)) # everyone has the same budget
# B = [1,2,3,4]
B = np.ones((n,))
B = B/np.sum(B) # normalize to sum(B) == 1

# SDA on beta
beta_min, beta_max = np.ones((n,)), 1/B
beta = (beta_min+beta_max)/2 
beta_ave = 0.01*np.ones((n,))/n
g_ave = 0.01*np.ones((n,))/n
T = 100000
for t in range(1, T+1):
    # sample theta
    theta = np.random.uniform(0, 1)
    # compute subgrad
    ii = np.argmax(beta*(c*theta + d))
    ggii = c[ii] * theta + d[ii]
    g_ave *= (t-1)/t
    g_ave[ii] += ggii/t
    # update beta and beta_ave
    beta = np.minimum(np.maximum(B/g_ave, beta_min), beta_max)
    beta_ave = (t-1)/t * beta_ave + beta/t
    if ( t%(T//20) ==0):
        print("t = {}, beta_ave = {}, B/beta_ave = {}".format(t, beta_ave, B/beta_ave))

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
        if isbp:
            breakpoints.append(thetaij)
            bpvals.append(valij)

# sort them and add the point at theta = 1
temp_argsort = np.argsort(breakpoints)
breakpoints = np.array(breakpoints)[temp_argsort]
bpvals = np.array(bpvals)[temp_argsort]
breakpoints = np.append(breakpoints, 1)
bpvals = np.append(bpvals, np.max(beta_ave*(c+d)))

# find winners corr. to the pieces
winners = [np.argmax(beta_ave * d)] # list of winners corr. to the line segments following the b.p.'s
for k in range(1, n): # there are exactly n breakpoints (including 0 excluding 1)
    theta_middle = (breakpoints[k] + breakpoints[k+1])/2
    # find the winner at theta_middle, which is the winner on segment k as well
    ik = np.argmax(beta_ave * (c * theta_middle + d))
    winners.append(ik)

# final division, i.e., division[i] = the interval buyer i gets
division = [None] * n
for k in range(n):
    i = winners[k]
    division[i] = (breakpoints[k], breakpoints[k+1])

# compute envy
umat = np.zeros((n,n)) # umat[i,j] is vi of buyer j's interval
for i in range(n):
    for j in range(n):
        umat[i,j] = integrate(c[i], d[i], *division[j])/B[j]

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

#############################################################################################
# plot
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), ]
x = np.linspace(0, 1, 1000)
p_discrete_approx = np.max(np.array([beta_ave[i] * (c[i]*x + d[i]) for i in range(n)]), axis=0)
# plot them
for i in range(n):
    plt.plot(x, beta_ave[i] * (c[i]*x + d[i]), label =  r"$\beta_{} v_{}: [{:.2f}, {:.2f}]$".format(i, i, *division[i]), linestyle = linestyles[i])
    # plt.plot(x, beta_ave[i] * (c[i]*x + d[i]), label =  r"$\beta_{} v_{}$", linestyle = linestyles[i])
plt.plot(x, p_discrete_approx, label = r"$ p := \max_i \beta_i v_i$", linewidth=2, color = 'brown')
plt.scatter(breakpoints, bpvals, marker = '|', color = 'brown')
plt.legend()
plt.xlabel(r"$\theta$")
plt.savefig("n-linear-distinct.pdf")

print(tuple(B))
print(tuple(c))
print(tuple(d))
tuple(envy*B)
print(tuple(envy))