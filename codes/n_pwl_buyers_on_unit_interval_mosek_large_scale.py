''' Fair division of [0,1] given n buyers with linear valuation functions '''

from functools import total_ordering
import numpy as np
import cvxpy as cp
import argparse
from numpy.core.arrayprint import dtype_is_implied

from numpy.core.shape_base import block

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", '-n', type=int, default=100)
    parser.add_argument("--K", '-K', type=int, default=50)
    parser.add_argument("--seed", '-sd', type=int, default=2021)
    args = parser.parse_args()
    n, K, seed = args.n, args.K, args.seed
except:
    n, K, seed = 100, 100, 1 # K = number of pieces of each buyer's valuation

# n,K,seed = 1000,1000,1

np.random.seed(seed)
print("n = {} buyers and K = {} pieces per buyer...".format(n, K))
bpts = np.random.exponential(size=(K+1,))
bpts = np.cumsum(bpts)
bpts /= bpts[-1]
bpts[0] = 0 # first bpt is 0, last bpt is 1

# permutations[1]
c, d = np.random.normal(size=(n, K)), 5 * np.random.normal(size=(n, K)) # coefficients of linear pieces of each buyer
# make sure all of them are >= 0
for i in range(n): 
    for k in range(K):
        min_val = min(c[i,k]*bpts[k] + d[i,k], c[i,k]*bpts[k+1] + d[i,k])
        d[i,k] += max(0, -min_val) # print(min(c[i,k]*bpts[k] + d[i,k], c[i,k]*bpts[k] + d[i,k]))

# buyers' budgest (sum(B) == 1 w.l.o.g.)
B = np.random.uniform(0, 1, n) + 2
B = B/np.sum(B)

############ solving process starts here ############
from time import time
begin = time()
# for every i, k, find constants
func = lambda i, k: c[i,k]/2 * (bpts[k+1]-bpts[k])**2 + (bpts[k+1]-bpts[k])*(c[i,k]*bpts[k]+d[i,k])
M = np.array([[func(i,k) for k in range(K)] for i in range(n)]) # Lambda in the paper
func = lambda i, k: (bpts[k+1]-bpts[k])**2 * c[i,k]/M[i,k]
c_hat = np.array([[ func(i,k) for k in range(K) ] for i in range(n)])
# c_hat[:, 0].shape
func = lambda i, k: (bpts[k+1]-bpts[k]) * (c[i,k]*bpts[k] + d[i,k])/M[i,k]
d_hat = np.array([[func(i,k) for k in range(K)] for i in range(n)]) # d_hat[:, 1] M[:,1]

permutations = []
for k in range(K):
    permutations.append(np.argsort(d_hat[:,k])[::-1].astype(np.int32)) # the top j-th buyer is i = permutation[k][j] 

for k in range(K):
    c_hat[:, k], d_hat[:, k] = c_hat[permutations[k], k], d_hat[permutations[k], k]

from mosek.fusion import *
model = Model('fair_division_unit_interval')
######################################################
# exponential cone auxiliary variables
q = model.variable('first exp-cone var', n, Domain.unbounded())
# qs = model.variable('second exp-cone var', n, Domain.unbounded())
u_total = model.variable('u[i]', n, Domain.inRange(0,1))
u = model.variable('u[i,k]', [n,K], Domain.inRange(0,1))
u_hat = model.variable('u_hat[i,k]', [n,K], Domain.inRange(0,1))
s = model.variable('s[i,k]', [n-1,K], Domain.unbounded())
t = model.variable('t[i,k]', [n-1,K], Domain.unbounded())
z = model.variable('z[i,k]', [n-1,K], Domain.inRange(0,1))
w = model.variable('w[i,k]', [n-1,K], Domain.inRange(-1,0))

# utility sum and transformation constraints
model.constraint('sum-u[i,k]-to-u_total[i]', Expr.sub(Expr.sum(u, 1), u_total), Domain.equalsTo(0.0))
# func = lambda i, k: 0.5 * c[i,k] * (bpts[k+1]**2 - bpts[k]**2) + d[i,k] *(bpts[k+1] - bpts[k]) # a function to compute Lambda / M
# func(permutations[k][j],k) == M[permutations[k][j],k]
import scipy as sp
import scipy.sparse as spsp

# allTransMatrices = [Matrix.sparse(n, n, permutations[k], np.arange(n), np.array([M[permutations[k][j], k] for j in range(n)])) for k in range(K)]
# u_hat.reshape
for k in range(K):
    vals, rows, cols = np.array([M[permutations[k][j], k] for j in range(n)], dtype=np.double), permutations[k].astype(np.int32), np.arange(n, dtype=np.int32)
    TransMat = Matrix.sparse(n, n, rows, cols, vals)
    model.constraint('TransMat @ u_hat[:,k] == u[:,k] {}'.format(k), Expr.sub(Expr.mul(TransMat, u_hat.slice([0,k], [n,k+1])), u.slice([0,k], [n,k+1])), Domain.equalsTo(0))

    # for j in range(n):
    #     lhs, rhs = Expr.mul(M[permutations[k][j], k],  u_hat.index(j, k)), u.index(permutations[k][j], k)
    #     model.constraint('u_hat[{},{}]-to-u[{},{}]'.format(j, k, permutations[k][j], k), Expr.sub(lhs, rhs), Domain.equalsTo(0))

# each interval k and u_hat representation
func = lambda i, k: np.array([[d_hat[i, k], c_hat[i, k]/2], [-d_hat[i+1, k], -c_hat[i+1, k]/2]])
G = np.array([[func(i,k) for k in range(K)] for i in range(n-1)]) # G[i,k] is a 2-by-2 matrix
# i, k = 1, 2

# z + w >= 0
model.constraint('z+w>=0 (all i,k)', Expr.add(z, w), Domain.greaterThan(0))

# the n constraints involving u, z, w
model.constraint('u_hat[0,:]<=z[0,:]', Expr.sub(u_hat.slice([0,0], [1,K]), z.slice([0,0], [1,K])), Domain.lessThan(0))
lhs, rhs = u_hat.slice([1,0], [n-1, K]), Expr.add(z.slice([1,0], [n-1,K]), w.slice([0,0], [n-2, K]))
model.constraint('u_hat[1:n-1,k] <= z[1:n-1,k] + w[1:n-1,k] for all k', Expr.sub(lhs, rhs), Domain.lessThan(0))
model.constraint('u_hat[n-1,k]<=1+w[n-2,k] for all k', Expr.sub(Expr.sub(u_hat.slice([n-1, 0], [n,K]), 1), w.slice([n-2, 0], [n-1,K])), Domain.lessThan(0)) # u_hat[n-1,k]

# G @ (s,t) == (z,w)

# allMats = [Matrix.sparse(G[i,k]) for i in range(n-1) for k in range(K)]
# len(allMats)
# blockDiagG = Matrix.diag(allMats)


# bd_sparse = spsp.block_diag(

# G.fla

# blockDiagG.numRows(), blockDiagG.numColumns()

# st_all = Expr.vstack([Expr.vstack(s.index(i,k), t.index(i,k)) for i in range(n-1) for k in range(K)])
# Expr.mul(blockDiagG, st_all)

# # [(a,b) for a in range(10) for b in range(4)]


# for i in range(n-1):
#     Gdiag = 

# GG = G.reshape((n-1)*K, 2, 2)

rows = np.concatenate([2*idx + np.array([0,0,1,1]) for idx in range((n-1)*K)])
cols = np.concatenate([2*idx + np.array([0,1,0,1]) for idx in range((n-1)*K)])
vals = G.flatten() # Gdiag = spsp.csc_matrix((vals, (rows, cols)), shape=(2*(n-1)*K, 2*(n-1)*K)) # same as spsp.block_diag([G[i,k] for i in range(n-1) for k in range(K)])
rows, cols, vals = rows.astype(np.int32), cols.astype(np.int32), vals.astype(np.double)
GdiagMosek = Matrix.sparse(2*(n-1)*K, 2*(n-1)*K, rows, cols, vals)
st_all = Expr.vstack([Expr.vstack(s.index(i,k), t.index(i,k)) for i in range(n-1) for k in range(K)])
zw_all = Expr.vstack([Expr.vstack(z.index(i,k), w.index(i,k)) for i in range(n-1) for k in range(K)])
model.constraint('G(s,t)<=(z,w) all', Expr.sub(Expr.mul(GdiagMosek, st_all), zw_all), Domain.equalsTo(0))

expression = Expr.hstack([Expr.constTerm(1/2 * np.ones((n-1)*K)), t.reshape((n-1)*K), s.reshape((n-1)*K)])
cd = Domain.inRotatedQCone(3)
cd.axis(1)
model.constraint('(s,t) conic all', expression, Domain.inRotatedQCone((n-1)*K, 3))

# for k in range(K):
    # constraints for u[i,k] and z[i,k], w[i,k]
    # model.constraint('u_hat[0,k]<=z[0,k] {}'.format(k), Expr.sub(u_hat.index(0,k), z.index(0,k)), Domain.lessThan(0)) # u_hat[0,k]
    # lhs, rhs = u_hat.slice([1,k], [n-1, k+1]), Expr.add(z.slice([1,k], [n-1,k+1]), w.slice([0,k], [n-2, k+1]))
    # model.constraint('u_hat[1:n-1,k] <= z[1:n-1,k] + w[1:n-1,k] {}'.format(k), Expr.sub(lhs, rhs), Domain.lessThan(0))
    # for i in range(1, n-1):
    #     lhs, rhs = Expr.sub(u_hat.index(i,k), z.index(i,k)), w.index(i-1,k)
    #     model.constraint('u_hat[i,k]<=z[i,k]+w[i-1,k] ({},{})'.format(i,k), Expr.sub(lhs, rhs), Domain.lessThan(0))    
    # model.constraint('u_hat[n-1,k]<=1+w[n-2,k] {}'.format(k), Expr.sub(Expr.sub(u_hat.index(n-1,k), 1), w.index(n-2,k)), Domain.lessThan(0)) # u_hat[n-1,k]
    # constraints for (z,w) and (s,t)
    # for i in range(n-1): # u_hat[1,k], ...u_hat[n-2,k]
    #     lhs, rhs = Expr.mul(G[i,k], Expr.vstack(s.index(i,k), t.index(i,k))), Expr.vstack(z.index(i,k), w.index(i,k))
    #     model.constraint('G@(s,t)==(z,w) ({},{})'.format(i,k),  Expr.sub(lhs, rhs), Domain.equalsTo(0))
        # model.constraint('z+w>=0 ({},{})'.format(i,k), Expr.add(z.index(i,k), w.index(i,k)), Domain.greaterThan(0))
        # model.constraint('(s,t) conic ({},{})'.format(i,k), Expr.vstack(1/2, t.index(i, k), s.index(i,k)), Domain.inRotatedQCone(3))

# objective <-> exponential cone
for i in range(n):
    model.constraint('(qi,q,ui) in Exp, i={}'.format(i), Expr.vstack(u_total.index(i), 1, q.index(i)), Domain.inPExpCone())

print('formulation time = {}'.format(time() - begin))
# actually solve
model.objective('maximization obj', ObjectiveSense.Maximize, Expr.dot(B, q))
before_mosek_solve = time()
model.solve()

print('mosek.solve() time = {}'.format(time() - before_mosek_solve))
try:
    u, u_total = u.level().reshape(n,K), u_total.level()
except:
    pass

u = np.maximum(u, 0)
tiny_num = max(1e-8, 1e-8 * np.min(M[M>0])) # a small number
u[u<= tiny_num] = 0

# u_total = np.sum(u, 1)

# compute a pure allocation
def eval_sub(i, k, ll, rr): # given (ll, rr) within k-th subinterval
    try:
        assert(bpts[k] <= ll <= rr <= bpts[k+1])
    except:
        print(k, bpts[k], ll, rr, bpts[k+1])
    return 0.5 * c[i,k] * (rr**2 - ll**2) + d[i,k] * (rr - ll)

def move_knife_sub(i, k, ui, ll): # given ll within k-th subinterval, find rr (also within) s.t. utility = ui
    if ui <= 1e-8: # buyer i gets nothing in k-th interval
        return ll
    assert(bpts[k] <= ll)
    if ll >= bpts[k+1]:
        return bpts[k+1]
    # try:
    #     assert(eval_sub(i, k, ll, bpts[k+1]) >= ui - 1e-7) # otherwise cannot attain ui
    # except:
    #     print(i, k, eval_sub(i, k, ll, bpts[k+1]), ui)
    aa, bb, cc = c[i,k]/2, d[i,k], - (c[i,k]/2 * ll**2 + d[i,k] * ll + ui)
    return min((-bb + np.sqrt(bb**2 - 4 * aa * cc))/(2*aa), bpts[k+1])

allocations = [[None] * K for i in range(n)] # record the interval of buyer i in k-th predefined interval
bpts_all = [] # for record...
for k in range(K):
    # print("predefin/ed interval {} is ({:.4f}, {:.4f})".format(k, bpts[k], bpts[k+1]))
    bpts_sub = [bpts[k]]
    for j in range(n):
        i = permutations[k][j]
        if j == n-1:
            bpts_sub.append(bpts[k+1])
        else:
            bpts_sub.append(move_knife_sub(i, k, u[i,k], bpts_sub[-1]))
        # print('{}-th buyer is {}, u[{},{}] = {:.4f}, its interval is ({:.4f}, {:.4f}), valuated utility = {:.4f}'.format(j, i, i, k, u[i,k], bpts_sub[-2], bpts_sub[-1], eval_sub(i, k, bpts_sub[-2], bpts_sub[-1])))
        allocations[i][k] = (bpts_sub[-2], bpts_sub[-1])
    bpts_all.append(bpts_sub)

total_time = time() - begin
print('total time = {}'.format(total_time))

# import csv, os
# mode = 'a' if os.path.exists('logs/large-scale.csv') else 'w'
# with open('logs/large-scale.csv', mode, newline='') as ff:
#     writer = csv.writer(ff, dialect='excel')
#     if mode == 'w':
#         writer.writerow(['n', 'K', 'seed', 'total_time'])
#     writer.writerow([n, K, seed, total_time])

u_buyers = np.sum(u, axis=1)
u_computed = np.array([sum(eval_sub(i, k, *allocations[i][k]) for k in range(K)) for i in range(n)])
print('indicator of numerical error: ||u_buyers - u_computed|| = {}'.format(np.linalg.norm(u_buyers - u_total)))
# u_total
# k = 0
# [allocations[permutations[k][j]][k] for j in range(n)], permutations[k], u[:, k]

# [eval_sub(i, k, *allocations[i][k]) for i in range(n)]

# u[:,44][u[:,44]>=1e-5]
# bpts[45]