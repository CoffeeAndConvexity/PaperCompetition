''' Fair division of [0,1] given n buyers with linear valuation functions '''

import numpy as np
import cvxpy as cp
import argparse
np.random.seed(2021)


from mosek.fusion import *
n, K = 10, 5
model = Model('utility-maximization-trial')
u = model.variable('u[i,k]', [n,K], Domain.inRange(0,1))
u_total = model.variable('u[i]', n, Domain.inRange(0,1))
model.objective('maximization obj', ObjectiveSense.Maximize, Expr.sum(u_total))
model.solve()
u_total.level()



# model = Model('ceo1')

# x = model.variable('x', 3, Domain.unbounded())

# # Create the constraint
# #      x[0] + x[1] + x[2] = 1.0
# model.constraint("lc", Expr.sum(x), Domain.equalsTo(1.0))

# # Create the conic exponential constraint
# expc = model.constraint("expc", x, Domain.inPExpCone())

# # Set the objective function to (x[0] + x[1])
# model.objective("obj", ObjectiveSense.Minimize, Expr.sum(x.slice(0,2)))

# # Solve the problem
# model.solve()

# model.writeTask('ceo1.ptf')
# # Get the linear solution values
# solx = x.level()
# print('x1,x2,x3 = %s' % str(solx))

# # Get conic solution of expc
# expcval  = expc.level()
# expcdual = expc.dual()
# print('expc levels                = %s' % str(expcval))
# print('expc dual conic var levels = %s' % str(expcdual))