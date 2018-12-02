# calculate markov bounds for cliques of size 1-n based on variables
from scipy.optimize import fsolve
import numpy as np

n = 1000
epsilon = 0.01

# define functions as val/epsilon - 1 = 0 instead of val - epsilon = 0
# this helps the solver for very small val (since otherwise its ~ epsilon)
func_one = lambda p: ((1-p) ** (k-1)) / epsilon - 1
func_any = lambda p: (1 - ((1- ((1-p) ** (k-1))) ** k)) / epsilon - 1
fsolve(func_one,0)[0]
fsolve(func_any,0)[0]

# loop over all n using previous value as seed
cache = np.zeros((n-1,3))
p_one = 1
p_all = 1
for k in range(2,n+1):
    p_one = fsolve(func_one,p_one)[0]
    p_all = fsolve(func_any,p_all)[0]
    cache[k-2] = [k,p_one,p_all]

np.savetxt('markov_bounds.csv', cache, fmt=['%.d','%.8f','%.8f'],header='k,p_one,p_any',comments='')

