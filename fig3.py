import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Results Fig. 3 - Eq 1 - 3 

'''
A = frequency of addicitive acts
V = addiiction vulnerability
S = self-control

q = value of A that corresponds with an extremely high consumptions level added over one week
d = unlearning parameter
b = contant of proportionality of the impact of A ('cue sensitivity')
S+ = maximum capacity occurs naturally 
p =  pyschological resilience parameter


Literature values
q = 0.8 (maximum pure alcohol consumption over one week 80 alcoholic beverages)
b = 2d/q = 0.5
p = 2d = 0.4
d = 0.2
S+ = 0.5
h = 0.2
k = 0.25

Equilibria for Eq 3 for E within (-0.5, 0.3)

'''

# Paper Parameters
q, b, d = 0.8, 0.5, 0.2 
p, h, k = 0.4, 0.2, 0.25
S_plus = 0.5 

def equilibria(E, C0, S0):
    C, S = C0, S0 # initial guess values
    for _ in range(500): # iterate over arbitrary high # of loops to get to equilibrium
        
        V = min(1, max(0, C - S - E))
        A = q * V # Equation 1
        C_new = C + b * min(1, 1 - C) * A - d * C # Equation 2
        S_new = S + p * max(0, S_plus - S) - h * C - k * A # Equation 3
        
        if abs(C_new - C) < 1e-6 and abs(S_new - S) < 1e-6: # check to for equilibrium convergence
            break
            
        C,S = C_new, S_new
    return C,S # convergence values for C and S
    
def unequilibria(E, C0, S0):
    def syseqs(vars): # iterate over arbitrary high # of loops to get to equilibrium
        C, S = vars
        V = min(1, max(0, C - S - E))
        A = q * V # Equation 1
        C_new = C + b * min(1, 1 - C) * A - d * C # Equation 2
        S_new = S + p * max(0, S_plus - S) - h * C - k * A # Equation 3
        return [C_new - C, S_new - S]
    return fsolve(syseqs, [C0, S0]) # use f solve to numerically solve sys for equlibrium 


# For piece-wise plotting
E_range1 = np.linspace(-1, 0, 200)  
E_range2 = np.linspace(-0.5, 0, 200)
E_range3 = np.linspace(-0.5, 1, 200)

# Stable Branch 1
Ceq, Seq = [], [] # intialize storage vecs
for E in E_range1:
    C, S = equilibria(E, 0.667, -0.333) # initial guesses 
    Ceq.append(C)
    Seq.append(S)

# Unstable Branch 2
E_unstable = []
C_unstable, S_unstable = [], []

for E in E_range2:
    eq1 = unequilibria(E, 0, 0)  # initial guesses 
    if 0 <= eq1[0] <= 1 and -0.75 <= eq1[1] <= S_plus:
        E_unstable.append(E)
        C_unstable.append(eq1[0]) # extract unstable C eq values
        S_unstable.append(eq1[1]) # extract unstable S eq values

# Stable Branch 2
Ceq3, Seq3 = [], [] # intialize storage vecs
for E in E_range3:
    C, S = equilibria(E,0, 0.667)  # initial guesses 
    Ceq3.append(C)
    Seq3.append(S)

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(E_range1, Ceq, label='C vs. E', color='red')
plt.plot(E_unstable, C_unstable, label='C vs. E', color='blue', linestyle="--")
plt.plot(E_range3, Ceq3, label='C vs. E', color='darkblue')
plt.xlabel('E (External Control)')
plt.ylabel('C (Craving)')
# plt.xlim(-1,1)
# plt.ylim(-0.1, 1)
plt.title('C vs. E')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(E_range1, Seq, label='S vs. E', color='red')
plt.plot(E_unstable, S_unstable, label='C vs. E', color='blue', linestyle="--")
plt.plot(E_range3, Seq3, label='C vs. E', color='darkblue')
plt.xlabel('E (External Control)')
plt.ylabel('S (Self-Control)')
# plt.xlim(-1,1)
# plt.ylim(-0.75, 0.75)
plt.title('S vs. E')
plt.grid(True)

plt.show()
