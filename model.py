import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# For Generating different figures
stochastic = True
fig_5 = True
sin_E = False


# stochastic process functions
def R_fn(lam):
    dist = stats.poisson(lam)
    return dist.rvs(1)


def F_fn(A, q):
    return np.minimum(A, q)


# Variable functions
def V_fn(C, S, E):
    return np.minimum(1, np.maximum(0, C-S-E))


def A_fn(V, stochastic, lam, q=0.8):
    if stochastic:
        return F_fn(
            q * V + R_fn(lam) / 7,
            q
        )
    else:
        return q * V


def gamma_fn(C, b):
    """
    b - cue sensitivity
    """
    return b * np.minimum(1, 1 - C)


# Difference Equations
def C_fn(C, A, b=0.5, d=0.2):
    """
    d - unlearning parameter
    """
    gamma = gamma_fn(C, b)
    return (1 - d) * C + gamma * A


def S_fn(C, S, A, h=0.2, k=0.25, S_plus=0.5, p=0.4):
    """
    p - psychological resiliance parameter
    """
    return S + p * np.maximum(0, S_plus - S) - h * C - k * A


def E_fn(E, dE=0.015):
    return E - dE


def E_sinusodal(E, t, freq=52, var=0.1):
    """
    Change to a sinusodal E
    """
    # convert to radians
    f = 2 * np.pi / freq

    E = E + f * np.cos(f * t) * 0.9

    # add a noise term
    noise = stats.norm(0, var).rvs()

    return E + noise


def lam_sinusodal(lam, t, phase_shift=np.pi, freq=52, var=0.1):
    """
    Also making lambda sinusodal as well
    """
    f = 2 * np.pi / freq

    lam = lam + f * np.cos(f * t + phase_shift)

    noise = stats.norm(0, var).rvs()

    return np.abs(lam + noise)


def lam_fn(lam, dlam=0.01):
    return lam + dlam


# running simulations
C_vals = []
S_vals = []
lam_vals = []
E_vals = []
A_vals = []
V_vals = []

# initial values (taken from their excel)
C_0 = np.array([2/3])
A_0 = np.array([4/5])
S_0 = np.array([-1/3])
E_0 = np.array([1])
V_0 = V_fn(C_0, S_0, E_0)
lam_0 = np.array([3])
t = 500  # weeks

# fig 5
if fig_5:
    E_0 = np.array([0.1525])
    lam_0 = np.array([1])
    S_0 = np.array([0.5])
    V_0 = np.array([0.3])
    t = 500  # weeks

C = C_0
A = A_0
S = S_0
V = V_0
E = E_0
lam = lam_0

for i in range(t):
    if stochastic:
        if fig_5:
            pass
        else:
            if sin_E:
                E = E_sinusodal(E, i)
                lam = lam_sinusodal(lam, i)
            else:
                E = E_fn(E)
                lam = lam_fn(lam)

    V = V_fn(C, S, E)
    A = A_fn(V, stochastic=stochastic, lam=lam)

    C = C_fn(C, A)
    S = S_fn(C, S, A)

    lam_vals.append(lam)
    E_vals.append(E)
    C_vals.append(C)
    S_vals.append(S)
    A_vals.append(A)
    V_vals.append(V)

print(f'final C: {C_vals[-1]}')
print(f'final S: {S_vals[-1]}')

# PLotting

# fig3a/b
if not stochastic:
    pass

# fig4a
fig, axs = plt.subplots()
axs.plot(range(len(E_vals)), E_vals, label='E')
axs.plot(range(len(lam_vals)), lam_vals, label='lambda')
axs.set_xlabel('time (weeks)')
fig.legend()
fig.tight_layout()

# fig4b
fig, axs = plt.subplots()
axs.plot(range(len(C_vals)), C_vals, label='C')
axs.plot(range(len(A_vals)), A_vals, label='A')
axs.set_xlabel('time (weeks)')
fig.legend()
fig.tight_layout()

# fig4c
fig, axs = plt.subplots()
axs.plot(range(len(S_vals)), S_vals, label='S', color='red', linestyle='--')
axs.plot(range(len(V_vals)), V_vals, label='V', color='blue')
axs.set_xlabel('time (weeks)')
fig.legend()
fig.tight_layout()

plt.show()
