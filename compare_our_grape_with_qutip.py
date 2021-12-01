import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
import grape as gp
import torch

### qutip pulse
ts = np.linspace(0, 1, 100)
H1 = qp.sigmax()
H0 = qp.qeye(2)
psi0 = qp.basis(2, 0)

# def H1_coeff(t, args):
#     return t
H1_coeff = ts
H = [H0,[H1, H1_coeff]]
result = qp.mesolve(H, psi0, ts)
print(len(result.states))

### ours grape
dt = 0.01
qubit_state_num = 2
Q_x = np.diag(np.sqrt(np.arange(1, qubit_state_num)), 1) \
        + np.diag(np.sqrt(np.arange(1, qubit_state_num)), -1)
H0 = np.eye(qubit_state_num)
Hs = [Q_x]
psi0 = [np.array([1,0])]
initial_states = [gp.Grape.c_to_r_vec(v) for v in psi0]
init_u = np.expand_dims(ts, -1)

initial_states = torch.tensor(initial_states).double().transpose(1, 0)
Hs = [torch.tensor(gp.Grape.c_to_r_mat(-1j * dt * H)) for H in Hs]
H0 = torch.tensor(gp.Grape.c_to_r_mat(-1j * dt * H0)) 
us = torch.tensor(init_u)

### compare results
grape = gp.Grape(taylor_terms=50)
final_u = grape.forward_simulate(init_u, H0, Hs, initial_states)

s = 40
n = 3
for a in grape.intermediate_states[s:s+n]:
    print(a)
for a in result.states[s+1:s+1+n]:
    print(a)
# print(result.states[:5])