import numpy as np
import qutip as qp

from sim_plain import SimulatorPlain as Sim

method_options = ["Finite-Diff", 'Ours']
method_name = method_options[1]

sim = Sim(basis='Legendre', n_basis=6, n_epoch=12, method_name=method_name, sampling_measure=False)

sim.logger.write_text("demo_MaxCut_{} ========".format(method_name))

n_qubit = 4
graph = [[0, 1], [0, 3], [1, 2], [2, 3]]
superposition = np.array([0] * 2**n_qubit)
for i in range(2**n_qubit):
    z = np.array([0] * 2**n_qubit)
    z[i] = 1
    superposition += z
superposition = superposition / np.sqrt(2.0**n_qubit)

Xs = []
I = np.array(
    [[1, 0], 
    [0, 1]])
X = np.array(
    [[0, 1], 
    [1, 0]])
Z = np.array(
    [[1, 0], 
    [0, -1]])

II = I
for i in range(n_qubit - 1):
    II = np.kron(II, I)

OO = II * 0.0

H0 = OO
H_cost = OO

omega0 = 1 * np.pi
omega1 = 1 * np.pi
n_layers = 1
sim.T = 2 * np.pi * (1. / omega0 + 1. / omega1) * n_layers
print("sim.T: ", sim.T)

sim.Pauli_M = []
for e in graph:
    if 0 in e:
        curr = Z
    else:
        curr = I
    for i in range(1, n_qubit):
        if i in e:
            curr = np.kron(curr, Z)
        else:
            curr = np.kron(curr, I)

    sim.Pauli_M.append([curr, 0.5])
    H_cost += II - curr
H_cost = - H_cost * 0.5
sim.Pauli_M.append([II, -0.5 * len(graph)])

for i in range(len(sim.Pauli_M)) :
    sim.Pauli_M[i].append(qp.Qobj(sim.Pauli_M[i][0]).eigenstates())

Hs = []
sim.omegas = []
for e in graph:
    H = sim.multi_kron(*[I if j not in e else Z for j in range(n_qubit)]) 
    Hs.append(H)
    sim.omegas.append(omega0)

for i in range(n_qubit):
    H = sim.multi_kron(*[I if j not in [i] else X for j in range(n_qubit)])
    Hs.append(H)
    sim.omegas.append(omega1)

Hs = [qp.Qobj(H) for H in Hs]

H_cost = qp.Qobj(H_cost)
H0 = qp.Qobj(H0)
superposition = qp.Qobj(superposition)
sim.train_energy_FD(H_cost, H0, Hs, superposition)

state, prob = sim.find_state(sim.final_state)
print("cut result is ", bin(state)[2:])
