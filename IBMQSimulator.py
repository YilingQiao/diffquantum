import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute, transpile, schedule, assemble, pulse
from qiskit.providers.aer import QasmSimulator, PulseSimulator
from qiskit.providers.aer.pulse import PulseSystemModel
from qiskit.tools.monitor import job_monitor
from scipy.special import legendre, expit
from math import pi, sqrt
from cmath import exp

import warnings
warnings.filterwarnings('ignore')

if IBMQ.active_account() is None:
    IBMQ.load_account()

provider = IBMQ.get_provider(hub='ibm-q-community', group='ibmquantumawards', project='open-science-22')
backend = provider.get_backend('ibmq_jakarta')
#sim_noisy_jakarta = QasmSimulator.from_backend(backend)
sim_backend = PulseSimulator()
backend_model = PulseSystemModel.from_backend(backend)
sim_backend.set_options(system_model=backend_model)


def normC(x, y) :
    l = sqrt(x ** 2 + y ** 2)
    coef = (2 * expit(l) - 1) / l
    return coef * x, coef * y

def dnormC(x, y) :
    l = sqrt(x ** 2 + y ** 2)
    sigl = expit(l)
    sigl21_l = (2 * sigl - 1) / l
    d_sigl21_l = 2 * sigl * (1 - sigl) - sigl21_l
    d_sigl21_l_dx = x * d_sigl21_l
    d_sigl21_l_dy = y * d_sigl21_l

    # Return (dX_dx, dX_dy), (dY_dx, dY_dy)
    return (sigl21_l + d_sigl21_l_dx * x, d_sigl21_l_dy * x), (d_sigl21_l_dx * y, sigl21_l + d_sigl21_l_dy * y)

def value(n, vR, vI, T, t) :
    r, i = 0, 0
    for j in range(n) :
        r += vR[j] * legendre(j)(2. * t / T - 1)
        i += vI[j] * legendre(j)(2. * t / T - 1)
    return normC(r, i)

def dvalue(n, vR, vI, T, t, dL_dR, dL_dI) :
    r, i = value(n, vR, vI, T, t)
    (dR_dr, dR_di), (dI_dr, dI_di) = dnormC(r, i)
    derivR = []
    derivI = []
    for j in range(n) :
        derivR.append((dL_dR * dR_dr + dL_dI * dI_dr) * legendre(j)(2. * t / T - 1))
        derivI.append((dL_dR * dR_di + dL_dI * dR_di) * legendre(j)(2. * t / T - 1))
    return np.array(derivR), np.array(derivI)

def mapC(x, y) :
    return x + 1j * y

def calcM(counts, shots) :
    # M = |1><1|
    if '1' not in counts.keys() :
        return 0
    else :
        return counts['1'] * 1. / shots

def experiment(n_qb, vRl, vIl, T, shots, backend, s = 0, k = -1, theta = 0, phi = 0, lam = 0) :
    with pulse.build(backend) as pulse_prog :
        for i in range(n_qb) :
            channel = pulse.drive_channel(i)
            seq = [mapC(*value(n_basis, vRl[i], vIl[i], T, t)) for t in range(0, s)]
            if seq != [] :
                pulse.play(seq, channel)
            if k != -1 :
                pulse.u3(theta, phi, lam, k) # apply u3(theta, phi, lam) on qubit k
            seq = [mapC(*value(n_basis, vRl[i], vIl[i], T, t)) for t in range(s, T)]
            if seq != [] :
                pulse.play(seq, channel)
            pulse.barrier(i)
            reg = pulse.measure(i)
    job = execute(pulse_prog, sim_backend, shots = shots)
    res = job.result()

    global sch
    sch = pulse_prog

    return calcM(res.get_counts(), shots)

n_epoch = 5000
n_qb = 1
n_basis = 7
lr = 0.1
shots = 1024
vRl = np.random.normal(0, 1e-1, [n_qb, n_basis])
vIl = np.random.normal(0, 1e-1, [n_qb, n_basis]) 
T = 100
qr = QuantumRegister(n_qb)
loss = []
# Gradient Ascent
for epoch in range(n_epoch) :
    print("Epoch " + str(epoch) + " is running...")
    loss.append(experiment(n_qb, vRl, vIl, T, shots, backend))
    print("Loss evaluated: " + str(loss[-1]))

    s = np.random.randint(T)

    # Estimate p- for Xk
    derivRl, derivIl = np.zeros([n_qb, n_basis], dtype = 'float'), np.zeros([n_qb, n_basis], dtype = 'float')
    for k in range(n_qb) :
        pm = experiment(n_qb, vRl, vIl, T, shots, backend, s, k, pi / 2, -pi / 2, pi / 2)  # G- = RX(pi/2)
        pp = experiment(n_qb, vRl, vIl, T, shots, backend, s, k, -pi / 2, -pi / 2, pi / 2) # G+ = RX(-pi/2)
        dL_dR = 1. / sqrt(2) * (pm - pp)
        
        pm = experiment(n_qb, vRl, vIl, T, shots, backend, s, k, pi / 2, 0, 0)  # G- = RY(pi/2)
        pp = experiment(n_qb, vRl, vIl, T, shots, backend, s, k, -pi / 2, 0, 0) # G+ = RY(-pi/2)
        dL_dI = 1. / sqrt(2) * (pm - pp)

        derivRl[k], derivIl[k] = dvalue(n_basis, vRl[k], vIl[k], T, s, dL_dR, dL_dI)

    log_file = open('log', 'a')
    print('Epoch ' + str(epoch) + ' loss: ' + str(loss[-1]) + ' dsum: ' + str(sum(sum(abs(derivRl))) + sum(sum(abs(derivIl)))) + ' vRl: ' + str(vRl.tolist()) + ' vIl: ' + str(vIl.tolist()), file = log_file)
    log_file.close()

    vRl += lr * derivRl
    vIl += lr * derivIl

