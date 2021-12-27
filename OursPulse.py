import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy 
from scipy.stats import unitary_group
from scipy.special import legendre, expit

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute, transpile, schedule, assemble, pulse
from qiskit.providers.aer import QasmSimulator, PulseSimulator
from qiskit.providers.aer.pulse import PulseSystemModel
from qiskit.tools.monitor import job_monitor



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

class OursPulse(object):
    """A class for using Fourier series to represent the amplitudes.
    The derivatives are computed by our method.
    Args:
        n_basis: number of Fourier basis.
    """
    def __init__(self, n_basis=5, basis='Fourier', n_epoch=200, n_step=100, 
        lr=2e-2, is_noisy=False, T=100, n_shots=8192):

        self.n_basis = n_basis
        self.log_dir = "./logs/"
        self.log_name = basis
        self.basis = basis
        self.n_epoch = n_epoch
        self.n_step = n_step
        self.lr = lr
        self.is_noisy = is_noisy
        self.T = T
        self.n_shots = n_shots
        if basis == 'Legendre':
            self.legendre_ps = [legendre(j) for j in range(self.n_basis)]

    def start_engine(self, hub='ibm-q-community', group='ibmquantumawards', project='open-science-22'):
        print("provider: ", hub, group, project)
        self.provider = IBMQ.get_provider(hub=hub, group=hub, project=project)
        self.backend = provider.get_backend('ibmq_jakarta')
        #sim_noisy_jakarta = QasmSimulator.from_backend(backend)
        self.sim_backend = PulseSimulator()
        self.backend_model = PulseSystemModel.from_backend(self.backend)
        self.sim_backend.set_options(system_model=self.backend_model)

    @staticmethod
    def experiemnt(vRI, T, n_shots, backend, sim_backend, 
        s=0, qbt=-1, theta=0, phi=0, lam=0):
        n_qubit = vRl.shape[0] 
        with pulse.build(backend) as pulse_prog :
            for i in range(n_qubit) :
                channel = pulse.drive_channel(i)
                seq = [mapC(*value(n_basis, vRI[i,:,0], vRI[i,:,1], T, t)) for t in range(0, s)]
                if seq != [] :
                    pulse.play(seq, channel)
                if k != -1 :
                    pulse.u3(theta, phi, lam, k) # apply u3(theta, phi, lam) on qubit k
                seq = [mapC(*value(n_basis, vRI[i,:,0], vRI[i,:,1], T, t)) for t in range(s, T)]
                if seq != [] :
                    pulse.play(seq, channel)
                pulse.barrier(i)
                reg = pulse.measure(i)
        job = execute(pulse_prog, sim_backend, shots = shots)
        res = job.result()

        # global sch
        # sch = pulse_prog
        return calcM(res.get_counts(), shots)


    def grad_energy_MC(self):
        grad_vRI = np.zeros(self.spectral_coeff.shape)
        vRI = self.spectral_coeff.detach().numpy()

        s = np.random.randint(T)

        for qbt in range(self.n_qubit) :
            pm = self.experiemnt(vRI, self.T, self.n_shots, self.backend, self.sim_backend, 
                s=s, qbt=qbt, theta=np.pi / 2, phi=-np.pi / 2, lam=np.pi / 2)
            pp = self.experiemnt(vRI, self.T, self.n_shots, self.backend, self.sim_backend, 
                s=s, qbt=qbt, theta=-np.pi / 2, phi=-np.pi / 2, lam=np.pi / 2)
            dL_dR = 1. / sqrt(2) * (pm - pp)


            pm = self.experiemnt(vRI, self.T, self.n_shots, self.backend, self.sim_backend, 
                s=s, qbt=qbt, theta=np.pi / 2, phi=0, lam=0)
            pp = self.experiemnt(vRI, self.T, self.n_shots, self.backend, self.sim_backend, 
                s=s, qbt=qbt, theta=-np.pi / 2, phi=0, lam=0)
            dL_dR = 1. / sqrt(2) * (pm - pp)
            dL_dI = 1. / sqrt(2) * (pm - pp)

            grad_vRI[qbt,:,0], grad_vRI[qbt,:,1] = dvalue(
                self.n_basis, vRI[qbt,:,0], vRI[qbt,:,1], T, s, dL_dR, dL_dI)
        return grad_vRI


    def train_energy():

        self.n_Hs = len(Hs)
        coeff = np.random.normal(0, 1e-3, [self.n_qubit ,self.n_basis, 2]) 
        # coeff = np.ones([self.n_qubit ,self.n_basis, 2])
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)
        qr = QuantumRegister(self.n_qubit)
        losses = []
        for epoch in range(self.n_epoch):
            vRI = self.spectral_coeff.detach().numpy()
            loss = self.experiemnt(vRI, self.T, self.n_shots, self.backend, self.sim_backend)

            optimizer.zero_grad()
            grad_vRI = self.grad_energy_MC()
            self.spectral_coeff.grad += grad_coeff
            optimizer.step()


            print("epoch: {:04d}, loss: {:.4f}, loss_energy: {:.4f}".format(
                epoch, 
                loss, 
                loss
            ))


    def demo_X():


        pass
if __name__ == '__main__':
    ours_pulse = OursPulse(basis='Legendre', n_basis=6, n_epoch=200)
    ours_pulse.start_engine()
    ours_pulse.demo_X()
    

