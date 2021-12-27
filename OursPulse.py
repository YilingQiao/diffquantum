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

import warnings
warnings.filterwarnings('ignore')


def normC(x, y) :
    l = np.sqrt(x ** 2 + y ** 2)
    coef = (2 * expit(l) - 1) / l
    return coef * x, coef * y

def dnormC(x, y) :
    l = np.sqrt(x ** 2 + y ** 2)
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

def enlarge16(seq) :
    if len(seq) % 16 != 0 :
        seq += [0j for i in range(16 - len(seq) % 16)]
    return seq

class OursPulse(object):
    """A class for using Legendre series to represent the amplitudes.
    The derivatives are computed by our method.
    Args:
        n_basis: number of Fourier basis.
    """
    def __init__(self, n_basis=5, basis='Legendre', n_epoch=200, n_step=100, 
                 lr=2e-2, is_noisy=False, T=128, n_shots=8192, n_qubit=1, pulse_simulation=True, init_coeff = None):

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
        self.n_qubit = n_qubit
        self.pulse_simulation = pulse_simulation
        self.init_coeff = init_coeff
        self.exps = []
        # if basis == 'Legendre':
        #     self.legendre_ps = [legendre(j) for j in range(self.n_basis)]
        self.start_engine()

    def start_engine(self, hub='ibm-q-community', group='ibmquantumawards', project='open-science-22'):
        
        if IBMQ.active_account() is None:
            IBMQ.load_account()

        print("provider: ", hub, group, project)
        self.provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        self.backend = self.provider.get_backend('ibmq_jakarta')
        #sim_noisy_jakarta = QasmSimulator.from_backend(backend)
        if self.pulse_simulation :
            self.sim_backend = PulseSimulator()
            self.backend_model = PulseSystemModel.from_backend(self.backend)
            self.sim_backend.set_options(system_model=self.backend_model)

    def clear_exps(self) :
        self.exps = []

    def add_experiment(self, vRI, s=0, qbt=-1, theta=0, phi=0, lam=0):
        n_qubit, n_basis = vRI.shape[0], vRI.shape[1]  

        with pulse.build(self.backend) as pulse_prog :
            for i in range(n_qubit) :
                channel = pulse.drive_channel(i)
                seq = enlarge16([mapC(*value(n_basis, vRI[i,:,0], vRI[i,:,1], self.T, t)) for t in range(0, s)])
                if seq != [] :
                    pulse.play(seq, channel)
                if qbt != -1 :
                    pulse.u3(theta, phi, lam, qbt) # apply u3(theta, phi, lam) on qubit qbt
                seq = enlarge16([mapC(*value(n_basis, vRI[i,:,0], vRI[i,:,1], self.T, t)) for t in range(s, self.T)])
                if seq != [] :
                    pulse.play(seq, channel)
                pulse.barrier(i)
                pulse.measure(i)
        self.exps.append(pulse_prog)

    def run_experiments(self) :
        self.counts_list = []
        if self.pulse_simulation :
            for i in range(len(self.exps)) :
                job = execute(self.exps[i], self.sim_backend, shots=self.n_shots)
                res = job.result()
                self.counts_list.append(res.get_counts())
        else :
            job = execute(self.exps, self.backend, shots=self.n_shots)
            res = job.result()
            for i in range(len(self.exps)) :
                self.counts_list.append(res.get_counts(i))
        return self.counts_list

    def calc_loss(self, counts) :
        # M = |0><0|
        # loss = <psi| |0><0| |psi>
        if '0' not in counts.keys() :
            return 0
        else :
            return counts['0'] * 1. / self.n_shots

    def grad_energy_MC(self):
        grad_vRI = np.zeros(self.spectral_coeff.shape)
        vRI = self.spectral_coeff.detach().numpy()

        self.clear_exps()
        s = np.random.randint(self.T)
        
        self.add_experiment(vRI)
        
        for qbt in range(self.n_qubit) :
            self.add_experiment(vRI, s=s, qbt=qbt, theta=np.pi / 2, phi=-np.pi / 2, lam=np.pi / 2)
            self.add_experiment(vRI, s=s, qbt=qbt, theta=-np.pi / 2, phi=-np.pi / 2, lam=np.pi / 2)

            self.add_experiment(vRI, s=s, qbt=qbt, theta=np.pi / 2, phi=0, lam=0)
            self.add_experiment(vRI, s=s, qbt=qbt, theta=-np.pi / 2, phi=0, lam=0)

        counts_list = self.run_experiments()

        loss = self.calc_loss(counts_list[0])

        for qbt in range(self.n_qubit) :
            pm = self.calc_loss(counts_list[4 * qbt + 1])
            pp = self.calc_loss(counts_list[4 * qbt + 2])
            dL_dR = 1. / np.sqrt(2) * (pm - pp)

            pm = self.calc_loss(counts_list[4 * qbt + 3])
            pp = self.calc_loss(counts_list[4 * qbt + 4])
            dL_dI = 1. / np.sqrt(2) * (pm - pp)

            grad_vRI[qbt,:,0], grad_vRI[qbt,:,1] = dvalue(
                self.n_basis, vRI[qbt,:,0], vRI[qbt,:,1], self.T, s, dL_dR, dL_dI)
        return loss, grad_vRI

    def order_1_norm(self, vRI, T):
        r, i = 0, 0
        for qbt in range(self.n_qubit):
            for t in range(T - 1):
                rt0, it0 = 0, 0
                rt1, it1 = 0, 0
                for j in range(self.n_basis):
                    rt0 += vRI[qbt, j, 0] * legendre(j)(2. * t / T - 1)
                    it0 += vRI[qbt, j, 1] * legendre(j)(2. * t / T - 1)
                    rt1 += vRI[qbt, j, 0] * legendre(j)(2. * (t + 1) / T - 1)
                    it1 += vRI[qbt, j, 1] * legendre(j)(2. * (t + 1) / T - 1)
                r += (rt1 - rt0)**2
                i += (it1 - it0)**2
        reg = 0
        reg += torch.sqrt(r / self.n_qubit / (T - 1))
        reg += torch.sqrt(i / self.n_qubit / (T - 1))        

        return reg


    def train_energy(self):
        if self.init_coeff == None :
            coeff = np.random.normal(0, 1e-3, [self.n_qubit, self.n_basis, 2])
        else :
            coeff = self.init_coeff
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)
        losses = []

        optimizer = torch.optim.Adam([self.spectral_coeff], lr=self.lr)

        for epoch in range(self.n_epoch):
            vRI = self.spectral_coeff.detach().numpy()
            loss_reg = 1e-2 * self.order_1_norm(self.spectral_coeff, self.T)

            optimizer.zero_grad()
            loss_reg.backward()
            loss, grad_vRI = self.grad_energy_MC()
            self.spectral_coeff.grad += torch.from_numpy(grad_vRI)
            optimizer.step()

            print("epoch: {:04d}, loss: {:.4f}".format(
                epoch, 
                loss, 
            ))
            print("vRI: ", self.spectral_coeff)


    def demo_X(self):
        self.train_energy()

if __name__ == '__main__':
    ours_pulse = OursPulse(basis='Legendre', n_basis=7, T=96)
    ours_pulse.demo_X()
    

