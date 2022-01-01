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
    derivR, derivI = [], []
    for j in range(n) :
        derivR.append((dL_dR * dR_dr + dL_dI * dI_dr) * legendre(j)(2. * t / T - 1))
        derivI.append((dL_dR * dR_di + dL_dI * dR_di) * legendre(j)(2. * t / T - 1))
    return np.array(derivR), np.array(derivI)

def dL_dv(n, vR, vI, T, t, omega_dt, dL_du) :
    return dvalue(n, vR, vI, T, t, dL_du * np.cos(2 * np.pi * omega_dt * t), -dL_du * np.sin(2 * np.pi * omega_dt * t))

def mapC(x, y) :
    return x + 1j * y

def enlarge(seq) :
    if len(seq) % 16 != 0 :
        seq += [0j for i in range(16 - len(seq) % 16)]
    if len(seq) < 64 :
        seq += [0j for i in range(64 - len(seq))]
    return seq

class OursPulse(object):
    """A class for using Legendre series to represent the amplitudes.
    The derivatives are computed by our method.
    Args:
        n_basis: number of basis.
    """
    def __init__(self, n_basis=5, basis='Legendre', n_epoch=40,
                 lr=2e-2, T=128, n_shot=8192, n_qubit=1, 
                 pulse_simulation=True, init_param = None,
                 load_checkpoint=False, save_path="model.pt"):

        self.n_basis = n_basis
        self.log_dir = "./logs/"
        self.log_name = basis
        self.basis = basis
        self.n_epoch = n_epoch
        self.lr = lr
        self.T = T
        self.n_shot = n_shot
        self.n_qubit = n_qubit
        self.pulse_simulation = pulse_simulation
        self.init_param = init_param
        self.exps = []
        self.load_checkpoint = load_checkpoint
        self.save_path = save_path
        # if basis == 'Legendre':
        #     self.legendre_ps = [legendre(j) for j in range(self.n_basis)]
        self.start_engine()

    def start_engine(self, hub='ibm-q-community', group='ibmquantumawards', project='open-science-22'):
        if IBMQ.active_account() is None:
            IBMQ.load_account()

        print("provider: ", hub, group, project)
        self.provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        self.backend = self.provider.get_backend('ibmq_jakarta')
        self.config = self.backend.configuration()
        self.defaults = self.backend.defaults()
               
        if self.pulse_simulation :
            self.sim_backend = PulseSimulator()
            self.backend_model = PulseSystemModel.from_backend(self.backend)
            self.sim_backend.set_options(system_model=self.backend_model)

    def clear_exps(self) :
        self.exps = []

    def add_experiment(self, vRI, phase_offset, s=0, qbt=-1, theta=0, phi=0, lam=0, pre = '00', post = '00'):
        n_qubit, n_basis = vRI.shape[0], vRI.shape[1]  

        with pulse.build(self.backend) as pulse_prog :
            # Pre-processing
            for i in range(n_qubit) :
                if pre[i] == '1' :
                    pulse.x(i)
                if pre[i] == '+' :
                    pulse.u3(np.pi / 2, 0, 0, i)
            pulse.barrier(*range(n_qubit))

            # Apply pulse
            for i in range(n_qubit) :
                channel = pulse.drive_channel(i)
                seq = enlarge([mapC(*value(n_basis, vRI[i,:,0], vRI[i,:,1], self.T, t)) for t in range(0, s)])
                if seq != [] :
                    pulse.play(seq, channel)
                if qbt != -1 :
                    pulse.u3(theta, phi, lam, qbt) # apply u3(theta, phi, lam) on qubit qbt
                seq = enlarge([mapC(*value(n_basis, vRI[i,:,0], vRI[i,:,1], self.T, t)) for t in range(s, self.T)])
                if seq != [] :
                    pulse.play(seq, channel)
                pulse.u3(0, 0, phase_offset[i], qbt)
            pulse.barrier(*range(n_qubit))

            # Post-processing
            for i in range(n_qubit) :
                if post[i] == '+' :
                    pulse.u3(np.pi / 2, 0, 0, i)
            pulse.barrier(*range(n_qubit))

            # Measure
            for i in range(n_qubit) :
                pulse.measure(i)
        self.exps.append(pulse_prog)

    def run_experiments(self) :
        self.counts_list = []
        if self.pulse_simulation :
            for i in range(len(self.exps)) :
                job = execute(self.exps[i], self.sim_backend, shots=self.n_shot)
                res = job.result()
                self.counts_list.append(res.get_counts())
        else :
            job = execute(self.exps, self.backend, shots=self.n_shot)
            res = job.result()
            for i in range(len(self.exps)) :
                self.counts_list.append(res.get_counts(i))
        return self.counts_list

    def calc_loss(self, counts, post = '0') :
        targ = ''
        for i in range(len(post)) :
            if post[i] == '+' :
                targ += '0'
            else :
                targ += post[i]
        failcnt = 0
        for k in counts.keys() :
            if k != targ :
               failcnt += counts[k]
        return failcnt * 1. / self.n_shot

    def grad_energy_MC(self):
        grad_vRI = np.zeros(self.spectral_coeff.shape)
        grad_po = np.zeros(self.n_qubit)
        vRI = self.spectral_coeff.detach().numpy()
        po = self.phase_offset.detach().numpy()

        self.clear_exps()
        n_term = 4
        pres = ['00', '01', '10', '++']
        posts = ['00', '01', '11', '++']

        for l in range(n_term) :
            self.add_experiment(vRI, po, pre=pres[l], post=posts[l])

        n_exp_X = 2
        n_exp_Z = 2
        n_sample = 8
        samples = np.random.randint(0, self.T, [n_sample, n_term, self.n_qubit])
        for sam in range(n_sample) :
            for l in range(n_term) :
                for qbt in range(self.n_qubit) :
                    s = samples[sam][l][qbt]
                    self.add_experiment(vRI, po, s=s, qbt=qbt, theta=np.pi / 2, phi=-np.pi / 2, lam=np.pi / 2, pre=pres[l], post=posts[l])
                    self.add_experiment(vRI, po, s=s, qbt=qbt, theta=-np.pi / 2, phi=-np.pi / 2, lam=np.pi / 2, pre=pres[l], post=posts[l])
                    
                    #s = np.random.randint(self.T)
                    #self.add_experiment(vRI, po, s=s, qbt=qbt, theta=np.pi / 2, phi=0, lam=0, pre=pres[l], post=posts[l])
                    #self.add_experiment(vRI, po, s=s, qbt=qbt, theta=-np.pi / 2, phi=0, lam=0, pre=pres[l], post=posts[l])

        for l in range(n_term) :
            for qbt in range(self.n_qubit) :
                self.add_experiment(vRI, po, s=self.T, qbt=qbt, theta=0, phi=0, lam=np.pi/2, pre=pres[l], post=posts[l])
                self.add_experiment(vRI, po, s=self.T, qbt=qbt, theta=0, phi=0, lam=-np.pi/2, pre=pres[l], post=posts[l])

        counts_list = self.run_experiments()

        loss_list = []
        for l in range(n_term) :
            loss_list.append(self.calc_loss(counts_list[l], post=posts[l]))
        loss = sum(loss_list)

        for sam in range(n_sample) :
            for l in range(n_term) : 
                for qbt in range(self.n_qubit) :
                    index = sam * n_term * self.n_qubit * n_exp_X + l * self.n_qubit * n_exp_X + qbt * n_exp_X + n_term
                    pm = self.calc_loss(counts_list[index + 0], post=posts[l])
                    pp = self.calc_loss(counts_list[index + 1], post=posts[l])
                    dL_du = pm - pp

                    grad_v = dL_dv(self.n_basis, vRI[qbt,:,0], vRI[qbt,:,1], self.T, samples[sam][l][qbt],
                                   self.defaults.qubit_freq_est[qbt] * self.config.dt, dL_du)
                    grad_vRI[qbt,:,0] += grad_v[0]
                    grad_vRI[qbt,:,1] += grad_v[1]
        grad_vRI /= n_sample
        
        for l in range(n_term) :
            for qbt in range(self.n_qubit) :
                index = n_sample * n_term * self.n_qubit * n_exp_X + l * self.n_qubit * n_exp_Z + qbt * n_exp_Z + n_term
                pm = self.calc_loss(counts_list[index + 0], post=posts[l])
                pp = self.calc_loss(counts_list[index + 1], post=posts[l])
                dL_dpo = pm - pp

                grad_po[qbt] += dL_dpo
        return loss, grad_vRI, grad_po, loss_list

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

    def save_ckpt(self, epoch, save_path):
        print('save ckpt to {} at epoch {}'.format(save_path, epoch))
        torch.save({
            'epoch': epoch + 1,
            'spectral_coeff': self.spectral_coeff,
            'phase_offset': self.phase_offset,
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)

    def load_ckpt(self, save_path):
        ckpt = torch.load(save_path)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        print('load ckpt from {} at epoch {}'.format(save_path, epoch))
        return epoch

    def step(self, epoch) :
        vRI = self.spectral_coeff.detach().numpy()
        loss_reg = 0 * self.order_1_norm(self.spectral_coeff, self.T) + 0 * torch.sum(self.phase_offset)

        self.optimizer.zero_grad()
        loss_reg.backward()
        loss, grad_vRI, grad_po, loss_list = self.grad_energy_MC()
        self.spectral_coeff.grad += torch.from_numpy(grad_vRI)
        self.phase_offset.grad += torch.from_numpy(grad_po)
        self.optimizer.step()

        print("epoch: {:04d}, loss: {:.4f}".format(
            epoch, 
            loss, 
        ))
        print("loss list: ", loss_list)
        print("param: ", (self.spectral_coeff, self.phase_offset))

        log_file = open('log', 'a')
        print("epoch: {:04d}, loss: {:.4f}".format(
            epoch, 
            loss, 
        ), file = log_file)
        print("loss list: ", loss_list, file = log_file)
        print("param: ", (self.spectral_coeff, self.phase_offset), file = log_file)
        log_file.close()        
            
        self.save_ckpt(epoch, self.save_path)

    def train_energy(self):
        if self.init_param == None :
            coeff = np.random.normal(0, 1e-1, [self.n_qubit, self.n_basis, 2])
            po = np.random.normal(0, 1e-1, [self.n_qubit])
        else :
            coeff = self.init_param[0]
            po = self.init_param[1]
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)
        self.phase_offset = torch.tensor(po, requires_grad=True)

        self.optimizer = torch.optim.Adam([self.spectral_coeff, self.phase_offset], lr=self.lr)

        init_epoch = 0
        if self.load_checkpoint:
            init_epoch = self.load_ckpt(self.save_path)
            
        for epoch in range(init_epoch, self.n_epoch + init_epoch):
            self.step(epoch)

    def demo_X(self):
        self.train_energy()

if __name__ == '__main__':
    op = OursPulse(basis='Legendre', n_basis=12, n_epoch=15, n_qubit=2, T=496, pulse_simulation = False, load_checkpoint = False)
    '''
    , init_param = (np.array([[[-2.7550e-02,  3.9761e-01], \
         [-3.5656e-02,  4.5210e-02], \
         [-1.9346e-02,  5.9561e-02], \
         [ 7.2024e-02,  7.0554e-03], \
         [ 4.3864e-02, -3.7871e-05]]]), np.array([-0.5586])))
    '''
    op.demo_X()
    

