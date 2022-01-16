import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy 
import time
from scipy.special import legendre
from scipy.stats import unitary_group
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix

import diffqc as dq 




class QubitControl(object):
    """A class for simulating single-qubit control on quantum hardware.
    The finest time-resolution dt will be rescaled to 1 in the numerical simulation.
    Parametrization for pulses: Legendre or NN.
    
    The derivatives are computed by our method.
    Args:
        n_basis: number of Fourier basis.
    """
    def __init__(self, dt=0.22, duration=96,
                 n_basis=5, basis='Legendre', n_epoch=200, lr=1e-2, 
                 is_sample_discrete=False, is_noisy=False, num_sample=1, is_sample_uniform=False,
                 per_step=10):
        self.dt = dt
        self.duration = duration
        
        
        self.n_basis = n_basis
        self.log_dir = "./logs/"
        self.log_name = basis
        self.basis = basis
        self.n_epoch = n_epoch
        self.lr = lr
        self.is_sample_discrete = is_sample_discrete
        self.is_noisy = is_noisy
        self.num_sample = num_sample
        self.is_sample_uniform = is_sample_uniform
        self.per_step = per_step

        if basis == 'Legendre':
            self.legendre_ps = [legendre(j) for j in range(self.n_basis)]
        
        self.I = np.array([[1.+ 0.j, 0], 
                    [0, 1.]])
        self.O = np.array([[0.+ 0.j, 0], 
                    [0, 0.]])
        self.X = np.array([[0 + 0.j, 1], 
                    [1, 0]])
        self.Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        self.Z = np.array([[1.0 + 0.j, 0], 
                    [0, -1.0]])

        solvers = [self.trotter_cpp, self.leapfrog, self.trotter]
        self.my_solver = solvers[0]
        # self.my_solver = trotter 

    @staticmethod
    def multi_kron(*args):
        ret = np.array([[1.0]])
        for q in args:
            ret = np.kron(ret, q)
        return ret

    @staticmethod
    def multi_dot(*args):
        for i, q in enumerate(args):
            if i == 0:
                ret = q
            else:
                ret = np.dot(ret, q)
        return ret

    @staticmethod
    def inst(curr, gate, n_qubit, idx=None):
        I = np.eye(2)
        if idx is not None:
            curr = OurSpectral.multi_dot(OurSpectral.multi_kron(
                *[gate if j == idx else I for j in range(n_qubit)]), curr)
        else:
            curr = OurSpectral.multi_dot(gate, curr)
        return curr

    @staticmethod
    def encoding_x(x, n_qubit):
        # g = np.array([1., 1.]) / np.sqrt(2.) 
        # g = np.array([1., 0.]) 
        I = np.eye(2)
        zero = np.array([[1.0],
                 [0.0]])
        RX = lambda theta: np.array([[np.cos(theta/2.0),-1j*np.sin(theta/2.0)],
                             [-1j*np.sin(theta/2.0),np.cos(theta/2.0)]])
        RY = lambda theta: np.array([[np.cos(theta/2.0),-np.sin(theta/2.0)],
                                     [np.sin(theta/2.0),np.cos(theta/2.0)]])
        RZ = lambda theta: np.array([[np.exp(-1j*theta/2.0),0],
                                     [0,np.exp(1j*theta/2.0)]])

        psi0 = OurSpectral.multi_kron(*[zero for j in range(n_qubit)]) 

        curr = OurSpectral.multi_kron(*[I for j in range(n_qubit)])   
        for j in range(n_qubit):
            curr = OurSpectral.inst(curr, RY(np.arcsin(x)), n_qubit, j)
            curr = OurSpectral.inst(curr, RZ(np.arccos(x**2)), n_qubit, j)

 
        psi0 = np.matmul(curr, psi0)
        psi0 = qp.Qobj(psi0)
        return psi0

    def trotter(self, H_, psi0_, T0, T):
        per_step = self.per_step
        psi = psi0_.full()
        
        n_steps = int(per_step * (T - T0))
        start = time.time()
        

        H = []
        for h in H_:
            if isinstance(h, list):
                H.append([csc_matrix(h[0].full()), h[1]])
            else:
                H.append(csc_matrix(h.full()))

        
        dt = (T - T0) / n_steps
        t = T0
        
        for k in range(n_steps):
            for h in H:
                if isinstance(h, list):
                    psi = expm_multiply(-1.j * dt * h[1](t,None) * h[0], psi)
                else:
                    psi = expm_multiply(-1.j * dt * h, psi)
            t += dt
            
        ans = qp.Qobj(psi)
        # print(T, T0, n_steps, time.time() - start)

        return ans

        
    def leapfrog(self, H_, psi0_, T0, T):
        per_step = self.per_step
        psi0 = psi0_.full()
        Re = np.real(psi0)
        Im = np.imag(psi0)
        
        n_steps = int(per_step * (T - T0))
        start = time.time()
        

        H = []
        for h in H_:
            if isinstance(h, list):
                H.append([h[0].full().real, h[1]])
            else:
                H.append(h.full().real)

        
        dt = (T - T0) / n_steps
        t = T0
        

        for k in range(n_steps):
            Im_half = Im
            for h in H:
                if isinstance(h, list):
                    Im_half -= np.matmul(0.5 * dt * h[1](t, None) * h[0], Re)
                else:
                    Im_half -= np.matmul(0.5 * dt * h, Re)
            t += dt/2
            for h in H:
                if isinstance(h, list):
                    Re += np.matmul(dt * h[1](t, None) * h[0], Im_half)
                else:
                    Re += np.matmul(dt * h, Im_half)
            Im = Im_half
            t += dt/2
            for h in H:
                if isinstance(h, list):
                    Im -= np.matmul(0.5 * dt * h[1](t, None) * h[0], Re)
                else:
                    Im -= np.matmul(0.5 * dt * h, Re)
        ans = Re + 1.j * Im
        ans = qp.Qobj(ans)
            
        # print(T, T0, n_steps, time.time() - start)
        return ans


    def trotter_cpp(self, H_, psi0_, T0, T):
        per_step = self.per_step
        psi0 = psi0_.full()
        psi = dq.trotter(psi0, T0, T, per_step, self.vv.detach().numpy())
        psi = np.array(psi).reshape([-1, 1])
        return qp.Qobj(psi)


    def full_pulse(self, vv, channels):
        """Generate the full driving pulse for H1 = X
        Args:
            vv: parameters for the ansatz
        Returns:
            _D: driving pulse D(t)
        """
        

        # j, omega, freq, idx

        def _D(t, args):
            # t ranges from [0, duration]
            ans = 0
            aa = time.time()

            for chan in channels:
                A = 0
                B = 0
                i, omega, w, idx = chan
                coeff_i = vv[:, idx, :]

                for j in range(self.n_basis):
                    A += coeff_i[0, j] * self.legendre_ps[j](2 * t / self.duration - 1)
                    B += coeff_i[1, j] * self.legendre_ps[j](2 * t / self.duration - 1)

                N = np.sqrt(A**2 + B**2)

                if N == 0:
                    ans += 0
                else:
                    ans += omega * (2 * scipy.special.expit(N) - 1)/N * (np.cos(w * t) * A + np.sin(w * t) * B)

            return ans


        return _D
    
    # ( 1 + 3 * n_H * num_sample)  * n_training
    # (1 + 3 * 2 * 2 * 5) *
    def get_integrand(self, H0, Hs, M, initial_state, s):
        
        integrand = np.zeros(self.vv.shape)
        
        # compute dDdv
        #param = torch.clone(self.vv)
        #param.requires_grad = True
        sgm = torch.nn.Sigmoid()

        for i in range(len(Hs.keys())):
            channels = Hs[i]['channels']
            for chan in channels:
                j, omega, w, idx = chan

                legendre_A = [self.vv[0,idx,j] * self.legendre_ps[j](2 * s / self.duration - 1) for j in range(self.n_basis)]
                legendre_B = [self.vv[1,idx,j] * self.legendre_ps[j](2 * s / self.duration - 1) for j in range(self.n_basis)]
                A = sum(legendre_A)
                B = sum(legendre_B)
                N = torch.sqrt(torch.square(A)+torch.square(B))
                Ds = omega * (2 * sgm(N) - 1)/N *(np.cos(w * s) * A + np.sin(w * s) * B)
                Ds.backward()

        dDdv = self.vv.grad.detach().numpy()

        H = [H0]
        for i in range(len(Hs.keys())):
            ham = Hs[i]['H']
            H.append([
                ham, self.full_pulse(self.vv.detach().numpy(), Hs[i]['channels'])])

        for i in range(1, len(H)):
            phi = self.my_solver(H, initial_state, 0, s)
            
            r = 1.
            d = initial_state.shape[0]
            gate_p = (qp.qeye(d) + r * 1.j * H[i][0]) / np.sqrt(1. + r**2)
            gate_m = (qp.qeye(d) - r * 1.j * H[i][0]) / np.sqrt(1. + r**2)
                
            ts1 = np.linspace(s, self.duration, 10)
            ket_p = self.my_solver(H, gate_p * phi, s, self.duration)
            ps_p = M.matrix_element(ket_p, ket_p)
            if self.is_noisy:
                ps_p += np.random.normal(scale=np.abs(ps_p.real) / 5)
                
            ket_m = self.my_solver(H, gate_m * phi, s, self.duration)
            ps_m = M.matrix_element(ket_m, ket_m)

            if self.is_noisy:
                ps_m += np.random.normal(scale=np.abs(ps_m.real) / 5)

            # print("--", ps_m, ps_p)
            ps = ((1 + r**2) / 2 / r * (ps_m - ps_p)).real

            channels = Hs[i-1]['channels']
            for chan in channels:
                j, omega, w, idx = chan
                dDdv[:,idx,:] = ps * dDdv[:,idx,:]
        return dDdv
        
    def compute_energy_grad_MC(self, H0, Hs, M, initial_state):
        # (M, H, initial_state, num_sample):
        """Compute the gradient of engergy function <psi(T)|M|psi(T)>, T = duration
        Args:
            vv0: current parameters
            M: A Hermitian matrix.
            initial_state: initial_state.
            num_sample = # of MC samples in [0,T]
        Returns:
            grad_coeff: (estimated) gradients of the parameters vv.
        """
        num_sample = self.num_sample

        if self.is_sample_uniform:
            sample_time = [
            (ss + 1.0) * self.duration / (num_sample + 1) for ss in range(num_sample)]
        elif self.is_sample_discrete == False:
            sample_time = np.random.uniform(0, self.duration, size=num_sample)
        else: 
            sample_time = 1 + np.random.randint(0, self.duration-1, size=num_sample)

        grad = np.zeros(self.vv.shape)
        for s in sample_time:
            if self.vv.grad != None :
                self.vv.grad.zero_()
            grad += self.get_integrand(H0, Hs, M, initial_state, s)
        
        return torch.from_numpy(self.duration * grad / num_sample)


    def compute_energy_grad_FD(self, H0, Hs, M, initial_state, delta=1e-4):
        coeff = self.vv.detach().numpy()
        grad_finite_diff = np.zeros(coeff.shape)

        # [2, self.n_funcs ,self.n_basis]

        def get_H(curr_coeff):
            H = [H0]
            for i in range(len(Hs.keys())):
                ham = Hs[i]['H']
                H.append([
                    ham, self.full_pulse(curr_coeff, Hs[i]['channels'])])
            return H

        def run_forward_sim(new_coeff):
            H = get_H(new_coeff)
            phi = self.my_solver(H, initial_state, 0, self.duration)
            loss_energy = M.matrix_element(phi, phi)
            if self.is_noisy:
                loss_energy += np.random.normal(scale=np.abs(loss_energy.real) / 5)
            return loss_energy.real


        for i_c in range(2):
            for i_Hs in range(self.n_funcs):
                for i_basis in range(self.n_basis):
                    new_coeff_p = coeff.copy()
                    new_coeff_p[i_c, i_Hs, i_basis] = coeff[i_c, i_Hs, i_basis] + delta
                    E_p = run_forward_sim(new_coeff_p)
                    new_coeff_m = coeff.copy()
                    new_coeff_m[i_c, i_Hs, i_basis] = coeff[i_c, i_Hs, i_basis] - delta
                    E_m = run_forward_sim(new_coeff_m)
                    grad_finite_diff[i_c, i_Hs, i_basis] = (E_p - E_m) / delta / 2.0

        return torch.from_numpy(grad_finite_diff)

    
    def compute_energy(self, H0, Hs, M, initial_state):   
        H = [H0]
        for i in range(len(Hs.keys())):
            ham = Hs[i]['H']
            H.append([
                ham, self.full_pulse(self.vv.detach().numpy(), Hs[i]['channels'])])
                # _H[i], self.full_pulse(self.vv.detach().numpy(), i=i-1)])

        psi_T = self.my_solver(H, initial_state, 0, self.duration)
        return np.real(M.matrix_element(psi_T, psi_T))
    
    

    def IBM_H(self, n_qubit):
        self.ibm_params = {
        'delta0': -2135551738.17207,
          'delta1': -2156392705.7817025,
          'delta2': -2146430374.8958619,
          'delta3': -2143302106.510836,
          'delta4': -2131591899.3490252,
          'delta5': -2144384268.1217923,
          'delta6': -2126003036.7621038,
          'jq0q1': 12286377.631357463,
          'jq1q2': 12580420.010373892,
          'jq1q3': 12895897.946888989,
          'jq3q5': 12535018.234570118,
          'jq4q5': 12857428.747059302,
          'jq5q6': 13142900.487599919,
          'omegad0': 955111374.7779446,
          # 'omegad0': 955000000,
          'omegad1': 987150040.8532522,
          'omegad2': 985715793.6078007,
          'omegad3': 978645256.0180163,
          'omegad4': 985963354.5772513,
          'omegad5': 1000100925.8075224,
          'omegad6': 976913592.7775077,
          'wq0': 32901013497.991684,
          # 'wq0': 32901000000,
          'wq1': 31504959831.439907,
          'wq2': 32092824583.27148,
          'wq3': 32536784568.16119,
          'wq4': 32756626771.431747,
          'wq5': 31813391726.380398,
          'wq6': 33300775594.753788}


        # raw_freq = 2 *  pi *
        self.get_qubit_lo_from_drift = np.array([
            5236376147.050786,
            5014084426.228487,
            5107774458.035009,
            5178450242.236394,
            5213406970.098254,
            5063177674.197013,
            5300001532.8865185])

        self.ws = self.get_qubit_lo_from_drift * 2 * np.pi * 1e-9 * self.dt


        self.hams = {
            0: [0, 1],
            1: [1, 0, 3, 2],
            2: [2, 1],
            3: [3, 1, 5],
            4: [4, 5],
            5: [5, 3, 6, 4],
            6: [6, 5]
        }

        H0 = self.multi_kron(*[self.O for j in range(n_qubit)])
        I = self.multi_kron(*[self.I for j in range(n_qubit)])

        for i in range(n_qubit):
            zi = self.multi_kron(*[self.I if j not in [i] else self.Z for j in range(n_qubit)])
            wq = 'wq{}'.format(i)
            H0 += 1 / 2. * (I - zi) * self.ibm_params[wq] * self.dt * 1e-9

        sm = np.array([[0, 1], 
                    [0, 0]])
        sp = np.array([[0, 0], 
                    [1, 0]])
        sps, sms = [], []

        for i in range(n_qubit):
            p = self.multi_kron(*[self.I if j not in [i] else sp for j in range(n_qubit)])
            m = self.multi_kron(*[self.I if j not in [i] else sm for j in range(n_qubit)])
            sps.append(p)
            sms.append(m)

        Js = [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]]
        for j in Js:
            if j[0] >= n_qubit or j[1] >= n_qubit:
                continue
            jqq =  'jq{}q{}'.format(j[0], j[1])
            H0 += (np.matmul(sps[j[0]], sms[j[1]]) + np.matmul(sms[j[0]], sps[j[1]])) * \
            (self.ibm_params[jqq] * 1e-9 * self.dt)

        self._channels = []
        self._Hs = []
        Hs = {}
        self.n_funcs = 0
        # j, omega, ws, idx
        for i in range(n_qubit):
            H = self.multi_kron(*[self.I if j not in [i] else self.X for j in range(n_qubit)])
            self._Hs.append(H)
            Hs[i] = {}
            Hs[i]['H'] = qp.Qobj(H)
            Hs[i]['channels'] = []
            for j in self.hams[i]:
                if j >= n_qubit:
                    continue
                omega = 'omegad{}'.format(i)
                Hs[i]['channels'].append([j, self.ibm_params[omega] * 1e-9 * self.dt, self.ws[j], self.n_funcs])
                self.n_funcs += 1

            self._channels.append(Hs[i]['channels'])
        print("self.n_funcs", self.n_funcs)

        self._H0 = H0
        dq.set_H(self._H0, self._Hs, self._channels, self.duration)
        
        return qp.Qobj(H0), Hs




    def save_plot(self, plot_name):
        return
        ts = np.linspace(0, 1, self.n_step) 
        fs = [self.generate_u(i, self.spectral_coeff.detach().numpy()) for i in range(self.n_Hs)]
        np_us = np.array([[f(x, None) for f in fs] for x in ts])
        plt.clf()
        for j in range(len(fs)):
            plt.plot(np_us[:, j], label='{} u_{}'.format(self.log_name,  j))

        plt.legend(loc="upper right")
        plt.savefig("{}{}_{}.png".format(self.log_dir, self.log_name, plot_name))

        

    def train_energy(self, vv0, H0, Hs, psi0, M):
        """Train the sepctral coefficients to minimize energy minimization.
        Args:
            M: A Hermitian matrix.
            initial_state: initial_state.
            vv0: initial guess of the parameters
        Returns:
            vv_final: optimized parameters
        """
 
        self.vv = torch.tensor(vv0, requires_grad=True)
        
        w_l2 = 0
        lr = self.lr
        optimizer = torch.optim.Adam([self.vv], lr=lr)
        psi0 = qp.Qobj(psi0)
        M = qp.Qobj(M)

        self.losses_energy = []
        for epoch in range(1, self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)
            
            loss_energy = self.compute_energy(H0, Hs, M, psi0)
            if self.is_noisy:
                loss_energy += np.random.normal(scale=np.abs(loss_energy) / 5)
            loss_l2 = ((self.vv**2).mean(0) * torch.tensor(
                [i**2 for i in range(self.n_basis)])).mean() * w_l2
            loss = loss_energy + loss_l2
            optimizer.zero_grad()
            # loss_l2.backward()
            grad_vv = self.compute_energy_grad_MC(H0, Hs, M, psi0)
            self.vv.grad = grad_vv
            optimizer.step()

            print("epoch: {:04d}, loss: {:.4f}, loss_energy: {:.4f}".format(
                epoch, 
                loss, 
                loss_energy
            ))
            self.losses_energy.append(loss_energy.real)
            #self.final_state = final_state
            
            
        return self.vv
            
    def train_fidelity(self, vv0, H0, Hs, initial_states, target_states):
        """Train the sepctral coefficients to minimize energy minimization.
        Args:
            M: A Hermitian matrix.
            initial_state: initial_state.
            vv0: initial guess of the parameters
        Returns:
            vv_final: optimized parameters
        """
        
        self.vv = torch.tensor(vv0, requires_grad=True)
        w_l2 = 0
        lr = self.lr
        optimizer = torch.optim.Adam([self.vv], lr=lr)

        self.losses_energy = []

        I = self.multi_kron(*[self.I for j in range(self.n_qubit)])
        I = qp.Qobj(I)
        initials = [qp.Qobj(v) for v in initial_states]
        targets = [qp.Qobj(v) for v in target_states]

        for epoch in range(1, self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)

            batch_losses = []
            idxs = np.arange(len(initial_states))
            np.random.shuffle(idxs)
            for i in idxs:
                print("batch id ", i)
                psi0 = initials[i]
                psi1 = targets[i]
                M = I - psi1 * psi1.dag() 
                loss_fidelity = self.compute_energy(H0, Hs, M, psi0)
                optimizer.zero_grad()
                grad_vv = self.compute_energy_grad_MC(H0, Hs, M, psi0) 

                self.vv.grad = grad_vv
                optimizer.step()

                batch_losses.append(loss_fidelity.real)
            print("batch_losses", batch_losses)

            batch_losses = np.array(batch_losses).mean()
            print("epoch: {:04d}, loss: {:.4f}, loss_fidelity: {:.4f}".format(
                epoch, 
                batch_losses, 
                batch_losses
            ))
            print("self.vv", self.vv)
            self.losses_energy.append(batch_losses) 
            
        return self.vv

    def demo_CNOT(self, method):
        n_qubit = 2
        self.n_qubit = n_qubit

        H0, Hs = self.IBM_H(n_qubit)
        self.n_Hs = len(Hs.keys()) 
        vv0 =  np.random.rand(2 * self.n_basis * self.n_funcs)
        vv0 = np.reshape(vv0, [2, self.n_funcs ,self.n_basis])

        g = np.array([1,0])
        e = np.array([0,1])
        # pres = ['00', '01', '10']
        # posts = ['00', '01', '11']
        h_ = 1/np.sqrt(2)*e + 1/np.sqrt(2)*g

        initial_states = [np.kron(g, g), np.kron(g, e), np.kron(e, e), np.kron(e, g), np.kron(h_, h_)]
        target_states = [np.kron(g, g), np.kron(g, e), np.kron(e, g), np.kron(e, e), np.kron(h_, h_)]
        print("initial_states", initial_states)
        print("target_states", target_states)

        self.train_fidelity(vv0, H0, Hs, initial_states, target_states)
        
    def demo_FD(self):
        self.is_sample_uniform = True
        n_qubit = 1
        self.n_qubit = n_qubit
        
        H0, Hs = self.IBM_H(n_qubit)

        self.n_Hs = len(Hs.keys()) 
        vv0 =  np.zeros([2 * self.n_basis * self.n_funcs]) + 0.1
        print(vv0)
        print(self.n_funcs)
        vv0 = np.reshape(vv0, [2, self.n_funcs ,self.n_basis])
        self.vv = torch.tensor(vv0, requires_grad=True)

        g = np.array([1,0])
        e = np.array([0,1])
        initial_states = [1/np.sqrt(2)*e + 1/np.sqrt(2)*g]
        target_states = [1/np.sqrt(2)*e + 1/np.sqrt(2)*g]

        I = self.multi_kron(*[self.I for j in range(self.n_qubit)])
        I = qp.Qobj(I)
        initials = [qp.Qobj(v) for v in initial_states]
        targets = [qp.Qobj(v) for v in target_states]

        psi0 = initials[0]
        psi1 = targets[0]
        M = I - psi1 * psi1.dag() 
        print("start FD")
        fd_grad_vv = self.compute_energy_grad_FD(H0, Hs, M, psi0)
        print("fd_grad_vv", fd_grad_vv)
        ours_grad_vv = self.compute_energy_grad_MC(H0, Hs, M, psi0) 
        print("ours_grad_vv", ours_grad_vv)

        return fd_grad_vv, ours_grad_vv


    def demo_X(self, method):
        n_qubit = 1
        self.n_qubit = n_qubit
        
        H0, Hs = self.IBM_H(n_qubit)
        # H = [H0] + Hs

        # print(H)

        self.n_Hs = len(Hs.keys()) 
        vv0 =  np.random.rand(2 * self.n_basis * self.n_funcs)
        print(self.n_funcs)
        vv0 = np.reshape(vv0, [2, self.n_funcs ,self.n_basis])
        # vv0 = np.random.normal(0, 1e-3, [self.n_Hs ,self.n_basis]) 

        g = np.array([1,0])
        e = np.array([0,1])
        initial_states = [1/np.sqrt(2)*e + 1/np.sqrt(2)*g, g]
        target_states = [1/np.sqrt(2)*e + 1/np.sqrt(2)*g, e]

        self.train_fidelity(vv0, H0, Hs, initial_states, target_states)


    def demo_H2(self):
        n_qubit = 2
        self.n_qubit = n_qubit
        
        H0, Hs = self.IBM_H(n_qubit)

        self.n_Hs = len(Hs.keys()) 
        vv0 =  np.random.rand(2 * self.n_basis * self.n_funcs)
        print(self.n_funcs)
        vv0 = np.reshape(vv0, [2, self.n_funcs ,self.n_basis])

        g = np.array([1,0])
        e = np.array([0,1])
        psi0 = np.kron(g, g)

        M = (-1.052373245772859 * np.kron(self.I, self.I)) + \
            (0.39793742484318045 * np.kron(self.I, self.Z)) + \
            (-0.39793742484318045 * np.kron(self.Z, self.I)) + \
            (-0.01128010425623538 * np.kron(self.Z, self.Z)) + \
            (0.18093119978423156 * np.kron(self.X, self.X))

        self.train_energy(vv0, H0, Hs, psi0, M)
        
    
    # def model_qubit(self, vv0, num_sample, method):
        
    #     I = np.array(
    #         [[1, 0], 
    #         [0, 1]])
    #     X = np.array(
    #         [[0, 1], 
    #         [1, 0]])
    #     Z = np.array(
    #         [[1, 0], 
    #         [0, -1]])
        
    #     M = qp.Qobj(0.5 * (I + Z))
    #     psi0 = qp.basis(2, 0)
        
    #     if method == 'plain':
    #         self.train_energy(M, psi0, vv0, num_sample)
    #     elif method == 'rwa':
    #         self.train_energy(M, psi0, vv0, num_sample)
    #     else:
    #         print("Method must be plain or rwa.")

    def plot_integrand(self, i, vv0):
        if i > self.n_basis:
            print("Index i is no more than n_basis.")
        
        self.vv = torch.tensor(vv0, requires_grad=True)
        
        I = np.array(
            [[1, 0], 
            [0, 1]])
        X = np.array(
            [[0, 1], 
            [1, 0]])
        Z = np.array(
            [[1, 0], 
            [0, -1]])
        
        M = qp.Qobj(0.5 * (I + Z))
        psi0 = qp.basis(2, 0)
        integrand = np.zeros(self.duration-2);
        
        for k in range(self.duration-2):
            s = k+1
            g = self.get_integrand(M, psi0, s)
            integrand[k] = g[i-1]
            display(k)
        
        return integrand
    

if __name__ == '__main__':
    np.random.seed(0)
    model = QubitControl(
        basis='Legendre', n_basis=16, dt=0.22, 
        duration=256, n_epoch=512, lr = 5e-3, num_sample=6, per_step=10)
  
    # vv0 = np.random.rand(model.n_basis)
    # num_sample = 1
    # g = model.model_qubit(vv0, num_sample, 'plain')
    # loss0 = model.losses_energy
    # model.demo_CNOT('plain')
    model.demo_H2()
    # model.demo_FD()
    # model.demo_X('plain')
    # model.demo_CNOT('plain')
