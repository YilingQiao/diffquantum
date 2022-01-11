import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy 
from scipy.special import legendre
from scipy.stats import unitary_group

class QubitControl(object):
    """A class for simulating single-qubit control on quantum hardware.
    The finest time-resolution dt will be rescaled to 1 in the numerical simulation.
    Parametrization for pulses: Legendre or NN.
    
    The derivatives are computed by our method.
    Args:
        n_basis: number of Fourier basis.
    """
    def __init__(self, dt=0.22, eps=32.901, Amax=0.955, freq=2*np.pi*5.236, duration=96,
                 n_basis=5, basis='Legendre', n_epoch=200, lr=1e-2, 
                 is_sample_discrete=False, is_noisy=False):
        self.dt = dt
        self.raw_eps = eps
        self.raw_Amax = Amax
        self.raw_freq = freq
        self.duration = duration
        
        self.eps = eps * dt
        self.Amax = Amax * dt
        self.w = freq * dt
        
        self.n_basis = n_basis
        self.log_dir = "./logs/"
        self.log_name = basis
        self.basis = basis
        self.n_epoch = n_epoch
        self.lr = lr
        self.is_sample_discrete = is_sample_discrete
        self.is_noisy = is_noisy

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

    def full_pulse(self, vv, i=None):
        """Generate the full driving pulse for H1 = X
        Args:
            vv: parameters for the ansatz
        Returns:
            _D: driving pulse D(t)
        """
        
        # def _D(t, args):
        #     if i is None:
        #         coeff_i = vv
        #     else:
        #         coeff_i = vv[i]
        #     # t ranges from [0, duration]
        #     u = 0
        #     for j in range(self.n_basis):
        #         u += coeff_i[j] * legendre(j)(2 * t / self.duration - 1)
            
        #     return np.cos(self.w * t) * (2 * scipy.special.expit(u) - 1)



        def _D(t, args):
            # t ranges from [0, duration]
            A = 0
            B = 0
            if i is None:
                coeff_i = vv
            else:
                coeff_i = vv[:, i,:]
            for j in range(self.n_basis):
                A += coeff_i[0, j] * legendre(j)(2 * t / self.duration - 1)
                B += coeff_i[1, j] * legendre(j)(2 * t / self.duration - 1)
            
            N = np.sqrt(A**2 + B**2)

            return self.Amax * (2 * scipy.special.expit(N) - 1)/N * (np.cos(self.w * t) * A + np.sin(self.w * t) * B)


        return _D
    
    
    def get_integrand(self, _H, M, initial_state, s):
        
        integrand = np.zeros(self.vv.shape)
        
        # compute dDdv
        #param = torch.clone(self.vv)
        #param.requires_grad = True
        sgm = torch.nn.Sigmoid()
        for i in range(1, len(_H)):

            legendre_A = [self.vv[0,i-1,j] * legendre(j)(2 * s / self.duration - 1) for j in range(self.n_basis)]
            legendre_B = [self.vv[1,i-1,j] * legendre(j)(2 * s / self.duration - 1) for j in range(self.n_basis)]
            A = sum(legendre_A)
            B = sum(legendre_B)
            N = torch.sqrt(torch.square(A)+torch.square(B))
            Ds = self.Amax * (2 * sgm(N) - 1)/N *(np.cos(self.w * s) * A + np.sin(self.w * s) * B)
            Ds.backward()

        dDdv = self.vv.grad.detach().numpy()

        H = [_H[0]]
        for i in range(1, len(_H)):
            H.append([_H[i], self.full_pulse(self.vv.detach().numpy(), i=i-1)])

        for i in range(1, len(_H)):
            t0s = np.linspace(0, s, 10)
            # print(self.duration)
            result = qp.mesolve(H, initial_state, t0s)
            # print("H", H)
            # print(initial_state)
            # print("------------------------------")
            phi = result.states[-1]
            
            r = 1 / 2
            d = initial_state.shape[0]
            gate_p = (qp.qeye(d) + r * 1.j * _H[i]) / np.sqrt(1. + r**2)
            gate_m = (qp.qeye(d) - r * 1.j * _H[i]) / np.sqrt(1. + r**2)
                
            ts1 = np.linspace(s, self.duration, 10)
            result = qp.mesolve(H, gate_p * phi, ts1)
            # print("_H[i]", _H[i])
            # print("phi", phi)
            # print("gate_p * phi", gate_p * phi)
            # print("gate_p ", gate_p )
            # print("======================")
            ket_p = result.states[-1]
            ps_p = M.matrix_element(ket_p, ket_p)
            if self.is_noisy:
                ps_p += np.random.normal(scale=np.abs(ps_p.real) / 5)
                
            result = qp.mesolve(H, gate_m * phi, ts1)
            # print("gate_m * phi", gate_m * phi)
            # print("gate_m ", gate_m )
            ket_m = result.states[-1]
            ps_m = M.matrix_element(ket_m, ket_m)

            # print("ket_p, ket_m", ket_p, ket_m)
            # exit()
            if self.is_noisy:
                ps_m += np.random.normal(scale=np.abs(ps_m.real) / 5)

            # print("--", ps_m, ps_p)
            ps = ((1 + r**2) / 2 / r * (ps_m - ps_p)).real
            dDdv[:,i-1,:] = ps * dDdv[:,i-1,:]
        
        return dDdv
        


    def compute_energy_grad_MC(self, _H, M, initial_state, num_sample):
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
        if self.is_sample_discrete == False:
            sample_time = np.random.uniform(0, self.duration, size=num_sample)
        else: 
            sample_time = 1 + np.random.randint(0, self.duration-1, size=num_sample)

        grad = np.zeros(self.vv.shape)
        for s in sample_time: 
            grad += self.get_integrand(_H, M, initial_state, s)
        
        return torch.from_numpy(self.duration * grad / num_sample)
    
    def compute_energy(self, _H, M, initial_state):    
        H = [_H[0]]
        for i in range(1, len(_H)):
            H.append([_H[i], self.full_pulse(self.vv.detach().numpy(), i=i-1)])
        tarray = np.linspace(0, self.duration, 10)
        result = qp.mesolve(H, initial_state, tarray)
        psi_T = result.states[-1]  
        return np.real(M.matrix_element(psi_T, psi_T))
    
    

    def IBM_H(self, n_qubit):
        self.ibm_params = {'delta0': -2135551738.17207,
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
          'omegad1': 987150040.8532522,
          'omegad2': 985715793.6078007,
          'omegad3': 978645256.0180163,
          'omegad4': 985963354.5772513,
          'omegad5': 1000100925.8075224,
          'omegad6': 976913592.7775077,
          'wq0': 32901013497.991684,
          'wq1': 31504959831.439907,
          'wq2': 32092824583.27148,
          'wq3': 32536784568.16119,
          'wq4': 32756626771.431747,
          'wq5': 31813391726.380398,
          'wq6': 33300775594.753788}


          # raw_freq = 2 *  pi *
          # h.get_qubit_lo_from_drift()
          #   [5236376147.050786,
          #    5014084426.228487,
          #    5107774458.035009,
          #    5178450242.236394,
          #    5213406970.098254,
          #    5063177674.197013,
          #    5300001532.8865185]


        H0 = self.multi_kron(*[self.O for j in range(n_qubit)])
        I = self.multi_kron(*[self.I for j in range(n_qubit)])

        for i in range(n_qubit):
            zi = self.multi_kron(*[self.I if j not in [i] else self.Z for j in range(n_qubit)])
            H0 += 1 / 2. * (I - zi) * self.eps

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

        if n_qubit > 1:
            Js = [[j, (j+1)%n_qubit] for j in range(n_qubit)]
            # Js = [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]]
            for j in Js:
                H0 += (np.matmul(sps[j[0]], sms[j[1]]) + np.matmul(sms[j[0]], sps[j[1]])) * \
                (self.ibm_params['jq0q1'] * 1e-9 * self.dt)

        Hs = []
        for i in range(n_qubit):
            Hs.append(self.multi_kron(*[self.I if j not in [i] else self.X for j in range(n_qubit)]))
            # H = self.multi_kron(*[self.I if j not in [i, (i + 1) % n_qubit] else self.Z for j in range(n_qubit)]) 
            # Hs.append(H)

        return H0, Hs




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

        
        
    def train_energy(self, M, initial_state, vv0, num_sample):
        """Train the sepctral coefficients to minimize energy minimization.
        Args:
            M: A Hermitian matrix.
            initial_state: initial_state.
            vv0: initial guess of the parameters
        Returns:
            vv_final: optimized parameters
        """
        if len(vv0) != self.n_basis:
            print("The length of initial guess must be the same as n_basis!")
        
        self.vv = torch.tensor(vv0, requires_grad=True)
        
        w_l2 = 0
        lr = self.lr
        optimizer = torch.optim.Adam([self.vv], lr=lr)

        self.losses_energy = []
        for epoch in range(1, self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)
            
            loss_energy = self.compute_energy(M, initial_state)
            if self.is_noisy:
                loss_energy += np.random.normal(scale=np.abs(loss_energy) / 5)
            loss_l2 = ((self.vv**2).mean(0) * torch.tensor(
                [i**2 for i in range(self.n_basis)])).mean() * w_l2
            loss = loss_energy + loss_l2
            optimizer.zero_grad()
            loss_l2.backward()
            grad_vv = self.compute_energy_grad_MC(M, initial_state, num_sample)
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
            
    def train_fidelity(self, vv0, H, initial_states, target_states, num_sample):
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
        H = [qp.Qobj(h) for h in H]

        for epoch in range(1, self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)

            batch_losses = []
            for i in range(len(initial_states)):
                psi0 = initials[i]
                psi1 = targets[i]
                M = I - psi1 * psi1.dag() 
                loss_fidelity = self.compute_energy(H, M, psi0)
                # result = qp.mesolve(H, psi0, ts)
                # final_state = result.states[-1]

                # inner_product_norm = M.matrix_element(final_state, final_state)
                # if self.is_noisy:
                #     inner_product_norm += np.random.normal(
                #         scale=np.abs(inner_product_norm.real) / 5)
                # loss_fidelity = 1 - inner_product_norm
                # loss_l2 = ((self.spectral_coeff**2).mean(0) * torch.tensor(
                #     [i**2 for i in range(self.n_basis)])).mean() * w_l2
                # loss = loss_fidelity + loss_l2
                optimizer.zero_grad()
                # loss_l2.backward()
                grad_vv = self.compute_energy_grad_MC(H, M, psi0, num_sample) 

                self.vv.grad = grad_vv
                optimizer.step()

                batch_losses.append(loss_fidelity.real)

            batch_losses = np.array(batch_losses).mean()
            print("epoch: {:04d}, loss: {:.4f}, loss_fidelity: {:.4f}".format(
                epoch, 
                batch_losses, 
                batch_losses
            ))
            self.losses_energy.append(batch_losses) 
            
        return self.vv

    def demo_CNOT(self, num_sample, method):
        n_qubit = 2
        self.n_qubit = n_qubit

        H0, Hs = self.IBM_H(n_qubit)
        H = [H0] + Hs
        self.n_Hs = len(Hs) 
        vv0 =  np.random.rand(2 * self.n_basis * self.n_Hs)
        vv0 = np.reshape(vv0, [2, self.n_Hs ,self.n_basis])

        g = np.array([1,0])
        e = np.array([0,1])
        # pres = ['00', '01', '10', '++']
        # posts = ['00', '01', '11', '++']
        initial_states = [np.kron(g, g), np.kron(g, e), np.kron(e, g)]
        target_states = [np.kron(g, g), np.kron(g, e), np.kron(e, e)]

        self.train_fidelity(vv0, H, initial_states, target_states, num_sample)
        
    def demo_X(self, num_sample, method):
        n_qubit = 1
        self.n_qubit = n_qubit
        
        H0, Hs = self.IBM_H(n_qubit)
        H = [H0] + Hs

        print(H)

        self.n_Hs = len(Hs) 
        vv0 =  np.random.rand(2 * self.n_basis * self.n_Hs)
        vv0 = np.reshape(vv0, [2, self.n_Hs ,self.n_basis])
        # vv0 = np.random.normal(0, 1e-3, [self.n_Hs ,self.n_basis]) 

        g = np.array([1,0])
        e = np.array([0,1])
        # pres = ['00', '01', '10', '++']
        # posts = ['00', '01', '11', '++']
        initial_states = [g, 1/np.sqrt(2)*e + 1/np.sqrt(2)*g]
        target_states = [e, 1/np.sqrt(2)*e + 1/np.sqrt(2)*g]

        self.train_fidelity(vv0, H, initial_states, target_states, num_sample)
        
    
    def model_qubit(self, vv0, num_sample, method):
        
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
        
        if method == 'plain':
            self.train_energy(M, psi0, vv0, num_sample)
        elif method == 'rwa':
            self.train_energy(M, psi0, vv0, num_sample)
        else:
            print("Method must be plain or rwa.")

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

    # np.random.seed(0)
    model = QubitControl(basis='Legendre', n_basis=5 , dt=0.22, duration=96, n_epoch=50, lr = 1e-2)
    num_sample = 5
    # vv0 = np.random.rand(model.n_basis)
    # num_sample = 1
    # g = model.model_qubit(vv0, num_sample, 'plain')
    # loss0 = model.losses_energy
    # model.demo_CNOT(num_sample, 'plain')
    # model.demo_X(num_sample, 'plain')
    model.demo_CNOT(num_sample, 'plain')