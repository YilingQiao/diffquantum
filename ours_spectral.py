import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
import torch

class OurSpectral(object):
    """A class for using Fourier series to represent the amplitudes.
    The derivatives are computed by our method.
    Args:
        n_basis: number of Fourier basis.
    """
    def __init__(self, n_basis=5, basis='Fourier', n_epoch=200, n_step=100):
        self.n_basis = n_basis
        self.log_dir = "./logs/"
        self.log_name = basis
        self.basis = basis
        self.n_epoch = n_epoch
        self.n_step = n_step
        if basis == 'Legendre':
            self.legendre_ps = [legendre(j) for j in range(self.n_basis)]

    def generate_u(self, i, spectral_coeff):
        """Generate the function u(i) for H_i
        Args:
            i: index of the H_i.
        Returns:
            _u: function u_i(t).
        """
        def _u(t, args):
            coeff_i = spectral_coeff[i]
            u = 0
            n = int(self.n_basis / 2) if self.basis == 'Fourier' else self.n_basis
            for j in range(n):
                if self.basis == 'poly':
                    u += coeff_i[j] * (t - 0.5)**j
                elif self.basis == 'Legendre':
                    u += coeff_i[j] * self.legendre_ps[j](2 * t - 1)
                elif self.basis == 'Fourier':
                    u += coeff_i[j] * np.cos(2 * np.pi * j * t) \
                        + coeff_i[j + n] * np.sin(2 * np.pi * j * t) 
            return u
        return _u

    def sample_multiple_times(self, M, H, initial_state, n_samples=100, is_MC=True):
        grads_coeffs = []
        for ss in range(n_samples):
            grad_coeff = np.zeros(self.spectral_coeff.shape)
            s = np.random.uniform() if is_MC else (ss + 1) * 1.0 / (n_samples + 1)
            t0s = np.linspace(0, s, self.n_step)
            result = qp.mesolve(H, initial_state, t0s)
            phi = result.states[-1]
            
            ts1 = np.linspace(s, 1, self.n_step)
            r = 1 / 2

            for i in range(self.n_Hs):
                d = initial_state.shape[0]
                gate_p = qp.qeye(d) + r * 1.j * H[i+1][0]
                gate_m = qp.qeye(d) - r * 1.j * H[i+1][0]
                
                result = qp.mesolve(H, gate_p * phi, ts1)
                ket_p = result.states[-1]
                ps_p = M.matrix_element(ket_p, ket_p)

                result = qp.mesolve(H, gate_m * phi, ts1)
                ket_m = result.states[-1]
                ps_m = M.matrix_element(ket_m, ket_m)

                ps = (0.5 / r * (ps_m - ps_p)).real
                
                n = int(self.n_basis / 2) if self.basis == 'Fourier' else self.n_basis 
                for j in range(n):
                    if self.basis == 'poly':
                        grad_coeff[i][j] = (s-0.5)**j * ps
                    elif self.basis == 'Legendre':
                        pj = legendre(j)
                        grad_coeff[i][j] = pj(2 * s - 1) * ps
                    elif self.basis == 'Fourier':
                        grad_coeff[i][j] = ps * np.cos(2 * np.pi * j * s) 
                        grad_coeff[i][j + n] =  ps * np.sin(2 * np.pi * j * s) 

            grads_coeffs.append(grad_coeff)

        return np.array(grads_coeffs)

    def compute_energy_grad_MC(self, M, H, initial_state, coeff=1.0):
        """Compute the gradient of engergy function <psi(1)|M|psi(1)>
        Args:
            M: A Hermitian matrix.
            H: Hamiltonians [H0, [H_i, u_i(t)], ...].
            initial_state: initial_state.
        Returns:
            grad_coeff: gradients of the spectral coefficients.
        """
        # self.sample_multiple_times(M, H, initial_state)

        grad_coeff = np.zeros(self.spectral_coeff.shape)
        s = np.random.uniform()
        t0s = np.linspace(0, s, self.n_step)
        result = qp.mesolve(H, initial_state, t0s)
        phi = result.states[-1]
        
        ts1 = np.linspace(s, 1, self.n_step)
        r = 1 / 2

        for i in range(self.n_Hs):
            d = initial_state.shape[0]
            gate_p = qp.qeye(d) + r * 1.j * H[i+1][0]
            gate_m = qp.qeye(d) - r * 1.j * H[i+1][0]
            
            result = qp.mesolve(H, gate_p * phi, ts1)
            ket_p = result.states[-1]
            ps_p = M.matrix_element(ket_p, ket_p)

            result = qp.mesolve(H, gate_m * phi, ts1)
            ket_m = result.states[-1]
            ps_m = M.matrix_element(ket_m, ket_m)

            ps = coeff * (0.5 / r * (ps_m - ps_p)).real

            n = int(self.n_basis / 2) if self.basis == 'Fourier' else self.n_basis  
            for j in range(n):
                if self.basis == 'poly':
                    grad_coeff[i][j] = (s-0.5)**j * ps
                elif self.basis == 'Legendre':
                    pj = legendre(j)
                    grad_coeff[i][j] = pj(2 * s - 1) * ps
                elif self.basis == 'Fourier':
                    grad_coeff[i][j] = ps * np.cos(2 * np.pi * j * s) 
                    grad_coeff[i][j + n] =  ps * np.sin(2 * np.pi * j * s) 
        return torch.from_numpy(grad_coeff)

    def save_plot(self, plot_name):
        ts = np.linspace(0, 1, self.n_step) 
        fs = [self.generate_u(i, self.spectral_coeff.detach().numpy()) for i in range(self.n_Hs)]
        np_us = np.array([[f(x, None) for f in fs] for x in ts])
        plt.clf()
        for j in range(len(fs)):
            plt.plot(np_us[:, j], label='{} u_{}'.format(self.log_name,  j))

        plt.legend(loc="upper right")
        plt.savefig("{}{}_{}.png".format(self.log_dir, self.log_name, plot_name))

    def train_energy(self, M, H0, Hs, initial_state):
        """Train the sepctral coefficients to minimize energy minimization.
        Args:
            M: A Hermitian matrix.
            H0: The drift Hamiltonian.
            Hs: Controlable Hamiltonians.
            initial_state: initial_state.
            n_step: number of time steps.
        Returns:
            spectral_coeff: sepctral coefficients.
        """
        self.n_Hs = len(Hs)
        coeff = np.random.normal(0, 1e-3, [self.n_Hs ,self.n_basis]) 
        # coeff = np.ones([self.n_Hs ,self.n_basis])
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = 2e-2
        w_l2 = 0
        I = qp.qeye(2)
        ts = np.linspace(0, 1, self.n_step) 
        optimizer = torch.optim.Adam([self.spectral_coeff], lr=lr)

        self.losses_energy = []
        for epoch in range(self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)
            H = [H0]
            for i in range(self.n_Hs):
                H.append([Hs[i], self.generate_u(i, self.spectral_coeff.detach().numpy())])

            result = qp.mesolve(H, initial_state, ts)
            final_state = result.states[-1]

            loss_energy = M.matrix_element(final_state, final_state)
            loss_l2 = ((self.spectral_coeff**2).mean(0) * torch.tensor(
                [i**2 for i in range(self.n_basis)])).mean() * w_l2
            loss = loss_energy + loss_l2
            optimizer.zero_grad()
            loss_l2.backward()
            grad_coeff = self.compute_energy_grad_MC(M, H, initial_state)
            self.spectral_coeff.grad = grad_coeff
            optimizer.step()

            print("epoch: {:04d}, loss: {:.4f}, loss_energy: {:.4f}".format(
                epoch, 
                loss.real, 
                loss_energy.real
            ))
            self.losses_energy.append(loss_energy.real)
            self.final_state = final_state
        return self.spectral_coeff


    def compute_energy_grad_FD(self, M, H, initial_state, delta=1e-3, coeff=1.0):
        """Compute the gradient of engergy function <psi(1)|M|psi(1)>
        Args:
            M: A Hermitian matrix.
            H: Hamiltonians [H0, [H_i, u_i(t)], ...].
            initial_state: initial_state.
        Returns:
            grad_coeff: gradients of the spectral coefficients.
        """
        coeff = self.spectral_coeff.detach().numpy()
        grad_finite_diff = np.zeros([self.n_Hs ,self.n_basis])

        ts = np.linspace(0, 1, self.n_step) 

        def get_H(curr_coeff):
            _H = [H[0]]
            for _i in range(self.n_Hs):
                _H.append([H[_i+1][0], self.generate_u(_i, curr_coeff)])
            return _H

        def run_forward_sim(new_coeff):
            _H = get_H(new_coeff)
            result = qp.mesolve(_H, initial_state, ts)
            final_state = result.states[-1]
            loss_energy = M.matrix_element(final_state, final_state)
            return loss_energy.real

        for i_Hs in range(self.n_Hs):
            for i_basis in range(self.n_basis):
                new_coeff_p = coeff.copy()
                new_coeff_p[i_Hs][i_basis] = coeff[i_Hs][i_basis] + delta
                E_p = run_forward_sim(new_coeff_p)
                new_coeff_m = coeff.copy()
                new_coeff_m[i_Hs][i_basis] = coeff[i_Hs][i_basis] - delta
                E_m = run_forward_sim(new_coeff_m)
                grad_finite_diff[i_Hs][i_basis] = (E_p - E_m) / delta / 2.0

        return torch.from_numpy(grad_finite_diff)

    def train_energy_FD(self, M, H0, Hs, initial_state, delta=1e-3):
        """Train the sepctral coefficients to minimize energy minimization.
        Args:
            M: A Hermitian matrix.
            H0: The drift Hamiltonian.
            Hs: Controlable Hamiltonians.
            initial_state: initial_state.
            n_step: number of time steps.
        Returns:
            spectral_coeff: sepctral coefficients.
        """
        self.n_Hs = len(Hs)
        coeff = np.random.normal(0, 1e-3, [self.n_Hs ,self.n_basis]) 
        # coeff = np.ones([self.n_Hs ,self.n_basis])
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = 2e-2
        w_l2 = 0
        I = qp.qeye(2)
        ts = np.linspace(0, 1, self.n_step) 
        optimizer = torch.optim.Adam([self.spectral_coeff], lr=lr)

        self.losses_energy = []
        for epoch in range(self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)
            H = [H0]
            for i in range(self.n_Hs):
                H.append([Hs[i], self.generate_u(i, self.spectral_coeff.detach().numpy())])

            result = qp.mesolve(H, initial_state, ts)
            final_state = result.states[-1]

            loss_energy = M.matrix_element(final_state, final_state)
            loss_l2 = ((self.spectral_coeff**2).mean(0) * torch.tensor(
                [i**2 for i in range(self.n_basis)])).mean() * w_l2
            loss = loss_energy + loss_l2
            optimizer.zero_grad()
            loss_l2.backward()

            grad_coeff = self.compute_energy_grad_FD(M, H, initial_state, delta=delta)
            self.spectral_coeff.grad = grad_coeff
            optimizer.step()

            print("epoch: {:04d}, loss: {:.4f}, loss_energy: {:.4f}".format(
                epoch, 
                loss.real, 
                loss_energy.real
            ))
            self.losses_energy.append(loss_energy.real)
            self.final_state = final_state
        return self.spectral_coeff

    def train_fidelity(self, H0, Hs, initial_states, target_states):
        """Train the sepctral coefficients to minimize energy minimization.
        Args:
            H0: The drift Hamiltonian.
            Hs: Controlable Hamiltonians.
            initial_states: initial states.
            target_states: target states.
        Returns:
            spectral_coeff: sepctral coefficients.
        """
        self.n_Hs = len(Hs)
        coeff = np.random.normal(0, 1e-3, [self.n_Hs ,self.n_basis]) 
        # coeff = np.ones([self.n_Hs ,self.n_basis])
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = 2e-2
        w_l2 = 0
        ts = np.linspace(0, 1, self.n_step) 
        optimizer = torch.optim.Adam([self.spectral_coeff], lr=lr)
        # I = qp.qeye(initial_states[0].shape[0])

        self.losses_energy = []
        for epoch in range(self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)

            batch_losses = []
            for i in range(len(initial_states)):
                H = [H0]
                for i in range(self.n_Hs):
                    H.append([Hs[i], self.generate_u(i, self.spectral_coeff.detach().numpy())])
                psi0 = initial_states[i]
                psi1 = target_states[i]
                M = psi1 * psi1.trans() 
                result = qp.mesolve(H, psi0, ts)
                final_state = result.states[-1]

                loss_fidelity = 1 - M.matrix_element(final_state, final_state)
                # loss_energy = M.matrix_element(final_state, final_state)
                loss_l2 = ((self.spectral_coeff**2).mean(0) * torch.tensor(
                    [i**2 for i in range(self.n_basis)])).mean() * w_l2
                loss = loss_fidelity + loss_l2
                optimizer.zero_grad()
                loss_l2.backward()
                grad_coeff = self.compute_energy_grad_MC(M, H, psi0, coeff=-1.0)

                self.spectral_coeff.grad = grad_coeff
                optimizer.step()

                batch_losses.append(loss_fidelity.real)

            batch_losses = np.array(batch_losses).mean()
            print("epoch: {:04d}, loss: {:.4f}, loss_fidelity: {:.4f}".format(
                epoch, 
                batch_losses, 
                batch_losses
            ))
            self.losses_energy.append(batch_losses)
        return self.spectral_coeff

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


    def train_learning(self, M, H0, Hs, X, Y, n_qubit):
        """Train the sepctral coefficients to minimize energy minimization.
        Args:
            H0: The drift Hamiltonian.
            Hs: Controlable Hamiltonians.
            initial_states: initial states.
            target_states: target states.
        Returns:
            spectral_coeff: sepctral coefficients.
        """
        self.n_Hs = len(Hs)
        coeff = np.random.normal(0, 1e-3, [self.n_Hs ,self.n_basis]) 
        # coeff = np.ones([self.n_Hs ,self.n_basis])
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = 2e-2
        w_l2 = 0
        ts = np.linspace(0, 1, self.n_step) 
        optimizer = torch.optim.Adam([self.spectral_coeff], lr=lr)
        
        self.losses_energy = []
        for epoch in range(self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)
            batch_losses = []

            # permutation = np.random.permutation(Y.shape[0])
            for k in range(Y.shape[0]):
                H = [H0]
                for i in range(self.n_Hs):
                    H.append([Hs[i], self.generate_u(i, self.spectral_coeff.detach().numpy())])
                psi0 = self.encoding_x(X[k], n_qubit)
                result = qp.mesolve(H, psi0, ts)
                final_state = result.states[-1]
                predict = M.matrix_element(final_state, final_state).real

                loss_fidelity = (Y[k] - predict)**2
                # loss_energy = M.matrix_element(final_state, final_state)

                loss_l2 = ((self.spectral_coeff**2).mean(0) * torch.tensor(
                    [i**2 for i in range(self.n_basis)])).mean() * w_l2
                loss = loss_fidelity + loss_l2
                optimizer.zero_grad()
                loss_l2.backward()
                coeff =   - 2 * (Y[k] - predict)
                grad_coeff = self.compute_energy_grad_MC(M, H, psi0, coeff=coeff)
                self.spectral_coeff.grad = grad_coeff
                optimizer.step()

                batch_losses.append(loss_fidelity.real)

            batch_losses = np.array(batch_losses).mean()
            print("epoch: {:04d}, loss: {:.4f}, loss_fidelity: {:.4f}".format(
                epoch, 
                batch_losses, 
                batch_losses
            ))
            self.losses_energy.append(batch_losses)
        return self.spectral_coeff

    @staticmethod
    def find_state(final_state):
        data = final_state.data
        prob = []
        for i in range(data.shape[0]):
            d = final_state[i]
            d = d.real**2 + d.imag**2
            prob.append(d)
        prob = np.array(prob)
        prob = np.reshape(prob, [-1])
        d = np.argmax(prob)
        return d, prob

    def demo_energy_qubit2(self):
        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1.j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])

        XI = np.kron(X, I)
        IX = np.kron(I, X)
        XX = np.kron(X, X)
        IZ = np.kron(I, Z)
        YY = np.kron(Y, Y)
        ZI = np.kron(Z, I)
        YI = np.kron(Y, I)
        ZZ = np.kron(Z, Z)
        OO = ZZ * 0
        I = qp.Qobj(I)
        X = qp.Qobj(X)
        Y = qp.Qobj(Y)
        Z = qp.Qobj(Z)
        XI = qp.Qobj(XI)
        IX = qp.Qobj(IX)
        XX = qp.Qobj(XX)
        YY = qp.Qobj(YY)
        IZ = qp.Qobj(IZ)
        ZI = qp.Qobj(ZI)
        YI = qp.Qobj(YI)
        ZZ = qp.Qobj(ZZ)
        OO = qp.Qobj(OO) 

        H0 = OO 
        Hs = [ZZ, IX, XI]
        # Hs = [XX, IZ, ZI]

        g = np.array([1,0])
        e = np.array([0,1])

        ee = np.kron(e, e)
        gg = np.kron(g, g)

        gg, ee = qp.Qobj(gg), qp.Qobj(ee)

        psi0 = gg
        # M = np.kron(np.array([[1, 3 + 1.j], [3 - 1.j, 2]]),
        #             np.eye(2))
        # M = qp.Qobj(M)

        M = 0.5*XX + 0.2*YY + ZZ + IZ
        M = qp.Qobj(M)

        self.train_energy(M, H0, Hs, psi0)

    def demo_energy_qubit1(self):
        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])

        I = qp.Qobj(I)
        X = qp.Qobj(X)
        Y = qp.Qobj(Y)
        Z = qp.Qobj(Z)

        H0 = I
        Hs = [X, Z]

        g = np.array([1,0])
        e = np.array([0,1])

        ee = np.kron(e, e)
        gg = np.kron(g, g)

        gg, ee = qp.Qobj(gg), qp.Qobj(ee)

        psi0 = qp.Qobj(g) 
        M = np.array([[1, 3 + 1.j], [3 - 1.j, 2]])
        M = qp.Qobj(M)
        self.train_energy(M, H0, Hs, psi0)


    def demo_fidelity(self):
        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])

        I = qp.Qobj(I)
        X = qp.Qobj(X)
        Y = qp.Qobj(Y)
        Z = qp.Qobj(Z)

        H0 = I
        Hs = [X]

        g = np.array([1,0])
        e = np.array([0,1])
        g = qp.Qobj(g) 
        e = qp.Qobj(e)

        initial_states = [g, e]
        target_states = [e, g]
        self.train_fidelity(H0, Hs, initial_states, target_states)

    def demo_learning(self):
        np.random.seed(0)
        n_training_size = 8
        n_qubit = 3
        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])

        curr = OurSpectral.multi_kron(*[I for j in range(n_qubit)])
        X0 = qp.Qobj(OurSpectral.inst(curr, X, n_qubit, 0))
        X1 = qp.Qobj(OurSpectral.inst(curr, X, n_qubit, 1))
        X2 = qp.Qobj(OurSpectral.inst(curr, X, n_qubit, 2))
        Z0 = qp.Qobj(OurSpectral.inst(curr, Z, n_qubit, 0))
        Z1 = qp.Qobj(OurSpectral.inst(curr, Z, n_qubit, 1))
        Z2 = qp.Qobj(OurSpectral.inst(curr, Z, n_qubit, 2))

        ZZI = np.kron(np.kron(Z, Z), I)
        ZIZ = np.kron(np.kron(Z, I), Z)
        IZZ = np.kron(np.kron(I, Z), Z)

        YYI = np.kron(np.kron(Y, Y), I)
        YIY = np.kron(np.kron(Y, I), Y)
        IYY = np.kron(np.kron(I, Y), Y)
        H0 = ZZI + ZIZ + IZZ + X0 + X1 + X2
        # H1 = YYI + YIY + IYY
        # H1 = np.kron(np.kron(Z, Z), I) + np.kron(np.kron(Z, I), Z) + np.kron(np.kron(I, Z), Z)
        H2 = X0 + X1 + X2

        I = qp.Qobj(I)
        X = qp.Qobj(X)
        Y = qp.Qobj(Y)
        Z = qp.Qobj(Z)

        H0 = qp.Qobj(H0)
        # Hs = [qp.Qobj(H1), qp.Qobj(H2)]
        # Hs = [ZZI, ZIZ, IZZ, X0, X1, X2, H1]
        Hs = [X0, X1, X2, Z0, Z1, Z2]
        Hs = [qp.Qobj(i) for i in Hs]

        x = np.linspace(-0.95, 0.95, n_training_size)
        x = x[::-1]
        # y = np.sin(x * np.pi)
        y = x**2 
        y = y + np.random.normal(0, 0.1, n_training_size)
        M = qp.Qobj(Z0)

        self.train_learning(M, H0, Hs, x, y, n_qubit)

    def demo_qaoa_max_cut4(self):
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
        H1 = OO
        H_cost = OO
        for i in range(n_qubit):
            if i == 0:
                curr = X
            else:
                curr = I
            for j in range(1, n_qubit):
                if j == i:
                    curr = np.kron(curr, X)
                else:
                    curr = np.kron(curr, I)
            H1 = H1 + curr

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
            H_cost += II - curr
        H_cost = - H_cost * 0.5

        H_cost = qp.Qobj(H_cost)
        H0 = qp.Qobj(H0)
        H1 = qp.Qobj(H1)
        superposition = qp.Qobj(superposition)
        self.train_energy(H_cost, H0, [H1], superposition)

        state, prob = self.find_state(self.final_state)
        print("cut result is ", bin(state)[2:])
        return state, prob


    def demo_finite_diff(self, n_samples=50, delta=0.001, is_MC=True):
        I = np.array(
            [[1, 0], 
            [0, 1]])
        X = np.array(
            [[0, 1], 
            [1, 0]])
        Y = (0+1j) * np.array(
            [[0, -1], 
            [1, 0]])
        Z = np.array(
            [[1, 0], 
            [0, -1]])

        I = qp.Qobj(I)
        X = qp.Qobj(X)
        Y = qp.Qobj(Y)
        Z = qp.Qobj(Z)

        H0 = I
        Hs = [X, Z]

        g = np.array([1,0])
        e = np.array([0,1])

        ee = np.kron(e, e)
        gg = np.kron(g, g)

        gg, ee = qp.Qobj(gg), qp.Qobj(ee)

        psi0 = qp.Qobj(g) 
        M = np.array([[1, 3 + 1.j], [3 - 1.j, 2]])
        M = qp.Qobj(M)

        self.n_Hs = len(Hs)
        coeff = np.ones([self.n_Hs ,self.n_basis])

        ts = np.linspace(0, 1, self.n_step) 

        self.spectral_coeff = torch.ones([self.n_Hs ,self.n_basis])
        H = [H0]
        for i in range(self.n_Hs):
            H.append([Hs[i], self.generate_u(i, self.spectral_coeff)])

        grad_ours = self.sample_multiple_times(M, H, psi0, n_samples=n_samples, is_MC=is_MC)
        grad_FD = self.compute_energy_grad_FD(M, H, psi0, delta=delta).detach().numpy()

        return grad_ours, grad_FD


    def demo_train_with_FD(self, delta=0.001):
        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1.j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])

        XI = np.kron(X, I)
        IX = np.kron(I, X)
        XX = np.kron(X, X)
        IZ = np.kron(I, Z)
        YY = np.kron(Y, Y)
        ZI = np.kron(Z, I)
        YI = np.kron(Y, I)
        ZZ = np.kron(Z, Z)
        OO = ZZ * 0
        I = qp.Qobj(I)
        X = qp.Qobj(X)
        Y = qp.Qobj(Y)
        Z = qp.Qobj(Z)
        XI = qp.Qobj(XI)
        IX = qp.Qobj(IX)
        XX = qp.Qobj(XX)
        YY = qp.Qobj(YY)
        IZ = qp.Qobj(IZ)
        ZI = qp.Qobj(ZI)
        YI = qp.Qobj(YI)
        ZZ = qp.Qobj(ZZ)
        OO = qp.Qobj(OO) 

        H0 = OO 
        Hs = [ZZ, IX, XI]
        # Hs = [XX, IZ, ZI]

        g = np.array([1,0])
        e = np.array([0,1])

        ee = np.kron(e, e)
        gg = np.kron(g, g)

        gg, ee = qp.Qobj(gg), qp.Qobj(ee)

        psi0 = gg
        # M = np.kron(np.array([[1, 3 + 1.j], [3 - 1.j, 2]]),
        #             np.eye(2))
        # M = qp.Qobj(M)

        M = 0.5*XX + 0.2*YY + ZZ + IZ
        M = qp.Qobj(M)

        self.train_energy_FD(M, H0, Hs, psi0, delta=delta)

        

if __name__ == '__main__':
    ours_spectral = OurSpectral(basis='Legendre', n_basis=6)
    # ours_spectral.demo_finite_diff(n_samples=50, delta=1e-5, is_MC=False)
    # ours_spectral = ours_spectral.demo_learning()
    # ours_spectral = ours_spectral.demo_qaoa_max_cut4()
    # ours_spectral.demo_fidelity()
    # ours_spectral.demo_energy_qubit2()
    ours_spectral.demo_train_with_FD()
    # ours_spectral.demo_energy()

