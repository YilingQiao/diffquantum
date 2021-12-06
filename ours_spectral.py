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

    def generate_u(self, i):
        """Generate the function u(i) for H_i
        Args:
            i: index of the H_i.
        Returns:
            _u: function u_i(t).
        """
        def _u(t, args):
            coeff_i = self.spectral_coeff.detach().numpy()[i]
            u = 0
            n = self.n_basis if self.basis == 'poly' else int(self.n_basis / 2)
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

    def sample_multiple_times(self, M, H, initial_state):
        n_samples = 100
        grads_coeffs = []

        for ss in range(n_samples):
            grad_coeff = np.zeros(self.spectral_coeff.shape)
            s = np.random.uniform()
            t0s = np.linspace(0, s, self.n_step)
            result = qp.mesolve(H, initial_state, t0s)
            phi = result.states[-1]
            
            ts1 = np.linspace(s, 1, self.n_step)
            r = 1

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

                n = int(self.n_basis / 2) if self.basis == 'poly' else self.n_basis  
                for j in range(n):
                    if self.basis == 'poly':
                        grad_coeff[i][j] = (s-0.5)**j * ps
                    elif self.basis == 'Legendre':
                        pj = legendre(j)
                        grad_coeff[i][j] = pj(2 * s - 1) * ps
                    elif self.basis == 'Fourier':
                        grad_coeff[i][j] = ps * (np.cos(2 * np.pi * j * s) \
                            + np.sin(2 * np.pi * j * s) )

            # grads_coeffs.append(np.expand_dims(grad_coeff, 0))
            grads_coeffs.append(grad_coeff)
        grads_coeffs = np.concatenate(grads_coeffs, 0)
        print(grads_coeffs.shape)
        print(grads_coeffs.mean(0))
        print(grads_coeffs.std(0))
        plt.clf()
        plt.hist(grads_coeffs[:,0])
        plt.show()

    def compute_energy_grad_MC(self, M, H, initial_state):
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
        r = 1

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
                    grad_coeff[i][j] = ps * (np.cos(2 * np.pi * j * s) \
                        + np.sin(2 * np.pi * j * s) )
        return torch.from_numpy(grad_coeff)

    def save_plot(self, plot_name):
        ts = np.linspace(0, 1, self.n_step) 
        fs = [self.generate_u(i) for i in range(self.n_Hs)]
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

        lr = 1e-1
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
                H.append([Hs[i], self.generate_u(i)])

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


    def compute_fidelity_grad_MC(self, I, H, initial_state, target_state, loss_fidelity_a):
        """Compute the gradient of engergy function <psi(1)|M|psi(1)>
        Args:
            M: A Hermitian matrix.
            H: Hamiltonians [H0, [H_i, u_i(t)], ...].
            initial_state: initial_state.
        Returns:
            grad_coeff: gradients of the spectral coefficients.
        """
        grad_coeff = np.zeros(self.spectral_coeff.shape)
        s = np.random.uniform()
        t0s = np.linspace(0, s, self.n_step)

        result = qp.mesolve(H, initial_state, t0s)
        psi = result.states[-1]
        phi = target_state

        
        ts1 = np.linspace(s, 1, self.n_step)
        r = 1

        for i in range(self.n_Hs):
            result = qp.mesolve(H, H[i+1][0] * psi, ts1)
            ket_p = result.states[-1]
            term1 = I.matrix_element(ket_p, phi) * np.conjugate(loss_fidelity_a)
            term2 = I.matrix_element(phi, ket_p) * loss_fidelity_a
            
            ps = (term1 + term2).real

            n = int(self.n_basis / 2) if self.basis == 'Fourier' else self.n_basis  
            for j in range(n):
                if self.basis == 'poly':
                    grad_coeff[i][j] = (s-0.5)**j * ps
                elif self.basis == 'Legendre':
                    pj = legendre(j)
                    grad_coeff[i][j] = pj(2 * s - 1) * ps
                elif self.basis == 'Fourier':
                    grad_coeff[i][j] = ps * (np.cos(2 * np.pi * j * s) \
                        + np.sin(2 * np.pi * j * s) )
        return torch.from_numpy(grad_coeff)

    def train_fidelity(self,H0, Hs, initial_states, target_states):
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
        # coeff = np.random.normal(0, 1e-3, [self.n_Hs ,self.n_basis]) 
        coeff = np.ones([self.n_Hs ,self.n_basis])
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = 2e-2
        w_l2 = 0
        ts = np.linspace(0, 1, self.n_step) 
        optimizer = torch.optim.Adam([self.spectral_coeff], lr=lr)
        I = qp.qeye(initial_states[0].shape[0])

        self.losses_energy = []
        for epoch in range(self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)
            H = [H0]
            for i in range(self.n_Hs):
                H.append([Hs[i], self.generate_u(i)])

            batch_losses = []
            for i in range(len(initial_states)):
                psi0 = initial_states[i]
                psi1 = target_states[i]
                result = qp.mesolve(H, psi0, ts)
                final_state = result.states[-1]

                loss_fidelity_a = I.matrix_element(psi1, final_state)
                loss_fidelity = 1 - loss_fidelity_a * np.conjugate(loss_fidelity_a)
                # loss_energy = M.matrix_element(final_state, final_state)
                loss_l2 = ((self.spectral_coeff**2).mean(0) * torch.tensor(
                    [i**2 for i in range(self.n_basis)])).mean() * w_l2
                loss = loss_fidelity + loss_l2
                optimizer.zero_grad()
                loss_l2.backward()
                grad_coeff = self.compute_fidelity_grad_MC(I, H, psi0, psi1, loss_fidelity_a)
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


if __name__ == '__main__':
    ours_spectral = OurSpectral(basis='Legendre', n_basis=3)
    ours_spectral = ours_spectral.demo_fidelity()
    # ours_spectral = ours_spectral.demo_qaoa_max_cut4()
    # ours_spectral.demo_energy_qubit1()
    # ours_spectral.demo_energy_qubit2()
    # ours_spectral.demo_energy()

