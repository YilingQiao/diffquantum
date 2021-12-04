import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
import torch

class OurSpectral(object):
    """A class for using Fourier series to represent the amplitudes.
    The derivatives are computed by our method.
    Args:
        n_basis: number of Fourier basis.
    """
    def __init__(self, n_basis=5, basis='poly'):
        self.n_basis = n_basis
        self.log_dir = "./logs/"
        self.log_name = 'OurSpectral'
        self.basis = basis

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
            for j in range(self.n_basis):
                if self.basis:
                    u += coeff_i[j] * (t - 0.5)**j
                else:
                    u += coeff_i[j] * np.cos(2 * np.pi * j * t)
            return u
        return _u

    def compute_energy_grad_MC(self, M, H, initial_state):
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

            for j in range(self.n_basis):
                if self.basis == 'poly':
                    grad_coeff[i][j] = (s-0.5)**j * ps
                else:
                    grad_coeff[i][j] = np.cos(2 * np.pi * j * s) * ps

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

    def train_energy(self, M, H0, Hs, initial_state, n_step):
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
        self.n_step = n_step
        self.n_Hs = len(Hs)
        coeff = np.random.normal(0, 1e-3, [self.n_Hs ,self.n_basis]) 
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = 2e-2
        n_epoch = 100
        w_l2 = 0
        I = qp.qeye(2)
        ts = np.linspace(0, 1, n_step) 
        optimizer = torch.optim.Adam([self.spectral_coeff], lr=lr)

        self.losses_energy = []
        for epoch in range(n_epoch + 1):
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

        return self.spectral_coeff

    def demo_energy_qubit2(self):
        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])

        XX = np.kron(X, X)
        IZ = np.kron(I, Z)
        ZI = np.kron(Z, I)
        YI = np.kron(Y, I)
        ZZ = np.kron(Z, Z)
        I = qp.Qobj(I)
        X = qp.Qobj(X)
        Y = qp.Qobj(Y)
        Z = qp.Qobj(Z)
        XX = qp.Qobj(XX)
        IZ = qp.Qobj(IZ)
        ZI = qp.Qobj(ZI)
        YI = qp.Qobj(YI)
        ZZ = qp.Qobj(ZZ) 

        H0 = ZZ
        Hs = [XX, IZ, YI]

        g = np.array([1,0])
        e = np.array([0,1])

        ee = np.kron(e, e)
        gg = np.kron(g, g)

        gg, ee = qp.Qobj(gg), qp.Qobj(ee)

        psi0 = gg
        n_step = 100
        M = np.kron(np.array([[1, 3 + 1.j], [3 - 1.j, 2]]),
                    np.eye(2))
        M = qp.Qobj(M)
        self.train_energy(M, H0, Hs, psi0, n_step)

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
        n_step = 100
        M = np.array([[1, 3 + 1.j], [3 - 1.j, 2]])
        M = qp.Qobj(M)
        self.train_energy(M, H0, Hs, psi0, n_step)

if __name__ == '__main__':
    ours_spectral = OurSpectral()
    # ours_spectral.demo_energy_qubit1()
    ours_spectral.demo_energy_qubit2()
    # ours_spectral.demo_energy()

