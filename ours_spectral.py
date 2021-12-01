from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from grape import *

class OurFourier(object):
    """Use Fourier series to represent the amplitude.
    The derivatives are computed by our method.
    """
    def __init__(self, n_basis=20):
        self.n_basis = n_basis
        return

    def generate_u(self, i):
        def _u(t, args):
            coeff_i = self.spectral_coeff.detach().numpy()[i]
            u = 0
            for j in range(self.n_basis):
                u += coeff_i[j] * np.cos(2 * np.pi * j * t)
            return u
        return _u


    def compute_grad_MC(self, H, initial_states, target_states, M):
        grad_coeff = np.zeros(self.spectral_coeff.shape)
        s = np.random.uniform()
        t0s = np.linspace(0, s, self.n_step)
        result = mesolve(H, initial_states, t0s)
        phi = result.states[-1]
        
        ts1 = np.linspace(s, 1, self.n_step)
        r = 1

        for i in range(self.n_Hs):
            gate_p = qeye(2) + r * 1.j * H[i+1][0]
            gate_m = qeye(2) - r * 1.j * H[i+1][0]
            
            result = mesolve(H, gate_p * phi, ts1)
            ket_p = result.states[-1]
            ps_p = M.matrix_element(ket_p, ket_p)

            result = mesolve(H, gate_m * phi, ts1)
            ket_m = result.states[-1]
            ps_m = M.matrix_element(ket_m, ket_m)

            ps = (0.5 / r * (ps_m - ps_p)).real

            for j in range(self.n_basis):
                grad_coeff[i][j] = np.cos(2 * np.pi * j * s) * ps

        self.spectral_coeff.grad = torch.from_numpy(grad_coeff)


    def train(self, H0, Hs, initial_states, target_states, n_step):
        self.n_step = n_step
        self.n_Hs = len(Hs)
        coeff = np.random.normal(0, 1, [self.n_Hs ,self.n_basis]) 
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = 8e-3
        n_epoch = 200
        w_l2 = 1

        M = np.array([[1, 1 + 1.j], [1 - 1.j, 2]])
        M = Qobj(M)

        I = qeye(2)
        ts = np.linspace(0, 1, n_step) 

        optimizer = torch.optim.Adam([self.spectral_coeff], lr=lr)

        for epoch in range(n_epoch):

            H = [H0]
            for i in range(self.n_Hs):
                H.append([Hs[i], self.generate_u(i)])
            
            result = mesolve(H, initial_states, ts)
            final_state = result.states[-1]

            loss_energy = M.matrix_element(final_state, final_state)
            loss = loss_energy

            print("epoch: {:04d}, loss: {:.4f}, loss_energy: {:.4f}".format(
                epoch, 
                loss.real, 
                loss_energy.real
            ))

            optimizer.zero_grad()
            self.compute_grad_MC(H, initial_states, target_states, M)
            optimizer.step()


if __name__ == '__main__':
    Hs = [sigmax()]
    H0 = qeye(2)
    g, e = basis(2, 0), basis(2, 1)
    psi0 = g
    psi1 = e
    n_step = 100

    ours = OurFourier()
    ours.train(H0, Hs, psi0, psi1, n_step)

