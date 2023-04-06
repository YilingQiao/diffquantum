import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy 
from scipy.special import legendre
from scipy.stats import unitary_group
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply
from logger import Logger
import time
import math

class SimulatorPlain(object):
    """A class for using function series to represent the amplitudes.
    The derivatives are computed by our method.
    Args:
        n_basis: number of function basis.
    """
    def __init__(self, n_basis=5, basis='BSpline', n_epoch=200, log_dir=None,
        n_step=100, lr=2e-2, is_noisy=False, measure_sample_times=1000, 
        method_name='Ours', sampling_measure=False, per_step=10):
        args = locals()
        self.n_basis = n_basis
        self.log_name = basis
        self.basis = basis
        self.n_epoch = n_epoch
        self.n_step = n_step
        self.lr = lr
        self.is_noisy = is_noisy
        self.sampling_measure = sampling_measure
        if basis == 'Legendre':
            self.legendre_ps = [legendre(j) for j in range(self.n_basis)]

        self.logger = Logger(name=method_name, path=log_dir)
        self.logger.write_text("no mod ========")
        self.logger.write_text("arguments ========")
        for k, v in args.items():
            if k == 'self':
                continue
            self.logger.write_text("{}: {}".format(k, v))
        self.per_step = per_step
        self.my_solver = self.trotter

        if basis == 'BSpline':
            self.get_func_bspline() 


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def get_func_bspline(self):
        tau = 1. / (self.n_basis - 2)
        tau_bs = [tau * (b - 1.5) for b in range(self.n_basis)]

        norm_factor = - (1.5 * tau) ** 2 
        
        def get_bspline_b(b):
            l = tau_bs[b] - 1.5 * tau
            r = tau_bs[b] + 1.5 * tau
            def bspline_b(t):
                if t >= r or t <= l:
                    ans = 0.0
                else:
                    ans = (t - l) * (t - r) / norm_factor
                return ans

            return bspline_b

        self.func_bsplines = [get_bspline_b(b) for b in range(self.n_basis)] 


    def generate_u(self, i, spectral_coeff):
        """Generate the function u(i) for H_i
        Args:
            i: index of the H_i.
        Returns:
            _u: function u_i(t).
        """
        sgm = torch.nn.Sigmoid()
        def _u(t, args):
            coeff_i = spectral_coeff[i]
            u = 0
            n = int(self.n_basis / 2) if self.basis == 'Fourier' else self.n_basis
            for j in range(n):
                if self.basis == 'poly':
                    u += coeff_i[j] * (t - 0.5)**j
                elif self.basis == 'Legendre':
                    u += coeff_i[j] * self.legendre_ps[j](2 * t / self.T - 1)
                elif self.basis == 'Fourier':
                    u += coeff_i[j] * np.cos(2 * np.pi * j * t) \
                        + coeff_i[j + n] * np.sin(2 * np.pi * j * t) 
                elif self.basis == 'BSpline':
                    u += coeff_i[j] * self.func_bsplines[j](t / self.T)

            sigmoid_u = self.sigmoid(u) * 2 - 1
            # print(u, sigmoid_u)
            return sigmoid_u * self.omegas[i]
        return _u

    def stochastic_measure(self, psi, per_Pauli=100) :
        psi_dag = psi.dag()
        ans = 0
        for i in range(len(self.Pauli_M)) :
            distr = []
            weight = self.Pauli_M[i][1]
            evals, estates = self.Pauli_M[i][2]
            for j in range(len(evals)) :
                distr.append((psi_dag * estates[j]).norm() ** 2)

            # print(sum(distr))
            res = np.random.choice(len(evals), per_Pauli, p = distr)
            for j in range(len(evals)) :
                freq = np.count_nonzero(res == j)
                #print(freq, freq / per_Pauli, distr[j], freq / per_Pauli - distr[j])
                ans += weight * evals[j] * freq / per_Pauli
        return ans

    def trotter(self, H_, psi0_, T0, T, **args):
        per_step = self.per_step
        psi = psi0_.full()
        
        n_steps = int(per_step * ((T - T0) + 1))
        start = time.time()

        H = []
        for h in H_:
            if isinstance(h, list):
                H.append([h[0].full(), h[1]])
            else:
                H.append(h.full())
        
        dt = (T - T0) / n_steps
        t = T0
        for k in range(n_steps):

            for h in H:
                if isinstance(h, list):
                    # psi = expm_multiply(-1.j * dt * h[1](t,None) * h[0], psi)
                    dH += -1.j * dt * h[1](t,None) * h[0]
                else:
                    # psi = expm_multiply(-1.j * dt * h, psi)
                    dH = -1.j * dt * h
            # print(time.time() - start_)
            expm = scipy.linalg.expm(dH)
            psi = np.matmul(expm , psi)
            # psi = expm_multiply(dH, psi)
            scipy.linalg.expm
            dH = dH * 0
            t += dt
            
        ans = qp.Qobj(psi)
        return ans

    def compute_G(self, M, H, initial_state):

        def compute_dDdv(s):
            self.spectral_coeff.grad.zero_()
            sgm = torch.nn.Sigmoid()
            for i in range(self.n_Hs):
                if self.basis == 'Legendre':
                    coeff_A = [self.spectral_coeff[i,j] * self.legendre_ps[j](2 * s / self.T - 1) \
                                    for j in range(self.n_basis)]
                elif self.basis == 'BSpline':
                    coeff_A = [self.spectral_coeff[i,j] * self.func_bsplines[j](s / self.T) \
                                    for j in range(self.n_basis)]

                A = sum(coeff_A)
                # Ds = A
                Ds = (sgm(A) * 2. - 1) * self.omegas[i] 
                Ds.backward()

            dDdv = self.spectral_coeff.grad.detach().numpy().copy()
            self.spectral_coeff.grad.zero_()
            return dDdv


        # term1
        s1 = np.random.uniform() * self.T
        s2 = np.random.uniform() * self.T

        smin, smax = min(s1, s2), max(s1, s2)

        n_parameters = self.n_Hs * self.n_basis
        term1 = np.zeros([n_parameters, n_parameters])
        term2 = np.zeros([n_parameters, n_parameters])

        phi_min = self.my_solver(H, initial_state, 0, smin)

        phi_1 = self.my_solver(H, initial_state, 0, s1)
        phi_2 = self.my_solver(H, initial_state, 0, s2)

        dDdv_smax = compute_dDdv(smax)
        dDdv_smin = compute_dDdv(smin)
        dDdv_s1 = compute_dDdv(s1)
        dDdv_s2 = compute_dDdv(s2)

        for ig in range(n_parameters):
            for jg in range(n_parameters):
                iH, ib = ig // self.n_basis, ig % self.n_basis
                jH, jb = jg // self.n_basis, jg % self.n_basis

                pHpThetai = H[iH+1][0] * dDdv_smax[iH, ib]
                pHpThetaj = H[jH+1][0] * dDdv_smin[jH, jb]
                
                # term 1
                size_H = H[iH+1][0].shape[0]
                P_p = np.eye(size_H) + H[jH+1][0]
                P_m = np.eye(size_H) - H[jH+1][0]

                ket_p = self.my_solver(H, P_p * phi_min, smin, smax)
                ket_m = self.my_solver(H, P_m * phi_min, smin, smax)

                res1 = np.real(pHpThetai.matrix_element(ket_p, ket_p))
                res2 = np.real(pHpThetai.matrix_element(ket_m, ket_m))
                term1[ig, jg] = (res1 - res2) / 4. 
                term1[ig, jg] = term1[ig, jg] * dDdv_smin[jH, jb] * dDdv_smax[iH, ib]


                # term 2

                pHpThetai_s1 = H[iH+1][0] * dDdv_s1[iH, ib]
                pHpThetaj_s2 = H[jH+1][0] * dDdv_s2[jH, jb]

                term2[ig, jg] = np.real(pHpThetai_s1.matrix_element(phi_1, phi_1)) *\
                    np.real(pHpThetaj_s2.matrix_element(phi_2, phi_2))

        G = term1 - term2 / (self.T**2)
        print("self.T", self.T)

        return torch.from_numpy(G)





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

        s = np.random.uniform() * self.T

        sgm = torch.nn.Sigmoid()
        for i in range(self.n_Hs):
            if self.basis == 'Legendre':
                coeff_A = [self.spectral_coeff[i,j] * self.legendre_ps[j](2 * s / self.T - 1) \
                                for j in range(self.n_basis)]
            elif self.basis == 'BSpline':
                coeff_A = [self.spectral_coeff[i,j] * self.func_bsplines[j](s / self.T) \
                                for j in range(self.n_basis)]

            A = sum(coeff_A)
            # Ds = A
            Ds = (sgm(A) * 2. - 1) * self.omegas[i] 
            Ds.backward()

        dDdv = self.spectral_coeff.grad.detach().numpy().copy()
        self.spectral_coeff.grad.zero_()


        grad_coeff = np.zeros(self.spectral_coeff.shape)
        t0s = np.linspace(0, s, self.n_step)
        # phi = qp.mesolve(H, initial_state, t0s)
        phi = self.my_solver(H, initial_state, 0, s)
        # phi = result.states[-1]
        
        ts1 = np.linspace(s, 1, self.n_step)
        r = 1 / 2

        for i in range(self.n_Hs):
            d = initial_state.shape[0]
            gate_p = (qp.qeye(d) + r * 1.j * H[i+1][0]) / np.sqrt(1. + r**2)
            gate_m = (qp.qeye(d) - r * 1.j * H[i+1][0]) / np.sqrt(1. + r**2)
            ket_p = self.my_solver(H, gate_p * phi, s, self.T)
            
            if self.sampling_measure :
                ps_p = self.stochastic_measure(ket_p)
            else :
                ps_p = M.matrix_element(ket_p, ket_p)

            if self.is_noisy:
                ps_p += np.random.normal(scale=np.abs(ps_p.real) / 5)

            ket_m = self.my_solver(H, gate_m * phi, s, self.T)

            if self.sampling_measure :
                ps_m = self.stochastic_measure(ket_m)
            else :
                ps_m = M.matrix_element(ket_m, ket_m)

            if self.is_noisy:
                ps_m += np.random.normal(scale=np.abs(ps_m.real) / 5)

            ps = coeff * ( (1 + r**2) / 2 / r * (ps_m - ps_p)).real

            n = int(self.n_basis / 2) if self.basis == 'Fourier' else self.n_basis  
            for j in range(n):
                if self.basis == 'poly':
                    grad_coeff[i][j] = (s-0.5)**j * ps
                elif self.basis in ['Legendre', 'BSpline']:
                    grad_coeff[i][j] = ps * dDdv[i, j]
                elif self.basis == 'Fourier':
                    grad_coeff[i][j] = ps * np.cos(2 * np.pi * j * s) 
                    grad_coeff[i][j + n] =  ps * np.sin(2 * np.pi * j * s) 
        return torch.from_numpy(grad_coeff)

    def save_plot(self, plot_name):
        return
        ts = np.linspace(0, 1, self.n_step) 
        fs = [self.generate_u(i, self.spectral_coeff.detach().numpy().copy()) for i in range(self.n_Hs)]
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
        self.logger.write_text("!!!! train_energy ========")

        self.n_Hs = len(Hs)
        coeff = np.random.normal(0, 1e-3, [self.n_Hs ,self.n_basis]) 
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = self.lr
        w_l2 = 0.
        I = qp.qeye(2)
        ts = np.linspace(0, self.T, 2)
        optimizer = torch.optim.Adam([self.spectral_coeff], lr=lr)

        self.losses_energy = []
        for epoch in range(1, self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)
            H = [H0]
            for i in range(self.n_Hs):
                H.append([Hs[i], self.generate_u(i, self.spectral_coeff.detach().numpy().copy())])

            final_state = self.my_solver(H, initial_state, 0, self.T)

            if self.sampling_measure :
                loss_energy = self.stochastic_measure(final_state).real
            else :
                loss_energy = M.matrix_element(final_state, final_state).real

            if self.is_noisy:
                loss_energy += np.random.normal(scale=np.abs(loss_energy.real) / 5)
            loss_l2 = ((self.spectral_coeff**2).mean(0) * torch.tensor(
                [i**2 for i in range(self.n_basis)])).mean() * w_l2
            loss = loss_energy + loss_l2
            optimizer.zero_grad()
            loss_l2.backward()
            grad_coeff = self.compute_energy_grad_MC(M, H, initial_state)
            
            # nature gradient
            G = self.compute_G(M, H, initial_state)
            grad_coeff = grad_coeff.reshape([-1])
            grad_coeff_ng = torch.linalg.solve(G, grad_coeff).reshape([self.n_Hs, self.n_basis])

            # print(torch.linalg.solve(G, G))
            # exit()
            
            print(grad_coeff - G.matmul(grad_coeff_ng.reshape([-1])))
            exit()

            # self.spectral_coeff = self.spectral_coeff - grad_coeff
            # optimizer.zero_grad()
            # use autograd
            self.spectral_coeff.grad = grad_coeff_ng
            optimizer.step()

            loss_energy = loss_energy - M.eigenenergies()[0]

            st = "epoch: {:04d}, loss: {}, loss_energy: {}".format(
                epoch, 
                loss, 
                loss_energy
            )

            self.logger.write_text(st)
            self.losses_energy.append(loss_energy)
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

            # loss_energy = M.matrix_element(final_state, final_state)
            if self.sampling_measure :
                loss_energy = self.stochastic_measure(final_state).real
            else :
                loss_energy = M.matrix_element(final_state, final_state).real

            if self.is_noisy:
                loss_energy += np.random.normal(scale=np.abs(loss_energy.real) / 5)
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
        self.logger.write_text("!!!! train_energy ========")
        self.n_Hs = len(Hs)
        coeff = np.random.normal(0, 1e-3, [self.n_Hs ,self.n_basis]) 
        # coeff = np.ones([self.n_Hs ,self.n_basis])
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = self.lr
        w_l2 = 0
        I = qp.qeye(2)
        ts = np.linspace(0, 1, self.n_step) 
        optimizer = torch.optim.Adam([self.spectral_coeff], lr=lr)

        self.losses_energy = []
        for epoch in range(1, self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)
            H = [H0]
            for i in range(self.n_Hs):
                H.append([Hs[i], self.generate_u(i, self.spectral_coeff.detach().numpy().copy())])

            result = qp.mesolve(H, initial_state, ts)
            final_state = result.states[-1]

            loss_energy = M.matrix_element(final_state, final_state).real
            if self.is_noisy:
                loss_energy += np.random.normal(scale=np.abs(loss_energy.real) / 5)
            loss_l2 = ((self.spectral_coeff**2).mean(0) * torch.tensor(
                [i**2 for i in range(self.n_basis)])).mean() * w_l2
            loss = loss_energy + loss_l2
            optimizer.zero_grad()
            loss_l2.backward()

            grad_coeff = self.compute_energy_grad_FD(M, H, initial_state, delta=delta)
            self.spectral_coeff.grad = grad_coeff
            optimizer.step()

            loss_energy = loss_energy - M.eigenenergies()[0]

            st = "epoch: {:04d}, loss: {}, loss_energy: {}".format(
                epoch, 
                loss, 
                loss_energy
            )
            self.logger.write_text(st)
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
        coeff = np.random.normal(0, 1, [self.n_Hs ,self.n_basis]) 
        # coeff = np.ones([self.n_Hs ,self.n_basis])
        self.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = self.lr
        w_l2 = 0
        ts = np.linspace(0, 1, self.n_step) 
        optimizer = torch.optim.Adam([self.spectral_coeff], lr=lr)
        # I = qp.qeye(initial_states[0].shape[0])

        self.losses_energy = []
        for epoch in range(1, self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch)

            batch_losses = []
            for i in range(len(initial_states)):
                H = [H0]
                for j in range(self.n_Hs):
                    H.append([Hs[j], self.generate_u(j, self.spectral_coeff.detach().numpy())])
                psi0 = initial_states[i]
                psi1 = target_states[i]
                M = psi1 * psi1.dag() 
                result = qp.mesolve(H, psi0, ts)
                final_state = result.states[-1]

                inner_product_norm = M.matrix_element(final_state, final_state)
                if self.is_noisy:
                    inner_product_norm += np.random.normal(
                        scale=np.abs(inner_product_norm.real) / 5)
                loss_fidelity = 1 - inner_product_norm
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
