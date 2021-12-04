import torch
import numpy as np
import matplotlib.pyplot as plt


class Grape(object):
    """A class for pulse-based simulation and optimization
    Args:
        taylor_terms: number of taylor expansion terms.
    """
    def __init__(self, taylor_terms=5, n_step=100):
        self.taylor_terms = taylor_terms
        self.n_step = n_step
        self.log_dir = "./logs/"
        self.log_name = 'grape'

    def train_fidelity(self, init_u, H0, Hs, initial_states, target_states, dt):
        """Control the systems to reach target states.
        Args:
            init_u: initial pulse.
            H0: a Hermitian matrix.
            Hs: a list of Hermitian matrics.
            initial_states: initial_states.
            target_states: target_states.
            dt: size of time step.
        Returns:
            us: optimized pulses.
        """
        lr = 3e-3
        n_epoch = 200
        w_l2 = 1
        max_amplitude = torch.from_numpy(4 * np.ones(len(Hs)))
        Hs = [torch.tensor(self.c_to_r_mat(-1j * dt * H)) for H in Hs]
        H0 = torch.tensor(self.c_to_r_mat(-1j * dt * H0)) 
        us = torch.tensor(init_u, requires_grad=True)
        optimizer = torch.optim.Adam([us], lr=lr)
        initial_states = torch.tensor(np.array(initial_states)).double().transpose(1, 0)
        target_states = torch.tensor(np.array(target_states)).double().transpose(1, 0)

        for epoch in range(n_epoch):
            modifled_us = torch.sin(us) * max_amplitude
            final_states = self.forward_simulate(modifled_us, H0, Hs, initial_states)
            loss_fidelity = 1 - self.get_inner_product_2D(final_states, target_states)
            loss_l2 = torch.sqrt((us**2).mean())
            loss = loss_fidelity + loss_l2 * w_l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch: {:04d}, loss: {:.4f}, loss_fid: {:.4f}".format(
                epoch, 
                float(loss.detach().numpy()), 
                float(loss_fidelity.detach().numpy()))
            )
        return us

    def save_plot(self, plot_name, us):
        np_us = us.detach().numpy()
        plt.clf()
        for j in range(us.shape[1]):
            plt.plot(np_us[:, j], label='{} u_{}'.format(self.log_name,  j))
        plt.legend(loc="upper right")
        plt.savefig("{}{}_{}.png".format(self.log_dir, self.log_name, plot_name))

    def train_energy(self, M, init_u, H0, Hs, psi0, dt):
        """Optimize the pulse to minimize the energy <psi(1)|M|psi(1)>
        Args:
            M: a Hermitian matrix.
            init_u: initial pulse.
            H0: a Hermitian matrix.
            Hs: a list of Hermitian matrics.
            psi0: initial state.
            dt: size of time step.
        Returns:
            us: optimized pulses.
        """
        lr = 2e-2
        n_epoch = 100
        w_l2 = 1e-3
        Hs = [torch.tensor(self.c_to_r_mat(-1j * dt * H)) for H in Hs]
        H0 = torch.tensor(self.c_to_r_mat(-1j * dt * H0)) 
        us = torch.tensor(init_u, requires_grad=True)
        optimizer = torch.optim.Adam([us], lr=lr)
        initial_states = torch.tensor(psi0).double().unsqueeze(-1)
        M = torch.from_numpy(M)

        self.losses_energy = []

        for epoch in range(n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch, us)
            v = self.forward_simulate(us, H0, Hs, initial_states)
            loss_energy = v.transpose(1, 0).matmul(M).matmul(v)
            loss_l2 = torch.sqrt((us**2).mean())
            loss = loss_energy + loss_l2 * w_l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch: {:04d}, loss: {:.4f}, loss_energy: {:.4f}".format(
                epoch, 
                float(loss.detach().numpy()), 
                float(loss_energy.detach().numpy()))
            )
            self.losses_energy.append(float(loss_energy.detach().numpy()))


        return us

    def forward_simulate(self, us, H0, Hs, initial_states):
        """Forward time evolution.
        Args:
            us: pulses.
            H0: a Hermitian matrix.
            Hs: a list of Hermitian matrics.
            initial_states: initial states.
        Returns:
            final_states: final states.
        """
        self.intermediate_states = []
        op_H0 = self.compute_op([1.], [H0])
        I = torch.eye(op_H0.shape[0]).double()
        final_states = initial_states
        for i in range(us.shape[0]):
            final_states = torch.matmul(op_H0, final_states)
            I = torch.matmul(op_H0, I)
            op = self.compute_op(us[i], Hs)
            final_states = torch.matmul(op, final_states)
            self.intermediate_states.append(final_states.detach().numpy())
            I = torch.matmul(op, I)

        return final_states

    def get_inner_product_2D(self,psi1,psi2):
        """compute the inner product |psi1.dag() * psi2|.
        Args:
            psi1: a state.
            psi2: a psi2.
        Returns:
            norm: norm of the complex number.
        """
        state_num = int(psi1.shape[0] / 2)
        
        psi_1_real = psi1[0:state_num,:]
        psi_1_imag = psi1[state_num:2*state_num,:]
        psi_2_real = psi2[0:state_num,:]
        psi_2_imag = psi2[state_num:2*state_num,:]

        ac = (psi_1_real * psi_2_real).sum(0) 
        bd = (psi_1_imag * psi_2_imag).sum(0) 
        bc = (psi_1_imag * psi_2_real).sum(0) 
        ad = (psi_1_real * psi_2_imag).sum(0) 
        reals = (ac + bd).sum()**2
        imags = (bc - ad).sum()**2
        norm = (reals + imags) / (psi1.shape[1]**2)
        return norm


    def compute_op(self, us, Hs):
        """Compute the time operator in a short time using taylor expansion.
        Args:
            us: n_Hs - amplitudes.
            Hs: n_Hs - hamiltonians.
            dt: size of time step.
        Returns:
            op: time operator.
        """
        H = torch.zeros_like(Hs[0])
        for i in range(len(us)):
            uHdt = us[i] * Hs[i]
            H += uHdt

        op = torch.eye(Hs[0].shape[0]).double()
        Hn = torch.eye(Hs[0].shape[0]).double()
        factorial = 1.

        for i in range(1, self.taylor_terms + 1):      
            factorial = factorial * i
            Hn = torch.matmul(H, Hn)
            op += Hn / factorial
        return op

    @staticmethod
    def c_to_r_mat(M):
        # complex to real isomorphism for matrix
        return np.asarray(np.bmat([[M.real, -M.imag],[M.imag, M.real]]))

    @staticmethod
    def c_to_r_vec(V):
        # complex to real isomorphism for vector
        new_v = []
        new_v.append(V.real)
        new_v.append(V.imag)
        return np.reshape(new_v, [2 * len(V)])
        
    @staticmethod
    def random_initialize_u(n_step, n_Hs):
        initial_mean = 0
        initial_stddev = 1e-3 # (1. / np.sqrt(n_step))
        u = np.random.normal(initial_mean, initial_stddev, [n_step ,n_Hs])
        return u

    def demo_fidelity(self):
        """demo of control the psi(1) to be the target states.
        """
        qubit_state_num = 2
        qubit_num = 1
        freq_ge = 3.9 # GHz
        dt = 0.01
        n_step = 100

        Q_x = np.diag(np.sqrt(np.arange(1, qubit_state_num)), 1) \
                + np.diag(np.sqrt(np.arange(1, qubit_state_num)), -1)
        Q_y = (0+1j) *(np.diag(np.sqrt(np.arange(1, qubit_state_num)), 1) \
                - np.diag(np.sqrt(np.arange(1, qubit_state_num)), -1))
        Q_z = np.diag(np.arange(0, qubit_state_num))
        ens = np.array([2 * np.pi * ii * freq_ge for ii in np.arange(qubit_state_num)])
        H_q = np.diag(ens)
        H0 = H_q
        H0 = np.eye(qubit_state_num)
        Hs = [Q_x]

        g = np.array([1,0])
        e = np.array([0,1])
        psi0 = [g, e]
            
        psi1 = [e, g]
        target_states = [self.c_to_r_vec(v) for v in psi1]
        initial_states = [self.c_to_r_vec(v) for v in psi0]

        init_u = self.random_initialize_u(n_step, len(Hs))
        final_u = self.train_fidelity(init_u, H0, Hs, initial_states, target_states, dt)


    def demo_energy_qubit1(self):
        """demo of optimizing an energy <psi(1)|M|psi(1)>.
        """
        qubit_state_num = 2
        qubit_num = 2
        dt = 1./self.n_step
        M = np.array([[1, 3 + 1.j], [3 - 1.j, 2]])
        # M = np.kron(np.array([[1, 3 + 1.j], [3 - 1.j, 2]]),
        #             np.eye(2))
        M = self.c_to_r_mat(M)

        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])
        H0 = I
        Hs = [X, Z]

        g = np.array([1,0])
        e = np.array([0,1])

        psi0 = self.c_to_r_vec(g)

        init_u = self.random_initialize_u(self.n_step, len(Hs))
        final_u = self.train_energy(M, init_u, H0, Hs, psi0, dt)

    def demo_energy_qubit2(self):
        """demo of optimizing an energy <psi(1)|M|psi(1)>.
        """
        qubit_state_num = 2
        qubit_num = 2
        dt = 1./self.n_step
        M = np.kron(np.array([[1, 3 + 1.j], [3 - 1.j, 2]]),
                    np.eye(2))
        M = self.c_to_r_mat(M)

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
        ZZ = np.kron(Z, Z)

        H0 = ZZ
        Hs = [XX, IZ]

        g = np.array([1,0])
        e = np.array([0,1])

        ee = np.kron(e, e)
        gg = np.kron(g, g)

        # print(np.kron(e, e))
        # print(np.kron(g, g))
        # exit()
        psi0 = self.c_to_r_vec(gg)

        init_u = self.random_initialize_u(self.n_step, len(Hs))
        final_u = self.train_energy(M, init_u, H0, Hs, psi0, dt)

if __name__ == '__main__':
    grape = Grape(taylor_terms=20)
    # grape.demo_fidelity()
    grape.demo_energy_qubit1()
    # grape.demo_energy_qubit2()