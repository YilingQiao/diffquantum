import torch
import numpy as np
import matplotlib.pyplot as plt


class Grape(object):
    """A class for pulse-based simulation and optimization
    Args:
        taylor_terms: number of taylor expansion terms.
    """
    def __init__(self, taylor_terms=5, n_step=100, n_epoch=200):
        self.taylor_terms = taylor_terms
        self.n_step = n_step
        self.log_dir = "./logs/"
        self.log_name = 'grape'
        self.n_epoch = n_epoch

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
        lr = 2e-2
        w_l2 = 1e-3
        max_amplitude = torch.from_numpy(4 * np.ones(len(Hs)))
        Hs = [torch.tensor(self.c_to_r_mat(-1j * dt * H)) for H in Hs]
        H0 = torch.tensor(self.c_to_r_mat(-1j * dt * H0)) 
        us = torch.tensor(init_u, requires_grad=True)
        optimizer = torch.optim.Adam([us], lr=lr)
        initial_states = torch.tensor(np.array(initial_states)).double().transpose(1, 0)
        target_states = torch.tensor(np.array(target_states)).double().transpose(1, 0)

        self.losses_energy = []
        for epoch in range(self.n_epoch):
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
            self.losses_energy.append(float(loss_fidelity.detach().numpy()))
        return us

    def save_plot(self, plot_name, us):
        np_us = us.detach().numpy()
        plt.clf()
        x = np.array([i * (1. / self.n_step) for i in range(self.n_step + 1)])
        for j in range(us.shape[1]):
            y = [i for i in np_us[:, j]]
            y = np.array([y[0]] + y)
            plt.step(x, y, label='{} u_{}'.format(self.log_name,  j))
            # plt.plot(np_us[:, j], label='{} u_{}'.format(self.log_name,  j))
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
        w_l2 = 1e-3
        Hs = [torch.tensor(self.c_to_r_mat(-1j * dt * H)) for H in Hs]
        H0 = torch.tensor(self.c_to_r_mat(-1j * dt * H0)) 
        us = torch.tensor(init_u, requires_grad=True)
        optimizer = torch.optim.Adam([us], lr=lr)
        initial_states = torch.tensor(psi0).double().unsqueeze(-1)
        M = torch.from_numpy(M)

        self.losses_energy = []
        for epoch in range(self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch, us)
            v = self.forward_simulate(us, H0, Hs, initial_states)
            loss_energy = v.transpose(1, 0).matmul(M).matmul(v)
            loss_l2 = torch.sqrt((us**2).mean())
            loss = loss_energy + loss_l2 * w_l2
            optimizer.zero_grad()
            loss.backward()
            # print(us.grad.norm())
            optimizer.step()

            print("epoch: {:04d}, loss: {:.4f}, loss_energy: {:.4f}".format(
                epoch, 
                float(loss.detach().numpy()), 
                float(loss_energy.detach().numpy()))
            )
            self.losses_energy.append(float(loss_energy.detach().numpy()))
            
        return us

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
            curr = Grape.multi_dot(Grape.multi_kron(
                *[gate if j == idx else I for j in range(n_qubit)]), curr)
        else:
            curr = Grape.multi_dot(gate, curr)
        return curr

    @staticmethod
    def encoding_x(x, n_qubit):
        g = np.array([1,1]) / np.sqrt(2.) 
        I = np.eye(2)
        zero = np.array([[1.0],
                 [0.0]])
        RX = lambda theta: np.array([[np.cos(theta/2.0),-1j*np.sin(theta/2.0)],
                             [-1j*np.sin(theta/2.0),np.cos(theta/2.0)]])
        RY = lambda theta: np.array([[np.cos(theta/2.0),-np.sin(theta/2.0)],
                                     [np.sin(theta/2.0),np.cos(theta/2.0)]])
        RZ = lambda theta: np.array([[np.exp(-1j*theta/2.0),0],
                                     [0,np.exp(1j*theta/2.0)]])

        psi0 = Grape.multi_kron(*[zero for j in range(n_qubit)]) 

        curr = Grape.multi_kron(*[I for j in range(n_qubit)])   
        for j in range(n_qubit):
            curr = Grape.inst(curr, RY(np.arcsin(x)), n_qubit, j)
            curr = Grape.inst(curr, RZ(np.arccos(x**2)), n_qubit, j)

 
        psi0 = np.matmul(curr, psi0)
        psi0 = torch.tensor(Grape.c_to_r_vec(psi0))
        return psi0.unsqueeze(-1)

    def train_learning(self, n_qubit, M, init_u, X, Y, H0, Hs, dt):
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
        w_l2 = 1e-3
        Hs = [torch.tensor(self.c_to_r_mat(-1j * dt * H)) for H in Hs]
        # print(Hs)
        # exit()
        H0 = torch.tensor(self.c_to_r_mat(-1j * dt * H0)) 
        us = torch.tensor(init_u, requires_grad=True)
        optimizer = torch.optim.Adam([us], lr=lr)
        M = torch.from_numpy(M)

        self.losses_energy = []
        for epoch in range(self.n_epoch + 1):
            if epoch % 20 == 0:
                self.save_plot(epoch, us)

            batch_losses = []
            permutation = np.random.permutation(X.shape[0])
            for k in permutation:
                psi0 = self.encoding_x(X[k], n_qubit) 
                v = self.forward_simulate(us, H0, Hs, psi0)
                pred = v.transpose(1, 0).matmul(M).matmul(v)

                loss_energy = (pred - Y[k])**2
                # print(float(pred.detach().numpy()), Y[k])

                loss_l2 = torch.sqrt((us**2).mean())
                loss = loss_energy + loss_l2 * w_l2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss_energy.detach().numpy())

            batch_losses = np.array(batch_losses).mean()
            print("epoch: {:04d}, loss: {:.4f}, loss_energy: {:.4f}".format(
                epoch, 
                batch_losses, 
                batch_losses)
            )
            self.losses_energy.append(batch_losses)
            
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
        # initial_mean = 0
        # initial_stddev = 1e-3 # (1. / np.sqrt(n_step))
        # u = np.random.normal(initial_mean, initial_stddev, [n_step ,n_Hs])

        def f(t):
            n = int(n_step / 2)
            u = 0   
            for j in range(n):
                    u += np.cos(2 * np.pi * j * t) \
                        + np.sin(2 * np.pi * j * t) 
            return u

        u = np.array([[f(t) for i in range(n_Hs)] for t in np.linspace(0, 1, n_step)])
        # plt.plot(u[:,0])
        # plt.show()
        # exit()
        return u

    def demo_fidelity(self):
        """demo of control the psi(1) to be the target states.
        """
        qubit_state_num = 2
        qubit_num = 1
        freq_ge = 3.9 # GHz
        dt = 1./self.n_step
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

        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])

        XI = np.kron(X, I)
        IX = np.kron(I, X)
        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        IZ = np.kron(I, Z)
        ZI = np.kron(Z, I)
        YI = np.kron(Y, I)
        ZZ = np.kron(Z, Z)
        OO = ZZ * 0
        H0 = OO

        M = 0.5*XX + 0.2*YY + ZZ + IZ
        # M = np.kron(np.array([[1, 3 + 1.j], [3 - 1.j, 2]]),
        #             np.eye(2))
        M = self.c_to_r_mat(M)

        Hs = [ZZ, IX, XI]
        # Hs = [XX, IZ, ZI]

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

    def demo_learning(self):
        n_qubit = 3
        dt = 1./self.n_step
        n_training_size = 8
        I = np.array([[1., 0], 
                    [0, 1.]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1.0, 0], 
                    [0, -1.0]])


        curr = Grape.multi_kron(*[I for j in range(n_qubit)])   
        X0 = Grape.inst(curr, X, n_qubit, 0)
        X1 = Grape.inst(curr, X, n_qubit, 1)
        X2 = Grape.inst(curr, X, n_qubit, 2)
        Z0 = Grape.inst(curr, Z, n_qubit, 0)
        Z1 = Grape.inst(curr, Z, n_qubit, 1)
        Z2 = Grape.inst(curr, Z, n_qubit, 2)


        H0 = X0 + X1 + X2 + Z0 + Z1 + Z2
        Hs = [X0, Z0, X0, X1, Z1, X1, X2, Z2, X2]
        M = Z0
        M = self.c_to_r_mat(M)


        x = np.linspace(-0.95, 0.95, n_training_size)
        x = x[::-1]
        y = x**2
        print(x)
        print(y)
        init_u = self.random_initialize_u(self.n_step, len(Hs))
        self.train_learning(n_qubit, M, init_u, x, y, H0, Hs, dt)

if __name__ == '__main__':
    grape = Grape(taylor_terms=20)
    grape.demo_learning()
    # grape.demo_energy_qubit1()
    # grape.demo_energy_qubit2()