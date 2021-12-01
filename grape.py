import torch
import numpy as np

def c_to_r_mat(M):
    # complex to real isomorphism for matrix
    return np.asarray(np.bmat([[M.real, -M.imag],[M.imag, M.real]]))

def c_to_r_vec(V):
    # complex to real isomorphism for vector
    new_v = []
    new_v.append(V.real)
    new_v.append(V.imag)
    return np.reshape(new_v, [2 * len(V)])

def random_initialize_u(n_step, n_Hs):
    initial_mean = 0
    initial_stddev = (1. / np.sqrt(n_step))
    u = np.random.normal(initial_mean, initial_stddev, [n_step ,n_Hs])
    return u


class Grape():
    """
    TODO: 
        measure

    """
    def __init__(self, 
        taylor_terms=5):

        # self.init_u = init_u
        # self.Hs = Hs
        self.taylor_terms = taylor_terms

    def train(self, init_u, H0, Hs, initial_states, target_states, dt):
        lr = 3e-3
        n_epoch = 200
        w_l2 = 1
        max_amplitude = torch.from_numpy(4 * np.ones(len(Hs)))
        Hs = [torch.tensor(c_to_r_mat(-1j * dt * H)) for H in Hs]
        H0 = torch.tensor(c_to_r_mat(-1j * dt * H0)) 
        us = torch.tensor(init_u, requires_grad=True)
        optimizer = torch.optim.Adam([us], lr=lr)
        initial_states = torch.tensor(initial_states).double().transpose(1, 0)
        target_states = torch.tensor(target_states).double().transpose(1, 0)

        for epoch in range(n_epoch):

            modifled_us = torch.sin(us) * max_amplitude
            final_states = self.forward_simulate(modifled_us, H0, Hs, initial_states)
            # print("final_states", final_states)
            # print("target_states", target_states)
            # exit()
            loss_fidelity = 1 - self.get_inner_product_2D(final_states, target_states)
            loss_l2 = torch.sqrt((us**2).mean())

            # print(final_states, target_states)

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

    def forward_simulate(self, us, H0, Hs, initial_states):
        """Forward time evolution.
        Args:
            initial_states: initial states.
            us: n_Time x n_Hs - amplitudes.
            Hs: n_Hs - hamiltonians.
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
        #Take 2 states psi1,psi2, calculate their overlap, for arbitrary number of vectors
        # psi1 and psi2 are shaped as (2*state_num, number of vectors)
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

    def inner_product(self, p0, p1):
        """Inner product of p0 p1.
        p0 = a + ib
        p1 = c + id
        Args:
            p0:
            p1:
        Returns:
            ans: <p0|p1>
        """
        p0 = torch.reshape(p0, [2, -1])
        p1 = torch.reshape(p1, [2, -1])
        a, b ,c, d = p0[0], p0[1], p1[0], p1[1]
        ac = (a * c).sum()
        bd = (b * d).sum()
        bc = (b * c).sum()
        ad = (a * d).sum()
        reals = (ac + bd)**2
        imags = (bc - ad)**2
        norm = (reals + imags) / (p0.shape[1]**2)

        return norm


    def compute_op(self, us, Hs):
        """Compute the time operator in a short time
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

if __name__ == '__main__':
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
    psi0=[g, e]
        
    psi1 = [e, g]
    target_states = [c_to_r_vec(v) for v in psi1]
    initial_states = [c_to_r_vec(v) for v in psi0]

    init_u = random_initialize_u(n_step, len(Hs))

    grape = Grape(taylor_terms=20)
    final_u = grape.train(init_u, H0, Hs, initial_states, target_states, dt)