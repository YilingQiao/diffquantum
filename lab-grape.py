import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import legendre
from scipy.stats import unitary_group
from numpy import kron
from logger import Logger


class Grape_lab(object):
    """A class for pulse-based simulation and optimization
    Args:
        taylor_terms: number of taylor expansion terms.
    """
    def __init__(self, n_step=100, n_epoch=200, lr=2e-2):
        args = locals()
        #self.taylor_terms = taylor_terms
        self.n_step = n_step
        self.log_dir = "./logs/"
        self.log_name = 'grape'
        self.n_epoch = n_epoch
        self.lr = lr

        self.logger = Logger()
        self.logger.write_text("arguments ========")
        for k, v in args.items():
            if k == 'self':
                continue
            self.logger.write_text("{}: {}".format(k, v))

    def train_fidelity_ibm1_lab(self, init_u, H0, Hs, initial_states, target_states):
        """Control the 1-qubit IBM system to reach target states.
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
        self.logger.write_text("!!!! train_fidelity (lab frame) ========")

        lr = self.lr
        w_l2 = 0
        
        H0 = torch.tensor(self.c_to_r_mat(-1.j * self.dt * H0))
        Hs = [torch.tensor(self.c_to_r_mat(-1.j * self.dt * hs)) for hs in Hs]
        us = torch.tensor(init_u, requires_grad=True)
        optimizer = torch.optim.Adam([us], lr=lr)
        initial_states = torch.tensor(np.array(initial_states)).double().transpose(1, 0)
        target_states = torch.tensor(np.array(target_states)).double().transpose(1, 0)

        self.losses_energy = []
        for epoch in range(self.n_epoch):
            st = "params: {}".format(us)
            self.logger.write_text_aux(st)
            final_states = self.forward_simulate_ibm1_lab(us, H0, Hs, initial_states)
            loss_fidelity = 1 - self.get_inner_product_2D(final_states, target_states)
            loss_l2 = torch.sqrt((us**2).mean())
            loss = loss_fidelity + loss_l2 * w_l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            st = "epoch: {:04d}, loss: {:.16f}, loss_fid: {:.16f}".format(
                epoch, 
                float(loss.detach().numpy()), 
                float(loss_fidelity.detach().numpy()))
            self.logger.write_text(st)

            self.losses_energy.append(float(loss_fidelity.detach().numpy()))
        
        return us
        
    
    
    def train_fidelity_ibm2_lab(self, init_u, H0, Hs, initial_states, target_states):
        """Control the 2-qubit IBM system to reach target states.
        Args:
            init_u: initial pulse.
            H0: a list of Hermitian matrices.
            Hs: a list of Hermitian matrices.
            initial_states: initial_states.
            target_states: target_states.
            dt: size of time step.
        Returns:
            us: optimized pulses.
        """
        self.logger.write_text("!!!! train_fidelity (lab frame) ========")

        lr = self.lr
        w_l2 = 0
        #max_amplitude = torch.from_numpy(np.ones(len(Hs)))
        H0 = torch.tensor(self.c_to_r_mat(-1.j * self.dt * H0)) 
        Hs = [torch.tensor(self.c_to_r_mat(-1.j * self.dt * hs)) for hs in Hs]
        us = torch.tensor(init_u, requires_grad=True)
        optimizer = torch.optim.Adam([us], lr=lr)
        initial_states = torch.tensor(np.array(initial_states)).double().transpose(1, 0)
        target_states = torch.tensor(np.array(target_states)).double().transpose(1, 0)

        self.losses_energy = []
        for epoch in range(self.n_epoch):
            st = "params: {}".format(us)
            self.logger.write_text_aux(st)
            
            final_states = self.forward_simulate_ibm2_lab(us, H0, Hs, initial_states)
            loss_fidelity = 1 - self.get_inner_product_2D(final_states, target_states)
            loss_l2 = torch.sqrt((us**2).mean())
            loss = loss_fidelity + loss_l2 * w_l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            st = "epoch: {:04d}, loss: {:.16f}, loss_fid: {:.16f}".format(
                epoch, 
                float(loss.detach().numpy()), 
                float(loss_fidelity.detach().numpy()))
            self.logger.write_text(st)

            self.losses_energy.append(float(loss_fidelity.detach().numpy()))
        
        return us

    
    def forward_simulate_ibm1_lab(self, us, H0, Hs, initial_states):
        """Forward time evolution: IBM 1-qubit system.
        Args:
            us: pulses.
            H0: a Hermitian matrix.
            Hs: a list of Hermitian matrics.
            initial_states: initial states.
        Returns:
            final_states: final states.
        """
        #self.intermediate_states = []
        
        n_step = us.shape[0]
        n_param = us.shape[1]
        
        if len(Hs) != 1:
            print("Hs must have 1 elements!")
            
        if n_param != 2:
            print("Num. of parameters must be 2!")

        final_states = initial_states
        sgm = torch.nn.Sigmoid()
        
        for j in range(n_step):
            t = j * self.dt
            Hjdt = H0.clone()
            N = torch.sqrt(torch.sum(torch.square(us[j])))
            Hjdt += (2*sgm(N)-1) / N * (np.cos(self.w0*t) * us[j][0] - np.sin(self.w0*t) * us[j][1]) * Hs[0].clone()
            final_states = torch.matmul(torch.matrix_exp(Hjdt), final_states)

        return final_states
    
    
    def forward_simulate_ibm2_lab(self, us, H0, Hs, initial_states):
        """Forward time evolution: IBM 2-qubit system.
        Args:
            us: pulses.
            H0: a Hermitian matrix.
            Hs: a list of Hermitian matrics.
            initial_states: initial states.
        Returns:
            final_states: final states.
        """
        #self.intermediate_states = []

        n_step = us.shape[0]
        n_param = us.shape[1]
        
        if len(Hs) != 2:
            print("Hs must have 2 elements!")
            
        if n_param != 8:
            print("Num. of parameters must be 8!")

        final_states = initial_states
        sgm = torch.nn.Sigmoid()

        for j in range(n_step):
            t = j * self.dt
            Nd0 = torch.sqrt(torch.sum(torch.square(us[j][0:2])))
            Nu0 = torch.sqrt(torch.sum(torch.square(us[j][2:4])))
            Nd1 = torch.sqrt(torch.sum(torch.square(us[j][4:6])))
            Nu1 = torch.sqrt(torch.sum(torch.square(us[j][6:8])))
            Hjdt = H0.clone()
            Hjdt += ((2*sgm(Nd0)-1) / Nd0 * (np.cos(self.w0 * t) * us[j][0] - np.sin(self.w0 * t) * us[j][1])
                     + (2*sgm(Nu0)-1) / Nu0 * (np.cos(self.w1 * t) * us[j][2] - np.sin(self.w1 * t) * us[j][3])) * Hs[0].clone()
            Hjdt += ((2*sgm(Nd1)-1) / Nd1 * (np.cos(self.w1 * t) * us[j][4] - np.sin(self.w1 * t) * us[j][5])
                     + (2*sgm(Nu1)-1) / Nu1 * (np.cos(self.w0 * t) * us[j][6] - np.sin(self.w0 * t) * us[j][7])) * Hs[1].clone()
            final_states = torch.matmul(torch.matrix_exp(Hjdt), final_states)

        return final_states

    def save_plot(self, plot_name, us):
        return
    

    def get_inner_product_2D(self,psi1,psi2):
        """compute the inner product |psi1.dag() * psi2|.
        Args:
            psi1: a state.
            psi2: a psi2.
        Returns:
            norm: norm of the complex number.
        """
        state_dim = int(psi1.shape[0] / 2)
        psi_1_real = psi1[0:state_dim,:]
        psi_1_imag = psi1[state_dim:2*state_dim,:]
        psi_2_real = psi2[0:state_dim,:]
        psi_2_imag = psi2[state_dim:2*state_dim,:]

        ac = (psi_1_real * psi_2_real).sum(0) 
        bd = (psi_1_imag * psi_2_imag).sum(0) 
        bc = (psi_1_imag * psi_2_real).sum(0) 
        ad = (psi_1_real * psi_2_imag).sum(0) 
        reals = (ac + bd).sum()**2
        imags = (bc - ad).sum()**2
        norm = (reals + imags) / (psi1.shape[1]**2)
        return norm
    
    
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
        #initial_stddev =  (1. / np.sqrt(n_step))
        initial_stddev = 2
        u = np.random.normal(initial_mean, initial_stddev, [n_step ,n_Hs])
        
        return u

    
    #====================================================================
    # Training methods
    #====================================================================
    def ibm_single_plus(self, duration):
        """synthesizing the plus state on the IBM machine
        """
        self.logger.write_text("ibm_single_plus========")

        n_step = self.n_step
        T = duration * 0.22
        self.dt = T / self.n_step
        self.w0 = 5236376147.050786 * 2 * np.pi * 1e-9
        eps0 = 32901013497.991684 * 1e-9
        Omega0 = 955111374.7779446 * 1e-9

        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])
        
        H0 = 0.5 * eps0 * (I-Z)
        Hs = [Omega0 * X]

        g = np.array([1., 0.])
        e = np.array([0., 1.])
        plus = 1/np.sqrt(2) * (g+e)
        
        psi_in = [g]
        psi_out = [plus]

        target_states = [self.c_to_r_vec(v) for v in psi_out]
        initial_states = [self.c_to_r_vec(v) for v in psi_in]

        init_u = self.random_initialize_u(n_step, 2*len(Hs))
        final_u = self.train_fidelity_ibm1_lab(init_u, H0, Hs, initial_states, target_states)
    
    
    def ibm_X(self, duration):
        """synthesizing the plus state on the IBM machine
        """
        self.logger.write_text("ibm_X========")

        n_step = self.n_step
        T = duration * 0.22
        self.dt = T / self.n_step
        self.w0 = 5236376147.050786 * 2 * np.pi * 1e-9
        eps0 = 32901013497.991684 * 1e-9
        Omega0 = 955111374.7779446 * 1e-9

        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])
        
        H0 = 0.5 * eps0 * (I-Z)
        Hs = [Omega0 * X]

        g = np.array([1., 0.])
        e = np.array([0., 1.])
        plus = 1/np.sqrt(2) * (g+e)
        
        psi_in = [g, e, plus]
        psi_out = [e, g, plus]

        target_states = [self.c_to_r_vec(v) for v in psi_out]
        initial_states = [self.c_to_r_vec(v) for v in psi_in]

        init_u = self.random_initialize_u(n_step, 2*len(Hs))
        final_u = self.train_fidelity_ibm1_lab(init_u, H0, Hs, initial_states, target_states)
    
        
        
    #====================================================================
    def ibm_bell_state(self, duration):
        """synthesizing the bell state on the IBM machine
        """
        self.logger.write_text("ibm_bell_state_lab ========")

        n_step = self.n_step
        T = duration * 0.22
        self.dt = T / self.n_step
        self.w0 = 5236376147.050786 * 2 * np.pi * 1e-9
        self.w1 = 5014084426.228487 * 2 * np.pi * 1e-9
        eps0 = 32901013497.991684 * 1e-9
        eps1 = 31504959831.439907 * 1e-9
        Omega0 = 955111374.7779446 * 1e-9
        Omega1 = 987150040.8532522 * 1e-9
        j01 = 12286377.631357463 * 1e-9

        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])
        
        Hint = j01 * np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]])
        Hsys_1q = 0.5 * (I-Z)
        Id = np.eye(2)
        H0 = eps0 * np.kron(Hsys_1q, Id) + eps1 * np.kron(Id, Hsys_1q) + Hint
        Hs = [Omega0 * kron(X,Id), Omega1 * kron(Id,X)]

        g = np.array([1., 0.])
        e = np.array([0., 1.])
        plus = 1/np.sqrt(2) * (g+e)
        state_in = kron(g,g)
        state_out = 1/np.sqrt(2) * (kron(g,g)+kron(e,e))
        psi_in = [state_in]
        psi_out = [state_out]

        target_states = [self.c_to_r_vec(v) for v in psi_out]
        initial_states = [self.c_to_r_vec(v) for v in psi_in]

        init_u = self.random_initialize_u(n_step, 8)  
        final_u = self.train_fidelity_ibm2_lab(init_u, H0, Hs, initial_states, target_states)
        
        return
    
    
    
    def ibm_CNOT(self, duration):
        """synthesizing the CNOT gate on the IBM machine
        """
        self.logger.write_text("ibm_bell_state_lab ========")

        n_step = self.n_step
        T = duration * 0.22
        self.dt = T / self.n_step
        self.w0 = 5236376147.050786 * 2 * np.pi * 1e-9
        self.w1 = 5014084426.228487 * 2 * np.pi * 1e-9
        eps0 = 32901013497.991684 * 1e-9
        eps1 = 31504959831.439907 * 1e-9
        Omega0 = 955111374.7779446 * 1e-9
        Omega1 = 987150040.8532522 * 1e-9
        j01 = 12286377.631357463 * 1e-9

        I = np.array([[1, 0], 
                    [0, 1]])
        X = np.array([[0, 1], 
                    [1, 0]])
        Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        Z = np.array([[1, 0], 
                    [0, -1]])
        
        Hint = j01 * np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]])
        Hsys_1q = 0.5 * (I-Z)
        Id = np.eye(2)
        H0 = eps0 * np.kron(Hsys_1q, Id) + eps1 * np.kron(Id, Hsys_1q) + Hint
        Hs = [Omega0 * kron(X,Id), Omega1 * kron(Id,X)]
        
        g = np.array([1., 0.])
        e = np.array([0., 1.])
        plus = 1/np.sqrt(2) * (g+e)
        gg = kron(g,g)
        ge = kron(g,e)
        eg = kron(e,g)
        ee = kron(e,e)
        pp = kron(plus,plus)
        psi_in = [gg, ge, eg, ee, pp]
        psi_out = [gg, ge, ee, eg, pp]

        target_states = [self.c_to_r_vec(v) for v in psi_out]
        initial_states = [self.c_to_r_vec(v) for v in psi_in]

        init_u = self.random_initialize_u(n_step, 8)  
        final_u = self.train_fidelity_ibm2_lab(init_u, H0, Hs, initial_states, target_states)
        
        return 

#====================================================================
if __name__ == '__main__':
    grape = Grape(n_step=1200, lr=5e-2, n_epoch=1000)
    # grape.ibm_single_plus(duration=20) # lr = 1e-2, epoch = 200
    # grape.ibm_X(duration=40) # lr = 5e-3, epoch = 200
    # grape.ibm_bell_state(duration=1200) # lr = 5e-2, epoch = 1000
    grape.ibm_CNOT(duration=1200) # lr = 3e-2, epoch = 1200