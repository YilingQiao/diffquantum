from ours_spectral import OurSpectral
import qutip as qp
import numpy as np
import torch

import cma
import scipy


# for duration in durations:
#     model = QubitControl(
#         basis='Legendre', n_basis=8 , dt=0.22, duration=duration, num_sample=6, solver=0,
#         per_step=100, n_epoch=4000, lr = 1e-2)
#     model.demo_H2()

class SimEnv:
    def __init__(self, DFO_Method="DFO", duration=720, n_qubit=2):

        self.model = OurSpectral(basis='Legendre', n_basis=6, n_epoch=200, method_name=DFO_Method)

        self.model.logger.write_text("!!!!{} ========".format(DFO_Method))
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
        H_cost = OO
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


        Hs = []
        # H = OurSpectral.multi_kron(*[O for j in range(n_qubit)])
        for e in graph:
            H = OurSpectral.multi_kron(*[I if j not in e else Z for j in range(n_qubit)]) 
            Hs.append(H)

        for i in range(n_qubit):
            H = OurSpectral.multi_kron(*[I if j not in [i] else X for j in range(n_qubit)])
            Hs.append(H)

        Hs = [qp.Qobj(H) for H in Hs]

        H_cost = qp.Qobj(H_cost)
        H0 = qp.Qobj(H0)
        superposition = qp.Qobj(superposition)

        self.M = H_cost
        self.H0 = H0
        self.Hs = Hs
        self.initial_state = superposition
        self.n_qubit = n_qubit
        self.n_Hs = len(self.Hs)
        self.n_basis = self.model.n_basis
        self.n_step = 0

    def run_with_coeff(self, coeff):

        coeff = coeff.reshape([self.n_Hs, self.n_basis]) 

        self.model.spectral_coeff = torch.tensor(coeff, requires_grad=True)

        lr = self.model.lr
        w_l2 = 0.
        I = qp.qeye(2)
        ts = np.linspace(0, 1, self.model.n_step) 

        H = [self.H0]
        for i in range(self.n_Hs):
            H.append([self.Hs[i], 
                self.model.generate_u(i, self.model.spectral_coeff.detach().numpy())])

        result = qp.mesolve(H, self.initial_state, ts)
        final_state = result.states[-1]
        loss_energy = self.M.matrix_element(final_state, final_state).real - self.M.eigenenergies()[0]

        st = "epoch: {:04d}, loss: {}, loss_energy: {}".format(
            self.n_step, 
            loss_energy, 
            loss_energy
        )
        self.model.logger.write_text(st)
        # if self.n_step > 205:
        #     exit()
        self.n_step += 1
        return loss_energy

def train_cmaes():
    env = SimEnv(DFO_Method="CMAES")
    n_qubit = env.n_qubit
    state_upper_bounds = np.ones([env.n_Hs * env.n_basis]) * 5.0
    state_lower_bounds = np.ones([env.n_Hs * env.n_basis]) * -5.0
    state_std = np.ones([env.n_Hs * env.n_basis]) * 2
    est_init_state = np.random.normal(0, 1e-3, [env.n_Hs * env.n_basis]) 
    optim_loss = 0.


    opts = cma.CMAOptions()
    opts.set('tolfun', optim_loss)
    opts.set('bounds', [state_lower_bounds, state_upper_bounds])
    opts.set('verb_filenameprefix', "./logs/")
    opts.set('CMA_stds', state_std)
    sigma = 1
    es = cma.CMAEvolutionStrategy(est_init_state, sigma, opts)

    step = 0
    while True:
        solutions = es.ask()
        es.tell(solutions, [env.run_with_coeff(x) for x in solutions])
        es.logger.add()
        es.disp()
        step += 1
        if es.result.fbest < optim_loss or step == 20:
            est_init_state = es.result.xbest
            break

def train_nelder_mead():
    DFO_Method="Nelder-Mead"
    env = SimEnv(DFO_Method=DFO_Method)
    n_qubit = env.n_qubit

    # coeff = np.random.normal(0, 1e-3, [self.n_Hs  *self.n_basis]) 
    est_init_state = np.random.normal(0, 1e-3, [env.n_Hs * env.n_basis]) 
    scipy.optimize.minimize(env.run_with_coeff, est_init_state, method=DFO_Method,
        options={'maxiter': 5})
    

def train_SLSQP():
    DFO_Method="SLSQP"
    env = SimEnv(DFO_Method=DFO_Method)
    n_qubit = env.n_qubit
    # coeff = np.random.normal(0, 1e-3, [self.n_Hs  *self.n_basis]) 
    est_init_state = np.random.normal(0, 1e-3, [env.n_Hs * env.n_basis]) 
    scipy.optimize.minimize(env.run_with_coeff, est_init_state, method=DFO_Method,
        options={'maxiter': 5})
    


if __name__ == '__main__':
    n_repeat = 5
    for i in range(n_repeat):
        train_cmaes()
        train_SLSQP()
        train_nelder_mead()