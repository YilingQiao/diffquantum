from ibmsim_icml22 import QubitControl
import qutip as qp
import numpy as np
import torch

import cma
import scipy

durations = [720]

# for duration in durations:
#     model = QubitControl(
#         basis='Legendre', n_basis=8 , dt=0.22, duration=duration, num_sample=6, solver=0,
#         per_step=100, n_epoch=4000, lr = 1e-2)
#     model.demo_H2()

class SimEnv:
    def __init__(self, DFO_Method="DFO", duration=720, n_qubit=2):

        self.model = QubitControl(
            basis='Legendre', n_basis=8 , dt=0.22, duration=duration, num_sample=6, solver=0,
            per_step=100, n_epoch=4000, lr = 1e-2)
        self.n_qubit = n_qubit
        self.H0, self.Hs = self.model.IBM_H(n_qubit)
        self.I = np.array([[1.+ 0.j, 0], 
                    [0, 1.]])
        self.O = np.array([[0.+ 0.j, 0], 
                    [0, 0.]])
        self.X = np.array([[0 + 0.j, 1], 
                    [1, 0]])
        self.Y = (0+1j) * np.array([[0, -1], 
                            [1, 0]])
        self.Z = np.array([[1.0 + 0.j, 0], 
                    [0, -1.0]])

        g = np.array([1,0])
        e = np.array([0,1])
        self.psi0 = np.kron(g, g)

        self.M = (-1.052373245772859 * np.kron(self.I, self.I)) + \
                (0.39793742484318045 * np.kron(self.I, self.Z)) + \
                (-0.39793742484318045 * np.kron(self.Z, self.I)) + \
                (-0.01128010425623538 * np.kron(self.Z, self.Z)) + \
                (0.18093119978423156 * np.kron(self.X, self.X))

        self.n_step = 0
        self.model.logger.write_text("{} ========".format(DFO_Method))
        self.model.logger.write_text("!!!! train_energy ========")


        vv0 = np.random.rand(2 * self.model.n_basis * self.model.n_funcs)
        
        self.n_vv = 2 * self.model.n_basis * self.model.n_funcs
        

    def run_with_psi0(self, x):
        vv0 = np.reshape(x, [2, self.model.n_funcs, self.model.n_basis])
        self.model.vv = torch.tensor(vv0, requires_grad=True)


        psi0 = qp.Qobj(self.psi0)
        M = qp.Qobj(self.M)
        loss_energy = self.model.compute_energy(self.H0, self.Hs, M, psi0)
        loss = loss_energy
        loss_energy = loss_energy - M.eigenenergies()[0]

        st = "epoch: {:04d}, loss: {}, loss_energy: {}".format(
            self.n_step, 
            loss, 
            loss_energy
        )
        self.model.logger.write_text(st)

        self.n_step += 1
        return loss_energy

def train_cmaes():
    env = SimEnv(DFO_Method="CMAES")
    n_qubit = env.n_qubit
    state_upper_bounds = np.ones([env.n_vv]) * 5.0
    state_lower_bounds = np.ones([env.n_vv]) * -5.0
    state_std = np.ones([env.n_vv]) * 2
    est_init_state = np.ones([env.n_vv]) * 0.001
    optim_loss = 0.


    opts = cma.CMAOptions()
    opts.set('tolfun', optim_loss)
    opts.set('bounds', [state_lower_bounds, state_upper_bounds])
    opts.set('verb_filenameprefix', "./logs/")
    opts.set('CMA_stds', state_std)
    sigma = 1
    es = cma.CMAEvolutionStrategy(est_init_state, sigma, opts)
    while True:
        solutions = es.ask()
        es.tell(solutions, [env.run_with_psi0(x) for x in solutions])
        es.logger.add()
        es.disp()
        if es.result.fbest < optim_loss:
            est_init_state = es.result.xbest
            break

def train_nelder_mead():
    DFO_Method="Nelder-Mead"
    env = SimEnv(DFO_Method=DFO_Method)
    n_qubit = env.n_qubit
    est_init_state = np.ones([env.n_vv]) * 0.001
    scipy.optimize.minimize(env.run_with_psi0, est_init_state, method=DFO_Method)
    

def train_SLSQP():
    DFO_Method="SLSQP"
    env = SimEnv(DFO_Method=DFO_Method)
    n_qubit = env.n_qubit
    est_init_state = np.ones([env.n_vv]) * 0.001
    scipy.optimize.minimize(env.run_with_psi0, est_init_state, method=DFO_Method)
    


if __name__ == '__main__':
    # train_cmaes()
    # train_SLSQP()
    train_nelder_mead()