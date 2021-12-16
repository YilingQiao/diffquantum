import numpy as np
import matplotlib.pyplot as plt
from ours_spectral import OurSpectral
from grape import Grape

n_parameters = 6
n_epoch = 200
delta = 1e-3
lr = 2e-2


ours_FD = OurSpectral(n_basis=n_parameters, basis='Legendre', n_epoch=n_epoch)
ours_FD.demo_train_with_FD(delta=delta)
loss_FD = ours_FD.losses_energy

ours_legendre = OurSpectral(n_basis=n_parameters, basis='Legendre', n_epoch=n_epoch)
ours_legendre.demo_energy_qubit2()
loss_legendre = ours_legendre.losses_energy


grape = Grape(taylor_terms=20, n_step=n_parameters, n_epoch=n_epoch)
grape.demo_energy_qubit2()
loss_grape = grape.losses_energy


plt.clf()
plt.plot(np.array(loss_grape) + 2, label='grape')
plt.plot(np.array(loss_FD) + 2, label='FD')
plt.plot(np.array(loss_legendre) + 2, label='Legendre')
plt.legend(loc="upper right")
plt.savefig("{}_{}.png".format(grape.log_dir, 'losses_energy'))
plt.yscale('log')
plt.savefig("{}_{}.png".format(grape.log_dir, 'losses_energy_log'))