import numpy as np
import matplotlib.pyplot as plt
from ours_spectral import OurSpectral
from grape import Grape

n_parameters = 8
n_epoch = 100
grape = Grape(taylor_terms=20, n_step=n_parameters, n_epoch=n_epoch)
grape.demo_fidelity()
loss_grape = grape.losses_energy


ours_legendre = OurSpectral(n_basis=n_parameters, basis='Legendre', n_epoch=n_epoch)
ours_legendre.demo_fidelity()
loss_legendre = ours_legendre.losses_energy


ours_fourier = OurSpectral(n_basis=n_parameters, basis='Fourier', n_epoch=n_epoch)
ours_fourier.demo_fidelity()
loss_fourier = ours_fourier.losses_energy

plt.clf()
plt.plot(loss_grape, label='grape')
plt.plot(loss_fourier, label='Fourier')
plt.plot(loss_legendre, label='Legendre')
plt.legend(loc="upper right")
plt.savefig("{}_{}.png".format(grape.log_dir, 'losses_energy'))
plt.yscale('log')
plt.savefig("{}_{}.png".format(grape.log_dir, 'losses_energy_log'))