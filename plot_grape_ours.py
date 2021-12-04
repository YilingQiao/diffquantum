import numpy as np
import matplotlib.pyplot as plt
from ours_spectral import OurSpectral
from grape import Grape

n_parameters = 4
ours_spectral = OurSpectral(n_basis=n_parameters)
ours_spectral.demo_energy()

grape = Grape(taylor_terms=20, n_step=n_parameters)
grape.demo_energy()

plt.clf()
plt.plot(grape.losses_energy, label='grape')
plt.plot(ours_spectral.losses_energy, label='ours_spectral')
plt.legend(loc="upper right")
plt.savefig("{}{}_{}.png".format(grape.log_dir, grape.log_name, ours_spectral.log_name))