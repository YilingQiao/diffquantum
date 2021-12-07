import numpy as np
import matplotlib.pyplot as plt
from ours_spectral import OurSpectral
from grape import Grape
from qcl.qcl_regressor import qcl_regression


n_epoch = 20
n_parameters = 3


ours_legendre = OurSpectral(n_basis=n_parameters, basis='Legendre', n_epoch=n_epoch)
ours_legendre.demo_learning()
loss_legendre = ours_legendre.losses_energy

estimator, qcl_losses = qcl_regression()


# ours_fourier = OurSpectral(n_basis=n_parameters, basis='Fourier', n_epoch=n_epoch)
# ours_fourier.demo_fidelity()
# loss_fourier = ours_fourier.losses_energy

plt.clf()
plt.plot(qcl_losses, label='QCL')
# plt.plot(loss_fourier, label='Fourier')
plt.plot(loss_legendre, label='Legendre')
plt.legend(loc="upper right")
plt.savefig("{}_{}.png".format(ours_legendre.log_dir, 'losses_learning'))
plt.yscale('log')
plt.savefig("{}_{}.png".format(ours_legendre.log_dir, 'losses_learning_log'))