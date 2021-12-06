import numpy as np
import matplotlib.pyplot as plt
from ours_spectral import OurSpectral
from maxcut_pennylane import qaoa_maxcut

n_layers = 2
n_parameters = n_layers * 2
n_epoch = 30

_, bit_strings, losses = qaoa_maxcut(n_epoch, n_layers=n_layers)

ours_legendre = OurSpectral(n_basis=n_parameters, basis='Legendre', n_epoch=n_epoch)
# ours_legendre = OurSpectral(n_basis=n_parameters, basis='Fourier', n_epoch=n_epoch)
state, prob = ours_legendre.demo_qaoa_max_cut4()
loss_legendre = ours_legendre.losses_energy

n = np.array(bit_strings).sum()
samples = []
for i in range(len(prob)):
    samples += [i] * int(prob[i] * n)


plt.clf()
plt.plot(losses, label='QAOA')
plt.plot(loss_legendre, label='Legendre')
plt.legend(loc="upper right")
plt.savefig("{}_{}.png".format(ours_legendre.log_dir, 'loss_qaoa'))

xticks = range(0, 16)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = np.arange(0, 17) - 0.5
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Ours")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(samples, bins=bins)
plt.subplot(1, 2, 2)
plt.title("QAOA")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bit_strings, bins=bins)
plt.tight_layout()
plt.savefig("{}_{}.png".format(ours_legendre.log_dir, 'qaoa_hist'))