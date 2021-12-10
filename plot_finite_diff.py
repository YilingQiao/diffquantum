import numpy as np
import matplotlib.pyplot as plt
from ours_spectral import OurSpectral

# ours_spectral = OurSpectral(basis='Fourier', n_basis=2)
ours_spectral = OurSpectral(basis='Legendre', n_basis=2)
grad_ours, grad_finite_diff = ours_spectral.demo_finite_diff(n_samples=3000, delta=1e-4)
print("basis: ", ours_spectral.basis)
print("finite difference: ")
print(grad_finite_diff)
print("ours: ")
print(grad_ours.mean(0))