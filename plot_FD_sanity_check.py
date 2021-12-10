import numpy as np
import matplotlib.pyplot as plt
from ours_spectral import OurSpectral

n_samples = 500
n_basis = 3
delta = 1e-5

ours_spectral = OurSpectral(basis='Legendre', n_basis=n_basis)

grad_ours_MC, grad_finite_diff = ours_spectral.demo_finite_diff(
    n_samples=n_samples, delta=delta, is_MC=True)

grad_ours_integrate, grad_finite_diff = ours_spectral.demo_finite_diff(
    n_samples=n_samples, delta=delta, is_MC=False)

print("basis: ", ours_spectral.basis)
print("n_samples: ", n_samples)
print("finite difference: ")
print(grad_finite_diff)
print("ours MC: ")
print(grad_ours_MC.mean(0))
print("ours integration: ")
print(grad_ours_integrate.mean(0))

a = grad_ours_integrate.mean(0)
b = grad_finite_diff

norm = lambda x: np.sqrt((x**2).sum(1))
cos_value = (a * b).sum(1) / norm(a) / norm(b)
angles = np.arccos(cos_value)
ours_over_FD = norm(a) / norm(b)
print("angles: ", angles)
print("ours_over_FD: ", ours_over_FD)
