import numpy as np
import matplotlib.pyplot as plt
from ibmsim_icml22 import QubitControl


model = QubitControl(
    basis='Legendre', n_basis=8 , dt=0.22, duration=64, num_sample=200, solver=1)

a, b = model.demo_FD()


a = a.detach().numpy()
b = b.detach().numpy()
a = a[:,0,:]
b = b[:,0,:]
norm = lambda x: np.sqrt((x**2).sum(1))
cos_value = (a * b).sum(1) / norm(a) / norm(b)
angles = np.arccos(cos_value)
ours_over_FD = norm(a) / norm(b)
print("a", a)
print("b", b)
print("cos_value", cos_value)
print("angles: ", angles)
print("ours_over_FD: ", ours_over_FD)
