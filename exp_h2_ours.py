from ibmsim_icml22 import QubitControl

durations = [360]

for duration in durations:
    model = QubitControl(
        basis='Legendre', n_basis=8 , dt=0.22, duration=duration, num_sample=6, solver=0,
        per_step=100, n_epoch=4000, lr = 1e-2)
    model.demo_H2()
