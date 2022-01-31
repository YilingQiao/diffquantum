from ibmsim_icml22 import QubitControl

durations = [720]
num_repeat = 3

for i in range(num_repeat):
    for duration in durations:
        model = QubitControl(
            basis='Legendre', n_basis=8 , dt=0.22, duration=duration, num_sample=6, solver=0,
            per_step=100, n_epoch=400, lr = 1e-2)
        model.demo_H2()
