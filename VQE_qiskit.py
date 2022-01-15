import numpy as np
from qiskit.opflow import Z, X, I, StateFn
from qiskit.opflow.gradients import Gradient

from qiskit import Aer
from qiskit.utils import QuantumInstance

from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import ADAM, COBYLA, CG, GradientDescent

#Circuit imports
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression

counts = []
values = []
def store_intermediate(eval_count, parameters, mean, std):
    print(mean)
    counts.append(eval_count)
    values.append(mean)

n = 2
d = 1
v = [[[None, None, None] for j in range(d + 1)] for i in range(n)]
for i in range(n) :
    for j in range(d + 1) :
        for k in range(3) :
            v[i][j][k] = Parameter('v{}{}{}'.format(i, j, k))

#hamil =0.011280 * (Z ^ Z) + 0.397936 * (Z ^ I) + 0.397936 * (I ^ Z) + 0.180931 * (X ^ X)

hamil = (-1.052373245772859 * I ^ I) + \
        (0.39793742484318045 * I ^ Z) + \
        (-0.39793742484318045 * Z ^ I) + \
        (-0.01128010425623538 * Z ^ Z) + \
        (0.18093119978423156 * X ^ X)

qr = QuantumRegister(n)
qc = QuantumCircuit(qr)
for j in range(d + 1) :
    if j > 0 :
        qc.rzx(np.pi, 0, 1)
    for i in range(n) :
        qc.rz(v[i][j][0], i)
        qc.rx(v[i][j][1], i)
        qc.rz(v[i][j][2], i)

grad = Gradient(grad_method='param_shift')

qi_sv = QuantumInstance(Aer.get_backend('aer_simulator_statevector'), shots=2)

#opt = CG(maxiter=10)
#opt = COBYLA(maxiter=10)
#opt = ADAM()
opt = GradientDescent(maxiter=10)

vqe = VQE(qc, callback=store_intermediate, optimizer=opt, gradient=grad, quantum_instance=qi_sv)

res = vqe.compute_minimum_eigenvalue(hamil)

#Reference value: -1.85728
