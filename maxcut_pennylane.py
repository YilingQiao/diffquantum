import pennylane as qml
from pennylane import numpy as np

np.random.seed(42)

##############################################################################
# Operators
# ~~~~~~~~~
# We specify the number of qubits (vertices) with ``n_wires`` and
# compose the unitary operators using the definitions
# above. :math:`U_B` operators act on individual wires, while :math:`U_C`
# operators act on wires whose corresponding vertices are joined by an edge in
# the graph. We also define the graph using
# the list ``graph``, which contains the tuples of vertices defining
# each edge in the graph.

n_wires = 4
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]

# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)


# unitary operator U_C with parameter gamma
def U_C(gamma):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


##############################################################################
# We will need a way to convert a bitstring, representing a sample of multiple qubits
# in the computational basis, to integer or base-10 form.

def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)


##############################################################################
# Circuit
# ~~~~~~~
# Next, we create a quantum device with 4 qubits.

dev = qml.device("default.qubit", wires=n_wires, shots=1)

##############################################################################
# We also require a quantum node which will apply the operators according to the
# angle parameters, and return the expectation value of the observable
# :math:`\sigma_z^{j}\sigma_z^{k}` to be used in each term of the objective function later on. The
# argument ``edge`` specifies the chosen edge term in the objective function, :math:`(j,k)`.
# Once optimized, the same quantum node can be used for sampling an approximately optimal bitstring
# if executed with the ``edge`` keyword set to ``None``. Additionally, we specify the number of layers
# (repeated applications of :math:`U_BU_C`) using the keyword ``n_layers``.

pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z, requires_grad=False)


@qml.qnode(dev)
def circuit(gammas, betas, edge=None, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(gammas[i])
        U_B(betas[i])
    if edge is None:
        # measurement phase
        return qml.sample()
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    a = qml.Hermitian(pauli_z_2, wires=edge)
    b = qml.expval(qml.Hermitian(pauli_z_2, wires=edge))
    return b#qml.expval(qml.Hermitian(pauli_z_2, wires=edge))


##############################################################################
# Optimization
# ~~~~~~~~~~~~
# Finally, we optimize the objective over the
# angle parameters :math:`\boldsymbol{\gamma}` (``params[0]``) and :math:`\boldsymbol{\beta}`
# (``params[1]``)
# and then sample the optimized
# circuit multiple times to yield a distribution of bitstrings. One of the optimal partitions
# (:math:`z=0101` or :math:`z=1010`) should be the most frequently sampled bitstring.
# We perform a maximization of :math:`C` by
# minimizing :math:`-C`, following the convention that optimizations are cast as minimizations
# in PennyLane.


def qaoa_maxcut(steps=10, n_layers=1):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = 0
        for edge in graph:
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers))
        # print("{.4f}".format(neg_obj.data), "--")
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params
    losses = []
    for i in range(steps):
        params = opt.step(objective, params)
        loss = -objective(params)
        losses.append(-loss)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -loss))

    # sample measured bitstrings 100 times
    bit_strings = []
    n_samples = 100
    
    for i in range(0, n_samples):
        circuit_result = circuit(params[0], params[1], edge=None, n_layers=n_layers)
        bitstring = bitstring_to_int(circuit_result)
        bit_strings.append(bitstring)

    # print optimal parameters and most frequently sampled bitstring

    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    print("Most frequently sampled bit string is: {:04b}".format(most_freq_bit_string))

    return -objective(params), bit_strings, losses

if __name__ == '__main__':
    # perform qaoa on our graph with p=1,2 and
    # keep the bitstring sample lists
    bitstrings1 = qaoa_maxcut(n_layers=1)[1]
    bitstrings2 = qaoa_maxcut(n_layers=2)[1]

    ##############################################################################
    # In the case where we set ``n_layers=2``, we recover the optimal
    # objective function :math:`C=4`

    ##############################################################################
    # Plotting the results
    # --------------------
    # We can plot the distribution of measurements obtained from the optimized circuits. As
    # expected for this graph, the partitions 0101 and 1010 are measured with the highest frequencies,
    # and in the case where we set ``n_layers=2`` we obtain one of the optimal partitions with 100% certainty.

    import matplotlib.pyplot as plt

    xticks = range(0, 16)
    xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
    bins = np.arange(0, 17) - 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("n_layers=1")
    plt.xlabel("bitstrings")
    plt.ylabel("freq.")
    plt.xticks(xticks, xtick_labels, rotation="vertical")
    plt.hist(bitstrings1, bins=bins)
    plt.subplot(1, 2, 2)
    plt.title("n_layers=2")
    plt.xlabel("bitstrings")
    plt.ylabel("freq.")
    plt.xticks(xticks, xtick_labels, rotation="vertical")
    plt.hist(bitstrings2, bins=bins)
    plt.tight_layout()
    plt.show()