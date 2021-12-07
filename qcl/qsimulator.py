import numpy as np

#==============================================================================
# Define gates
#==============================================================================
RX = lambda theta: np.array([[np.cos(theta/2.0),-1j*np.sin(theta/2.0)],
                             [-1j*np.sin(theta/2.0),np.cos(theta/2.0)]])
RY = lambda theta: np.array([[np.cos(theta/2.0),-np.sin(theta/2.0)],
                             [np.sin(theta/2.0),np.cos(theta/2.0)]])
RZ = lambda theta: np.array([[np.exp(-1j*theta/2.0),0],
                             [0,np.exp(1j*theta/2.0)]])
I = np.eye(2)
X = np.array([[0.0,1.0],
              [1.0,0.0]])
Z = np.array([[1.0,0.0],
              [0.0,-1.0]])

#==============================================================================
# Define qubit states in computational basis
#==============================================================================
zero = np.array([[1.0],
                 [0.0]])
one = np.array([[0.0],
                [1.0]])

#==============================================================================
# Utilities
#==============================================================================  
def multi_kron(*args):
    ret = np.array([[1.0]])
    for q in args:
        ret = np.kron(ret, q)
    return ret
  
def multi_dot(*args):
    for i, q in enumerate(args):
        if i == 0:
            ret = q
        else:
            ret = np.dot(ret, q)
    return ret

#==============================================================================
# Mini quantum simulator with limited features. Used to speed up simulations.
#==============================================================================    
class Program(object):
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gate = multi_kron(*[I for j in range(self.n_qubits)])    
        
    def inst(self, gate, idx=None):
        if idx is not None:
            self.gate = multi_dot(multi_kron(*[gate if j == idx else I for j in range(self.n_qubits)]), self.gate)
        else:
            self.gate = multi_dot(gate, self.gate)
    
    def __add__(self, add):
        self.gate = multi_dot(add.gate, self.gate)
        return self

    def wavefunction(self):
        wav = multi_kron(*[zero for j in range(self.n_qubits)]) 
        return multi_dot(self.gate, wav)
        
    def expectation(self, operator_programs):
        wav = self.wavefunction()
        exp = []
        for op in operator_programs:
            exp.append(np.conj(wav).T.dot(op.gate.dot(wav)).real[0][0])
        return np.array(exp)
        