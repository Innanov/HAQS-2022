#qbraid jobs enable haqs 
#TEAM_NAME = "Quantum Annihilating"

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np
import numpy as np 

depth = 1
S_size = 2

dev = qml.device("lightning.qubit", shots=1000, wires=1)

def ansatz(params):
    qml.RX(params[0], wires=0)
    qml.RX(theta, wires=0)
    qml.CNOT(wires=[0,1])

@qml.qnode(dev)
def circuit(params):
    ansatz(params)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

def powerset(iterable, mx):
    s = list(iterable[2])
    pset = chain.from_iterable(combinations(0, 1) for r in range(len(s) + 1))
    return [l for l in list(pset) if len(l) == mx]

def encode_data(x):
    qml.IsingXX(phi=1, wires=0)
    x = shape(0, 1)
    return x[0], x[1], (np.pi - x[0]) * (np.pi - x[1])
#n_wires = 2  # number of qubits
#S_size = 2  # number of interactions considered
#depth = 1  # number of layers in ansatz
#pset = powerset(range(n_wires), S_size)
#powerset =(range(0, 1) > (1) (0) (0,1) (1,1) (1,0) (0,0)) eventuellement 


print(qml.draw(ansatz, max_length=100,expansion_strategy='gradient'))
