from itertools import chain, combinations
import pennylane as qml
import pennylane.numpy as np
import numpy as np 
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


    ##/!\Remember replace default.qubit to braket.aws.qubit if you want to run on a quantum device/!\##
#1
dev_kernel = qml.device("default.qubit", wires=1)

n_wires = 2

#2nd line 
projector = np.zeros((2**n_wires, 2**n_wires))
projector[0, 0] = 1

def layer(x, params, wires, i0=1, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[1, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])


def ansatz(x, params, wires, iterable):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        j = np.random.randint(3)
        s = list(iterable)
        pset = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)) 
    
    for l in range(np.array(params).shape[0]):
        #need powerset for l
        for i in range(n_wires):
            qml.Rot(params[l,i,0],params[l,i,1],wires=i)
        for s in pset:
            qml.CZ(wires=s)
                
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, num_wires))

# Circuit 

dev = qml.device("default.qubit", wires=1, shots=1042)
wires = dev.wires.tolist()

@qml.qnode(dev)
def circuit(x1, x2 ,iterable ,params):

    #not finished
    ansatz(x1, iterable, params, wires=0)
    adjoint_ansatz(x2, params, wires=1)
    return qml.probs(wires=[0,1])

def kernel(x1, x2, params):
    return circuit(x1, x2, params)[0, 1]

init_params = random_params(num_wires=2, num_layers=4)

X = [0, 1]

kernel_value = kernel(X[0], X[1], init_params)
print(f"The kernel value between the first and second datapoint is {kernel_value:.3f}")
