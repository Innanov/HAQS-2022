from itertools import chain, combinations
import pennylane as qml
import pennylane.numpy as np
import numpy as np 
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from pyrsistent import l, pset

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


def powerset(iterable, mx):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    pset = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    return [l for l in list(pset) if len(l) == mx]

def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        
        #need powerset for l
        for i in range(n_wires):
            qml.Rot(params[l,i,0],params[l,i,1],params[l,i,2],wires=i)
           
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
def circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)

def kernel(x1, x2, params):
    return circuit(x1, x2, params)[0, 1]

init_params = random_params(num_wires=2, num_layers=4)

kernel_value = kernel(x1=[0], x2=[1])
print(f"The kernel value between the first and second datapoint is {kernel_value:.3f}")


"""def training( R = 500 , i = 1, j = 1, shot = 1, T = ):
    
    for i in range 

@qml.qnode(dev_kernel)
def kernel(x, z):
    Compute quantum kernel element for two feature vectors.

    Args:
        x : shape (2,) tensor containing one input data vector
        z : shape (2,) tensor containing one input data vector

    

    x_enc = encode_data(x)
    z_enc = encode_data(z)

    for _ in range(S_size):
        for i in range(n_wires):
            qml.Hadamard(wires=i)
        embedding(x_enc)

    for _ in range(S_size):
        qml.adjoint(embedding)(z_enc)
        for i in range(n_wires):
            qml.Hadamard(wires=i)

    return qml.expval(qml.Hermitian(projector, wires=range(n_wires)))

def kernel_ideal(A, B):
    Ideal kernel matrix for sets A and B.
    return np.array([[kernel(a, b) for b in B] for a in A])

k_ideal = kernel_ideal(X_train, X_train)

im = plt.imshow(k_ideal, extent=(0, samples_per_set, 0, samples_per_set))
plt.colorbar(im)
plt.xticks([0, samples_per_set])
plt.yticks([0, samples_per_set])
plt.title("$K$ (ideal)")
plt.show()"""