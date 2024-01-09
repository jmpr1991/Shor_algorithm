from math import radians, cos, sin, asin, sqrt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.tools.visualization import plot_histogram, plot_bloch_multivector

import numpy as np
import matplotlib.pyplot as plt

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points on the
    earth (specified in decimal degrees), returns the distance in
    kilometers.
    All arguments must be of equal length.
    :param lon1: longitude of first place
    :param lat1: latitude of first place
    :param lon2: longitude of second place
    :param lat2: latitude of second place
    :return: distance in kilometers between the two sets of coordinates
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

def qft_jmp(q_circuit: 'classmethod', list_qubits: 'int' = None) -> 'classmethod':
    """
    Calculate the Quantum Fourier Transform of the input circuit
    and returns the circuit
    :param q_circuit: input circuit
    :param list_qubits: [optional parameter] number of qubits of the QFT circuit
    :return: circuit with the QFT added
    """

    # Create the circuit for the QFT depending on the list_qubit input
    if list_qubits==None:
        num_qubits = q_circuit.num_qubits
        qft_circuit = QuantumCircuit(num_qubits)
    else:
        num_qubits = len(list_qubits)
        qft_circuit = QuantumCircuit(num_qubits)

    # Build the QFT (Nielsen & Chuang section 5.1)
    for qubit_i in reversed(range(num_qubits)):
        # implement Hadamard gate
        qft_circuit.h(qubit_i)

        # implement control phase gates
        for qubit_j in (range(qubit_i)):
            phase = 2*np.pi/(2**(qubit_i - qubit_j + 1))
            qft_circuit.cp(phase, qubit_j, qubit_i)
        qft_circuit.barrier()

    # swap qubits
    n_qubits = np.floor(num_qubits/2)
    for qubit_i in range(int(n_qubits)):
        qft_circuit.swap(qubit_i, num_qubits-1-qubit_i)

    # Compose the final circuit
    if list_qubits==None:
        circuit = q_circuit.compose(qft_circuit, range(num_qubits))
    else:
        circuit = q_circuit.compose(qft_circuit, list_qubits)

    return circuit

qubit_list = [0, 1, 2]
qr = QuantumRegister(4)
cr = ClassicalRegister(len(qubit_list))
circuit = QuantumCircuit(qr, cr)

# test case
circuit.x([0, 2])

circuit = qft_jmp(circuit, qubit_list)
print(circuit)

simulator = Aer.get_backend('statevector_simulator')

statevector = simulator.run(circuit).result().get_statevector()
print(statevector)
