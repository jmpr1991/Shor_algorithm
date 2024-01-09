from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
#from qiskit.tools.visualization import plot_histogram, plot_bloch_multivector

import numpy as np
#import matplotlib.pyplot as plt


def qft_jmp(q_circuit: 'classmethod', list_qubits: 'int' = None) -> 'classmethod':
    """
    Calculate the Quantum Fourier Transform of the input circuit
    and returns the circuit
    :param q_circuit: input circuit
    :param list_qubits: [optional parameter] number of qubits of the QFT circuit
    :return: circuit with the QFT added
    """

    # Create the QFT circuit depending on the list_qubit input
    if list_qubits is None:
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
    if list_qubits is None:
        circuit = q_circuit.compose(qft_circuit, range(num_qubits))
    else:
        circuit = q_circuit.compose(qft_circuit, list_qubits)

    return circuit


def qft_optimized_jmp(q_circuit: 'classmethod', list_qubits: 'int' = None) -> 'classmethod':
    """
    Calculate the Quantum Fourier Transform of the input circuit
    and returns the circuit. This version does not use swap gates
    :param q_circuit: input circuit
    :param list_qubits: [optional parameter] number of qubits of the QFT circuit
    :return: circuit with the QFT added
    """

    # Create the QFT circuit depending on the list_qubit input
    if list_qubits is None:
        num_qubits = q_circuit.num_qubits
        qft_circuit = QuantumCircuit(num_qubits)
    else:
        num_qubits = len(list_qubits)
        qft_circuit = QuantumCircuit(num_qubits)

    # Build the QFT (Nielsen & Chuang section 5.1)
    for qubit_i in range(num_qubits):
        # implement Hadamard gate
        qft_circuit.h(qubit_i)

        # implement control phase gates
        for qubit_j in range(qubit_i+1, num_qubits):
            phase = 2*np.pi/(2**(qubit_j - qubit_i + 1))
            qft_circuit.cp(phase, qubit_j, qubit_i)
        qft_circuit.barrier()

    # Compose the final circuit
    if list_qubits is None:
        circuit = q_circuit.compose(qft_circuit, range(num_qubits))
    else:
        circuit = q_circuit.compose(qft_circuit, list_qubits)

    return circuit


def inverse_qft_jmp(q_circuit: 'classmethod', list_qubits: 'int' = None) -> 'classmethod':

    """
    Calculate the Quantum Fourier Transform inverse of the input circuit
    and returns the circuit. This version does not use swap gates
    :param q_circuit: input circuit
    :param list_qubits: [optional parameter] number of qubits of the QFT circuit
    :return: circuit with the QFT added
    """

    # Create the inverse QFT circuit depending on the list_qubit input
    if list_qubits is None:
        num_qubits = q_circuit.num_qubits
        qft_circuit = QuantumCircuit(num_qubits)
    else:
        num_qubits = len(list_qubits)
        qft_circuit = QuantumCircuit(num_qubits)

    # Build the QFT (Nielsen & Chuang section 5.1)
    for qubit_i in range(num_qubits):
        # implement Hadamard gate
        qft_circuit.h(qubit_i)

        # implement control phase gates
        for qubit_j in range(qubit_i + 1, num_qubits):
            phase = 2 * np.pi / (2 ** (qubit_j - qubit_i + 1))
            qft_circuit.cp(phase, qubit_j, qubit_i)
        qft_circuit.barrier()

    # inverse the QFT circuit
    inverse_qft_circuit = qft_circuit.inverse()

    # Compose the final circuit
    if list_qubits is None:
        circuit = q_circuit.compose(inverse_qft_circuit, range(num_qubits))
    else:
        circuit = q_circuit.compose(inverse_qft_circuit, list_qubits)

    return circuit
