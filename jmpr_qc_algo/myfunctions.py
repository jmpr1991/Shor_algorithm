from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.quantum_info.operators import Operator
from qiskit.tools.visualization import plot_histogram, plot_bloch_multivector
from qiskit.circuit.library import QFT

import numpy as np
import matplotlib.pyplot as plt


def qft_jmp(q_circuit: 'classmethod', list_qubits: 'int' = None, qft_inverse: 'bool' = False) -> 'classmethod':
    """
    Calculate the Quantum Fourier Transform of the input circuit
    and returns the circuit
    :param q_circuit: input circuit
    :param list_qubits: [optional parameter] number of qubits of the QFT circuit
    :param qft_inverse: [optional parameter] apply the inverse of the QFT is it is set to True. It is false by default
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
        for qubit_j in reversed(range(qubit_i)):
            phase = 2*np.pi/(2**(qubit_i - qubit_j + 1))
            qft_circuit.cp(phase, qubit_j, qubit_i)
        qft_circuit.barrier()

    # swap qubits
    n_qubits = np.floor(num_qubits/2)
    for qubit_i in range(int(n_qubits)):
        qft_circuit.swap(qubit_i, num_qubits-1-qubit_i)

    # Make the qft inverse if requested
    if qft_inverse is True:
        qft_circuit = qft_circuit.inverse()

    # Compose the final circuit
    if list_qubits is None:
        circuit = q_circuit.compose(qft_circuit, range(num_qubits))
    else:
        circuit = q_circuit.compose(qft_circuit, list_qubits)

    return circuit


def phase_estimation_jmp(estimator_register_size: 'int', phi_register_size: 'int', phase: 'float') -> 'classmethod':

    # Create the Quantum Circuit
    estimator_register = QuantumRegister(estimator_register_size, name='reg')
    classical_register = ClassicalRegister(estimator_register_size, name='meas')
    phi_register = QuantumRegister(phi_register_size, name='phi')

    circuit = QuantumCircuit(estimator_register, phi_register, classical_register)

    # Initialization of the quantum register with H gates
    circuit.h(estimator_register)
    circuit.x(phi_register)

    # Phase estimation procedure
    step = 0
    for control_qubit in range(estimator_register_size):

        for exponent in range(2**step):
            circuit.cp(2*np.pi * phase, estimator_register[control_qubit], phi_register)

        step = step + 1

    # add the inverse QFT
    phase_estimator_circuit = qft_jmp(circuit, estimator_register, qft_inverse=True)
    phase_estimator_circuit.barrier()

    # Measure the estimation register
    phase_estimator_circuit.measure(estimator_register, classical_register)

    return phase_estimator_circuit


def phase_estimation_operator_jmp(estimator_circuit: 'classmethod', phi_circuit: 'classmethod', operator: 'float') \
        -> 'classmethod':

    # Create the Quantum Circuit
    estimator_qubits = estimator_circuit.num_qubits
    phi_qubits = phi_circuit.num_qubits
    circuit = QuantumCircuit(estimator_qubits + phi_qubits, estimator_qubits)

    circuit = circuit.compose(estimator_circuit, range(estimator_qubits))
    circuit = circuit.compose(phi_qubits, range(estimator_qubits, estimator_qubits + phi_qubits))

    # Initialization of the quantum register with H gates
    circuit.h(range(estimator_qubits))
    circuit.x(range(estimator_qubits, estimator_qubits + phi_qubits))

    # Phase estimation procedure
    step = 0
    for control_qubit in range(estimator_qubits):

        for exponent in range(2**step):
            circuit.cp(2*np.pi * phase, estimator_register[control_qubit], phi_register)

        step = step + 1

    # add the inverse QFT
    phase_estimator_circuit = qft_jmp(circuit, estimator_register, qft_inverse=True)
    phase_estimator_circuit.barrier()

    # Measure the estimation register
    phase_estimator_circuit.measure(estimator_register, classical_register)

    return phase_estimator_circuit


def modular_exponentiation_jmp(number_to_factor: 'int', number_module: 'int') -> 'list':
    """
    This function computes the list of operator of the modular exponentiation to be used in Shor algorithm
    :param number_to_factor: number which is going to be factorized
    :param number_module: module number to be used in the computations
    :return operators: list of operators following the modular exponentiation
    """

    # compute the number of bits of the number to factor
    length = len(bin(number_to_factor)) - 2

    # initialize state parameter
    state = []
    state_binary = []
    initial_state = 1
    state.append(initial_state)
    state_binary.append(bin(initial_state))

    iter = 1
    while state[iter-1] != 1 or iter == 1:
        exponentiate = (number_to_factor * state[iter-1]) % number_module
        state.append(exponentiate)
        state_binary.append(bin(state[iter]))

        iter += 1
    # eliminate the last element of the list. It is redundant (state[last]=1)
    state.pop()
    state_binary.pop()

    # calculate the period
    period = len(state)

    circuit_size = int(np.ceil(np.log2(number_module)))

    # create the equivalent circuit and operators
    circuits = []
    operators = []
    for iter in range(period):
        circuit_iter = QuantumCircuit(circuit_size)

        qubit = 0
        for iter_binary in reversed(range(len(state_binary[iter]))):

            if state_binary[iter][iter_binary] == '1':
                circuit_iter.x(qubit)
            elif state_binary[iter_binary] == 'b':
                break

            circuits.append(circuit_iter)
            qubit += 1

        operators.append(Operator(circuits[iter]))

    return operators, circuits


operators, circuits = modular_exponentiation_jmp(3, 35)
print(type(operators))
print(circuits[0])
