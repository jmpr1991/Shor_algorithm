from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.quantum_info.operators import Operator
from qiskit.tools.visualization import plot_histogram, plot_bloch_multivector
from qiskit.circuit.library import QFT, TGate, SGate

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
    """
    This function has been built to experiment before creating the function bellow
    :param estimator_register_size:
    :param phi_register_size:
    :param phase:
    :return:
    """

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


def phase_estimation_operator_jmp(estimator_qubits: 'int', operator: 'class list') -> 'classmethod':
    """
    This function computes the phase estimation of the input operator provided a number of qubits for the estimator
    :param estimator_qubits: number of qubits to estimate the phase/eigenvalue
    :param operator: operator to be used in the phase estimation algorithm
    :return: phase_estimation_circuit
    """

    # Create the Quantum Circuit
    phi_qubits = np.log2(operator.dim[0])

    estimator_register = QuantumRegister(estimator_qubits, name='reg')
    classical_register = ClassicalRegister(estimator_qubits, name='meas')
    phi_register = QuantumRegister(phi_qubits, name='phi')

    circuit = QuantumCircuit(estimator_register, phi_register, classical_register)

    # Initialization of the quantum register with H gates
    circuit.h(estimator_register)
    circuit.x(phi_register[0])

    # Phase estimation procedure
    step = 0
    for control_qubit in range(estimator_qubits):

        # create controlled operator
        exponential_operator = operator ** (2 ** control_qubit)
        control_operator = exponential_operator.to_instruction()
        control_operator.name = f"U^2^{control_qubit}"
        control_operator = control_operator.control(1)

        # append the control operator
        circuit.append(control_operator, [control_qubit, phi_register])

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


def plain_adder(a: 'int', b: 'int'):

    # compute size of the inputs numbers in binary
    a_bin = bin(a)
    b_bin = bin(b)
    a_bit_size = len(a_bin)
    b_bit_size = len(b_bin)

    register_size = max(a_bit_size - 2, b_bit_size - 2)

    # build the circuit
    a_register = QuantumRegister(register_size, name='a')
    b_register = QuantumRegister(register_size + 1, name='b')
    carry_register = QuantumRegister(register_size, name='c')

    circuit = QuantumCircuit(a_register, b_register, carry_register)

    # build the numbers inside the circuit
    qubit = 0
    for iter in reversed(range(a_bit_size)):

        if a_bin[iter] == '1':
            circuit.x(a_register[qubit])
        elif a_bin[iter] == 'b':
            break
        qubit += 1

    qubit = 0
    for iter in reversed(range(b_bit_size)):
        if b_bin[iter] == '1':
            circuit.x(b_register[qubit])
        elif b_bin[iter] == 'b':
            break
        qubit += 1

    # create the basic operators for the addition
    carry_circuit = QuantumCircuit(4)
    carry_circuit.ccx(1, 2, 3)
    carry_circuit.cx(1, 2)
    carry_circuit.ccx(0, 2, 3)

    carry_operator = Operator(carry_circuit).to_instruction()
    carry_operator.name = "carry"
    carry_operator_inverse = Operator(carry_circuit.inverse()).to_instruction()
    carry_operator_inverse.name = "carry-I"

    sum_circuit = QuantumCircuit(3)
    sum_circuit.cx(1, 2)
    sum_circuit.cx(0, 2)

    sum_operator = Operator(sum_circuit).to_instruction()
    sum_operator.name = "sum"

    for iter in range(register_size):
        if iter < register_size - 1:
            circuit.append(carry_operator, [carry_register[iter], a_register[iter], b_register[iter], carry_register[iter+1]])
        else:
            circuit.append(carry_operator, [carry_register[iter], a_register[iter], b_register[iter], b_register[iter+1]])

    circuit.cx(a_register[register_size - 1], b_register[register_size - 1])
    circuit.append(sum_operator, [carry_register[register_size - 1], a_register[register_size - 1], b_register[register_size - 1]])

    for iter in reversed(range(register_size -1)):

        circuit.append(carry_operator_inverse, [carry_register[iter], a_register[iter], b_register[iter], carry_register[iter+1]])
        circuit.append(sum_operator, [carry_register[iter], a_register[iter], b_register[iter]])

    plain_adder_operator = Operator(circuit).to_instruction()
    return circuit, plain_adder_operator


def drappper_adder(a: 'int', b: 'int'):
    """
    This function add two integers in the b register using the fourier basis (Draper adder [addition on a Quantum Computer])
    :param a: integer 1
    :param b: integer 2
    :return: circuit and operator with b register with value phi(a + b)
    """

    # compute size of the inputs numbers in binary
    a_bin = bin(a)
    b_bin = bin(b)
    a_bit_size = len(a_bin)
    b_bit_size = len(b_bin)

    register_size = b_bit_size - 2 + 1

    b_register = QuantumRegister(register_size, name='b')
    circuit = QuantumCircuit(b_register)

    # build "b" inside the circuit
    qubit = 0
    for iter in reversed(range(b_bit_size)):
        if b_bin[iter] == '1':
            circuit.x(b_register[qubit])
        elif b_bin[iter] == 'b':
            break
        qubit += 1

    # Apply QFT
    circuit = circuit.compose(QFT(num_qubits=register_size, do_swaps=False))

    # create the adder circuit
    circuit_adder = QuantumCircuit(b_register)
    qubit = 0
    for b_iter in reversed(range(a_bit_size)):

        if a_bin[b_iter] == '1':
            qubit_j = 0
            for i in range(qubit, register_size):
                phase = 2 * np.pi / 2 ** (qubit_j + 1)
                circuit_adder.p(phase, b_register[i])
                qubit_j += 1
        elif a_bin[b_iter] == 'b':
            break

        qubit += 1

    #compose both circuits
    operator_adder = Operator(circuit_adder).to_instruction()
    operator_adder.name = 'D-adder'
    operator_adder_inverse = Operator(circuit_adder.inverse()).to_instruction()
    operator_adder_inverse.name = 'inverse D-adder'
    circuit = circuit.compose(circuit_adder)
    #circuit.append(operator_adder, range(register_size))

    return circuit, operator_adder, operator_adder_inverse


def modular_adder(a: 'int', b: 'int', n: 'int'):
    # compute size of the inputs numbers in binary
    a_bin = bin(a)
    b_bin = bin(b)
    n_bin = bin(n)
    a_bit_size = len(a_bin)
    b_bit_size = len(b_bin)
    n_bit_size = len(n_bin)

    register_size = b_bit_size - 2 + 1

    b_register = QuantumRegister(register_size, name='b')
    c_register = QuantumRegister(2, name='c')
    aux_register = QuantumRegister(1, name='|0>')
    circuit = QuantumCircuit(c_register, b_register, aux_register)

    # build "b" inside the circuit
    qubit = 0
    for iter in reversed(range(b_bit_size)):
        if b_bin[iter] == '1':
            circuit.x(b_register[qubit])
        elif b_bin[iter] == 'b':
            break
        qubit += 1

    # Apply QFT
    circuit = circuit.compose(QFT(num_qubits=register_size, do_swaps=False), b_register)

    # create the modular_adder circuit
    _, operator_adder_a, operator_adder_inverse_a = drappper_adder(a, b)
    control_operator_adder_a = operator_adder_a.control(2)
    control_operator_adder_inverse_a = operator_adder_inverse_a.control(2)
    _, operator_adder_n, operator_adder_inverse_n = drappper_adder(n, b)
    control_operator_adder_n = operator_adder_n.control(1)

    modular_adder_circuit = QuantumCircuit(c_register, b_register, aux_register)
    modular_adder_circuit.x(c_register)
    modular_adder_circuit.append(control_operator_adder_a, [c_register, b_register])
    modular_adder_circuit.append(operator_adder_inverse_n, [b_register])






#circuit, operator = plain_adder(4, 3)
circuit, operator = drappper_adder(1,4)
circuit = circuit.compose(QFT(4, inverse=True, do_swaps=False))
circuit.measure_all()
print(circuit)

simulator = Aer.get_backend('qasm_simulator')
results = simulator.run(circuit.decompose(reps=6)).result().get_counts()
print(results)