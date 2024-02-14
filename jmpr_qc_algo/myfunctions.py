from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import QFT

import numpy as np
import pandas as pd
from fractions import Fraction
# import matplotlib.pyplot as plt


def qft_jmp(q_circuit: 'QuantumCircuit', list_qubits: 'QuantumRegister' = None, qft_inverse: 'bool' = False,
            do_swaps: 'bool' = True) -> 'QuantumCircuit':
    """
    Calculate the Quantum Fourier Transform of the input circuit
    and returns the circuit
    :param q_circuit: input circuit
    :param list_qubits: [optional parameter] number of qubits of the QFT circuit
    :param qft_inverse: [optional parameter] apply the inverse of the QFT is it is set to True. It is false by default
    :param do_swaps: [optional parameter] do swaps in the QFT circuit
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
    if do_swaps is True:
        n_qubits = np.floor(num_qubits/2)
        for qubit_i in range(int(n_qubits)):
            qft_circuit.swap(qubit_i, num_qubits-1-qubit_i)

    # Make the qft inverse if requested
    if qft_inverse is True:
        qft_circuit = qft_circuit.inverse()

    # Compose the final circuit
    if list_qubits is None:
        circ = q_circuit.compose(qft_circuit, range(num_qubits))
    else:
        circ = q_circuit.compose(qft_circuit, list_qubits)

    return circ


def phase_estimation_jmp(estimator_register_size: 'int', phi_register_size: 'int', phase: 'float') -> 'QuantumCircuit':
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

    circ = QuantumCircuit(estimator_register, phi_register, classical_register)

    # Initialization of the quantum register with H gates
    circ.h(estimator_register)
    circ.x(phi_register)

    # Phase estimation procedure
    step = 0
    for control_qubit in range(estimator_register_size):

        for exponent in range(2**step):
            circ.cp(2*np.pi * phase, estimator_register[control_qubit], phi_register)

        step = step + 1

    # add the inverse QFT
    phase_estimator_circuit = qft_jmp(circ, estimator_register, qft_inverse=True)
    phase_estimator_circuit.barrier()

    # Measure the estimation register
    phase_estimator_circuit.measure(estimator_register, classical_register)

    return phase_estimator_circuit


def phase_estimation_operator_jmp(estimator_qubits: 'int', operator: 'class list', input_state: 'int') \
        -> 'QuantumCircuit':
    """
    This function computes the phase estimation of the input operator provided a number of qubits for the estimator
    :param input_state:
    :param estimator_qubits: number of qubits to estimate the phase/eigenvalue
    :param operator: operator to be used in the phase estimation algorithm
    :return: phase_estimation_circuit
    """

    # Create the Quantum Circuit
    phi_qubits = np.log2(operator.dim[0])

    estimator_register = QuantumRegister(estimator_qubits, name='reg')
    classical_register = ClassicalRegister(estimator_qubits, name='meas')
    phi_register = QuantumRegister(phi_qubits, name='phi')

    circ = QuantumCircuit(estimator_register, phi_register, classical_register)

    # Initialization of the quantum register with H gates
    circ.h(estimator_register)

    # build input state inside the circuit
    input_state_bin = bin(input_state)
    input_state_size = len(input_state_bin)
    qubit = 0
    for i in reversed(range(input_state_size)):
        if input_state_bin[i] == '1':
            circ.x(phi_register[qubit])
        elif input_state_bin[i] == 'b':
            break
        qubit += 1

    # Phase estimation procedure
    step = 0
    for control_qubit in range(estimator_qubits):

        # create controlled operator
        exponential_operator = operator ** (2 ** control_qubit)
        control_operator = exponential_operator.to_instruction()
        control_operator.name = f"U^2^{control_qubit}"
        control_operator = control_operator.control(1)

        # append the control operator
        circ.append(control_operator, [control_qubit, phi_register])

        step = step + 1

    # add the inverse QFT
    phase_estimator_circuit = qft_jmp(circ, estimator_register, qft_inverse=True)
    phase_estimator_circuit.barrier()

    # Measure the estimation register
    phase_estimator_circuit.measure(estimator_register, classical_register)

    return phase_estimator_circuit


def drappper_adder(a: 'int', b: 'int', n: 'int' = 0):
    """
    This is the controlled multiplier function as described in "circuit for Shor's algorithm using 2n+3 qubits" by
    S. Beauegard. This function add two integers in the b register using the fourier basis (Draper adder [addition on a
    Quantum Computer])
    :param a: integer 1 to be added with 'b'
    :param b: integer 2 to be added with 'a'
    :param n: integer 3 optional parameter to be used to compute the size of the quantum register, useful for shor
    algorithm
    :return: circuit and operators with b register with value phi(a + b)
    """

    # compute size of the inputs numbers in binary
    a_bin = bin(a)
    b_bin = bin(b)
    n_bin = bin(n)
    a_bit_size = len(a_bin)
    b_bit_size = len(b_bin)
    n_bin_size = len(n_bin)

    register_size = max(b_bit_size - 2 + 1, a_bit_size - 2 + 1, n_bin_size - 2 + 1)

    b_register = QuantumRegister(register_size, name='b')
    circ = QuantumCircuit(b_register)

    # build "b" inside the circuit
    qubit = 0
    for i in reversed(range(b_bit_size)):
        if b_bin[i] == '1':
            circ.x(b_register[qubit])
        elif b_bin[i] == 'b':
            break
        qubit += 1

    # Apply QFT
    circ = circ.compose(QFT(num_qubits=register_size, do_swaps=False))

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

    gate_adder = circuit_adder.to_gate()
    gate_adder.name = 'D-adder'
    gate_adder_inverse = (circuit_adder.inverse()).to_gate()
    gate_adder_inverse.name = 'inverse D-adder'

    # compose both circuits
    operator_adder = Operator(circuit_adder).to_instruction()
    operator_adder.name = 'D-adder'
    operator_adder_inverse = Operator(circuit_adder.inverse()).to_instruction()
    operator_adder_inverse.name = 'inverse D-adder'
    circ = circ.compose(circuit_adder)

    return circ, gate_adder, gate_adder_inverse


def modular_adder(a: 'int', b: 'int', n: 'int'):
    """
    This is the modular adder function as described in "circuit for Shor's algorithm using 2n+3 qubits" by
    S. BeauegardThis function add two integers in the b register using the fourier basis (Draper adder) and computed the
    modulo of the result
    :param a: integer 1 to be added with b
    :param b: integer 2 to be added with a
    :param n: integer 3 to be used to compute the module of a+b
    :return: circuit and operators with b register with value (a + b)mod(n)
    """

    # b should be less than n
    if b >= n:
        print("b should be less than n!!")
        assert False

    # recompute a in case a >= n to optimize number of qubits
    if a >= n:
        a = a % n

    # compute size of the inputs numbers in binary
    a_bin = bin(a)
    b_bin = bin(b)
    n_bin = bin(n)
    a_bit_size = len(a_bin)
    b_bit_size = len(b_bin)
    n_bit_size = len(n_bin)

    register_size = max(b_bit_size - 2 + 1, a_bit_size - 2 + 1, n_bit_size - 2 + 1)  # 1 bit extra added

    b_register = QuantumRegister(register_size, name='b')
    aux_register = QuantumRegister(1, name='|0>')
    circ = QuantumCircuit(b_register, aux_register)

    # build "b" inside the circuit
    qubit = 0
    for i in reversed(range(b_bit_size)):
        if b_bin[i] == '1':
            circ.x(b_register[qubit])
        elif b_bin[i] == 'b':
            break
        qubit += 1

    # Apply QFT
    circ = circ.compose(QFT(num_qubits=register_size, do_swaps=False), b_register)
    # TODO: use the qft_jmp function instead the iqskit repository one
    # circuit = circuit.compose(qft_jmp(q_circuit=circuit, list_qubits=b_register, do_swaps=False), b_register)

    # create the modular_adder circuit
    _, operator_adder_a, operator_adder_inverse_a = drappper_adder(a, b, n)
    _, operator_adder_n, operator_adder_inverse_n = drappper_adder(n, b)
    control_operator_adder_n = operator_adder_n.control(1)

    modular_adder_circuit = QuantumCircuit(b_register, aux_register)
    modular_adder_circuit.append(operator_adder_a, b_register)
    modular_adder_circuit.append(operator_adder_inverse_n, b_register)
    modular_adder_circuit = modular_adder_circuit.compose(QFT(num_qubits=register_size, inverse=True, do_swaps=False),
                                                          b_register)
    modular_adder_circuit.cx(b_register[-1], aux_register)
    modular_adder_circuit = modular_adder_circuit.compose(QFT(num_qubits=register_size, do_swaps=False), b_register)
    modular_adder_circuit.append(control_operator_adder_n, aux_register[:] + b_register[:])
    modular_adder_circuit.append(operator_adder_inverse_a, b_register)
    modular_adder_circuit = modular_adder_circuit.compose(QFT(num_qubits=register_size, inverse=True, do_swaps=False),
                                                          b_register)
    modular_adder_circuit.x(b_register[-1])
    modular_adder_circuit.cx(b_register[-1], aux_register)
    modular_adder_circuit.x(b_register[-1])
    modular_adder_circuit = modular_adder_circuit.compose(QFT(num_qubits=register_size, do_swaps=False), b_register)
    modular_adder_circuit.append(operator_adder_a, b_register)

    # create gates to be used in subsequent function to support shor algorithm
    modular_adder_gate = modular_adder_circuit.to_gate()
    modular_adder_gate.name = 'mod_adder'

    modular_adder_gate_inverse = (modular_adder_circuit.inverse()).to_gate()
    modular_adder_gate_inverse.name = 'inverse_mod_adder'

    # compose the final circuit
    circ.append(modular_adder_gate, b_register[:] + aux_register[:])

    return circ, modular_adder_gate, modular_adder_gate_inverse


def controlled_multiplier(a: 'int', b: 'int', n: 'int', x: 'int'):
    """
    This is the controlled multiplier function as described in "circuit for Shor's algorithm using 2n+3 qubits" by
    S. Beauegard. This gate take 3 inputs |x>, |b>, 'a' and 'n', and return |x>|(b+ax)mod(n)>
    :param a: integer 1 to be added with b
    :param b: integer 2 to be added with ax
    :param n: integer 3 modulo to compute
    :param x: integer 4 integer to multiply by a
    :return: circuit and operators  with value (b + ax)mod(n)
    """

    # recompute 'a' in case a >= n to optimize number of qubits
    if a >= n:
        a_effective = a % n
    else:
        a_effective = a

    # compute size of the inputs numbers in binary
    a_bin = bin(a_effective)
    b_bin = bin(b)
    n_bin = bin(n)
    x_bin = bin(x)
    a_bit_size = len(a_bin)
    b_bit_size = len(b_bin)
    n_bit_size = len(n_bin)
    x_bit_size = len(x_bin)

    # one additional qubit added to the register
    register_size = max(b_bit_size - 2 + 1, a_bit_size - 2 + 1, n_bit_size - 2 + 1)

    b_register = QuantumRegister(register_size, name='b')
    # x_register has the same size as b_register to make easier future computations
    x_register = QuantumRegister(register_size, name='x')
    aux_register = QuantumRegister(1, name='|0>')
    circ = QuantumCircuit(x_register, b_register, aux_register)

    # build "x" inside the circuit
    qubit = 0
    for i in reversed(range(x_bit_size)):
        if x_bin[i] == '1':
            circ.x(x_register[qubit])
        elif x_bin[i] == 'b':
            break
        qubit += 1

    # build "b" inside the circuit
    qubit = 0
    for i in reversed(range(b_bit_size)):
        if b_bin[i] == '1':
            circ.x(b_register[qubit])
        elif b_bin[i] == 'b':
            break
        qubit += 1

    # build the multiplier circuit
    control_multiplier = QuantumCircuit(x_register, b_register, aux_register)
    control_multiplier = control_multiplier.compose(QFT(num_qubits=register_size, inverse=False, do_swaps=False),
                                                    b_register)

    for i in range(register_size):
        # compute the modular adder operator
        _, modular_adder_circuit, _ = modular_adder(2**i * a, b, n)

        control_modular_adder_circuit = modular_adder_circuit.control(1)

        control_multiplier.append(control_modular_adder_circuit, [i, *range(register_size, 2 * register_size + 1)])  # x_bit_size -2

    control_multiplier = control_multiplier.compose(QFT(num_qubits=register_size, inverse=True, do_swaps=False),
                                                    b_register)

    # convert circuits to gates
    control_multiplier_gate = control_multiplier.to_gate()
    control_multiplier_gate.name = 'C-mult(a)'
    control_multiplier_gate_inverse = (control_multiplier.inverse()).to_gate()
    control_multiplier_gate_inverse.name = 'C-mult(a)-inv'

    # compose the final circuit
    circ = circ.compose(control_multiplier, x_register[:] + b_register[:] + aux_register[:])

    return circ, control_multiplier_gate, control_multiplier_gate_inverse


def U_a(a: 'int', b: 'int', n: 'int', x: 'int'):
    """
    This is the controlled U_a function as described in "circuit for Shor's algorithm using 2n+3 qubits" by
    S. Beauegard. This gate take 3 inputs |x>, |b> (0 for show algo), 'a' and 'n', and return |(ax)mod(n)>
    :param a: integer 1 to be added with b
    :param b: integer 2 to be added with ax
    :param n: integer 3 modulo to compute
    :param x: integer 4 integer to multiply by a
    :return: circuit and operator with value (ax)mod(n)
    """

    # recompute 'a' in case a >= n to optimize number of qubits
    if a >= n:
        a_effective = a % n
    else:
        a_effective = a

    # compute size of the inputs numbers in binary
    a_bin = bin(a_effective)
    b_bin = bin(b)
    n_bin = bin(n)
    x_bin = bin(x)
    a_bit_size = len(a_bin)
    b_bit_size = len(b_bin)
    n_bit_size = len(n_bin)
    x_bit_size = len(x_bin)

    # one additional qubit added to the register
    register_size = max(b_bit_size - 2 + 1, a_bit_size - 2 + 1, n_bit_size - 2 + 1)

    b_register = QuantumRegister(register_size, name='b')
    x_register = QuantumRegister(register_size, name='x')
    aux_register = QuantumRegister(1, name='|0>')
    circ = QuantumCircuit(x_register, b_register, aux_register)

    # build "x" inside the circuit
    qubit = 0
    for i in reversed(range(x_bit_size)):
        if x_bin[i] == '1':
            circ.x(x_register[qubit])
        elif x_bin[i] == 'b':
            break
        qubit += 1

    # build "b" inside the circuit
    qubit = 0
    for i in reversed(range(b_bit_size)):
        if b_bin[i] == '1':
            circ.x(b_register[qubit])
        elif b_bin[i] == 'b':
            break
        qubit += 1

    # build the U circuit
    U_a = QuantumCircuit(x_register, b_register, aux_register)

    _, c_mult, c_mult_inv = controlled_multiplier(a, b, n, x)

    U_a.append(c_mult, x_register[:] + b_register[:] + aux_register[:])

    # apply swap gate
    for i in range(register_size):
        U_a.swap(x_register[i], b_register[i])

    # last step after swap
    a_inv = pow(a, -1, n)
    _, _, c_mult_inv = controlled_multiplier(a_inv, x, n, (b + a * x) % n)
    U_a.append(c_mult_inv, x_register[:] + b_register[:] + aux_register[:])

    # convert to gate
    u_gate = U_a.to_gate()
    u_gate.name = 'U_a'

    circ = circ.compose(U_a, x_register[:] + b_register[:] + aux_register[:])

    return circ, u_gate


def shor_algo(n: 'int' = None, a: 'int' = None, estimator_qubits: 'int' = None):

    # request input if needed
    if n is None:
        print("Inject the number you want to factorize. "
              "Remember to introduce a number which is the product of 2 primes (if test type 15)")
        n = int(input())
        print("")

    if n % 2 == 0:
        print("the number is multiple of 2")
        exit()

    if a is None:
        print("Inject a number less than the number you want to factorize and with no common factors with it "
              "(commonly called in literature a) (if test type 7)")
        a = int(input())
        print("")

    if estimator_qubits is None:
        print("Inject the number of qubits you want to use to estimate the factor. "
              "Please note that the more qubits injected the more precision and the more computation time "
              "(if test type 4)")
        estimator_qubits = int(input())
    print("Computation started. Please wait...")

    # compute the modular exponentiation gate using predefined functions
    _, u_a_gate = U_a(a, 0, n, 1)
    phi_qubits = u_a_gate.num_qubits

    # create quantum registers for the quantum circuit
    estimator_register = QuantumRegister(estimator_qubits, name='est')
    phi_register = QuantumRegister(phi_qubits, name='phi')
    classic_bits = ClassicalRegister(estimator_qubits)

    # initialize circuit
    circ = QuantumCircuit(estimator_register, phi_register, classic_bits)
    circ.h(estimator_register)
    circ.x(phi_register[0])

    # Phase estimation procedure
    qubit_count = 0
    for control_qubit in range(estimator_qubits):

        # create controlled operator
        _, u_a_gate = U_a(a ** (2 ** qubit_count), 0, n, 1)
        u_a_gate.name = f'U_{a}^{2 ** qubit_count}'
        u_a_gate_control = u_a_gate.control(1)
        circ.append(u_a_gate_control, [control_qubit, *range(estimator_qubits, estimator_qubits + phi_qubits)])

        qubit_count += 1

    # add the inverse QFT
    circ = circ.compose(QFT(num_qubits=estimator_qubits, inverse=True), estimator_register)

    # measure the estimators qubits
    circ.measure(estimator_register, classic_bits)
    print(circ)
    print("")
    print("The period is being computed now. Please wait...")

    # compute results
    simulator = Aer.get_backend('qasm_simulator')
    counts = simulator.run(circ.decompose(reps=6), shots=500).result().get_counts()
    print(counts)

    # The core of the code bellow for computing the period is imported from qiskit textbook - shor section
    # (https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/shor.ipynb)
    rows, measured_phases = [], []
    for output in counts:
        decimal = int(output, 2)  # Convert (base 2) string to decimal
        phase = decimal / (2 ** estimator_qubits)  # Find corresponding eigenvalue
        measured_phases.append(phase)
        # Add these values to the rows in our table:
        rows.append([f"{output}(bin) = {decimal:>3}(dec)",
                     f"{decimal}/{2 ** estimator_qubits} = {phase:.2f}"])
    # Print the rows in a table
    headers = ["Register Output", "Phase"]
    df = pd.DataFrame(rows, columns=headers)
    print(df)

    rows, r = [], []
    for phase in measured_phases:
        frac = Fraction(phase).limit_denominator(int(np.floor(n/2)))
        rows.append([phase,
                     f"{frac.numerator}/{frac.denominator}",
                     frac.denominator])
        r.append(int(frac.denominator))
    # Print as a table
    headers = ["Phase", "Fraction", "Guess for r"]
    df = pd.DataFrame(rows, columns=headers)
    print(df)
    print("")

    # compute period
    period = 0
    for i in range(len(r)):
        if a**r[i] % n == 1 and r[i] > 0:
            print(f"the period is {r[i]}")
            period = r[i]
            break
    print("")
    print("The factors are being computed now. Please wait...")

    if period % 2 != 0:
        print("period is not even number. "
              "Please run the program with another value of a until obtaining an even number of r")
        shor_algo(n=n, estimator_qubits=estimator_qubits)
        exit()
    elif period == 0:
        print("Computation went wrong. Please relaunch the program again injecting more qubits in the estimation")
        exit()

    # compute factors
    factor1 = np.gcd(a ** int(period / 2) - 1, n)
    factor2 = np.gcd(a ** int(period / 2) + 1, n)
    print(f"The factor of {n} are {factor1} and {factor2}")

    return

