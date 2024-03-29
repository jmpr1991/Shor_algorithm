from jmpr_qc_algo import myfunctions
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import QFT, TGate, SGate
from qiskit.quantum_info.operators import Operator

import numpy as np


def test_qft_jmp_1():
    """
    This test check that the qft_jmp function build a QFT circuit on top of the input circuit. No list of qubits is
    provided in this test
    """
    qr = QuantumRegister(4)
    cr = ClassicalRegister(4)
    circuit = QuantumCircuit(qr, cr)

    # test case
    circuit.x([0, 2])

    circuit = myfunctions.qft_jmp(circuit)

    full_circuit = myfunctions.qft_jmp(circuit)
    full_circuit_test = circuit.compose(QFT(4, insert_barriers=True))

    simulator = Aer.get_backend('statevector_simulator')

    statevector = simulator.run(full_circuit).result().get_statevector()
    statevector_test = simulator.run(full_circuit_test.decompose()).result().get_statevector()

    for i in range(len(statevector_test)):
        assert np.round(statevector[i], 4) == np.round(statevector_test[i], 4)


def test_qft_jmp_2():
    """
    This test check that the qft_jmp function build a QFT circuit on top of the input circuit. A list of Qubits is
    provided in this test
    """
    qubit_list = [0, 1, 2]
    qr = QuantumRegister(4)
    cr = ClassicalRegister(len(qubit_list))
    circuit = QuantumCircuit(qr, cr)

    # test case
    circuit.x([0, 2])

    circuit = myfunctions.qft_jmp(circuit, qubit_list)
    full_circuit = myfunctions.qft_jmp(circuit, qubit_list)
    full_circuit_test = circuit.compose(QFT(num_qubits=3, insert_barriers=True))

    simulator = Aer.get_backend('statevector_simulator')

    statevector = simulator.run(full_circuit).result().get_statevector()
    statevector_test = simulator.run(full_circuit_test.decompose()).result().get_statevector()

    for i in range(len(statevector_test)):
        assert np.round(statevector[i], 4) == np.round(statevector_test[i], 4)


def test_qft_jmp_3():
    """
    This test check that the qft_jmp function build an inverse QFT circuit on top of the input circuit. No list of qubits
    is provided in this test
    """
    qr = QuantumRegister(5)
    cr = ClassicalRegister(5)
    circuit = QuantumCircuit(qr, cr)

    # test case
    circuit.x([0, 2])

    full_circuit = myfunctions.qft_jmp(circuit,  qft_inverse=True)
    full_circuit_test = circuit.compose(QFT(5, insert_barriers=True, inverse=True))

    # simulation of the state vector
    simulator = Aer.get_backend('statevector_simulator')

    state_vector = simulator.run(full_circuit).result().get_statevector()
    state_vector_test = simulator.run(full_circuit_test.decompose()).result().get_statevector()

    for i in range(len(state_vector_test)):
        assert np.round(state_vector[i], 4) == np.round(state_vector_test[i], 4)


def test_qft_jmp_4():
    """
    This test check that the inverse_qft_jmp function build an inverse QFT circuit on top of the input circuit. A list
    of Qubits is provided in this test
    """
    qubit_list = [0, 1, 2]
    qr = QuantumRegister(5)
    cr = ClassicalRegister(len(qubit_list))
    circuit = QuantumCircuit(qr, cr)

    # test case
    circuit.x([0, 2])

    full_circuit = myfunctions.qft_jmp(circuit, qubit_list, qft_inverse=True)
    full_circuit_test = circuit.compose(QFT(3, insert_barriers=True, inverse=True))
    # simulation of the state vector
    simulator = Aer.get_backend('statevector_simulator')

    statevector = simulator.run(full_circuit).result().get_statevector()
    statevector_test = simulator.run(full_circuit_test.decompose()).result().get_statevector()

    for i in range(len(statevector_test)):
        assert np.round(statevector[i], 4) == np.round(statevector_test[i], 4)


def test_phase_estimation_jmp_1():
    """
    This test check the phase estimation function computes exactly the estimated phase
    """

    # test case phase = 1/8
    circuit_size = 3
    phi_size = 1
    phase = 1/8

    # Build the circuit
    circuit_p = myfunctions.phase_estimation_jmp(circuit_size, phi_size, phase)

    # simulation of the state vector
    simulator = Aer.get_backend('qasm_simulator')
    results = simulator.run(circuit_p).result().get_counts()

    # post-process of the results
    keys = list(results.keys())
    values = list(results.values())
    sorted_values = sorted(values)
    max_index = values.index(sorted_values[-1])

    if len(sorted_values) == 1:
        phase_computed = int(keys[max_index], 2) / 2 ** circuit_size
    else:
        second_max_index = values.index(sorted_values[-2])
        phase_computed = (int(keys[max_index], 2) / 2 ** circuit_size + int(keys[second_max_index], 2)
                          / 2 ** circuit_size) / 2

    assert phase == phase_computed


def test_phase_estimation_jmp_2():
    """
    This test check the phase estimation function computes exactly the estimated phase
    """

    # test case phase = 1/3
    circuit_size = 5
    phi_size = 1
    phase = 1/3

    # Build the circuit
    circuit_p = myfunctions.phase_estimation_jmp(circuit_size, phi_size, phase)

    # simulation of the state vector
    simulator = Aer.get_backend('qasm_simulator')
    results = simulator.run(circuit_p).result().get_counts()

    # post-process of the results
    keys = list(results.keys())
    values = list(results.values())
    sorted_values = sorted(values)
    max_index = values.index(sorted_values[-1])

    if len(sorted_values) == 1:
        phase_computed = int(keys[max_index], 2) / 2 ** circuit_size
    else:
        second_max_index = values.index(sorted_values[-2])
        phase_computed = (int(keys[max_index], 2) / 2 ** circuit_size + int(keys[second_max_index], 2)
                          / 2 ** circuit_size) / 2

    if abs(phase - phase_computed) <= 1/circuit_size**2:
        assert True
    else:
        assert False


def test_phase_estimation_operator_jmp():

    # Build the circuit
    circuit_1 = myfunctions.phase_estimation_operator_jmp(3, Operator(TGate()), 1)
    circuit_2 = myfunctions.phase_estimation_operator_jmp(3, Operator(SGate()), 1)

    phase = 1/3
    operator_test = Operator([[1, 0], [0, np.exp(2 * np.pi * 1j * phase)]])
    circuit_3 = myfunctions.phase_estimation_operator_jmp(3, operator_test, 1)

    # simulation of the state vector
    simulator = Aer.get_backend('qasm_simulator')
    results_1 = simulator.run(circuit_1.decompose(reps=6)).result().get_counts()
    results_2 = simulator.run(circuit_2.decompose(reps=6)).result().get_counts()
    results_3 = simulator.run(circuit_3.decompose(reps=6)).result().get_counts()

    # post-process of the results
    keys_1 = list(results_1.keys())
    values_1 = list(results_1.values())
    sorted_values_1 = sorted(values_1)
    max_index_1 = values_1.index(sorted_values_1[-1])

    phase_computed_1 = int(keys_1[max_index_1], 2) / 2 ** 3

    keys_2 = list(results_2.keys())
    values_2 = list(results_2.values())
    sorted_values_2 = sorted(values_2)
    max_index_2 = values_2.index(sorted_values_2[-1])

    phase_computed_2 = int(keys_2[max_index_2], 2) / 2 ** 3

    keys_3 = list(results_3.keys())
    values_3 = list(results_3.values())
    sorted_values_3 = sorted(values_3)
    max_index_3 = values_3.index(sorted_values_3[-1])
    second_max_index = values_3.index(sorted_values_3[-2])

    phase_computed_3 = (int(keys_3[max_index_3], 2) / 2 ** 3 + int(keys_3[second_max_index], 2)
                      / 2 ** 3) / 2

    assert phase_computed_1 == 1/8, phase_computed_2 == 1/4

    if abs(phase - phase_computed_3) <= 1 / 3**2:
        assert True
    else:
        assert False


def test_draper_adder():
    """
    This function test Draper adder
    """
    # test parameters

    for a in range(10):
        for b in range(10):

            # compute size of the inputs numbers in binary
            a_bin = bin(a)
            b_bin = bin(b)
            a_bit_size = len(a_bin)
            b_bit_size = len(b_bin)

            register_size = max(b_bit_size - 2 + 1, a_bit_size - 2 + 1)

            # build circuit
            circuit, operator, operator_inverse = myfunctions.drappper_adder(a, b)
            circuit = circuit.compose(QFT(register_size, inverse=True, do_swaps=False))
            circuit.measure_all()

            simulator = Aer.get_backend('qasm_simulator')
            results = simulator.run(circuit.decompose(reps=6)).result().get_counts()

            keys = list(results.keys())

            assert int(keys[-1], 2) == a + b


def test_modular_adder():
    """
    This function test the modular adder
    """

    # test parameters
    for n in [2, 4, 7]:
        for a in range(10):
            for b in range(n):

                # recompute in case a>n
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

                # build circuit
                circuit, _, _ = myfunctions.modular_adder(a, b, n)
                circuit = circuit.compose(QFT(register_size, inverse=True, do_swaps=False))
                circuit.measure_all()

                simulator = Aer.get_backend('qasm_simulator')
                results = simulator.run(circuit.decompose(reps=10), shots=10).result().get_counts()

                keys = list(results.keys())

                assert int(keys[-1], 2) == (a + b) % n


def test_controlled_multiplier():
    """
    This function test the controlled multiplier
    """

    # test parameters
    for n in [5, 8]:
        for x in [2, 5]:
            for a in range(0, 10, 3):
                for b in range(0, n, 3):

                    # recompute in case a>n
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

                    # build circuit
                    circuit, _, _ = myfunctions.controlled_multiplier(a, b, n, x)
                    circuit.measure_all()

                    simulator = Aer.get_backend('qasm_simulator')
                    results = simulator.run(circuit.decompose(reps=10), shots=1).result().get_counts()

                    keys = list(results.keys())

                    assert int(keys[-1][0:register_size+1], 2) == (b + a*x) % n


def test_U_a():
    """
    This function test the U_a circuit used for shor algorithm
    """

    # test parameters
    for n in [5, 8]:
        for x in [2, 4]:
            for a in range(1, n, 2):
                b = 0
                # build circuit
                circuit, _ = myfunctions.U_a(a, 0, n, x)
                circuit.measure_all()

                simulator = Aer.get_backend('qasm_simulator')
                results = simulator.run(circuit.decompose(reps=10), shots=1).result().get_counts()

                keys = list(results.keys())

                assert int(keys[-1], 2) == (b + a*x) % n
