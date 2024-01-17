from jmpr_qc_algo import myfunctions
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import QFT

import numpy as np


def test_qft_jmp_1():
    """
    This test check that the qft_jmp function build a QFT circuit on top of the input circuit. No list of qubits is
    is provided in this test
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

    #test case
    circuit.x([0, 2])

    full_circuit = myfunctions.qft_jmp(circuit,  qft_inverse=True)
    full_circuit_test = circuit.compose(QFT(5, insert_barriers=True, inverse=True))

    # simulation of the state vector
    simulator = Aer.get_backend('statevector_simulator')

    statevector = simulator.run(full_circuit).result().get_statevector()
    statevector_test = simulator.run(full_circuit_test.decompose()).result().get_statevector()

    for i in range(len(statevector_test)):
        assert np.round(statevector[i], 4) == np.round(statevector_test[i], 4)


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
    circuit_size = 3
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
