from jmpr_qc_algo import myfunctions
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer

import numpy as np
def test_qft_jmp_1():
    """
    This test check that the qft_jmp function build a QFT circuit on top of the input circuit. No list of qubits is
    is provided in this test
    """
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    circuit = QuantumCircuit(qr, cr)

    #test case
    circuit.x([0, 2])

    circuit = myfunctions.qft_jmp(circuit)

    simulator = Aer.get_backend('statevector_simulator')

    statevector = simulator.run(circuit).result().get_statevector()
    statevecto_test = [3.53553391e-01-8.65956056e-17j,
             -2.50000000e-01-2.50000000e-01j,
              1.08244507e-16+3.53553391e-01j,
              2.50000000e-01-2.50000000e-01j,
             -3.53553391e-01+8.65956056e-17j,
              2.50000000e-01+2.50000000e-01j,
             -1.08244507e-16-3.53553391e-01j,
             -2.50000000e-01+2.50000000e-01j]
    for i in range(len(statevecto_test)):
        assert np.round(statevector[i], 4) == np.round(statevecto_test[i], 4)

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

    simulator = Aer.get_backend('statevector_simulator')

    statevector = simulator.run(circuit).result().get_statevector()
    statevecto_test = [3.53553391e-01-8.65956056e-17j,
             -2.50000000e-01-2.50000000e-01j,
              1.08244507e-16+3.53553391e-01j,
              2.50000000e-01-2.50000000e-01j,
             -3.53553391e-01+8.65956056e-17j,
              2.50000000e-01+2.50000000e-01j,
             -1.08244507e-16-3.53553391e-01j,
             -2.50000000e-01+2.50000000e-01j,
              0.00000000e+00+0.00000000e+00j,
              0.00000000e+00+0.00000000e+00j,
              0.00000000e+00+0.00000000e+00j,
              0.00000000e+00+0.00000000e+00j,
              0.00000000e+00+0.00000000e+00j,
              0.00000000e+00+0.00000000e+00j,
              0.00000000e+00+0.00000000e+00j,
              0.00000000e+00+0.00000000e+00j]
    for i in range(len(statevecto_test)):
        assert np.round(statevector[i], 4) == np.round(statevecto_test[i], 4)
