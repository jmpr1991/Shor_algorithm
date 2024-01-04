from jmpr_qc_algo import myfunctions
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer

import numpy as np

def test_haversine():
    # Amsterdam to Berlin
    assert myfunctions.haversine(
        4.895168, 52.370216, 13.404954, 52.520008
    ) == 576.6625818456291

def test_qft_jmp():
    # 3 qubits QFT
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    circuit = QuantumCircuit(qr, cr)

    #test case
    circuit.x([0, 2])

    circuit = myfunctions.qft_jmp(circuit, 3)
    # circuit_2.measure(qr,cr)
    print(circuit)

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

