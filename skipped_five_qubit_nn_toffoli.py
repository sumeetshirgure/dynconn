import numpy as np
import itertools

from bqskit.ir.gates.constant.cx import CNOTGate
# from dc_instantiation import generate_measurement_operators
# from bqskit.qis import UnitaryMatrix
from bqskit import Circuit
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import ZGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates import CCXGate, CXGate, RZZGate, RXXGate, RYYGate, CZGate
from bqskit.ir.gates import SwapGate
from scipy.optimize import minimize
import qiskit

from acdc import unpack_params, get_composite_unitary, cost_function, grad_function

if __name__ == '__main__':

    num_qubits = 5
    # create a target CNOT gate
    target_circuit = Circuit(num_qubits)
    # Long range entanglement gate
    target_circuit.append_gate(CCXGate(), (0, 2, 4))
    target_unitary = target_circuit.get_unitary()
 
    # Base circuit
    qc_main = Circuit(num_qubits)
    for i in range(num_qubits):
        qc_main.append_gate(U3Gate(), i)

    for __ in range(2) :
        qc_main.append_gate(CNOTGate(), (0, 1))
        qc_main.append_gate(CNOTGate(), (2, 3))
        qc_main.append_gate(U3Gate(), 0)
        qc_main.append_gate(U3Gate(), 1)
        qc_main.append_gate(U3Gate(), 2)
        qc_main.append_gate(U3Gate(), 3)

        qc_main.append_gate(CNOTGate(), (1, 2))
        qc_main.append_gate(CNOTGate(), (3, 4))
        qc_main.append_gate(U3Gate(), 1)
        qc_main.append_gate(U3Gate(), 2)
        qc_main.append_gate(U3Gate(), 3)
        qc_main.append_gate(U3Gate(), 4)

    qc_main.append_gate(CNOTGate(), (0, 1))
    qc_main.append_gate(CNOTGate(), (2, 3))
    qc_main.append_gate(U3Gate(), 0)
    qc_main.append_gate(U3Gate(), 1)
    qc_main.append_gate(U3Gate(), 2)
    qc_main.append_gate(U3Gate(), 3)


    # Branch circuits
    ancillas = [1, 3] # we put mid-circuit measurement on q1 and q2
    branch_0 = Circuit(num_qubits) # measurement result 00
    for i in range(num_qubits):
        if i not in ancillas :
            branch_0.append_gate(U3Gate(), i)
    branch_1 = branch_0.copy() # measurement result 01
    branch_2 = branch_0.copy() # measurement result 10
    branch_3 = branch_0.copy() # measurement result 11
    branch_circuits = [branch_0, branch_1, branch_2, branch_3]

    # Get parameters for all the circuits including main and branch circuits
    # Do this by concatenating the parameters for qc_main with those of branch circuits
    num_params = qc_main.num_params + sum(branch_circuit.num_params for branch_circuit in branch_circuits) + 2 * 2 ** len(ancillas) 

    multi_starts = 500
    best_cost = np.inf

    for _ in range(multi_starts):
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=qc_main.num_params)
        gamma = np.random.uniform(low=-np.pi, high=np.pi, size=sum(branch_circuit.num_params for branch_circuit in branch_circuits))
        phi   = np.random.uniform(low=-np.pi, high=np.pi, size=2**len(ancillas))
        alpha = np.random.uniform(low=0, high=1, size=2**len(ancillas))

        initial_params = np.concat([theta, gamma, phi, alpha])

        result = minimize(cost_function, initial_params, method='BFGS',
                          jac=grad_function,
                          args=(qc_main, branch_circuits, ancillas, target_unitary),
                          )
        cost = cost_function(result.x, qc_main, branch_circuits, ancillas, target_unitary)
        # grad = grad_function(result.x, qc_main, branch_circuits, ancillas, target_unitary)
        # print('---cost---', cost, grad, np.linalg.norm(grad))
        print(_, '---cost---', cost)
        if np.abs(cost) <= 1e-6:
            best_cost = cost
            best_params = result.x
            break

    theta, gamma, phi, alpha = unpack_params(best_params, qc_main, branch_circuits, ancillas)
    print(theta, gamma, phi, alpha)

    # Verify synthesized unitary
    U = get_composite_unitary(qc_main, branch_circuits, ancillas, theta, gamma)

    UUd = U @ np.matrix.getH(U)
    print(np.allclose(np.eye(2**num_qubits), UUd)) # Assert U is unitary

    ket = np.zeros(2**num_qubits).reshape((2**num_qubits, 1))
    ket[21, 0] = 1
    bra = np.matrix.getH(ket)
    dm = ket @ bra # |10101><10101|

    fdm = U @ dm @ np.matrix.getH(U)
    rfdm = qiskit.quantum_info.partial_trace(fdm, ancillas)
    print(rfdm)

    edm = target_unitary.numpy @ dm @ np.matrix.getH(target_unitary.numpy)
    redm = qiskit.quantum_info.partial_trace(edm, ancillas)
    print(redm)

