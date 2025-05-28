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


def unpack_params(params: list, qc: Circuit, branch_circuits: list[Circuit], ancillas) :
    theta = np.array(params[0:qc.num_params])
    gamma = []
    index = qc.num_params
    for bc in branch_circuits :
        gamma.append(np.array(params[index:index+bc.num_params]))
        index += bc.num_params
    phi = np.array(params[-2*2**len(ancillas):-2**len(ancillas)])
    alpha = np.array(params[-2**len(ancillas):])
    return theta, gamma, phi, alpha


def get_composite_unitary(qc: Circuit, branch_circuits: list[Circuit], ancillas, theta, gamma) :
    Ub = qc.get_unitary(theta)
    proj = [np.array([[1., 0.], [0., 0.]]), np.array([[0., 0.], [0., 1.]])]
    Usum = np.zeros_like(Ub)
    base = [[np.eye(2)] if i not in ancillas else proj for i in range(qc.num_qudits)]
    index = 0
    for combination in itertools.product(*base) :
        conditional = combination[0]
        for i in range(1, len(combination)) :
            conditional = np.kron(conditional, combination[i])
        Usum += (branch_circuits[index].get_unitary(gamma[index]) @ conditional)
        index += 1
    U = Usum @ Ub
    return U


def get_parametrized_target(qc, T, ancillas, phi, alpha) :
    norm = np.linalg.norm(alpha)
    braket0 = [np.array([[1., 0.], [0., 0.]]), np.array([[0., 0.], [1., 0.]])]
    base = [[np.eye(2)] if i not in ancillas else braket0 for i in range(qc.num_qudits)]
    index = 0
    W = np.zeros_like(T)
    for combination in itertools.product(*base) :
        string = combination[0]
        for i in range(1, len(combination)) :
            string = np.kron(string, combination[i])
        W += alpha[index] * np.exp(1.j*phi[index]) * string / norm
        index += 1
    V = T @ W
    return V


def cost_function(params: list, qc: Circuit, branch_circuits: list[Circuit], ancillas, T: np.matrix|np.ndarray):
    """
    params: parameters of qc (base circuit before measurement) + parameters of branch circuits
    qc: the base circuits
    branch_circuits: circuits after mid-circuit measurement
    T: target state or unitary
    ancillas: the qubits which we put mid-circuit measurement
    """

    theta, gamma, phi, alpha = unpack_params(params, qc, branch_circuits, ancillas)

    U = get_composite_unitary(qc, branch_circuits, ancillas, theta, gamma)

    V = get_parametrized_target(qc, T, ancillas, phi, alpha)

    cost = 1 - np.abs((V.conj() * U).sum()) / (2**(qc.num_qudits - len(ancillas)))

    print("---cost---", cost, end='\r')
    return cost


def grad_function(params: list, qc: Circuit, branch_circuits: list[Circuit], ancillas, T: np.matrix|np.ndarray):
    """
    params: parameters of qc (base circuit before measurement) + parameters of branch circuits
    qc: the base circuit
    branch_circuits: circuits after mid-circuit measurement
    T: target state or unitary
    ancillas: the qubits which we put mid-circuit measurement
    """
    theta, gamma, phi, alpha = unpack_params(params, qc, branch_circuits, ancillas)

    Ub, dUb = qc.get_unitary_and_grad(theta)

    proj = [np.array([[1., 0.], [0., 0.]]), np.array([[0., 0.], [0., 1.]])]
    Usum = np.zeros_like(Ub)
    base = [[np.eye(2)] if i not in ancillas else proj for i in range(qc.num_qudits)]
    dUbranch = list(branch_circuits[j].get_grad(gamma[j]) for j in range(len(branch_circuits)))
    dUdGammaj = []
    index = 0
    for combination in itertools.product(*base) :
        conditional = combination[0]
        for i in range(1, len(combination)) :
            conditional = np.kron(conditional, combination[i])
        Usum += (branch_circuits[index].get_unitary(gamma[index]) @ conditional)
        dUdGammaj.append(dUbranch[index] @ conditional)
        index += 1
    U = Usum @ Ub
    dUdGamma = (np.concat(dUdGammaj) @ Ub)

    norm = np.linalg.norm(alpha)
    braket0 = [np.array([[1., 0.], [0., 0.]]), np.array([[0., 0.], [1., 0.]])]
    base = [[np.eye(2)] if i not in ancillas else braket0 for i in range(qc.num_qudits)]
    index = 0
    W = np.zeros_like(T)
    dWdPhi = []
    dWdAlpha = []
    for combination in itertools.product(*base) :
        string = combination[0]
        for i in range(1, len(combination)) :
            string = np.kron(string, combination[i])
        Wi = alpha[index] * np.exp(1.j*phi[index]) * string / norm
        W += Wi
        dWdPhi.append( 1.j * phi[index] * Wi )
        index += 1

    V = T @ W
    dWdPhi = np.array(dWdPhi)
    dVdPhi = T @ dWdPhi

    index = 0
    for combination in itertools.product(*base) :
        string = combination[0]
        for i in range(1, len(combination)) :
            string = np.kron(string, combination[i])
        dWdAlpha.append( np.exp(1.j * phi[index]) * string / norm - alpha[index] * W / norm**2 )
        index += 1
    dWdAlpha = np.array(dWdAlpha)
    dVdAlpha = T @ dWdAlpha

    z = (V.conj() * U).sum()
    # cost = 1 - np.abs(z) / (2**(qc.num_qudits - len(ancillas)))
    den = (np.abs(z) * 2**(qc.num_qudits - len(ancillas)))

    dzdTheta = (V.conj() * (Usum @ dUb)).sum(axis=(1, 2))
    dTheta = - (z.real * dzdTheta.real + z.imag * dzdTheta.imag) / den

    dzdGamma = (V.conj() * dUdGamma).sum(axis=(1, 2))
    dGamma = - (z.real * dzdGamma.real + z.imag * dzdGamma.imag) / den

    dzdPhi   = (dVdPhi.conj() * U).sum(axis=(1, 2))
    dPhi   = - (z.real * dzdPhi.real + z.imag * dzdPhi.imag)     / den

    dzdAlpha = (dVdAlpha.conj() * U).sum(axis=(1, 2))
    dAlpha = - (z.real * dzdAlpha.real + z.imag * dzdAlpha.imag) / den

    # print(dTheta.shape, dGamma.shape, dPhi.shape, dAlpha.shape)
    return np.concat([dTheta, dGamma, dPhi, dAlpha])


if __name__ == '__main__':

    num_qubits = 5
    target_circuit = Circuit(num_qubits)
    # Long range entanglement gate
    target_circuit.append_gate(CNOTGate(), (0, 4))
    target_unitary = target_circuit.get_unitary()
  
    # Base circuit
    qc_main = Circuit(num_qubits)
    for i in range(num_qubits):
        qc_main.append_gate(U3Gate(), i)

    for _ in range(1) :
        qc_main.append_gate(CZGate(), (0, 1))
        qc_main.append_gate(U3Gate(), 0)
        qc_main.append_gate(U3Gate(), 1)
        qc_main.append_gate(CZGate(), (2, 3))
        qc_main.append_gate(U3Gate(), 2)
        qc_main.append_gate(U3Gate(), 3)

        qc_main.append_gate(CZGate(), (1, 2))
        qc_main.append_gate(U3Gate(), 1)
        qc_main.append_gate(U3Gate(), 2)
        qc_main.append_gate(CZGate(), (3, 4))
        qc_main.append_gate(U3Gate(), 3)
        qc_main.append_gate(U3Gate(), 4)

    qc_main.append_gate(CZGate(), (0, 1))
    qc_main.append_gate(U3Gate(), 0)
    qc_main.append_gate(U3Gate(), 1)
    qc_main.append_gate(CZGate(), (2, 3))
    qc_main.append_gate(U3Gate(), 2)
    qc_main.append_gate(U3Gate(), 3)

    # Branch circuits
    ancillas = [1, 2, 3]
    branch_0 = Circuit(num_qubits)
    for i in range(num_qubits):
        if i not in ancillas :
            branch_0.append_gate(U3Gate(), i)
    branch_1 = branch_0.copy()
    branch_2 = branch_0.copy()
    branch_3 = branch_0.copy()
    branch_4 = branch_0.copy()
    branch_5 = branch_0.copy()
    branch_6 = branch_0.copy()
    branch_7 = branch_0.copy()
    branch_circuits = [branch_0, branch_1, branch_2, branch_3, branch_4, branch_5, branch_6, branch_7]

    # Get parameters for all the circuits including main and branch circuits
    # Do this by concatenating the parameters for qc_main with those of branch circuits
    num_params = qc_main.num_params + sum(branch_circuit.num_params for branch_circuit in branch_circuits) + 2 * 2 ** len(ancillas) 

    multi_starts = 50
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
        if cost <= 1e-6:
            best_cost = cost
            best_params = result.x
            break

    theta, gamma, phi, alpha = unpack_params(best_params, qc_main, branch_circuits, ancillas)
    print(theta, gamma, phi, alpha)
