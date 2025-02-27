import numpy as np
import pennylane as qml
from braket.jobs import hybrid_job
from braket.jobs.metrics import log_metric

@hybrid_job(device="local:braket/default")
def hybrid_braket_aws(num_steps=1, stepsize=0.5):
    device = qml.device("braket.local.qubit", wires=1)

    @qml.qnode(device)
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    params = np.array([0.5, 0.75])

    for i in range(num_steps):
        # update the circuit parameters
        params = opt.step(circuit, params)
        expval = circuit(params)

        log_metric(metric_name="expval", iteration_number=i, value=expval)

    return params
