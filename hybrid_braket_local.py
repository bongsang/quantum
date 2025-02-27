"""
Creating your first Hybrid Job
This tutorial provides an introduction to running hybrid quantum-classical algorithms using PennyLane on Amazon Braket . With Amazon Braket, you gain access to both real quantum devices and scalable classical compute, enabling you to push the boundaries of your algorithm.

In this tutorial, we'll walk through how to create your first hybrid quantum-classical algorithms on AWS. With a single line of code, we'll see how to scale from PennyLane simulators on your laptop to running full-scale experiments on AWS that leverage both powerful classical compute and quantum devices. You'll gain understanding of the hybrid jobs queue, including QPU priority queuing, and learn how to scale classical resources for resource-intensive tasks. We hope these tools will empower you to start experimenting today with hybrid quantum algorithms!

Amazon Braket Hybrid Jobs
Amazon Braket Hybrid Jobs offers a way for you to run hybrid quantum-classical algorithms that require both classical resources and quantum processing units (QPUs). Hybrid Jobs is designed to spin up the requested classical compute, run your algorithm, and release the instances after completion so you only pay for what you use. This workflow is ideal for long-running iterative algorithms involving both classical and quantum resources. Simply package up your code into a single function, create a hybrid job with a single line of code, and Braket will schedule it to run as soon as possible without interruption.

Hybrid jobs have a separate queue from quantum tasks, so once your algorithm starts running, it will not be interrupted by variations in the quantum task queue. This helps your long-running algorithms run efficiently and predictably. Any quantum tasks created from a running hybrid job will be run before any other quantum tasks in the queue. This is particularly beneficial for iterative hybrid algorithms where subsequent tasks depend on the outcomes of prior quantum tasks. Examples of such algorithms include the Quantum Approximate Optimization Algorithm (QAOA), Variational Quantum Eigensolver (VQE), or Quantum Machine Learning (QML). You can also monitor your algorithm's progress in near-real time, enabling you to keep track of costs, budget, or custom metrics such as training loss or expectation values.

Importantly, on specific QPUs, running your algorithm in Hybrid Jobs benefits from parametric compilation. This reduces the overhead associated with the computationally expensive compilation step by compiling a circuit only once and not for every iteration in your hybrid algorithm. This reduces the total runtime for many variational algorithms by up to 10x. For long-running hybrid jobs, Braket automatically uses the updated calibration data from the hardware provider when compiling your circuit to ensure the highest quality results.
"""

# Use Braket SDK Cost Tracking to estimate the cost to run this example
from braket.tracking import Tracker

t = Tracker().start()

"""
Getting started with PennyLane
Let’s setup an algorithm that makes use of both classical and quantum resources. We adapt the PennyLane qubit rotation tutorial.

First, we define a quantum simulator to run the algorithm on. In this example, we will use the Braket local simulator before moving onto a QPU.
"""
import pennylane as qml
from pennylane import numpy as np

device = qml.device("braket.local.qubit", wires=1)

# Now we define a circuit with two rotation gates and measure the expectation value in the Z-basis
@qml.qnode(device)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))


"""
Finally, we create a classical-quantum loop that uses gradient descent to minimize the expectation value.

We add the log_metric function from Braket to record the training progress (see metrics documentation). When running on AWS, log_metric records the metrics in Amazon CloudWatch, which is accessible through the Braket console page or the Braket SDK. When running locally on your laptop, log_metric prints the iteration numbers.
"""
from braket.jobs.metrics import log_metric


def qubit_rotation(num_steps=10, stepsize=0.5):
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    params = np.array([0.5, 0.75])

    for i in range(num_steps):
        # update the circuit parameters
        params = opt.step(circuit, params)
        expval = circuit(params)

        log_metric(metric_name="expval", iteration_number=i, value=expval)

    return params

# To run the entire algorithm, we call the `qubit_rotation`` function to see that it runs correctly.
qubit_rotation(5, stepsize=0.5)

"""
Metrics - timestamp=1739897940.479213; expval=0.38894534132396147; iteration_number=0;
Metrics - timestamp=1739897940.587559; expval=0.12290715413453952; iteration_number=1;
Metrics - timestamp=1739897940.6713548; expval=-0.09181374013482183; iteration_number=2;
Metrics - timestamp=1739897940.746251; expval=-0.2936094099948541; iteration_number=3;
Metrics - timestamp=1739897940.809514; expval=-0.5344079938678081; iteration_number=4;
tensor([0.67679672, 2.32609342], requires_grad=True)
Great! We see the expectation value change with each iteration number and the final parameters were returned as a list. Now, instead of running on our laptop, let’s submit this same function to be run on AWS.
"""