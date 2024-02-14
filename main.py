from jmpr_qc_algo import myfunctions


# by running the main file the shor algorithm will be triggered. You can inject some inputs to the function,
# otherwise the function will ask the user to inject the needed parameters to proceed with the computations.
# The function will print some outputs to the user while running.

myfunctions.shor_algo(n=35, a=3, estimator_qubits=6)
