# Introduction 

This is my implementation of the famous Shor´s algorithm  using modular exponentiation, as described in [**"Circuit 
for Shor’s algorithm using 2n+3 qubits"**](https://arxiv.org/pdf/quant-ph/0205095.pdf) by Stephane Beauregard.

The objective of this work is to better understand Quantum Computing foundations and to get practice with Qiskit, so if 
you are interested in using the code please consider that it could be optimized/improved.

The code is currently working in IBM simulators but not in IBM real quantum hardware. I am working on it.

# Shor's algorithm

Shor's algorithm is a quantum algorithm devised by mathematician Peter Shor in 1994. It is designed to efficiently 
factorize large composite numbers into their prime factors. Factoring large numbers into primes is a fundamental problem
in number theory and is known to be a challenging task for classical computers, particularly when the numbers are very 
large and the factors are large prime numbers.

Shor's algorithm exploits the inherent parallelism and superposition properties of quantum computing to significantly 
speed up the factoring process. The algorithm has the potential to break widely used public-key crypto-systems, such as 
RSA, which rely on the difficulty of factoring large numbers for their security.

The efficiency of Shor's algorithm makes it a significant development in the field of quantum computing, as it 
demonstrates a clear advantage over classical algorithms for certain types of problems. However, it's important to note 
that practical implementation of large-scale quantum computers capable of running Shor's algorithm is still a 
challenging task, and current quantum computers are not yet at a scale where they can threaten widely used cryptographic
systems.

# How to use it

Clone the project
```bash
git clone https://https://github.com/jmpr1991/Shor_algorithm.git
```

Run the main.py. The program will ask the user the different needed parameters to proceed with the computation

# Output
The program will print the most relevant outputs while running (quantum circuit, phases, fractions, period, factors).
additionally the shor function output the quantum circuit, the period and factors
