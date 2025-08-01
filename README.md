# IGT: Initilization, Generation, and Training for optimal control

## 📌 Overview

We use neural networks to obtain a rough approximation of the value function, which serves as an initial guess for solving the two-point boundary value problem derived from Pontryagin’s maximum principle, and subsequently generates accurate data. The network is then trained using a loss function that incorporates this dataset and penalizes deviations from the Hamilton-Jacobi-Bellman (HJB) equation.

This code includes:
- Initialization "HJB solver" using DGM or C-DGM
- Data generation using TPBVP solvers
- Value function training

## 🔧 Structure

- `main.py` — Training and evaluation pipeline
- `model.py` — Neural network architectures V_Net
- `src/solve.py` — Functions for solving TPBVP and Training routines for HJB
- `src/problem/prb.py` — problem definitions (Hamiltonian, dynamics, costs, etc...)


# IGT-MFG: Machine learning method to solve Mean Field Games

## 📌 Overview

To approximate the equilibria in a first-order Mean Field Game (MFG), we combine IGT with the fictitious play algorithm. These equilibria arise from a coupled system consisting of a first-order HJB equation and a continuity equation. To approximate the solution of the continuity equation, we employ a second neural network that learns the flow map transporting the initial agent distribution. This network is trained on trajectories generated by solving the associated ODEs for a batch of initial conditions. 

This code includes:
- IGT for approximating value function
- Neural generator training to approximate the best response
- Fictitious play iterations
- Evaluation of exploitability and Sinkhorn divergence

## 🔧 Structure

- `main.py` — Training and evaluation pipeline
- `model.py` — Neural network architectures (V_Net and G_Net)
- `src/solve.py` — Functions for Training routines, HJB, TPBVP, and generator
- `src/problem/prb.py` — problem definitions (Hamiltonian, dynamics, costs, etc...)

## 📄 Article

📝 If you're interested in the full theoretical and experimental details, please see our paper:  
**[INITIALIZATION-DRIVEN NEURAL GENERATION AND TRAINING FOR
HIGH-DIMENSIONAL OPTIMAL CONTROL AND FIRST-ORDER MEAN FIELD
GAMES](https://arxiv.org/pdf/2507.15126)**

## Contact

If you have any questions, suggestions, or feedback,  feel free to contact me at [mouhcine.assouli@unilim.fr](mailto:mouhcine.assouli@unilim.fr) — I'm happy to help!
