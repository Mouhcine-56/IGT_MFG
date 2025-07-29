# IGT-MFG: Iterative Generator Training for Mean Field Games

This repository contains the code used in our work:

**"INITIALIZATION-DRIVEN NEURAL GENERATION AND TRAINING FOR
HIGH-DIMENSIONAL OPTIMAL CONTROL AND FIRST-ORDER MEAN FIELD
GAMES"**  
📄 [arXiv:2507.15126](https://arxiv.org/pdf/2507.15126)

## 📌 Overview

We propose a framework based on neural network approximation for solving mean field games (MFG) using a generator-based population simulator. The method alternates between solving Hamilton-Jacobi-Bellman (HJB) equations and training generators to match population distributions.

This code includes:
- Initialisation "HJB solver" using DGM
- Best response generation using TPBVP solvers
- Neural generator training and simulation
- Evaluation of exploitability and Wasserstein distance

## 🔧 Structure

- `main.py` — Training and evaluation pipeline
- `model.py` — Neural network architectures (V_Net and G_Net)
- `src/solve.py` — Training routines for HJB, TPBVP, and generator
- `src/problem/prb.py` — problem definitions (dynamics, costs, etc.)

## 📄 Paper

📝 If you're interested in the full theoretical and experimental details, please see our paper:  
**[INITIALIZATION-DRIVEN NEURAL GENERATION AND TRAINING FOR
HIGH-DIMENSIONAL OPTIMAL CONTROL AND FIRST-ORDER MEAN FIELD
GAMES](https://arxiv.org/pdf/2507.15126)**


