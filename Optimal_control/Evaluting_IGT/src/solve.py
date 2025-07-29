import torch
import time
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import torch.optim as optim
from src.problem.prb import Analytic

import torch
import torch.optim as optim
from src.problem.prb import Analytic

def Solve_HJB(V_NN, num_epoch, t, lr, num_samples, device):

    """
    Parameters:
        V_NN         : Neural network approximator of V(t, x)
        num_epoch    : Number of training epochs
        t            : Time grid (tensor of shape [num_samples, 1])
        lr           : Learning rate for optimizer
        num_samples  : Number of spatial samples
        device       : Torch device (CPU or CUDA)

    Returns:
        V_NN         : Trained neural network
    """
    an = Analytic(device)
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)

    # Initial spatial samples (x), held constant or resampled during training
    x_rand = an.sample_x0(num_samples).requires_grad_(True)
    T = an.TT * torch.ones(num_samples, 1, device=device)

    # Training loop
    tracked_loss = []
    best_loss = float("inf")

    for epoch in range(num_epoch + 1):
        V_NN.train()

        # Ensure t has requires_grad for ∂_t V
        t = t.clone().detach().requires_grad_(True)

        # Forward pass
        V_pred = V_NN(t, x_rand)

        # Compute ∂_t V
        V_t = torch.autograd.grad(
            outputs=V_pred, inputs=t,
            grad_outputs=torch.ones_like(V_pred),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Compute ∇_x V
        V_x = torch.autograd.grad(
            outputs=V_pred, inputs=x_rand,
            grad_outputs=torch.ones_like(V_pred),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # HJB residual loss
        residual = V_t + an.ham(t, x_rand, V_x)
        loss_val = torch.mean(residual ** 2)

        # Optimization step
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Periodic logging
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {loss_val.item():.4e}")
            tracked_loss.append(loss_val.item())

            # Optional: resample x_rand if improvement occurs
            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                x_rand = an.sample_x0(num_samples).requires_grad_(True)

    return V_NN


def Approximate_v(V_NN, data, num_epoch, t, lr, num_samples, Round, device):
    """
    Train the value network V_NN using both:
    - the HJB residual loss (unsupervised)
    - supervised data from best response (value and gradient)
    
    Args:
        V_NN         : neural network approximator of V(t,x)
        data         : dictionary with keys 't', 'X', 'V', 'A'
        num_epoch    : number of training iterations
        t            : time samples (tensor)
        lr           : learning rate
        num_samples  : number of points for HJB residual
        Round        : current training round (for logging)
        device       : torch device (CPU or CUDA)

    Returns:
        V_NN         : updated neural network
    """
    # Instantiate analytic tools (e.g., Hamiltonian and sampling)
    an = Analytic(device)
    V_NN.train()

    # Optimizer
    optimizer = optim.Adam(V_NN.parameters(), lr)

    # Initial spatial samples for the HJB loss term
    x_rand = an.sample_x0(num_samples).requires_grad_(True)
    T = an.TT * torch.ones(num_samples, 1, device=device)

    # Loss tracking
    old_loss = 1
    loss = []

    for epoch in range(num_epoch + 1):
        # Ensure gradient tracking on time
        t = t.requires_grad_(True)

        # Forward pass for V(t, x) on HJB points
        V_nn = V_NN(t, x_rand)

        # Compute ∂_t V
        V_nn_t = torch.autograd.grad(
            outputs=V_nn, inputs=t,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Compute ∇_x V
        V_nn_x = torch.autograd.grad(
            outputs=V_nn, inputs=x_rand,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # HJB residual loss: ∂_t V + H = 0
        Loss_hjb = torch.mean((V_nn_t + an.ham(t, x_rand, V_nn_x)) ** 2)

        # Supervised loss on value: V ≈ V^BR
        Loss_v = torch.mean((V_NN(data['t'], data['X']) - data['V']) ** 2)

        # Supervised loss on gradient: ∇V ≈ ∇V^BR
        Loss_v_x = torch.mean((V_NN.get_grad(data['t'], data['X']) - data['A']) ** 2)

        # Total loss
        Loss_total = Loss_hjb + Loss_v + Loss_v_x

        # Backpropagation
        optimizer.zero_grad()
        Loss_total.backward()
        optimizer.step()

        # Periodic logging and potential resampling
        if epoch % 1000 == 0:
            loss.append(old_loss)
            new_loss = Loss_total.item()

            print(f"Iteration {epoch}: "
                  f"Loss_V = {Loss_v.item():.4e}, "
                  f"Loss_V_x = {Loss_v_x.item():.4e}, "
                  f"Loss_HJB = {Loss_hjb.item():.4e}, "
                  f"Loss_total = {Loss_total.item():.4e}")

            # Resample spatial points if improvement
            if new_loss > min(loss):
                x_rand = x_rand  # keep current points
            else:
                x_rand = an.sample_x0(num_samples).requires_grad_(True)  # resample

            old_loss = new_loss

    return V_NN



def generate_data(V_NN, num_samples, t, lr, num_samples_hjb, device, num_epoch=1000):
    """
    Generate training data by solving TBVP using the current V_NN as a guess.
    If early failures happen, V_NN is retrained via Solve_HJB.

    Args:
        V_NN             : Value function neural network
        num_samples      : Number of BVP trajectories to generate
        t                : Time grid for HJB residual (for retraining fallback)
        lr               : Learning rate for retraining
        num_samples_hjb  : Number of points for HJB residual training
        device           : torch.device('cuda') or torch.device('cpu')
        num_epoch        : Epochs for fallback Solve_HJB retraining (default=1000)

    Returns:
        data             : Dict with keys 't', 'X', 'A', 'V' (all torch tensors)
    """
    an = Analytic(device)

    # Helper functions
    def eval_u(t, x):
        return -0.5 * V_NN.get_grad(t, x).detach().cpu().numpy()

    def bvp_guess(t, x):
        V_NN.eval()
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        V = V_NN(t_tensor, x_tensor).detach().cpu().numpy()
        V_x = V_NN.get_grad(t, x).detach().cpu().numpy()
        return V, V_x

    # Initialization
    dim = an.dim
    X_OUT = np.empty((0, dim))
    A_OUT = np.empty((0, dim))
    V_OUT = np.empty((0, 1))
    t_OUT = np.empty((0, 1))

    Ns_sol = 0
    failure_count = 0
    x0_int = an.gen_x0(num_samples, Torch=False)

    max_failures = num_samples  # safety cap

    start_time = time.time()

    while Ns_sol < num_samples:
        print(f'Solving BVP #{Ns_sol + 1}...', end='\r')

        X0 = x0_int[Ns_sol]
        bc = an.make_bc(X0)


        # Integrate system forward using current NN policy
        SOL = solve_ivp(an.dynamics, [0., an.TT], X0,
                        method=an.ODE_solver,
                        args=(eval_u,),
                        rtol=1e-8)

        V_guess, A_guess = bvp_guess(SOL.t.reshape(-1, 1), SOL.y.T)
        X_aug_guess = np.vstack((SOL.y, A_guess.T, V_guess.T))

        try:
            # Attempt to solve the augmented BVP
            SOL = solve_bvp(an.aug_dynamics, bc, SOL.t, X_aug_guess,
                            verbose=0,
                            tol=an.data_tol,
                            max_nodes=an.max_nodes)

            if SOL.success:
                # Successful solution → store it
                V = SOL.y[-1:] + an.terminal_cost(SOL.y[:dim, -1])
                t_OUT = np.vstack((t_OUT, SOL.x.reshape(-1, 1)))
                X_OUT = np.vstack((X_OUT, SOL.y[:dim].T))
                A_OUT = np.vstack((A_OUT, SOL.y[dim:2*dim].T))
                V_OUT = np.vstack((V_OUT, V.T))
                Ns_sol += 1  # Increment only on success

            else:
                # Failure: log and handle retry
                print(f"Solver failed ({failure_count + 1}/{max_failures}): {SOL.message}")
                failure_count += 1

                # Early fallback retraining (only if within first 32 and parameters provided)
                if Ns_sol < 32: 
                    print("  --> Retrying with retrained V_NN via Solve_HJB...")
                    V_NN = Solve_HJB(V_NN, num_epoch, t, lr, num_samples_hjb, device)

                if failure_count >= max_failures:
                    print("Maximum failures reached. Exiting early.")
                    break

                continue  # Skip to next trial without incrementing Ns_sol

        except Warning as e:
            print(f"Warning encountered: {str(e)}. Skipping...")
            failure_count += 1
            if failure_count >= max_failures:
                print("Maximum warnings reached. Exiting.")
                break

    print(f'\nGenerated {X_OUT.shape[0]} data points from {Ns_sol} BVP solutions '
          f'in {time.time() - start_time:.1f} sec.\n')

    # Convert data to torch tensors
    data = {
        't': torch.tensor(t_OUT, dtype=torch.float32, device=device),
        'X': torch.tensor(X_OUT, dtype=torch.float32, device=device),
        'A': torch.tensor(A_OUT, dtype=torch.float32, device=device),
        'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)
    }

    return data
    
   
   


