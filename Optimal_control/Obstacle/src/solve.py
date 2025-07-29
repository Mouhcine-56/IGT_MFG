import torch
import time
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import torch.optim as optim
from src.problem.prb import Obstacle

def Solve_HJB(V_NN, num_epoch, t, lr, num_samples, device):
    """
    Solve the HJB equation by minimizing the PDE residual along closed-loop trajectories.
    
    Args:
        V_NN         : neural network approximator of V(t, x)
        num_epoch    : number of training iterations
        t            : time placeholder (ignored here, dynamically generated)
        lr           : learning rate
        num_samples  : number of trajectories to simulate
        device       : torch device (CPU or CUDA)
        
    Returns:
        V_NN         : trained neural network
    """
    obs = Obstacle(device)

    def eval_u(t, x):
        """Closed-loop control: u = ∇V(t,x)"""
        return V_NN.get_grad(t, x).detach().cpu().numpy()

    optimizer = optim.Adam(V_NN.parameters(), lr)
    old_loss = 1e5
    loss = []

    for epoch in range(num_epoch + 1):
        # Generate new initial condition and simulate forward
        x0 = obs.gen_x0(num_samples, Torch=False).flatten()
        SOL = solve_ivp(
            obs.dynamics, [0., obs.TT], x0,
            method=obs.ODE_solver,
            t_eval=np.linspace(0, obs.TT, 21),
            args=(eval_u,),
            rtol=1e-8
        )

        # Convert state and time trajectories to torch tensors
        x_rand = torch.tensor(SOL.y.T, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
        t = torch.tensor(SOL.t.reshape(-1, 1), dtype=torch.float32, device=device).clone().detach().requires_grad_(True)

        # Forward pass
        V_nn = V_NN(t, x_rand)

        # Compute ∂V/∂t
        V_nn_t = torch.autograd.grad(
            outputs=V_nn, inputs=t,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Compute ∇V
        V_nn_x = torch.autograd.grad(
            outputs=V_nn, inputs=x_rand,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Compute residual loss from HJB equation
        Loss = torch.mean((V_nn_t + obs.ham(t, x_rand, V_nn_x)) ** 2)

        # Backpropagation
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        # Logging
        if epoch % 50 == 0:
            print(f"Iteration {epoch}: Loss = {Loss.item():.4e}")

    return V_NN



def Approximate_v(V_NN, data, num_epoch, t, lr, num_samples, Round, device):
    """
    Train V_NN to fit the best-response data (value and gradient),
    while still minimizing the HJB residual.

    Args:
        V_NN        : neural network for V(t, x)
        data        : dict with keys 't', 'X', 'V', 'A' (torch tensors)
        num_epoch   : number of training epochs
        t           : placeholder time grid (not used directly)
        lr          : learning rate
        num_samples : unused (present for compatibility)
        Round       : current training round index (for display/logging)
        device      : CPU or CUDA device

    Returns:
        V_NN        : updated network after training
    """
    obs = Obstacle(device)
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)

    old_loss = 1
    loss = []

    # Require gradients for HJB term
    t = data['t'].requires_grad_(True)
    x_rand = data['X'].requires_grad_(True)

    for epoch in range(num_epoch + 1):
        # Forward pass
        V_nn = V_NN(t, x_rand)

        # ∂V/∂t
        V_nn_t = torch.autograd.grad(
            outputs=V_nn, inputs=t,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # ∇V
        V_nn_x = torch.autograd.grad(
            outputs=V_nn, inputs=x_rand,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Loss terms
        Loss_hjb = torch.mean((V_nn_t + obs.ham(t, x_rand, V_nn_x)) ** 2)
        Loss_v = torch.mean((V_NN(data['t'], data['X']) - data['V']) ** 2)
        Loss_v_x = torch.mean((V_NN.get_grad(data['t'], data['X']) - data['A']) ** 2)

        # Total loss
        Loss_total = Loss_hjb + Loss_v + Loss_v_x

        # Backpropagation
        optimizer.zero_grad()
        Loss_total.backward()
        optimizer.step()

        # Logging
        if epoch % 1000 == 0:
            print(
                f"Iteration {epoch}: "
                f"Loss_V = {Loss_v.item():.4e}, "
                f"Loss_V_x = {Loss_v_x.item():.4e}, "
                f"Loss_HJB = {Loss_hjb.item():.4e}, "
                f"Loss_total = {Loss_total.item():.4e}"
            )

    print('\n')
    return V_NN

def generate_data(V_NN, num_samples, t, lr, num_samples_hjb, device, num_epoch=50):
    
    obs = Obstacle(device)
    
    def eval_u(t, x):
        u =  V_NN.get_grad(t, x).detach().cpu().numpy()
        return u
    
    def bvp_guess(t, x):
        V_NN.eval()
        V = V_NN(torch.tensor(t, dtype=torch.float32, device=device), 
                 torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        V_x = V_NN.get_grad(t, x).detach().cpu().numpy()
        return V, V_x

    print('Generating data...')

    dim = obs.dim

    X_OUT = np.empty((0, dim))
    A_OUT = np.empty((0, dim))
    V_OUT = np.empty((0, 1))
    t_OUT = np.empty((0, 1))

    Ns_sol = 0  # Initialize the number of successful solutions
    failure_count = 0
   

    start_time = time.time()
    x0_int = obs.gen_x0(num_samples, Torch=False)

    # ----------------------------------------------------------------------

    while Ns_sol < num_samples:
        
        print('Solving BVP #', Ns_sol+1, '...', end='\r')
        
        max_failures = num_samples - Ns_sol

        X0 = x0_int[Ns_sol,:]
        bc = obs.make_bc(X0)

        # Integrates the closed-loop system (NN controller)

        SOL = solve_ivp(obs.dynamics, [0., obs.TT], X0,
                        method=obs.ODE_solver,
                        args=(eval_u,),
                        rtol=1e-08)
        

        V_guess, A_guess = bvp_guess(SOL.t.reshape(1,-1).T, SOL.y.T)

        try:
            # Solves the two-point boundary value problem

            X_aug_guess = np.vstack((SOL.y, A_guess.T, V_guess.T))
            SOL = solve_bvp(obs.aug_dynamics, bc, SOL.t, X_aug_guess,
                            verbose=0,
                            tol=obs.data_tol,
                            max_nodes=obs.max_nodes)
            # Save only successful solutions
            if SOL.success:
                V = SOL.y[-1:] + obs.terminal_cost(SOL.y[:dim, -1])
                t_OUT = np.vstack((t_OUT, SOL.x.reshape(1, -1).T))
                X_OUT = np.vstack((X_OUT, SOL.y[:dim].T))
                A_OUT = np.vstack((A_OUT, SOL.y[dim:2*dim].T))
                V_OUT = np.vstack((V_OUT, V.T))
                Ns_sol += 1
                #failure_count = 0  # Reset failure count on success
            else:
                # Failure: log and handle retry
                print(f"Solver failed ({failure_count + 1}/{max_failures}): {SOL.message}")
                failure_count += 1

                # Early fallback retraining (only if within first 32 and parameters provided)
                if Ns_sol < 10: 
                    print("  --> Retrying with retrained V_NN via Solve_HJB...")
                    V_NN = Solve_HJB(V_NN, num_epoch, t, lr, num_samples_hjb, device)

                if failure_count >= max_failures:
                    print("Maximum failures reached. Exiting early.")
                    break
                # Generate a new initial condition for retry
                X0 = x0_int[Ns_sol+failure_count,:]


        except Warning as e:
            print(f"Warning encountered: {str(e)}. Skipping...")
            failure_count += 1
            if failure_count >= max_failures:
                print("Maximum warnings reached. Exiting.")
                break

    print('Generated', X_OUT.shape[0], 'data from', Ns_sol,
        'BVP solutions in %.1f' % (time.time() - start_time), 'sec \n')

    data = {'t': torch.tensor(t_OUT, dtype=torch.float32, device=device), 
            'X': torch.tensor(X_OUT, dtype=torch.float32, device=device), 
            'A': torch.tensor(A_OUT, dtype=torch.float32, device=device), 
            'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)}
    
    return data

   
# with time marchine
#import warnings
#warnings.filterwarnings('error')
def generate_data_TM(V_NN, num_samples, t, lr, num_samples_hjb, device, num_epoch=1000):
    
    obs = Obstacle(device)
    
    print('Generating data...')

    dim = obs.dim

    X_OUT = np.empty((0, dim))
    A_OUT = np.empty((0, dim))
    V_OUT = np.empty((0, 1))
    t_OUT = np.empty((0, 1))

    Ns_sol = 0
    failure_count = 0
    
    start_time = time.time()
    x0_int = obs.gen_x0(num_samples)

    # ----------------------------------------------------------------------

    while Ns_sol < num_samples:
        
        print('Solving BVP #', Ns_sol+1, '...', end='\r')
        
        max_failures = num_samples - Ns_sol

        X0 = x0_int[Ns_sol,:]
        bc = obs.make_bc(X0)


        # Integrates the closed-loop system (NN controller)

        start_time = time.time()

        try:
            # Initial guess is zeros
            t_guess = np.array([0.])
            X_guess = np.vstack((X0.reshape(-1,1),
                                 np.zeros((dim+1, 1))))

            tol = 1e-01

            ##### Time-marching to build from t0 to tf #####
            for k in range(obs.tseq.shape[0]):
                if tol >= 2.*obs.data_tol:
                    tol /= 2.
                if k == obs.tseq.shape[0] - 1:
                    tol = obs.data_tol

                t_guess = np.concatenate((t_guess, obs.tseq[k:k+1]))
                X_guess = np.hstack((X_guess, X_guess[:,-1:]))
                #print(np.shape(X_guess))
                SOL = solve_bvp(obs.aug_dynamics, bc, t_guess, X_guess,
                                verbose=0, tol=tol, max_nodes=obs.max_nodes)

                if not SOL.success:
                    print(SOL.message)
                    warnings.warn(Warning())
                t_guess = SOL.x
                X_guess = SOL.y

            #sol_time.append(time.time() - start_time)
            
            #print( problem.terminal_cost(SOL.y[:N_states,-1]))

            V = SOL.y[-1:] + obs.terminal_cost(SOL.y[:dim,-1])
            
            t_OUT = np.vstack((t_OUT, SOL.x.reshape(1, -1).T))
            X_OUT = np.vstack((X_OUT, SOL.y[:dim].T))
            A_OUT = np.vstack((A_OUT, SOL.y[dim:2*dim].T))
            V_OUT = np.vstack((V_OUT, V.T))

            N_sol += 1


        except Warning:
            X0 = x0_int[Ns_sol,:]
            failure_count += 1
            if failure_count >= max_failures:
                print("Maximum warnings reached. Exiting.")
                break


    print('Generated', X_OUT.shape[0], 'data from', Ns_sol,
        'BVP solutions in %.1f' % (time.time() - start_time), 'sec \n')
   
 

    data = {'t': torch.tensor(t_OUT, dtype=torch.float32, device=device), 
            'X': torch.tensor(X_OUT, dtype=torch.float32, device=device), 
            'A': torch.tensor(A_OUT, dtype=torch.float32, device=device), 
            'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)}


    return data

