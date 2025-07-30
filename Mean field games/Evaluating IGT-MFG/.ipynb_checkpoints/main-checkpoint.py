import multiprocessing
import numpy as np
import argparse
import time
import sys
import torch

from model import *
from src.solve import *
from src.problem.prb import Analytic


# ==========================
#       Logging Setup
# ==========================
class Tee(object):
    """Class to write output to both a file and terminal in real-time."""
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

sys.stdout = Tee("training_output.log")


# ==========================
#     Argument Parser
# ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--t0', type=int, default=0)
parser.add_argument('--t_final', type=int, default=1)
parser.add_argument('--N', type=int, default=20)
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--num_samples_hjb', type=int, default=256)
parser.add_argument('--num_samples_bvp', type=int, default=32)
parser.add_argument('--num_samples_gen', type=int, default=128)
parser.add_argument('--num_samples_conv', type=int, default=128)
parser.add_argument('--num_points_test', type=int, default=1000)
parser.add_argument('--Max_Round', type=int, default=5)
parser.add_argument('--num_epoch_hjb', type=int, default=1000)
parser.add_argument('--num_epoch_v', type=int, default=1000)
parser.add_argument('--num_epoch_gen', type=int, default=1000)
parser.add_argument('--freq', type=float, default=1000)
parser.add_argument('--ns_v', type=int, default=128)
parser.add_argument('--ns_g', type=int, default=128)
parser.add_argument('--lr_v', type=float, default=1e-4)
parser.add_argument('--lr_v2', type=float, default=1e-5)
parser.add_argument('--lr_g', type=float, default=1e-4)
parser.add_argument('--betas', default=(0.5, 0.9))
parser.add_argument('--weight_decay', default=1e-3)
parser.add_argument('--act_func_v', default=lambda x: torch.tanh(x))
parser.add_argument('--act_func_g', default=lambda x: torch.relu(x))
parser.add_argument('--hh', type=float, default=0.5)
args = parser.parse_args()


# ==========================
#     Random Seeds
# ==========================
torch_seed = np.random.randint(low=-sys.maxsize - 1, high=sys.maxsize)
torch.random.manual_seed(torch_seed)

np_seed = np.random.randint(low=0, high=2 ** 32 - 1)
np.random.seed(np_seed)


# ==========================
#       Device Setup
# ==========================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    print('NOTE: USING ONLY THE CPU')


# ==========================
#     Utility Functions
# ==========================
def psi_func(xx_inp):
    return torch.zeros(xx_inp.size(0), device=device)

def V_exact(x, T, mu, t):
    Pi_t = (np.exp(2 * T - t) - np.exp(t)) / (np.exp(2 * T - t) + np.exp(t))
    s_t = -Pi_t * mu
    c_t = 0.5 * Pi_t * mu**2
    return 0.5 * Pi_t * x**2 + s_t * x + c_t

def relative_error(u_exact, u_predicted):
    norm_exact = torch.norm(u_exact)
    error = torch.norm(u_exact - u_predicted)
    return error / norm_exact if norm_exact != 0 else torch.tensor(0.0, device=u_exact.device)

def relative_linf_error(u_exact, u_predicted):
    norm_exact = torch.norm(u_exact, p=float('inf'))
    error = torch.norm(u_exact - u_predicted, p=float('inf'))
    return error / norm_exact if norm_exact != 0 else torch.tensor(0.0, device=u_exact.device)

def test(V_NN, num_points, dim, T, mu):
    x = torch.linspace(-2, 2, num_points, device=device).unsqueeze(1)
    with torch.no_grad():
        V_NN.eval()
        for t_val in [0.0, 0.5, 1.0]:
            t = t_val * torch.ones(num_points, 1, device=device)
            v_pred = V_NN(t, x)
            v_exact = V_exact(x, T, mu, t=t_val)
            print(f"Time t = {t_val:.1f}")
            print(f"  Relative L2 Error    = {relative_error(v_exact, v_pred).item():.4e}")
            print(f"  Relative Linf Error  = {relative_linf_error(v_exact, v_pred).item():.4e}\n")

def W_gem(an, G_history, G_k, x0_ini, m0, t_grid):
    W_k = wasserstein_fp(an, G_history, G_k, x0_ini, m0, t_grid, p=1, blur=0.5, device=device)
    print(f"k {an.n:2d} │ L1={np.linalg.norm(W_k,1):.3e} │ L2={np.linalg.norm(W_k,2):.3e} │ L∞={W_k.max():.3e}")
    
    return W_k.max()


# ==========================
#        Data Loading
# ==========================
data = np.load('data1.npz')
M0 = np.load('m0.npz')
m0 = torch.tensor(M0['x'], dtype=torch.float32, device=device)
x0_ini = torch.tensor(data['x'], dtype=torch.float32, device=device)

mu = x0_ini.mean(axis=0)
std = torch.sqrt(x0_ini.var(axis=0))


# ==========================
#         Main Loop
# ==========================
if __name__ == "__main__":

    V_NN = V_Net(args.dim, args.ns_v, args.act_func_v, args.hh, device, psi_func, args.t_final).to(device)
    V_NN2 = V_Net(args.dim, args.ns_v, args.act_func_v, args.hh, device, psi_func, args.t_final).to(device)
    G_NN = G_Net(args.dim, args.ns_g, args.act_func_g, args.hh, device, mu, std, args.t_final).to(device)

    t = torch.linspace(args.t0, args.t_final, args.num_samples_hjb, device=device).unsqueeze(1)
    t_g = torch.linspace(args.t0, args.t_final, args.num_samples_gen, device=device).unsqueeze(1)
    tt = torch.linspace(args.t0, args.t_final, 21, device=device).unsqueeze(1)
    
    tol = 1e-6
    Delta = 1+tol

    for Round in range(args.Max_Round):
    
        if Delta <= tol:
            break

        print(f'\n===== Round {Round} =====\n')
        G_NN_list = []
        delta = 1 + tol
        if Round != 0:
            G_NN_list.append(copy.deepcopy(G_NN_new))

        nmax = 20
        for n in range(nmax):
        
            if delta <= tol:
                break

            print(f"\n--- Best Response {n} ---\n")
            start_time = time.time()

            # Phase 1: Initialisation
            an = Analytic(G_NN_list, Round, n, x0_ini, device, VV=1)
            if Round == 0 and n == 0:
                print("Solving HJB...")
                V_NN = Solve_HJB(an, V_NN, args.num_epoch_hjb, t, args.lr_v, args.num_samples_hjb, device)

            # Phase 2: Generating
            data = generate_data(an, V_NN, args.num_samples_bvp, device)

            # Phase 3: Train 
            print("Training V_NN on TPBVP data...")
            V_NN = Approximate_v(an, V_NN, data, args.num_epoch_v, t, args.lr_v, args.num_samples_hjb, Round, device)

            # Phase 4: Simulate trajectories
            t_tr, X_tr, t_OUT, X_OUT = sim_points(an, V_NN, args.num_samples_gen, args.N, args.t0, args.t_final, device)

            # Phase 5: Train generator
            G_NN, G_NN_new = Train_Gen(an, G_NN, V_NN, t_tr, X_tr, X_OUT, args.num_epoch_gen, t_g, args.lr_g, args.num_samples_gen, Round, device)

            # Phase 6: Evaluate
            test(V_NN, args.num_points_test, args.dim, T=1, mu=0.1)
            delta = W_gem(an, G_NN_list, G_NN_new, x0_ini, m0, tt)

            # Save generator and update list
            G_NN_list.append(copy.deepcopy(G_NN_new))
            torch.save(G_NN_new.state_dict(), f"G_NNN_round{Round}_n{n}.pth")

            # Compute J (cost-to-go from policy)
            J = Comp_J0(an, tt, x0_ini, V_NN)

            # Phase 7: Solve second value function
            print("\nTraining second value function (alpha1)...")
            an = Analytic(G_NN_list, Round, n, x0_ini, device, VV=2)
            data = generate_data(an, V_NN, args.num_samples_bvp, device)
            V_NN2 = Approximate_v2(an, V_NN2, data, args.num_epoch_v, t, args.lr_v2, args.num_samples_hjb, Round, device)

            # Exploitability
            V0 = Comp_V(x0_ini, V_NN2)
            exploi = abs(V0 - J)
            print(f"Exploitability: {exploi:.4e}")

            G_NN = copy.deepcopy(G_NN_new)
            
        Delta = delta

