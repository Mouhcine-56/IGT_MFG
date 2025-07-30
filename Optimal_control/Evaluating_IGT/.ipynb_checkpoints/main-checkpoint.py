# ==========================
#      Imports
# ==========================
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
#      Logging Setup
# ==========================
class Tee(object):
    """Class to write output to both a file and the terminal in real-time."""
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
#      Argument Parser
# ==========================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t0', type=int, default=0)
    parser.add_argument('--t_final', type=int, default=1)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--num_samples_hjb', type=int, default=1000)
    parser.add_argument('--num_samples_bvp', type=int, default=128)
    parser.add_argument('--num_points_test', type=int, default=1000)
    parser.add_argument('--Max_Round', type=int, default=3)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--num_itr', type=int, default=10000)
    parser.add_argument('--freq', type=int, default=1000)
    parser.add_argument('--ns', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrr', type=float, default=1e-4)
    parser.add_argument('--betas', type=tuple, default=(0.5, 0.9))
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--act_func', default=lambda x: torch.tanh(x))
    parser.add_argument('--hh', type=float, default=0.5)
    return parser.parse_args()


# ==========================
#     Utility Functions
# ==========================
def create_meshgrid_2d(num_points, device):
    x = torch.linspace(-1, 1, num_points, device=device)
    y = torch.linspace(-1, 1, num_points, device=device)
    X, Y = torch.meshgrid(x, y)
    return torch.stack([X.flatten(), Y.flatten()], dim=1)

# Relative Error E_2
def relative_error(u_exact, u_pred):
    return torch.norm(u_exact - u_pred) / torch.norm(u_exact)

def test(V_NN, num_points, dim, analytic, device):
    if dim == 1:
        x = torch.linspace(-1, 1, num_points, device=device).unsqueeze(1)
    elif dim == 2:
        x = create_meshgrid_2d(num_points, device)
    else:
        x = 2 * torch.rand(num_points, dim, device=device) - 1

    V_NN.eval()
    for t_val in [0.0, 0.5, 1.0]:
        t = t_val * torch.ones(x.shape[0], 1, device=device)
        with torch.no_grad():
            v_pred = V_NN(t, x)
            v_exact = analytic.V_exact(x, t=t_val)
            err = relative_error(v_exact, v_pred)
            print(f"Relative_Error_t{t_val} = {err.item():.6e}")


# ==========================
#           Main
# ==========================
def main():
    args = get_args()

    # Random seeds
    torch_seed = np.random.randint(low=-sys.maxsize - 1, high=sys.maxsize)
    torch.manual_seed(torch_seed)
    np_seed = np.random.randint(low=0, high=2 ** 32 - 1)
    np.random.seed(np_seed)

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cpu':
        print("NOTE: USING ONLY THE CPU")

    # Analytic Problem Setup
    analytic = Analytic(device)

    # Initialize Value Network
    V_NN = V_Net(
        dim=args.dim, ns=args.ns, act_func=args.act_func, hh=args.hh,
        device=device, psi_func=analytic.psi_func, TT=analytic.TT
    ).to(device)

    t_grid = torch.linspace(args.t0, args.t_final, args.num_samples_hjb, device=device).unsqueeze(1)

    print("Initialisation...\n")
    V_NN = Solve_HJB(V_NN, args.num_epoch, t_grid, args.lr, args.num_samples_hjb, device)

    for Round in range(args.Max_Round):
        print(f"\n=== Round {Round} ===\n")

        print("Generating data...\n")
        data = generate_data(V_NN, args.num_samples_bvp, t_grid, args.lr, args.num_samples_hjb, device)

        print("Training...\n")
        V_NN = Approximate_v(V_NN, data, args.num_itr, t_grid, args.lrr, args.num_samples_hjb, Round, device)

        print("Testing relative error...\n")
        test(V_NN, args.num_points_test, args.dim, analytic, device)

    total_time = time.time() - start_time
    print(f"\nTotal Time = {total_time:.1f} sec")

    torch.save(V_NN.state_dict(), 'V_NN_1d.pth')


if __name__ == "__main__":
    start_time = time.time()
    main()

