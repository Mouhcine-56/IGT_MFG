import multiprocessing
import numpy as np
import argparse
import time
from model import *
from src.solve import *
import sys
from src.problem.prb import Analytic

parser = argparse.ArgumentParser()
parser.add_argument('--t0', type=int, default=0)
parser.add_argument('--t_final', type=int, default=1)
parser.add_argument('--dt', type=int, default=0.05)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--num_samples_hjb', type=int, default=1)
parser.add_argument('--num_samples_bvp', type=int, default=32)
parser.add_argument('--num_points_test', type=int, default=1000)
parser.add_argument('--Max_Round', type=float, default=2)
parser.add_argument('--num_epoch', type=float, default=100)
parser.add_argument('--num_itr', type=float, default=10000)
parser.add_argument('--freq', type=float, default=1000)
parser.add_argument('--ns', default=128, help='Network size')
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--lrr', default=1e-4)
parser.add_argument('--betas', default=(0.5, 0.9), help='Adam only')
parser.add_argument('--weight_decay',default=1e-3)
parser.add_argument('--act_func',default= lambda x: torch.tanh(x), help='Activation function for discriminator')
parser.add_argument('--hh', default=0.5, help='ResNet step-size')
args = parser.parse_args()

torch_seed = np.random.randint(low=-sys.maxsize - 1, high=sys.maxsize)
torch.random.manual_seed(torch_seed)
np_seed = np.random.randint(low=0, high=2 ** 32 - 1)
np.random.seed(np_seed)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    print('NOTE: USING ONLY THE CPU')

an = Analytic(device)    

if __name__ == "__main__":
    
    #act_funcs = lambda x: torch.tanh(x)
    start_time = time.time()
    V_NN = V_Net(dim=args.dim, ns=args.ns, act_func=args.act_func, hh=args.hh,
                            device=device, psi_func=an.psi_func, TT=an.TT).to(device)
    
    
    t = torch.linspace(args.t0, args.t_final, args.num_samples_hjb, device=device).unsqueeze(1)
    
    print('Approximate solution by HJB \n')
    

#     V_NN = Solve_HJB(V_NN, args.num_epoch, t, args.lr, args.num_samples_hjb, device)
    
    for Round in range(args.Max_Round):
        
        print('Round: ', Round, '\n')
        
        data = generate_data(V_NN, args.num_samples_bvp, t, args.lr, args.num_samples_hjb, device, num_epoch=10)


        print('Approximate solution of BVP \n')
        V_NN = Approximate_v(V_NN, data, args.num_itr, t, args.lrr, args.num_samples_hjb, Round, device)

    print('Time = %.1f' % (time.time() - start_time), 'sec \n')    
    torch.save(V_NN.state_dict(), 'V_NN.pth')
    
    
