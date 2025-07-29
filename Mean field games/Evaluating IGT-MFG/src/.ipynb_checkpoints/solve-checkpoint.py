import torch
import time
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import torch.optim as optim
import copy
from scipy.stats import ks_2samp
import torch.nn as nn
from scipy.stats import wasserstein_distance
import  ot
from geomloss import SamplesLoss
#================================ DGM_HJB =========================================#

def Solve_HJB(an, V_NN, num_epoch, t, lr, num_samples, device):
   
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)

    x_rand = an.sample_x0(num_samples).requires_grad_(True)

    old_loss = 1
    loss = []  

    for epoch in range(num_epoch+1):

        t = t.requires_grad_(True)

        V_nn =  V_NN(t, x_rand)

        V_nn_t = torch.autograd.grad(outputs=V_nn, inputs=t,
                                          grad_outputs=torch.ones_like(V_nn),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]

        V_nn_x = torch.autograd.grad(outputs=V_nn, inputs=x_rand,
                                          grad_outputs=torch.ones_like(V_nn),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]


        Loss = torch.mean(( V_nn_t + an.ham(t,x_rand,V_nn_x))**2) #+  torch.mean((V_NN(T, x_rand)-an.psi_func(x_rand))**2)


        optimizer.zero_grad()


        Loss.backward()


        optimizer.step()


        if epoch % 1000 == 0:
        #loss.append(old_loss)    
        #new_loss = Loss.item()
            print(f"Iteration {epoch}: Loss = {Loss.item():.4e}")
        #if new_loss>min(loss):
        #    x_rand = x_rand
        #else:   
        #    x_rand = an.sample_x0(num_samples).requires_grad_(True)
        #old_loss = new_loss    

    print('\n') 
    return V_NN

#================================  BVP + HJB =========================================#

def Approximate_v(an, V_NN, data, num_epoch, t, lr, num_samples, Round, device):
   
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)

    x_rand = an.sample_x0(num_samples).requires_grad_(True)

#     old_loss = 1
#     loss = []


    for epoch in range(num_epoch+1):

        t = t.requires_grad_(True)

        V_nn =  V_NN(t, x_rand)

        V_nn_t = torch.autograd.grad(outputs=V_nn, inputs=t,
                                          grad_outputs=torch.ones_like(V_nn),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]

        V_nn_x = torch.autograd.grad(outputs=V_nn, inputs=x_rand,
                                          grad_outputs=torch.ones_like(V_nn),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]

        Loss_hjb = torch.mean(( V_nn_t + an.ham(t,x_rand,V_nn_x))**2) 
        Loss_v = torch.mean((V_NN(data['t'], data['X']) - data['V'])**2) 
        Loss_v_x = torch.mean((V_NN.get_grad(data['t'], data['X']) - data['A'])**2)

        Loss_total =   Loss_hjb + Loss_v + Loss_v_x

        optimizer.zero_grad()


        Loss_total.backward()


        optimizer.step()


        if epoch % 1000 == 0:
            #loss.append(old_loss)    
            #new_loss = Loss_total.item()
            print(f"Iteration {epoch}: Loss_V = {Loss_v.item():.4e}, Loss_V_x = {Loss_v_x.item():.4e}, Loss_HJB = {Loss_hjb.item():.4e}, Loss_total = {Loss_total.item():.4e}")
            #if new_loss>min(loss):
            #    x_rand = x_rand
            #else:   
            #    x_rand = an.sample_x0(num_samples).requires_grad_(True)
            #old_loss = new_loss  


    print('\n')      
    return V_NN

def Approximate_v2(an, V_NN, data, num_epoch, t, lr, num_samples, Round, device):
   
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)

    x_rand = an.sample_x0(num_samples).requires_grad_(True)

    for epoch in range(num_epoch+1):

        Loss_v = torch.mean((V_NN(data['t'], data['X']) - data['V'])**2) 
        Loss_v_x = torch.mean((V_NN.get_grad(data['t'], data['X']) - data['A'])**2)

        Loss_total =  Loss_v + Loss_v_x

        optimizer.zero_grad()


        Loss_total.backward()


        optimizer.step()


        if epoch % 1000 == 0:

            print(f"Iteration {epoch}: Loss_V = {Loss_v.item():.4e}, Loss_V_x = {Loss_v_x.item():.4e}, Loss_total = {Loss_total.item():.4e}")

    print('\n')      
    return V_NN

#================================ solve BVP ===========================================#

def generate_data(an, V_NN, num_samples, device):
    
    def eval_u(t, x):
        u = - V_NN.get_grad(t, x).detach().cpu().numpy()
        return u
    
    def bvp_guess(t, x):
        V_NN.eval()
        V = V_NN(torch.tensor(t, dtype=torch.float32, device=device), 
                 torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        V_x = V_NN.get_grad(t, x).detach().cpu().numpy()
        return V, V_x

    print('Generating data_OC...')

    dim = an.dim

    X_OUT = np.empty((0, dim))
    A_OUT = np.empty((0, dim))
    V_OUT = np.empty((0, 1))
    t_OUT = np.empty((0, 1))

    Ns_sol = 0
    start_time = time.time()
    x0_int = an.gen_x0(num_samples, Torch=False)
#     data = np.load('data.npz')
#     x0_int = data['x']

    # ----------------------------------------------------------------------

    while Ns_sol < num_samples:
        
        #print('Solving BVP #', Ns_sol+1, '...', end='\r')

        #X0 = an.gen_x0(1).flatten()
        X0 = x0_int[Ns_sol,:]
        bc = an.make_bc(X0)

        # Integrates the closed-loop system (NN controller)

        SOL = solve_ivp(an.dynamics, [0., an.TT], X0,
                        method=an.ODE_solver,
                        args=(eval_u,),
                        rtol=1e-08)

        V_guess, A_guess = bvp_guess(SOL.t.reshape(1,-1).T, SOL.y.T)
        
        #print(SOL.y)

        try:
            # Solves the two-point boundary value problem
            
            X_aug_guess = np.vstack((SOL.y, A_guess.T, V_guess.T))

            SOL = solve_bvp(an.aug_dynamics, bc, SOL.t, X_aug_guess,
                            verbose=0,
                            tol=an.data_tol,
                            max_nodes=an.max_nodes)
            if not SOL.success:
                warnings.warn(Warning())

            Ns_sol += 1

            V = SOL.y[-1:] + an.terminal_cost(SOL.y[:dim,-1])

            t_OUT = np.vstack((t_OUT, SOL.x.reshape(1,-1).T))
            X_OUT = np.vstack((X_OUT, SOL.y[:dim].T))
            A_OUT = np.vstack((A_OUT, SOL.y[dim:2*dim].T))
            V_OUT = np.vstack((V_OUT, V.T))

        except Warning:
            pass

    print('Generated', X_OUT.shape[0], 'data from', Ns_sol,
        'BVP solutions in %.1f' % (time.time() - start_time), 'sec \n')

    data = {'t': torch.tensor(t_OUT, dtype=torch.float32, device=device), 
            'X': torch.tensor(X_OUT, dtype=torch.float32, device=device), 
            'A': torch.tensor(A_OUT, dtype=torch.float32, device=device), 
            'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)}

    return data
   
#==========================   Simulate points   ==========================================#   


def sim_points(an, V_NN, num_samples, N, t0, tf,  device):
    
    def eval_u(t, x):
        u = - V_NN.get_grad(t, x).detach().cpu().numpy()
        return u
    
    X_OUT = np.empty((0, an.dim))
    t_OUT = np.empty((0, 1))
    
    data = an.gen_x0(num_samples, Torch=False)
    
    Ns_sol = 0
    start_time = time.time()
    
    print('Generating data_MFG...')
    
    while Ns_sol < num_samples:
        
        #print('Solving IVP #', Ns_sol+1, '...', end='\r')
        
        X0 = data[Ns_sol, :]
        
        # Integrates the closed-loop system (NN controller)

        SOL = solve_ivp(an.dynamics, [0., 1], X0,
                        method= 'RK23', t_eval=np.linspace(0,1,N+1),
                        args=(eval_u,),
                        rtol=1e-08)

        

        Ns_sol += 1

        t_OUT = np.vstack((t_OUT, SOL.t.reshape(1,-1).T))
        X_OUT = np.vstack((X_OUT, SOL.y.T))
    
    t_train = t_OUT.reshape(num_samples, N+1)
    t_tr = t_train.T.flatten().reshape(-1, 1)
    x_tr = np.tile(data[0:num_samples, :], (N+1, 1))
    X_OUT = X_OUT.reshape(num_samples, N+1, an.dim)
    X_OUT = X_OUT.transpose(1, 0, 2).reshape((N+1) * num_samples, an.dim)
    
    print('Generated', X_OUT.shape[0], 'data from', Ns_sol,
        'IVP solutions in %.1f' % (time.time() - start_time), 'sec \n')
    
        
    return t_tr, x_tr, t_OUT, X_OUT


#================================  Train Generator =========================================#

def Train_Gen(an, G_NN, V_NN, t_tr, x_tr, X_OUT, num_epoch, t, lr, num_samples, Round, device):
    
    G_NN.eval()
    
    G_NN_original = copy.deepcopy(G_NN)
    
    x_rand = an.gen_x0(num_samples, Torch=True)
    t = t.requires_grad_(True)

    t_train = torch.tensor(t_tr, dtype=torch.float32, device=device).requires_grad_(True)
    X_train = torch.tensor(x_tr, dtype=torch.float32, device=device).requires_grad_(True)
    X_OUT = torch.tensor(X_OUT, dtype=torch.float32, device=device).requires_grad_(True)
    
    
    G_NN.train()
    optimizer = optim.Adam(G_NN.parameters(), lr)
    
    old_loss = 1
    loss = []

    for epoch in range(num_epoch+1):
        
        gen_samples = G_NN(t,x_rand)

        G_nn_t =  G_NN.grad_t(t, x_rand)
       
        Loss_ode = torch.mean(( G_nn_t - an.dynamics_torch(t,G_NN(t, x_rand),V_NN))**2)
        Loss_G = torch.mean((G_NN(t_train, X_train) - X_OUT)**2) 

        Loss_total =  Loss_G   + 0.5*Loss_ode  

        optimizer.zero_grad()

        Loss_total.backward()

        optimizer.step()


        if epoch % 1000 == 0:
            loss.append(old_loss)    
            new_loss = Loss_total.item()
            print(f"Iteration {epoch}:  Loss_G = {Loss_G.item():.4e},  Loss_ODE = {Loss_ode.item():.4e},  Loss_total = {Loss_total.item():.4e}")
            if new_loss>min(loss):
               x_rand = x_rand
            else:   
               x_rand = an.gen_x0(num_samples, Torch=True).requires_grad_(True)
            old_loss = new_loss    
    
    print('\n')
    return G_NN_original, G_NN



#==================================Compute J and V ==============================#

def Comp_J0(an, t, x, V_NN):
    
    t_expanded = t.repeat_interleave(x.shape[0]).view(-1,1)
    x0_expanded = x.repeat(t.shape[0],1)
    
    Xn = an.G_NN_list[-1](t_expanded, x0_expanded)

    mean = torch.mean(Xn.reshape(t.shape[0], x.shape[0]),
                            dim=1
                        )
    
    F = 0.5 * (Xn.reshape(t.shape[0], x.shape[0])-mean.view(-1,1))**2


    u = - V_NN.get_grad(t_expanded, Xn)

    l = 0.5 * (u**2).reshape(t.shape[0], x.shape[0]) +  F
    J = torch.mean(1/t.shape[0] * torch.sum(l, dim=0))
    
    return J.item()

def Comp_J1(an, t, x, V_NN, G_NN2):
    
    t_expanded = t.repeat_interleave(x.shape[0]).view(-1,1)
    x0_expanded = x.repeat(t.shape[0],1)
    
    
    Xn = G_NN2(t_expanded, x0_expanded)
    
    dist_0 = an.G_NN_list[-1](t_expanded, x0_expanded)

    mean = torch.mean(dist_0.reshape(t.shape[0], x.shape[0]),
                            dim=1
                        )
    
    F = 0.5 * (Xn.reshape(t.shape[0], x.shape[0])-mean.view(-1,1))**2


    u = - V_NN.get_grad(t_expanded, Xn)

    l = 0.5 * (u**2).reshape(t.shape[0], x.shape[0]) +  F
    J = torch.mean(1/t.shape[0] * torch.sum(l, dim=0))
    
    return J.item()


def Comp_V(x, V_NN):
    
    t0 = torch.zeros_like(x)
    V0 = torch.mean(V_NN(t0, x)) 
    
    return V0.item()
    
  

#==================================Update==============================#
def wasserstein_distance_1(model1, model2, t, x0):
    
    
    t_expanded = t.repeat_interleave(x0.shape[0]).view(-1,1)
    x0_expanded = x0.repeat(t.shape[0],1)
    

    M_old = model1(t_expanded, x0_expanded).reshape(t.shape[0], x0.shape[0])
    
    M_new = model2(t_expanded, x0_expanded).reshape(t.shape[0], x0.shape[0])
    
    # Ensure tensors are on CPU and converted to numpy
    M_old_np = M_old.detach().cpu().numpy()
    M_new_np = M_new.detach().cpu().numpy()

    # Compute Wasserstein distances per time step
    distances = []
    for i in range(M_old_np.shape[0]):
        d = wasserstein_distance(M_old_np[i], M_new_np[i])
        distances.append(d)

    distances = np.array(distances)

    # Compute norms of the distance vector
    norm_l1 = np.linalg.norm(distances, ord=1)
    norm_l2 = np.linalg.norm(distances, ord=2)
    norm_linf = np.linalg.norm(distances, ord=np.inf)

    # Print
    print("\n=== Wasserstein Distance Norms over Time ===")
    print(f"L1 norm    : {norm_l1:.4e}")
    print(f"L2 norm    : {norm_l2:.4e}")
    print(f"Linf norm  : {norm_linf:.4e}")
    print("============================================\n")
    
def wasserstein_population_vs_br(an, GNN_list, GNN_new, m0, t, x0, p=1):

    T      = t.shape[0]
    N0, d  = x0.shape
    k      = len(GNN_list)              
    distances = []

    # boucle sur les pas de temps
    for ti in t:
        
        # 1) bar(m) from générateurs historiques
        if an.Round==0:
            pop_pts = [m0.cpu()]
        else:
             pop_pts = []
        for G in GNN_list:
            with torch.no_grad():
                ti_b = ti.repeat_interleave(N0 ).view(-1, 1)   # (N0,1)
                pts  = G(ti_b, x0).cpu().numpy()              # (N0,d)
            pop_pts.append(pts)

        X = np.concatenate(pop_pts, axis=0)                   # ((k+1)*N0 , d)
        a = np.ones(X.shape[0]) / X.shape[0]                  # poids uniformes

        # 2) mesure du best-response μ^k
        with torch.no_grad():
            ti_b = ti.repeat_interleave(N0).view(-1, 1)
            Y    = GNN_new(ti_b, x0).cpu().numpy()           # (N0,d)
        b = np.ones(Y.shape[0]) / Y.shape[0]

        # 3) Wasserstein-p
        C   = ot.dist(X, Y, metric='euclidean') #** p
        Wp  = ot.emd2(a, b, C) #** (1.0 / p)
        distances.append(Wp)

    distances = np.array(distances)

    # Compute norms of the distance vector
    norm_l1 = np.linalg.norm(distances, ord=1)
    norm_l2 = np.linalg.norm(distances, ord=2)
    norm_linf = np.linalg.norm(distances, ord=np.inf)

    # Print
    print("\n=== Wasserstein Distance Norms over Time ===")
    print(f"L1 norm    : {norm_l1:.4e}")
    print(f"L2 norm    : {norm_l2:.4e}")
    print(f"Linf norm  : {norm_linf:.4e}")
    print("============================================\n")
    
def wasserstein_population_vs_br_geom(an, GNN_list, GNN_new, m0, t, x0, p=1, blur=0.05):
    """
    Compute Wasserstein-p distances over time between the population distribution (historical generators)
    and the best response, using GeomLoss for efficiency.
    """

    T      = t.shape[0]
    N0, d  = x0.shape
    k      = len(GNN_list)
    device = x0.device

    distances = []

    # Define the Sinkhorn-based loss
    loss_fn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend="tensorized")

    for ti in t:
        # 1) Compute population distribution (bar(m))
        if an.Round == 0:
            pop_pts = [m0]
        else:
            pop_pts = []

        for G in GNN_list:
            with torch.no_grad():
                ti_b = ti.repeat_interleave(N0).view(-1, 1).to(device)
                pts = G(ti_b, x0)  # shape (N0, d)
            pop_pts.append(pts)

        X = torch.cat(pop_pts, dim=0)  # shape ((k+1)*N0, d) if Round > 0

        # 2) Best-response samples
        with torch.no_grad():
            ti_b = ti.repeat_interleave(N0).view(-1, 1).to(device)
            Y = GNN_new(ti_b, x0)  # shape (N0, d)

        # 3) Compute Wasserstein distance using GeomLoss
        Wp = loss_fn(X, Y).item()
        distances.append(Wp)

    # Convert to numpy array for norm computations
    distances = np.array(distances)

    # Compute norms
    norm_l1 = np.linalg.norm(distances, ord=1)
    norm_l2 = np.linalg.norm(distances, ord=2)
    norm_linf = np.linalg.norm(distances, ord=np.inf)

    # Print results
    print("\n=== Wasserstein Distance Norms over Time (GeomLoss) ===")
    print(f"L1 norm    : {norm_l1:.4e}")
    print(f"L2 norm    : {norm_l2:.4e}")
    print(f"Linf norm  : {norm_linf:.4e}")
    print("=========================================================\n")
    
    
# import numpy as np, torch, ot

# def wasserstein_population_vs_br(an,
#                                  GNN_list,
#                                  GNN_new,
#                                  m0,        # ndarray (N0,d) déjà float64
#                                  t,         # (T,) tensor
#                                  x0,        # (N0,d) tensor  (peut être float32)
#                                  p: int = 2,
#                                  use_sinkhorn: bool = False,
#                                  reg: float = 1e-3):
#     """
#     Calcule W_p( \bar m^k , μ^k ) pour chaque pas de temps ti.
#     - Conversion systématique en float64 pour éviter les résidus numériques.
#     - Si use_sinkhorn=True : ot.sinkhorn2 (plus rapide quand |X||Y| est grand).
#     Retourne distances (T,) + normes L1, L2, Linf.
#     """
#     # ------------------------------------------------------------------
#     # 0. préparation
#     # ------------------------------------------------------------------
#     T          = t.shape[0]
#     N0, d      = x0.shape
#     k          = len(GNN_list)
#     distances  = []

#     # convertit x0 en float64 une seule fois
#     x0_np64 = x0.detach().cpu().numpy().astype(np.float64)

#     # ------------------------------------------------------------------
#     # 1. boucle sur les temps
#     # ------------------------------------------------------------------
#     for ti in t:                          # ti est un scalaire tensor
#         # 1.a population \bar m^k
#         pop_pts = []
#         if an.Round == 0:
#             pop_pts.append(m0.astype(np.float64))   # m0 déjà float64

#         for G in GNN_list:
#             with torch.no_grad():
#                 ti_b = ti.repeat_interleave(N0).view(-1, 1)
#                 pts  = G(ti_b, x0).cpu().numpy().astype(np.float64)
#             pop_pts.append(pts)

#         X = np.concatenate(pop_pts, axis=0)         # ((k+1)*N0, d) float64
#         a = np.ones(len(X), dtype=np.float64) / len(X)

#         # 1.b best-response μ^k
#         with torch.no_grad():
#             ti_b = ti.repeat_interleave(N0).view(-1, 1)
#             Y = GNN_new(ti_b, x0).cpu().numpy().astype(np.float64)
#         b = np.ones(len(Y), dtype=np.float64) / len(Y)

#         # 1.c coût et W_p
#         C = ot.dist(X, Y, metric='euclidean').astype(np.float64)
#         if p != 1:
#             C_p = C ** p
#         else:
#             C_p = C

#         if use_sinkhorn:
#             Wp_val = ot.sinkhorn2(a, b, C_p, reg=reg)
#             if p != 1:
#                 Wp_val = Wp_val ** (1 / p)
#         else:
#             Wp_val = ot.emd2(a, b, C_p)
#             if p != 1:
#                 Wp_val = Wp_val ** (1 / p)

#         distances.append(Wp_val)

#     distances = np.asarray(distances)           # (T,)

#     # ------------------------------------------------------------------
#     # 2. normes
#     # ------------------------------------------------------------------
#     L1, L2, Linf = map(float, [
#         np.linalg.norm(distances, 1),
#         np.linalg.norm(distances, 2),
#         np.linalg.norm(distances, np.inf)
#     ])

#     # ------------------------------------------------------------------
#     # 3. affichage
#     # ------------------------------------------------------------------
#     print("\n=== Wasserstein-{} : normes sur {} pas de temps ===".format(p, T))
#     print(f"L1   = {L1:.4e}")
#     print(f"L2   = {L2:.4e}")
#     print(f"Linf = {Linf:.4e}")
#     print("===============================================\n")

#     return distances, L1, L2, Linf

def wasserstein_fp(an, G_NN_list,              # BR passés (longueur k)
                   G_NN_new,               # BR courant   (G_k)
                   x0, m0,                    # (N,d)  échantillon initial
                   t_grid,                 # (T,1)  instants
                   p          = 1,
                   blur       = 0.05,
                   device     = "cpu"):
    """
    W_p(  m̄_k(t) , BR_k(t) )  pour t ∈ t_grid,   avec
        m̄_k = (1/(k+1)) [ δ_{x0} + Σ_{i=0}^{k-1} δ_{G_i} ].
    Retourne  np.ndarray shape (T,)  des distances.
    """

    T       = t_grid.shape[0]
    N, d    = x0.shape
    k       = len(G_NN_list)
    loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur, backend="tensorized")

    # ---------- pré-calcul des trajectoires BR_k -------------------
    with torch.no_grad():
        t_big = t_grid.repeat_interleave(N, dim=0)          # (T*N,1)
        x_rep = x0.repeat(T, 1)                             # (T*N,d)
        br_k  = G_NN_new(t_big, x_rep).view(T, N, d)        # (T,N,d)

    # ---------- pré-calcul des trajectoires passées ----------------
    past_traj = []  
    for G in G_NN_list:
        with torch.no_grad():
            past = G(t_big, x_rep).view(T, N, d)
        past_traj.append(past)
    if past_traj:
        past_traj = torch.stack(past_traj, dim=0)           # (k,T,N,d)

    # ---------- distances -----------------------------------------
    dists = torch.empty(T, device="cpu")
    for j in range(T):
        # population : x0 + BR passés (chacun N pts)
        if an.Round == 0:
            parts = [m0]                                        # (N,d)
        else:
            parts = [] 
        if k:
            parts.append(past_traj[:, j, :, :].reshape(k*N, d))
        X = torch.cat(parts, dim=0)                         # ((k+1)*N,d)
        Y = br_k[j]                                         # (N,d)
        dists[j] = loss_fn(X, Y).cpu()

    return dists.numpy()        # shape (T,)


    
