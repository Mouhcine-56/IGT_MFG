import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

bias_bool = True

class V_Net(nn.Module):
    def __init__(self, dim, ns, act_func, hh, psi_func, device, TT):
        super().__init__()
        
        # Core network layers
        self.lin1 = nn.Linear(dim + 1, ns, bias=bias_bool)
        self.lin2 = nn.Linear(ns, ns, bias=bias_bool)
        self.lin3 = nn.Linear(ns, ns, bias=bias_bool)
        self.linlast = nn.Linear(ns, 1)

        # Activation function and residual step
        self.act_func = act_func
        self.hh = hh

        # Problem-specific final condition
        self.psi_func = psi_func

        # Time interpolation (unused?)
        self.lintt = nn.Linear(1, dim)  # Not used in forward currently

        self.dim = dim
        self.TT = TT
        self.device = device

    def forward(self, t, inp):
        """
        Forward pass with time input `t` and spatial input `inp`.
        Implements residual MLP followed by time-interpolated blending
        between learned dynamics and terminal condition.
        """
        # Concatenate time and space
        out = torch.cat((t, inp), dim=1)

        # ResNet-style layers
        out = self.act_func(self.lin1(out))
        out = self.act_func(out + self.hh * self.lin2(out))
        out = self.act_func(out + self.hh * self.lin3(out))
        out = self.linlast(out)

        # Time-dependent interpolation coefficients (φ1, φ2)
        # φ1 = 1 - exp(t - 1), φ2 = exp(t - 1)
        c1 = 1 - torch.exp(t.view(-1, 1) - 1)
        c2 = torch.exp(t.view(-1, 1) - 1)

        # Alternate linear interpolation:
        # c1 = 1 - t.view(-1, 1)
        # c2 = t.view(-1, 1)

        # Final output: interpolated between learned output and ψ(x)
        return c1 * out + c2 * self.psi_func(inp).view(-1, 1)

    def get_grad(self, t, inp):
        """
        Compute ∇_x V(t, x) using autograd.
        Inputs:
            - t: scalar or tensor of shape (batch_size, 1)
            - inp: tensor of shape (batch_size, dim)
        Returns:
            - gradient of output with respect to `inp`
        """
        self.eval()

        # Prepare input tensors with gradients
        t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device).clone().detach()
        inp_tensor = torch.tensor(inp, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)

        # Forward pass
        output = self(t_tensor, inp_tensor)

        # Backward pass
        grad_outputs = torch.ones_like(output)
        grad_inputs = torch.autograd.grad(
            outputs=output,
            inputs=inp_tensor,
            grad_outputs=grad_outputs,
            create_graph=True
        )[0]

        return grad_inputs

