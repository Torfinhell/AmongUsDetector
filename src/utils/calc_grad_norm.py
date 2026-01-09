from torch import nn
def calculate_norm_grad(module:nn.Module):
        total_sq_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_sq_norm += param_norm ** 2
        return total_sq_norm.sqrt()