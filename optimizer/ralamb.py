# Credits : https://github.com/mgrankin/over9000
import torch, math
from torch.optim.optimizer import Optimizer

# RAdam + LARS + GC
class Ralamb(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, gc_conv_only=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("Ralamb does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True), alpha=-1)

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # trying yogi
                # grad_squared = grad.mul(grad)
                # exp_avg_sq.mul_(beta2).addcmul_(-(1 - beta2), torch.sign(exp_avg_sq - grad_squared), grad_squared)

                state["step"] += 1
                buffered = self.buffer[int(state["step"] % 10)]

                if state["step"] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state["step"])
                    buffered[2] = radam_step_size

                # if group["weight_decay"] != 0:
                #    p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)

                # more conservative since it's an approximated value
                radam_step = p_data_fp32.clone()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    radam_step.addcdiv_(exp_avg, denom, value=-radam_step_size * group["lr"])
                    # GC
                    G_grad = exp_avg.div(denom)
                else:
                    radam_step.add_(exp_avg, alpha=-radam_step_size * group["lr"])
                    # GC
                    G_grad = exp_avg

                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state["weight_norm"] = weight_norm
                state["adam_norm"] = radam_norm
                state["trust_ratio"] = trust_ratio

                # GC operation for Conv layers and FC layers
                if G_grad.dim() > self.gc_gradient_threshold:
                    G_grad.add_(G_grad.mean(dim=tuple(range(1, G_grad.dim())), keepdim=True), alpha=-1)

                p_data_fp32.add_(G_grad, alpha=-radam_step_size * group["lr"] * trust_ratio)

                p.data.copy_(p_data_fp32)

        return loss
