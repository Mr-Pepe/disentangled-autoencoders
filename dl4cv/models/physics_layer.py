import torch
import torch.nn as nn


class PhysicsPVA(nn.Module):
    """
    Equation of motion for a system which has position (P), velocity (V)
    and acceleration (A). Propagates the system one timestep (dt) forward
    assuming constant acceleration.
    """
    def __init__(self, dt: float):
        super(PhysicsPVA, self).__init__()
        d2 = 0.5 * (dt**2)
        self.forwardMatrix = torch.tensor(
            [[1., 0., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0.],
             [dt, 0., 1., 0., 0., 0.],
             [0., dt, 0., 1., 0., 0.],
             [d2, 0., dt, 0., 1., 0.],
             [0., d2, 0., dt, 0., 1.]],
            requires_grad=False
        )

    def forward(self, x):
        return torch.mm(x, self.forwardMatrix)
