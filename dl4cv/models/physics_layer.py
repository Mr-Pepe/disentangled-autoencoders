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
        self.forwardMatrix = nn.Parameter(
            torch.tensor(
                [[1., 0.],
                 [0., 1.],
                 [dt, 0.],
                 [0., dt],
                 [d2, 0.],
                 [0., d2]]
            )
        )
        self.forwardMatrix.requires_grad = False
        self.num_latents_in = 6
        self.num_latents_out = 2

    def forward(self, x):
        """
        x.shape: [batch, 6, 1, 1]
        return shape: [batch, 2, 1, 1]
        """
        return torch.mm(
            x.flatten(start_dim=1),
            self.forwardMatrix
        )[:, :, None, None]


class PhysicsPV(nn.Module):
    """
        Equation of motion for a system which has position (P) and
        velocity (V). Propagates the system one timestep (dt) forward
        assuming constant velocity.
    """
    def __init__(self, dt: float):
        super(PhysicsPV, self).__init__()
        self.forwardMatrix = nn.Parameter(
            torch.tensor(
                [[1., 0.],
                 [0., 1.],
                 [dt, 0.],
                 [0., dt]]
            )
        )
        self.forwardMatrix.requires_grad = False
        self.num_latents_in = 4
        self.num_latents_out = 2

    def forward(self, x):
        """
        x.shape: [batch, 4, 1, 1]
        return shape: [batch, 2, 1, 1]
        """
        return torch.mm(
            x.flatten(start_dim=1),
            self.forwardMatrix
        )[:, :, None, None]


"""
pos_x(t+1) = pos_x(t) + vel_x(t) * dt + acc_x(t) * dt^2
pos_y(t+1) = pos_y(t) + vel_y(t) * dt + acc_y(t) * dt^2
vel_x(t+1) = vel_x(t) + acc_x(t) * dt
vel_y(t+1) = vel_y(t) + acc_y(t) * dt
acc_x(t+1) = acc_x(t)
acc_y(t+1) = acc_y(t)

[pos_x(t+1)]        [1., 0., dt, 0., d2, 0.]        [pos_x(t)]
[pos_y(t+1)]        [0., 1., 0., dt, 0., d2]        [pos_y(t)]
[vel_x(t+1)]   =    [0., 0., 1., 0., dt, 0.]    *   [vel_x(t)] 
[vel_y(t+1)]        [0., 0., 0., 1., 0., dt]        [vel_y(t)]
[acc_x(t+1)]        [0., 0., 0., 0., 1., 0.]        [acc_x(t)] 
[acc_y(t+1)]        [0., 0., 0., 0., 0., 1.]        [acc_y(t)] 
"""
