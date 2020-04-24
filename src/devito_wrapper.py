import torch
from torch.autograd import Function
import numpy as np
import copy


class ForwardBorn(Function):

    @staticmethod
    def forward(ctx, input, model, geometry, device, solver):
        input = input.to('cpu')
        # Save modeling parameters for backward pass
        ctx.solver = solver
        ctx.model = model
        ctx.geometry = geometry
        ctx.device = device

        # Prepare input
        input = torch.nn.ReplicationPad2d((ctx.model.nbl))(input).detach().cpu().numpy()

        # Linearized forward modeling
        d_lin = ctx.solver.born(input[0, 0, :, :])[0].data

        return torch.from_numpy(d_lin).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = grad_output[:]
        # Adjoint linearized modeling
        u0 = ctx.solver.forward(save=True)[1]
        g = ctx.solver.gradient(rec, u0)[0].data

        # Remove padding
        nb = ctx.model.nbl
        g = torch.from_numpy(g[nb:-nb, nb:-nb]).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None

class AdjointBorn(Function):

    @staticmethod
    def forward(ctx, input, model, geometry, device, solver):

        # Save modeling parameters for backward pass
        ctx.solver = solver
        ctx.model = model
        ctx.geometry = geometry
        ctx.device = device

        # Adjoint born modeling
        input = input.detach().cpu().numpy()
        u0 = ctx.solver.forward(save=True)[1]
        g = gradient(input.data[:], u=u0)[0].data

        # Remove padding
        nb = model.nbl
        g = torch.from_numpy(g[nb:-nb, nb:-nb]).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1])

    @staticmethod
    def backward(ctx, grad_output):

        # Prepare input
        grad_output = torch.nn.ReplicationPad2d((ctx.model.nbl))(input).detach().cpu().numpy()

        # Linearized forward modeling
        d_lin = ctx.solver.born(input[0, 0, :, :])[0].data

        return torch.from_numpy(d_lin).to(ctx.device), None, None, None, None


class ForwardModeling(Function):

    @staticmethod
    def forward(ctx, input, model, geometry, device, solver):
        # Save modeling parameters for backward pass
        # from IPython import embed; embed()
        ctx.model = model
        ctx.geometry = copy.deepcopy(geometry)
        ctx.device = device

        # Prepare input
        input = torch.nn.ReplicationPad2d((ctx.model.nbl))(input).detach().cpu().numpy()
        ctx.model.vp.data[:] = input**(-0.5)
        
        ctx.solver = solver
        # Nonlinear forward modeling
        d_nonlin, ctx.u0 = ctx.solver.forward(save=True)[:2]

        return torch.from_numpy(d_nonlin.data).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach().cpu().numpy()

        g = ctx.solver.gradient(input.data[:], u=ctx.u0)[0].data

        # Remove padding
        nb = ctx.model.nbl
        g = torch.from_numpy(g[nb:-nb, nb:-nb]).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None
