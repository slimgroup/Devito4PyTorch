import torch
from torch.autograd import Function
import numpy as np


class ForwardBorn(Function):

    @staticmethod
    def forward(ctx, input, model, geometry, solver, device):

        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device

        # Prepare input
        input = torch.nn.ReplicationPad2d((ctx.model.nbl))(input).detach().cpu().numpy()

        # Linearized forward modeling
        d_lin = ctx.solver.born(input[0, 0, :, :])[0].data

        return torch.from_numpy(np.array(d_lin)).to(ctx.device)

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
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None


class AdjointBorn(Function):

    @staticmethod
    def forward(ctx, input, model, geometry, solver, device):

        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device

        # Adjoint born modeling
        input = input.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = input[:]

        u0 = ctx.solver.forward(save=True)[1]
        g = ctx.solver.gradient(rec, u=u0)[0].data

        # Remove padding
        nb = ctx.model.nbl
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1])

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = torch.nn.ReplicationPad2d((ctx.model.nbl))(grad_output)
        grad_output = grad_output.detach().cpu().numpy()[0, 0, :, :]

        # Linearized forward modeling
        d_lin = ctx.solver.born(grad_output)[0].data

        return torch.from_numpy(np.array(d_lin)).to(ctx.device), None, None, None, None


class ForwardModeling(Function):

    @staticmethod
    def forward(ctx, input, model, geometry, solver, device):

        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device

        # Prepare input
        input = input[0, 0, ...].detach().cpu().numpy()
        ctx.model.vp = input**(-0.5)

        # Nonlinear forward modeling
        d_nonlin, ctx.u0 = ctx.solver.forward(save=True)[:2]

        return torch.from_numpy(np.array(d_nonlin.data)).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = grad_output[:]

        g = ctx.solver.gradient(rec, u=ctx.u0)[0].data

        # Remove padding
        nb = ctx.model.nbl
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None


class AdjointModeling(Function):

    @staticmethod
    def forward(ctx, input, model, geometry, solver, device):

        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device

        # Prepare input
        input = input[0, 0, ...].detach().cpu().numpy()
        ctx.model.vp = input**(-0.5)

        # Nonlinear forward modeling
        d_nonlin, ctx.u0 = ctx.solver.forward(save=True)[:2]

        return torch.from_numpy(np.array(d_nonlin.data)).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = grad_output[:]

        g = ctx.solver.gradient(rec, u=ctx.u0)[0].data

        # Remove padding
        nb = ctx.model.nbl
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None
