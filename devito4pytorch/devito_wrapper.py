import torch
import numpy as np
from devito.builtins import initialize_function
from devito import Function


class ForwardBorn(torch.autograd.Function):
    """
    Wrapping the forward-born operator
    """
    @staticmethod
    def forward(ctx, input, model, geometry, solver, device):

        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device

        # Prepare input
        input = torch.nn.ReplicationPad2d((ctx.model.nbl))(input)
        input = input.detach().cpu().numpy()

        # Linearized forward modeling
        d_lin = ctx.solver.jacobian(input[0, 0, :, :])[0].data

        return torch.from_numpy(np.array(d_lin)).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = grad_output[:]

        # Adjoint linearized modeling
        u0 = ctx.solver.forward(save=True)[1]
        g = ctx.solver.jacobian_adjoint(rec, u0)[0].data

        # Remove padding
        nb = ctx.model.nbl
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None


class AdjointBorn(torch.autograd.Function):
    """
    Wrapping the adjoint-born operator
    """
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
        g = ctx.solver.jacobian_adjoint(rec, u=u0)[0].data

        # Remove padding
        nb = ctx.model.nbl
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1])

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = torch.nn.ReplicationPad2d((ctx.model.nbl))(grad_output)
        grad_output = grad_output.detach().cpu().numpy()[0, 0, :, :]

        # Linearized forward modeling
        d_lin = ctx.solver.jacobian(grad_output)[0].data

        return (torch.from_numpy(np.array(d_lin)).to(ctx.device), None,
                None, None, None)


class ForwardModeling(torch.autograd.Function):
    """
    Wrapping forward-modeling operator
    """
    @staticmethod
    def forward(ctx, input, model, geometry, solver, device):

        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device

        # Prepare input
        input = input[0, 0, ...].detach().cpu().numpy()
        vp = Function(name='vp', grid=ctx.model.grid,
                      space_order=ctx.model.space_order)
        initialize_function(vp, input**(-0.5), ctx.model.nbl)
        ctx.model.vp = vp

        # Nonlinear forward modeling
        d_nonlin, ctx.u0 = ctx.solver.forward(save=True)[:2]

        return torch.from_numpy(np.array(d_nonlin.data)).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = grad_output[:]

        g = ctx.solver.jacobian_adjoint(rec, u=ctx.u0)[0].data

        # Remove padding
        nb = ctx.model.nbl
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None


class AdjointModeling(torch.autograd.Function):
    """
    Wrapping adjoint-modeling operator
    """
    @staticmethod
    def forward(ctx, input, model, geometry, solver, device):

        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device

        # Prepare input
        input = input[0, 0, ...].detach().cpu().numpy()
        vp = Function(name='vp', grid=ctx.model.grid,
                      space_order=ctx.model.space_order)
        initialize_function(vp, input**(-0.5), ctx.model.nbl)
        ctx.model.vp = vp

        # Nonlinear forward modeling
        d_nonlin, ctx.u0 = ctx.solver.forward(save=True,
                                              vp=ctx.model.vp)[:2]

        return torch.from_numpy(np.array(d_nonlin.data)).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = grad_output[:]

        g = ctx.solver.jacobian_adjoint(rec, u=ctx.u0)[0].data

        # Remove padding
        nb = ctx.model.nbl
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None
