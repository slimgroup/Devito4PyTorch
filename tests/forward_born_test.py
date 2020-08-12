import torch
import numpy as np
from devito4pytorch import devito_wrapper
from devito import gaussian_smooth
from examples.seismic import demo_model, setup_geometry
from examples.seismic.acoustic import AcousticWaveSolver

if not torch.cuda.is_available():
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class ForwardBornLayer(torch.nn.Module):
    """
    Creates a wrapped forward-born operator
    """
    def __init__(self, model, geometry, device):
        super(ForwardBornLayer, self).__init__()
        self.forward_born = devito_wrapper.ForwardBorn()
        self.model = model
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model, self.geometry,
                                         space_order=4)

    def forward(self, x):
        return self.forward_born.apply(x, self.model, self.geometry,
                                       self.solver, self.device)


def test_forward_born():

    tn = 1000.
    shape = (101, 101)
    nbl = 40
    model = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                       spacing=(10., 10.), nbl=nbl, nlayers=2)
    model0 = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                        spacing=(10., 10.), nbl=nbl, nlayers=2)

    gaussian_smooth(model0.vp, sigma=(1, 1))
    geometry0 = setup_geometry(model0, tn)
    geometry = setup_geometry(model, tn)

    # Pure Devito
    solver = AcousticWaveSolver(model, geometry, space_order=4)
    solver0 = AcousticWaveSolver(model0, geometry0, space_order=4)

    d = solver.forward(vp=model.vp)[0]
    d0, u0 = solver0.forward(save=True, vp=model0.vp)[:2]
    d_lin = d.data - d0.data

    rec = geometry0.rec
    rec.data[:] = -d_lin[:]
    grad_devito = solver0.jacobian_adjoint(rec, u0)[0].data
    grad_devito = np.array(grad_devito)[nbl:-nbl, nbl:-nbl]

    # Devito4PyTorch
    d_lin = torch.from_numpy(np.array(d_lin)).to(device)

    forward_born = ForwardBornLayer(model0, geometry0, device)
    dm_est = torch.zeros([1, 1, shape[0], shape[1]], requires_grad=True,
                         device=device)

    loss = 0.5*torch.norm(forward_born(dm_est) - d_lin)**2
    grad = torch.autograd.grad(loss, dm_est, create_graph=False)[0]

    # Test
    rel_err = np.linalg.norm(grad.cpu().numpy()
                             - grad_devito) / np.linalg.norm(grad_devito)
    assert np.isclose(rel_err, 0., atol=1.e-6)
