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


class AdjointBornLayer(torch.nn.Module):
    """
    Creates a wrapped forward-born operator
    """
    def __init__(self, model, geometry, device):
        super(AdjointBornLayer, self).__init__()
        self.forward_born = devito_wrapper.AdjointBorn()
        self.model = model
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model, self.geometry,
                                         space_order=8)

    def forward(self, x):
        return self.forward_born.apply(x, self.model, self.geometry,
                                       self.solver, self.device)


def test_adjoint_born():

    tn = 1000.
    shape = (101, 101)
    nbl = 40
    model = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                       spacing=(10., 10.), nbl=nbl, nlayers=2,
                       space_order=8)
    model0 = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                        spacing=(10., 10.), nbl=nbl, nlayers=2,
                        space_order=8)

    gaussian_smooth(model0.vp, sigma=(1, 1))
    geometry0 = setup_geometry(model0, tn)

    # Pure Devito
    solver0 = AcousticWaveSolver(model0, geometry0, space_order=8)
    dm = model.vp.data**(-2) - model0.vp.data**(-2)

    grad_devito = np.array(solver0.jacobian(dm)[0].data)

    # Devito4PyTorch
    dm = torch.from_numpy(np.array(dm[nbl:-nbl, nbl:-nbl])).to(device)

    d_est = torch.zeros(geometry0.rec.data.shape, requires_grad=True,
                        device=device)
    adjoint_born = AdjointBornLayer(model0, geometry0, device)

    loss = 0.5*torch.norm(adjoint_born(d_est) - dm)**2

    # Negative gradient to be compared with linearized data computed above
    grad = -torch.autograd.grad(loss, d_est, create_graph=False)[0]

    # Test
    rel_err = np.linalg.norm(grad.cpu().numpy()
                             - grad_devito) / np.linalg.norm(grad_devito)
    assert np.isclose(rel_err, 0., atol=1.e-6)
