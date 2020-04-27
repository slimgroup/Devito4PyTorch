import torch
import numpy as np
from devito4pytorch import devito_wrapper
from examples.seismic import demo_model, setup_geometry
from examples.seismic.acoustic import AcousticWaveSolver
from scipy import ndimage

if not torch.cuda.is_available():
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class AdjointBornLayer(torch.nn.Module):
    def __init__(self, model, geometry, device):
        super(AdjointBornLayer, self).__init__()
        self.forward_born = devito_wrapper.AdjointBorn()
        self.model = model
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model, self.geometry, space_order=8)

    def forward(self, x):
        return self.forward_born.apply(x, self.model, self.geometry,
                                       self.solver, self.device)


def test_adjoint_born():

    tn = 1000.
    shape = (101, 101)
    model = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                       spacing=(10., 10.), nbl=40, nlayers=5)
    model0 = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                        spacing=(10., 10.), nbl=40, nlayers=5)
    nb = model.nbl
    model0.vp = ndimage.gaussian_filter(model0.vp.data[nb:-nb, nb:-nb],
                                        sigma=(1, 1), order=0)
    geometry0 = setup_geometry(model0, tn)

    # Pure Devito
    solver0 = AcousticWaveSolver(model0, geometry0, space_order=8)
    dm = (model.vp.data**(-2) - model0.vp.data**(-2))

    grad_devito = np.array(solver0.born(-dm)[0].data)

    # Devito4PyTorch
    dm = torch.from_numpy(np.array(dm[nb:-nb, nb:-nb])).to(device)

    d_est = torch.zeros(geometry0.rec.data.shape, requires_grad=True, device=device)
    adjoint_born = AdjointBornLayer(model0, geometry0, device)

    loss = 0.5*torch.norm(adjoint_born(d_est) - dm)**2
    grad = torch.autograd.grad(loss, d_est, create_graph=False)[0]

    # Test
    assert np.isclose(grad.cpu().numpy() - grad_devito, 0., atol=1.e-8).all()
