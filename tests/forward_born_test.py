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

class ForwardBornLayer(torch.nn.Module):
    def __init__(self, model, geometry, device):
        super(ForwardBornLayer, self).__init__()
        self.forward_born = devito_wrapper.ForwardBorn()
        self.model = model
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model, self.geometry, space_order=8)

    def forward(self, x):
        return self.forward_born.apply(x, self.model, self.geometry, self.solver, self.device)


if __name__ == '__main__':

    tn = 1000.
    shape = (101, 101)
    model = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                          spacing=(10., 10.), nbl=40, nlayers=5)
    model0 = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                          spacing=(10., 10.), nbl=40, nlayers=5)
    model0.vp = ndimage.gaussian_filter(model0.vp.data[nb:-nb, nb:-nb], sigma=(1, 1), order=0)
    geometry0 = setup_geometry(model0, tn)
    geometry = setup_geometry(model, tn)

    ### Pure Devito
    solver = AcousticWaveSolver(model, geometry, space_order=8)
    solver0 = AcousticWaveSolver(model0, geometry0, space_order=8)

    d = solver.forward()[0]
    d0, u0 = solver0.forward(save=True)[:2]
    d_lin = d.data - d0.data

    rec = geometry0.rec
    rec.data[:] = -d_lin[:]
    nb = model.nbl
    grad_devito = np.array(solver0.gradient(rec, u0)[0].data[nb:-nb, nb:-nb])

    ### Deito4PyTorch
    d_lin = torch.from_numpy(np.array(d_lin)).to(device)
    
    forward_born = ForwardBornLayer(model0, geometry0, device)
    dm_est = torch.zeros([1, 1, shape[0], shape[1]], requires_grad=True, device=device)

    loss = 0.5*torch.norm(forward_born(dm_est) - d_lin)**2
    grad = torch.autograd.grad(loss, dm_est, create_graph=False)[0]

    # Test
    assert np.isclose(grad.cpu().numpy() - grad_devito, 0., atol=1.e-8).all()