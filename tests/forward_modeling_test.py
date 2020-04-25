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

class ForwardModelingLayer(torch.nn.Module):
    def __init__(self, model, geometry, device):
        super(ForwardModelingLayer, self).__init__()
        self.forward_modeling = devito_wrapper.ForwardModeling()
        self.model = model
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model, self.geometry, space_order=8)

    def forward(self, x):
        return self.forward_modeling.apply(x, self.model, self.geometry, self.solver, self.device)


if __name__ == '__main__':

    tn = 1000.
    shape = (101, 101)
    model = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                          spacing=(10., 10.), nbl=40, nlayers=5)
    model0 = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                          spacing=(10., 10.), nbl=40, nlayers=5)
    model0.vp.data[:] = ndimage.gaussian_filter(model0.vp.data, sigma=(1, 1), order=0) 
    geometry0 = setup_geometry(model0, tn)
    geometry = setup_geometry(model, tn)

    ### Pure Devito
    solver = AcousticWaveSolver(model, geometry, space_order=8)
    solver0 = AcousticWaveSolver(model0, geometry0, space_order=8)

    d = solver.forward()[0]
    d0, u0 = solver0.forward(save=True)[:2]
    residual = d.data - d0.data

    rec = geometry0.rec
    rec.data[:] = -residual[:]
    nb = model.nbl
    grad_devito = np.array(solver0.gradient(rec, u0)[0].data)[nb:-nb, nb:-nb]

    ### Deito4PyTorch
    d = torch.from_numpy(np.array(d.data)).to(device)

    forward_modeling = ForwardModelingLayer(model0, geometry0, device)

    m0 = np.array(model0.vp.data**(-2))[nb:-nb, nb:-nb]
    m0 = torch.Tensor(m0).unsqueeze(0).unsqueeze(0).to(device)
    m0.requires_grad = True

    loss = 0.5*torch.norm(forward_modeling(m0) - d)**2
    grad = torch.autograd.grad(loss, m0, create_graph=False)[0]

    # Test
    assert np.isclose(grad.cpu().numpy() - grad_devito, 0., atol=1.e-8).all()