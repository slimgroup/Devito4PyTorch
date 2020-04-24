import torch
import numpy as np
from devito_wrapper import *
from examples.seismic import demo_model, setup_geometry
from examples.seismic.acoustic import AcousticWaveSolver
from scipy import ndimage
import matplotlib.pyplot as plt

if not torch.cuda.is_available():
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class ForwardModelingLayer(torch.nn.Module):
    def __init__(self, model, geometry, device):
        super(ForwardModelingLayer, self).__init__()
        
        self.forward_modeling = ForwardModeling()

        self.model = model
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model, self.geometry, space_order=8)

    def forward(self, x):
        data = self.forward_modeling.apply(x, self.model, self.geometry, self.solver, self.device)
        return data

class ForwardBornLayer(torch.nn.Module):
    def __init__(self, model, geometry, device):
        super(ForwardBornLayer, self).__init__()
        
        self.forward_born = ForwardBorn()

        self.model = model
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model, self.geometry, space_order=8)

    def forward(self, x):
        data = self.forward_born.apply(x, self.model, self.geometry, self.solver, self.device)
        return data

if __name__ == '__main__':

    tn = 1000.
    model = demo_model('layers-isotropic', origin=(0., 0.), shape=(101, 101),
                          spacing=(10., 10.), nbl=40, nlayers=5)
    model0 = demo_model('layers-isotropic', origin=(0., 0.), shape=(101, 101),
                          spacing=(10., 10.), nbl=40, nlayers=5)
    model0.vp.data[:] = ndimage.gaussian_filter(model0.vp.data, sigma=(1, 1), order=0) 
    geometry0 = setup_geometry(model0, tn)
    geometry = setup_geometry(model, tn)

    nb = model.nbl
    m = np.float32(model.vp.data**(-2))[nb:-nb, nb:-nb]
    m = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device)
    m0 = np.float32(model0.vp.data**(-2))[nb:-nb, nb:-nb]
    m0 = torch.from_numpy(m0).unsqueeze(0).unsqueeze(0).to(device)

    forward_modeling = ForwardModelingLayer(model, geometry, device)
    d_obs = forward_modeling(m)
    d_0 = forward_modeling(m0)
    d_lin0 = d_obs - d_0

    forward_born = ForwardBornLayer(model0, geometry0, device)
    dm_est = torch.zeros(m.size(), requires_grad=True, device=device)

    loss = 0.5*torch.norm(forward_born(dm_est) - d_lin0)**2
    grad = torch.autograd.grad(loss, dm_est, create_graph=False)[0]

    plt.imshow(grad[0, 0, ...].detach().cpu().T, vmin=-10, vmax=10); plt.show()


