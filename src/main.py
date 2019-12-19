from wave_solver import ForwardBorn, AdjointBorn
from PySource import RickerSource, Receiver
from load_vel import judi_model
from PyModel import Model
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
sfmt=ticker.ScalarFormatter(useMathText=True) 
sfmt.set_powerlimits((0, 0))

class setup_devito(nn.Module):
    def __init__(self, shots, shape, origin, spacing, m0, dm, noise=0.0):
        super(setup_devito, self).__init__()
        
        self.forward_born = ForwardBorn()
        self.noise = noise
        self.dm = dm
        self.spacing = spacing
        epsilon = np.sqrt(noise)*np.random.randn(shots.shape[0], shots.shape[1], shots.shape[2])
        self.shots = shots + epsilon

        self.model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=1/np.sqrt(m0))
        self.T = self.mute_top(dm, mute_end=20, length=5)
        t0 = 0.
        tn = 2000.0
        self.dt = self.model0.critical_dt
        nt = int(1 + (tn-t0) / self.dt)
        self.time_range = np.linspace(t0,tn,nt)
        self.f0 = 0.008
        self.nsrc = 369
        self.nsimsrc = 369
        self.src = RickerSource(name='src', grid=self.model0.grid, f0=self.f0, time=self.time_range, npoint=self.nsimsrc)
        self.src.coordinates.data[:,0] = np.linspace(0, self.model0.domain_size[0], num=self.nsimsrc)
        self.src.coordinates.data[:,-1] = 2.0*spacing[1]
        nrec = 369
        self.rec = Receiver(name='rec', grid=self.model0.grid, npoint=nrec, ntime=nt)
        self.rec.coordinates.data[:, 0] = np.linspace(0, self.model0.domain_size[0], num=nrec)
        self.rec.coordinates.data[:, 1] = 2.0*spacing[1]

    def forward(self, x, model, src, rec):
        data = self.forward_born.apply(x, model, src, rec)
        return data

    def mute_top(self, image, mute_end=20, length=1):

        mute_start = mute_end - length
        damp = torch.zeros(image.shape, device='cpu', requires_grad=False)
        damp[:, :, :, mute_end:] = 1.
        damp[:, :, :, mute_start:mute_end] = (1. + torch.sin((np.pi/2.0*torch.arange(0, length))/(length)))/2.
        def T(image, damp=damp): return damp*image
        return T

    def create_operators(self):

        self.mixing_weights = np.zeros([self.nsrc], dtype=np.float32)
        self.src = RickerSource(name='src', grid=self.model0.grid, f0=self.f0, time=self.time_range, npoint=self.nsimsrc)
        self.src.coordinates.data[:,0] = np.linspace(0, self.model0.domain_size[0], num=self.nsimsrc)
        self.src.coordinates.data[:,-1] = 2.0*self.spacing[1]
        s = np.random.choice(self.nsrc, 1)[0]
        self.mixing_weights[s] = 1.0
        self.src.data[...] *= self.mixing_weights
        def f(dm=self.dm, model0=self.model0, src=self.src, rec=self.rec): return self.forward(self.T(dm), model0, src, rec)

        return f

    def mix_data(self):

        y = np.zeros(self.shots.shape[1:], dtype=np.float32)
        for s in range(self.nsrc):
            y += self.mixing_weights[s]*self.shots[s, :, :]

        return y



if __name__ == '__main__':

    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

    data_path = os.path.join(os.path.expanduser('~'), 'data/nonlinear-seq-shots.hdf5')
    if not os.path.isfile(data_path):
        os.system("wget https://www.dropbox.com/s/04a3xblk0634mm4/nonlinear-seq-shots.hdf5 -O" + data_path)            
    y = h5py.File(data_path, 'r')["data"][...]
    
    m0, m, dm, spacing, shape, origin = judi_model()
    x = dm.to(device)
    wave_solver = setup_devito(y, shape, origin, spacing, m0, dm, noise=2.7)

    A = wave_solver.create_operators()
    y = wave_solver.mix_data()
    y = torch.from_numpy(y)

    x_est = torch.zeros(x.size()).to('cpu')
    x_est.requires_grad = True
    pred = A(x_est)

    extent = np.array([0., dm.shape[2]*spacing[0], pred.shape[1]*wave_solver.dt, 0.])/1.0e3
    fig = plt.figure("predicted data", dpi=100, figsize=(6, 10))
    plt.imshow(pred, vmin=-1.01, vmax=1.01, cmap="RdGy", interpolation='bicubic', \
        extent=extent, aspect='auto')
    plt.title(r"$\delta {d}$")
    plt.show()

    from IPython import embed; embed()


    x_loss = torch.norm(pred.reshape(-1) - y.reshape(-1))**2
    grad_x = torch.autograd.grad(x_loss, [x_est], create_graph=False)[0]