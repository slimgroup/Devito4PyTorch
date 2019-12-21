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

class DevitoLayer(nn.Module):
    def __init__(self, shots, shape, origin, spacing, m0, dm, noise=0.0, device='cpu'):
        super(DevitoLayer, self).__init__()
        
        self.forward_born = ForwardBorn()
        self.noise = noise
        self.device = device
        self.dm = dm
        self.spacing = spacing
        epsilon = np.sqrt(noise)*np.random.randn(shots.shape[0], shots.shape[1], shots.shape[2])
        self.shots = shots + epsilon

        self.model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=1/np.sqrt(m0))
        self.T = self.mute_top(dm, mute_end=20, length=5)
        t0 = 0.
        tn = 2000.0
        dt = self.model0.critical_dt
        nt = int(1 + (tn-t0) / dt)
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

    def forward(self, x, model, src, rec, device='cpu'):
        data = self.forward_born.apply(x, model, src, rec, self.device)
        return data

    def mute_top(self, image, mute_end=20, length=1):

        mute_start = mute_end - length
        damp = torch.zeros(image.shape, device=self.device, requires_grad=False)
        damp[:, :, :, mute_end:] = 1.
        damp[:, :, :, mute_start:mute_end] = (1. + torch.sin((np.pi/2.0*torch.arange(0, length))/(length)))/2.
        def T(image, damp=damp): return damp*image
        return T

    def create_operators(self):

        self.mixing_weights = np.zeros([self.nsrc], dtype=np.float32)
        self.src = RickerSource(name='src', grid=self.model0.grid, f0=self.f0, time=self.time_range, npoint=self.nsimsrc)
        self.src.coordinates.data[:,0] = np.linspace(0, self.model0.domain_size[0], num=self.nsimsrc)
        self.src.coordinates.data[:,-1] = 2.0*self.spacing[1]
        for s in range(self.nsrc):
            self.mixing_weights[s] = np.random.randn()
            self.src.data[:, s] *= self.mixing_weights[s]
        def f(dm=self.dm, model0=self.model0, src=self.src, rec=self.rec):
            return self.forward(self.T(dm), model0, src, rec, device=self.device)

        return f

    def mix_data(self):

        y = np.zeros(self.shots.shape[1:], dtype=np.float32)
        for s in range(self.nsrc):
            y += self.mixing_weights[s]*self.shots[s, :, :]

        return y



if __name__ == '__main__':

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print(' [*] GPU is available')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    data_path = os.path.join(os.path.expanduser('~'), 'data/nonlinear-seq-shots.hdf5')
    if not os.path.isfile(data_path):
        os.system("wget https://www.dropbox.com/s/04a3xblk0634mm4/nonlinear-seq-shots.hdf5 -O" + data_path)            
    y = h5py.File(data_path, 'r')["data"][...]
    
    m0, m, dm, spacing, shape, origin = judi_model()
    x = dm.to(device)
    wave_solver = DevitoLayer(y, shape, origin, spacing, m0, dm, noise=0.0, device=device)

    x_est = torch.zeros(x.size()).to(device)
    x_est.requires_grad = True
    zero_vec = torch.zeros(x_est.size(), device=device)

    optim = torch.optim.ASGD([x_est], lr=0.001, weight_decay=1.0e10)
    l2_loss = torch.nn.MSELoss()

    for j in range(10):
        optim.zero_grad()
        A = wave_solver.create_operators()
        y = wave_solver.mix_data()
        y = torch.from_numpy(y)
        y = y.to(device)

        pred = A(x_est)
        x_loss = l2_loss(pred.reshape(-1), y.reshape(-1))# + 1e4*l2_loss(x_est, zero_vec)
        # grad_x = torch.autograd.grad(x_loss, [x_est], create_graph=False)
        # for param, grad in zip([x_est], grad_x):
        #     param.grad = grad

        x_loss.backward()
        optim.step()

        fig = plt.figure("G(z_0)", dpi=100, figsize=(7, 2.5))
        plt.imshow(x_est[0, 0, :, :].cpu().detach().t().numpy(), vmin=-3.0/90.0, vmax=5.0/90.0)
        plt.colorbar(fraction=0.014, pad=0.03)
        plt.savefig(os.path.join("figs/", "X0_" + 
            str(j) + ".png"), format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(("Itr: [%d/%d] | loss: %4.8f | model loss: %4.8f" % (j+1, 10, x_loss, l2_loss(x_est, x))))

    from IPython import embed; embed()

    extent = np.array([0., dm.shape[2]*spacing[0], pred.shape[1]*wave_solver.dt, 0.])/1.0e3
    fig = plt.figure("predicted data", dpi=100, figsize=(6, 10))
    plt.imshow(pred.detach().numpy(), vmin=-1.01, vmax=1.01, cmap="RdGy", interpolation='bicubic', \
        extent=extent, aspect='auto')
    plt.title(r"$\hat{\delta {d}}$")
    fig = plt.figure("true data", dpi=100, figsize=(6, 10))
    plt.imshow(y.detach().numpy(), vmin=-1.01, vmax=1.01, cmap="RdGy", interpolation='bicubic', \
        extent=extent, aspect='auto')
    plt.title(r"$\delta {d}$")

    extent_m = np.array([0., x.shape[2]*spacing[0], x.shape[3]*spacing[1], 0.])/1.0e3
    fig = plt.figure("Gradient", dpi=100, figsize=(7, 2.5))
    plt.imshow(grad_x[0, 0, :, :].detach().t().numpy(), aspect=1, extent=extent_m)
    plt.title("gradient of " + r"$J^{T} (Jx - \delta d)$");
    plt.colorbar(fraction=0.0145, pad=0.01, format=sfmt); plt.xlabel("Horizontal distance (km)")
    plt.ylabel("Depth (km)")
    plt.show()
    