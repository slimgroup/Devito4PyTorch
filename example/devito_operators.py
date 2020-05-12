import torch
import numpy as np
from utils import *
from devito4pytorch import devito_wrapper
from examples.seismic import Model, AcquisitionGeometry, TimeAxis
from examples.seismic.acoustic import AcousticWaveSolver
from devito import configuration
configuration['log-level'] = 'WARNING'

class ForwardBornLayer(torch.nn.Module):
    def __init__(self, model, geometry, device):
        super(ForwardBornLayer, self).__init__()
        self.forward_born = devito_wrapper.ForwardBorn()
        self.model = model
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model, self.geometry, space_order=16)

    def forward(self, x):
        return self.forward_born.apply(x, self.model, self.geometry,
                                        self.solver, self.device)

class ForwardOpConstructor(object):
    """docstring for ForwardOpConstructor"""
    def __init__(self, tn, f0, nsrc, nrec, device):
        super(ForwardOpConstructor, self).__init__()
        
        self.device = device
        self.tn = 1500.0
        self.f0 = 0.030
        self.nsrc = 205
        self.nrec = 410

        self.setup_model()
        self.forward_op = self.wrap_op()

    def mute_op(self, dm, mute_end=10, length=5):
        mute_start = mute_end - length
        damp = torch.zeros_like(dm)
        damp[:, :, :, mute_end:] = 1.
        damp[:, :, :, mute_start:mute_end] = (1. + torch.sin((np.pi/2.0*torch.arange(0, 
            length))/(length)))/2.
        return damp*dm

    def setup_model(self):
        # Load velocity
        m0, dm, spacing, shape, origin = get_velocity()
        self.dm = torch.from_numpy(dm).unsqueeze(0).unsqueeze(0)
        self.model = Model(space_order=16, vp=1.0/np.sqrt(m0), origin=origin, 
                        shape=m0.shape, dtype=np.float32, spacing=spacing, 
                        nbl=40)

    def wrap_op(self):
        # Receiver geometry
        rec_coordinates = np.empty((self.nrec, len(self.model.spacing)), 
                                dtype=np.float32)
        rec_coordinates[:, 0] = np.linspace(0, self.model.domain_size[0], 
                                            num=self.nrec)
        rec_coordinates[:, 1] = 2.0*self.model.spacing[1]

        # Source geometry
        src_coordinates = np.empty((1, len(self.model.spacing)))
        src_coordinates[:,0] = 0.0
        src_coordinates[:,-1] = 2.0*self.model.spacing[1]

        # Create geometry
        geometry = AcquisitionGeometry(self.model, rec_coordinates, src_coordinates, 
                                    t0=0.0, tn=self.tn, src_type='Ricker', f0=self.f0)
        return ForwardBornLayer(self.model, geometry, self.device)

    def create_op(self, src_indx):
        # Receiver geometry
        self.forward_op.geometry.src_positions[:, 0] = src_indx*self.model.spacing[0]
        def J(x): return self.forward_op(self.mute_op(x))
        return J
