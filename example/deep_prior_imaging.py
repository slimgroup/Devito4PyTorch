import torch
import numpy as np
from utils import *
from devito_operators import *
from generator import generator
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.FloatTensor')
device = 'cpu'

if __name__ == '__main__':

    # Download simulated data
    d_obs = torch.from_numpy(get_data()).to(device)

    # Noise variance
    var = 0.01

    # Add noise to get observed data
    d_obs = d_obs + np.sqrt(var)*torch.randn(d_obs.shape, device=device)

    # Acquisition details
    tn = 1500.0
    f0 = 0.030
    nsrc = 205
    nrec = 410

    # Setup devito wrapper
    op_constructor = ForwardOpConstructor(tn, f0, nsrc, nrec, device)
    dm = op_constructor.dm

    # Define the deep prior and its fixed input
    z = torch.randn((1, 3, 256, 192), device=device)
    G = generator(
                dm.size(),
                num_input_channels=3, num_output_channels=1,
                num_channels_down = [16, 32, 64],
                num_channels_up   = [16, 32, 64],
                num_channels_skip = [0, 0, 64],
                upsample_mode = 'nearest',
                need1x1_up = True,
                filter_size_down=5,
                filter_size_up=5,
                filter_skip_size = 1,
                need_sigmoid=False,
                need_bias=True,
                pad='reflection',
                act_fun='LeakyReLU').to(device)


    # Define the optimizer
    optim = torch.optim.RMSprop(G.parameters(), 1e-4, weight_decay=1e0)

    max_itr = 3000
    for itr in range(max_itr):

        # Select a random source index
        idx = np.random.choice(nsrc, 1, replace=False)[0]

        # Create born operator
        J = op_constructor.create_op(idx)

        # Compute predicted data
        dm_est = G(z)
        d_pred = J(dm_est)

        # Compute the objective function
        obj = (d_obs.shape[0]/(var * 2.0))*torch.norm(d_pred - d_obs[idx])**2

        # Compute the gradient w.r.t deep prior unknowns and update them
        obj.backward()
        optim.step()

        print(("Iteration: [%d/%d] | objective: %4.4f | model error %4.4f" % \
            (itr+1, max_itr, obj, ((dm_est.detach()-dm).norm()**2).item())))