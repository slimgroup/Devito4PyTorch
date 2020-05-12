import h5py
import os
import numpy as np

def get_velocity(input_data=os.path.join(os.path.expanduser("~"), "data")):
    
    if not os.path.exists(os.path.join(os.path.expanduser('~'), 'data/')):
        os.makedirs(os.path.join(os.path.expanduser('~'), 'data/'))
    vel_path = os.path.join(os.path.expanduser('~'), 
        'data/parihaka_model_high-freq.h5')
    if not os.path.isfile(vel_path):
        os.system("wget https://www.dropbox.com/s/eouo2awl156vc94/parihaka_model_high-freq.h5 -O" + vel_path)
             
    m0 = np.transpose(h5py.File(vel_path, 'r')['m0'][...])
    dm = np.transpose(h5py.File(vel_path, 'r')['dm'][...])
    shape = h5py.File(vel_path, 'r')['n'][...][::-1]
    origin = (0., 0.)
    spacing = (25.0, 12.5)

    return m0, dm, spacing, shape, origin


def get_data():
    if not os.path.exists(os.path.join(os.path.expanduser('~'), 'data/')):
        os.makedirs(os.path.join(os.path.expanduser('~'), 'data/'))
    vel_path = os.path.join(os.path.expanduser('~'), 
        'data/devito_parihaka_linear-seq-shots.hdf5')
    if not os.path.isfile(vel_path):
        os.system("wget https://www.dropbox.com/s/won86mni35hf8ac/devito_parihaka_linear-seq-shots.hdf5 -O" + vel_path)

    return h5py.File(vel_path, 'r')["data"][...]


def save(G):

    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')

    torch.save({'model_state_dict': G.state_dict(),
        'z': z}, os.path.join(save_dir,
        'checkpoint.pth'))
