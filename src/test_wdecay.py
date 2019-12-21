import torch
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(19)
torch.manual_seed(19)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

x = torch.randn([1000, 1], device=device, requires_grad=False)
A = torch.randn([10000, 1000], device=device, requires_grad=False)
y = torch.mm(A, x)

x_est = torch.zeros([1000, 1], device=device, requires_grad=True)

optim = torch.optim.SGD([x_est], lr=0.1, weight_decay=10.0)
l2_loss = torch.nn.MSELoss()

loss_log = []
model_loss_log = []
for j in range(10000):
    optim.zero_grad()
    pred = torch.mm(A, x_est)
    loss = l2_loss(pred.reshape(-1), y.reshape(-1))
    grad_x = torch.autograd.grad(loss, [x_est], create_graph=False)
    for param, grad in zip([x_est], grad_x):
        param.grad = grad
    x_loss =  l2_loss(x_est, x)

    optim.step()

    loss_log.append(loss.item())
    model_loss_log.append(x_loss.item())

    # print(("Itr: [%d/%d] | loss: %4.8f | model loss: %4.8f" % (j+1, 10, loss, x_loss)))


fig = plt.figure("training logs - net", dpi=100, figsize=(7, 2.5))
plt.semilogy(loss_log); plt.title(r"$\|\|y- A x_{est}\|\|_2^2$"); plt.grid()

fig = plt.figure("training logs - model", dpi=100, figsize=(7, 2.5))
plt.semilogy(model_loss_log); plt.title(r"$\|\|x - x_{est})\|\|_2^2$"); plt.grid()

plt.show()

from IPython import embed; embed()
