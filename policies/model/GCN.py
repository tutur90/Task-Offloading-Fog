from collections import OrderedDict

import torch
from torch import nn


from model.modules.graph import GCN
# from model.modules.mlp import MLP
# from model.modules.tcn import MultiScaleTCN

class MGCN(nn.Module):
    """
    MotionAGFormer, the main class of our model.
    """
    def __init__(self, dim_in, dim_out, num_joints, neighbour_num=4):
        super(MGCN, self).__init__()
        
        self.gcn = GCN(dim_in, dim_out, num_joints, neighbour_num, mode='temporal')
        
        self.u = nn.Linear(dim_in, dim_out)
        self.v = nn.Linear(num_joints, num_joints)

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C] (T=27, J=17, C=3)
        :param return_rep: Returns motion representation feature volume (In case of using this as backbone)
        """
        
        # x = self.u(x)
        # x = self.v(x.transpose(2, 3)).transpose(2, 3)
        
        x = self.gcn(x)


        return x


def _test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 27, 17
    random_x = torch.randn((b, t, j, c)).to(device)

    model = MGCN(dim_in=3, dim_out=3, num_joints=17).to(device)
    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params:,}")
    print(f"Model FLOPS #: {profile_macs(model, random_x):,}")

    # Warm-up to avoid timing fluctuations
    for _ in range(10):
        _ = model(random_x)
        # print(_)

    import time
    num_iterations = 100 
    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations

    # Calculate FPS
    fps = 1.0 / average_inference_time

    print(f"FPS: {fps}")
    
    

    out = model(random_x)

    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()
