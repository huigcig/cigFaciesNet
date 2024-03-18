import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class EqualizedLR_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = np.sqrt(2/(in_ch * kernel_size[0] * kernel_size[1]))

        self.weight = Parameter(torch.Tensor(out_ch, in_ch, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_ch))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.conv2d(x, self.weight*self.scale, self.bias, self.stride, self.padding)

class Pixel_norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        b = a / torch.sqrt(torch.sum(a**2, dim=1, keepdim=True)+ 10e-8)
        return b

class Minibatch_std(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        size = list(x.size())
        size[1] = 1

        std = torch.std(x, dim=0)
        mean = torch.mean(std)
        return torch.cat((x, mean.repeat(size)),dim=1)    

class FromRGB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)

class ToRGB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1,1), stride=(1, 1))

    def forward(self, x):

        return self.conv(x)

class G_Block(nn.Module):
    def __init__(self, in_ch, out_ch, initial_block=False):
        super().__init__()
        if initial_block:
            self.upsample = None
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.LeakyReLU(0.2)
        self.pixelwisenorm = Pixel_norm()
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
    def forward(self, x):

        if self.upsample is not None:
            x = self.upsample(x)
        # x = self.conv1(x*scale1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pixelwisenorm(x)
        # x = self.conv2(x*scale2)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pixelwisenorm(x)
        return x    

class D_Block(nn.Module):
    def __init__(self, in_ch, out_ch, initial_block=False):
        super().__init__()

        if initial_block:
            self.minibatchstd = Minibatch_std()
            self.conv1 = EqualizedLR_Conv2d(in_ch+1, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(4, 4), stride=(1, 1))
            self.outlayer = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(out_ch, 1)
                                    )
        else:
            self.minibatchstd = None
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.outlayer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.relu = nn.LeakyReLU(0.2)
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        if self.minibatchstd is not None:
            x = self.minibatchstd(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.outlayer(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, latent_size, out_res, conv_dim):
        super(Generator, self).__init__()
        self.depth = 1
        self.alpha = 1
        self.fade_iters = 0
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.current_net = nn.ModuleList([G_Block(latent_size, conv_dim, initial_block=True)])
        self.toRGBs = nn.ModuleList([ToRGB(conv_dim, 1)])
        # __add_layers(out_res)
        for d in range(2, int(np.log2(out_res))):
            if d < 6:
                ## low res blocks 8x8, 16x16, 32x32 with conv_dim channels
                in_ch, out_ch = conv_dim, conv_dim
            else:
                ## from 64x64(5th block), the number of channels halved for each block
                in_ch, out_ch = int(conv_dim / 2**(d - 6)), int(conv_dim / 2**(d - 5))
            self.current_net.append(G_Block(in_ch, out_ch))
            self.toRGBs.append(ToRGB(out_ch, 1))


    def forward(self, x):
        for block in self.current_net[:self.depth-1]:
            x = block(x)
        out = self.current_net[self.depth-1](x)
        x_rgb = self.toRGBs[self.depth-1](out)
        if self.alpha < 1:
            x_old = self.upsample(x)
            old_rgb = self.toRGBs[self.depth-2](x_old)
            x_rgb = (1-self.alpha)* old_rgb + self.alpha * x_rgb

            self.alpha += self.fade_iters

        return x_rgb


    def growing_net(self, num_iters):

        self.fade_iters = 1/num_iters
        self.alpha = 1/num_iters

        self.depth += 1


class Discriminator(nn.Module):
    def __init__(self, latent_size, out_res, conv_dim):
        super(Discriminator, self).__init__()
        self.depth = 1
        self.alpha = 1
        self.fade_iters = 0

        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.current_net = nn.ModuleList([D_Block(conv_dim, conv_dim, initial_block=True)])
        self.fromRGBs = nn.ModuleList([FromRGB(1, conv_dim)])
        for d in range(2, int(np.log2(out_res))):
            if d < 6:
                in_ch, out_ch = conv_dim, conv_dim
            else:
                in_ch, out_ch = int(conv_dim / 2**(d - 5)), int(conv_dim / 2**(d - 6))
            self.current_net.append(D_Block(in_ch, out_ch))
            self.fromRGBs.append(FromRGB(1, in_ch))

    def forward(self, x_rgb):
        x = self.fromRGBs[self.depth-1](x_rgb)
        x = self.current_net[self.depth-1](x)

        if self.alpha < 1:
            x_rgb = self.downsample(x_rgb)
            x_old = self.fromRGBs[self.depth-2](x_rgb)
            x = (1-self.alpha)* x_old + self.alpha * x
            self.alpha += self.fade_iters
        for block in reversed(self.current_net[:self.depth-1]):
            x = block(x)

        return x

    def growing_net(self, num_iters):

        self.fade_iters = 1/num_iters
        self.alpha = 1/num_iters

        self.depth += 1
   
    
    
    
    
    
    
    
