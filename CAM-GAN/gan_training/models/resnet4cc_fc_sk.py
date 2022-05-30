import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed




class Generator(nn.Module):
    def __init__(self, z_dim=256, nlabels=1, size=256, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 64
        nf = self.nf = nfilter
        self.z_dim = z_dim

        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, 16*nf*s0*s0)


        self.emb1=nn.Embedding(1,16*nf*s0*s0)
        self.emb2=nn.Embedding(1,16*nf*s0*s0)

        self.resnet_0_0 = ResnetBlockG(16*nf, 16*nf)
        self.resnet_1_0 = ResnetBlockG(16*nf, 16*nf)
        self.resnet_2_0 = ResnetBlockG(16*nf, 8*nf)
        self.resnet_3_0 = ResnetBlockG(8*nf, 4*nf)
        self.resnet_4_0 = ResnetBlockG(4*nf, 2*nf)
        self.resnet_5_0 = ResnetBlockG(2*nf, 1*nf)
        self.resnet_6_0 = ResnetBlockG(1*nf, 1*nf)
        self.conv_img = nn.Conv2d(nf, 3, 7, padding=3)


    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        yembed = self.embedding(y)
        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        # print('outfc:  '+str(out.shape))
        out=out*self.emb1(torch.cuda.LongTensor([0]))+out*self.emb2(torch.cuda.LongTensor([0]))
        # print(out.shape)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)
        # print('outfcview:  '+str(out.shape))
        

        out = self.resnet_0_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_1_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_2_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_3_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_4_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_5_0(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_6_0(out)
        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out

class Het_conv(nn.Module):
    def __init__(self, in_channels, out_channels,if_skip=False):
        super(Het_conv, self).__init__()
        # Groupwise Convolution
        #print("P-----------")
        if(in_channels==1024):
            f_size=64
        elif(in_channels==512):
            f_size=32
        else:
            f_size=16

        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=int(in_channels/f_size), bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=int(8),bias=False)


    def forward(self, x):
        return self.gwc(x) + self.pwc(x)

class conv_dw(nn.Module):
    def __init__(self, inp, oup):
        super(conv_dw, self).__init__()
        self.dwc1 = nn.Conv2d(inp, oup, 3, 1, 1, groups=int(inp/8), bias=False)      
        self.sig2 = nn.Tanh()            
    def forward(self, x):
        x = self.dwc1(x)    
        x = self.sig2(x)
        return x


class ResnetBlockG(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)

        self.hc0=Het_conv(self.fhidden,self.fhidden)
        self.hc1=Het_conv(self.fout,self.fout)

        self.calib0 = conv_dw(self.fin, self.fhidden)
        self.calib1 = conv_dw(self.fhidden, self.fout)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)
            self.conv_s_adapt = nn.Conv2d(self.fin, self.fout, 3, stride=1, padding=1,groups=int(self.fin/4), bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)

        dx = self.conv_0(actvn(x))
        dx1 = self.hc0(actvn(dx))+self.calib0(x)

        dx = self.conv_1(actvn(dx1))
        dx = self.hc1(actvn(dx))+self.calib1(dx1)
        out = x_s + 0.1*dx
    

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)+self.conv_s_adapt(x)
            # x_s=self.conv_s_adapt(x_s)
        else:
            x_s = x
        return x_s


class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 64
        nf = self.nf = nfilter

        # Submodules
        self.conv_img = nn.Conv2d(3, 1*nf, 7, padding=3)

        self.resnet_0_0 = ResnetBlock(1*nf, 1*nf)
        self.resnet_1_0 = ResnetBlock(1*nf, 2*nf)
        self.resnet_2_0 = ResnetBlock(2*nf, 4*nf)
        self.resnet_3_0 = ResnetBlock(4*nf, 8*nf)
        self.resnet_4_0 = ResnetBlock(8*nf, 16*nf)
        self.resnet_5_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_6_0 = ResnetBlock(16*nf, 16*nf)

        self.fc = nn.Linear(16*nf*s0*s0, nlabels)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet_0_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_6_0(out)

        out = out.view(batch_size, 16*self.nf*self.s0*self.s0)
        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
