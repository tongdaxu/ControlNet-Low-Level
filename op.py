import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from resizer import Resizer
from op_seg import ModelBuilder
from op_edge import Sobel
import kornia
from torchvision.transforms.functional import to_pil_image
from functools import partial
import random
import numpy as np
import scipy
import math
from roomlayout import LayoutSeg
from real_world_utils import *
from motionblur.motionblur import Kernel
from fastmri_utils import fft2c_new
import yaml

def init_kernel_torch(kernel, device="cuda:0"):
    h, w = kernel.shape
    kernel = Variable(torch.from_numpy(kernel).to(device), requires_grad=True)
    kernel = kernel.view(1, 1, h, w)
    kernel = kernel.repeat(1, 3, 1, 1)
    return kernel


def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type="instance"):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization
                            layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and
    track running statistics (mean/stddev).

    For InstanceNorm, we do not use learnable affine
    parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError(
            f"normalization layer {norm_type}\
                                    is not found"
        )
    return norm_layer

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) --previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            up = [uprelu, upconv, nn.Tanh()]
            down = [downconv]
            self.down = nn.Sequential(*down)
            self.submodule = submodule
            self.up = nn.Sequential(*up)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            self.down = nn.Sequential(*down)
            self.up = nn.Sequential(*up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up += [nn.Dropout(0.5)]

            self.down = nn.Sequential(*down)
            self.submodule = submodule
            self.up = nn.Sequential(*up)

    def forward(self, x, noise):

        if self.outermost:
            return self.up(self.submodule(self.down(x), noise))
        elif self.innermost:  # add skip connections
            if noise is None:
                noise = torch.randn((1, 512, 8, 8)).cuda() * 0.0007
            return torch.cat((self.up(torch.cat((self.down(x), noise), dim=1)), x), dim=1)
        else:
            return torch.cat((self.up(self.submodule(self.down(x), noise)), x), dim=1)


# The function G in the paper
class KernelAdapter(nn.Module):
    def __init__(self, opt):
        super(KernelAdapter, self).__init__()
        input_nc = opt["nf"]
        output_nc = opt["nf"]
        ngf = opt["nf"]
        norm_layer = get_norm_layer(opt["Adapter"]["norm"])

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, x, k):
        """Standard forward"""
        return self.model(x, k)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding
                                   layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer,
                              and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                        is not implemented"
            )

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                      is not implemented"
            )
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=False)
        out = self.conv2(out)
        return identity + out

# The function G in the paper
class KernelAdapter(nn.Module):
    def __init__(self, opt):
        super(KernelAdapter, self).__init__()
        input_nc = opt["nf"]
        output_nc = opt["nf"]
        ngf = opt["nf"]
        norm_layer = get_norm_layer(opt["Adapter"]["norm"])

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, x, k):
        """Standard forward"""
        return self.model(x, k)


class KernelExtractor(nn.Module):
    def __init__(self, opt):
        super(KernelExtractor, self).__init__()

        nf = opt["nf"]
        self.kernel_dim = opt["kernel_dim"]
        self.use_sharp = opt["KernelExtractor"]["use_sharp"]
        self.use_vae = opt["use_vae"]

        # Blur estimator
        norm_layer = get_norm_layer(opt["KernelExtractor"]["norm"])
        n_blocks = opt["KernelExtractor"]["n_blocks"]
        padding_type = opt["KernelExtractor"]["padding_type"]
        use_dropout = opt["KernelExtractor"]["use_dropout"]
        if type(norm_layer) == partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        input_nc = nf * 2 if self.use_sharp else nf
        output_nc = self.kernel_dim * 2 if self.use_vae else self.kernel_dim

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(nf),
            nn.ReLU(True),
        ]

        n_downsampling = 5
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            inc = min(nf * mult, output_nc)
            ouc = min(nf * mult * 2, output_nc)
            model += [
                nn.Conv2d(inc, ouc, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(nf * mult * 2),
                nn.ReLU(True),
            ]

        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(
                    output_nc,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        self.model = nn.Sequential(*model)

    def forward(self, sharp, blur):
        output = self.model(torch.cat((sharp, blur), dim=1))
        if self.use_vae:
            return output[:, : self.kernel_dim, :, :], output[:, self.kernel_dim :, :, :]

        return output, torch.zeros_like(output).cuda()


class KernelWizard(nn.Module):
    def __init__(self, opt):
        super(KernelWizard, self).__init__()
        lrelu = nn.LeakyReLU(negative_slope=0.1)
        front_RBs = opt["front_RBs"]
        back_RBs = opt["back_RBs"]
        num_image_channels = opt["input_nc"]
        nf = opt["nf"]

        # Features extraction
        resBlock_noBN_f = partial(ResidualBlock_noBN, nf=nf)
        feature_extractor = []

        feature_extractor.append(nn.Conv2d(num_image_channels, nf, 3, 1, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)

        for i in range(front_RBs):
            feature_extractor.append(resBlock_noBN_f())

        self.feature_extractor = nn.Sequential(*feature_extractor)

        # Kernel extractor
        self.kernel_extractor = KernelExtractor(opt)

        # kernel adapter
        self.adapter = KernelAdapter(opt)

        # Reconstruction
        recon_trunk = []
        for i in range(back_RBs):
            recon_trunk.append(resBlock_noBN_f())

        # upsampling
        recon_trunk.append(nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True))
        recon_trunk.append(nn.PixelShuffle(2))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True))
        recon_trunk.append(nn.PixelShuffle(2))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(64, num_image_channels, 3, 1, 1, bias=True))

        self.recon_trunk = nn.Sequential(*recon_trunk)

    def adaptKernel(self, x_sharp, kernel):
        B, C, H, W = x_sharp.shape
        base = x_sharp

        x_sharp = self.feature_extractor(x_sharp)

        out = self.adapter(x_sharp, kernel)
        out = self.recon_trunk(out)
        out += base

        return out

    def forward(self, x_sharp, x_blur):
        x_sharp = self.feature_extractor(x_sharp)
        x_blur = self.feature_extractor(x_blur)

        output = self.kernel_extractor(x_sharp, x_blur)
        return output

class NonlinearBlurOperator(nn.Module):
    def __init__(self, opt_yml_path="./bkse/options/generate_blur/default.yml"):
        super(NonlinearBlurOperator, self).__init__()
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(data.shape[0], 512, 4, 4).to(data.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

    def y_channel(self):
        return 3
    
    def to_pil(self, y):
        y = (y[0] + 1.0) / 2.0
        y = torch.clip(y, 0, 1)
        if (len(y.shape)==4):
            assert(y.shape[0]==1)
            y = y[0]
        y = to_pil_image(y, 'RGB')
        return y

class PhaseRetrievalOperator(nn.Module):
    def __init__(self, oversample=2.0):
        super().__init__()
        self.pad = int((oversample / 8.0) * 512)
        
    def forward(self, data, keep_shape=False, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()

        if keep_shape == True:
            amplitude = F.interpolate(amplitude, 512)
        return amplitude

    def y_channel(self):
        return 3
    
    def to_pil(self, y):
        y = y / torch.max(y)
        y = torch.clip(y, 0, 1)
        if (len(y.shape)==4):
            assert(y.shape[0]==1)
            y = y[0]
        y = to_pil_image(y, 'RGB')
        return y

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k

class SuperResolutionOperator(nn.Module):
    def __init__(self, in_shape, scale_factor):
        super(SuperResolutionOperator, self).__init__()
        self.scale_factor = scale_factor
        self.down_sample = Resizer(in_shape, 1/scale_factor)
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)

    def forward(self, x, keep_shape=False):
        x = (x + 1.0) / 2.0
        y = self.down_sample(x)
        y = (y - 0.5) / 0.5
        if keep_shape:
            y = F.interpolate(y, scale_factor=self.scale_factor, mode='bicubic')
        return y

    def transpose(self, y):
        return self.up_sample(y)

    def y_channel(self):
        return 3
    
    def to_pil(self, y):
        y = (y[0] + 1.0) / 2.0
        y = torch.clip(y, 0, 1)
        y = to_pil_image(y, 'RGB')
        return y
    
class AEDSegOperator(nn.Module):
    def __init__(self):
        super(AEDSegOperator, self).__init__()
        self.encoder = ModelBuilder.build_encoder(arch="mobilenetv2dilated",fc_dim=320,weights="./models/models_seg/mobilenetv2-c1/encoder_epoch_20.pth")
        self.decoder = ModelBuilder.build_decoder(arch="c1_deepsup",fc_dim=320,num_class=150,weights="./models/models_seg/mobilenetv2-c1/decoder_epoch_20.pth",use_softmax=True)
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False
        for _, param in self.decoder.named_parameters():
            param.requires_grad = False
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def forward(self, x, **kwargs):

        x = (x + 1) / 2.0
        x = self.transform(x)
        y = self.decoder(self.encoder(x, return_feature_maps=True), segSize=(512,512))
        assert 'mode' in kwargs
        if kwargs['mode'] == 'init':
            return torch.argmax(y, dim=1, keepdim=True)
        else:
            return y

    def y_channel(self):
        return 3


class EdgeOperator(nn.Module):

    def __init__(self):
        super(EdgeOperator, self).__init__()
        # self.sobel = Sobel()
        self.canny = kornia.filters.Canny()

    def forward(self, data, **kwargs):
        # out = self.sobel(torch.mean((data + 1.0) / 2.0, dim=1, keepdim=True))
        out = self.canny((data + 1.0) / 2.0)[0]

        return out

    def y_channel(self):
        return 1

    def to_pil(self, y):
        y = torch.cat([y, y, y], dim=1)[0]
        y = y / torch.max(y)
        y = to_pil_image(y, 'RGB')
        return y

class GaussialBlurOperator(nn.Module):
    def __init__(self, kernel_size=61, intensity=3.0):
        super(GaussialBlurOperator, self).__init__()

        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, data, **kwargs):
        return self.conv(data)

    def y_channel(self):
        return 3

    def transpose(self, data, **kwargs):
        return data

    def to_pil(self, y):
        y = (y[0] + 1.0) / 2.0
        y = torch.clip(y, 0, 1)
        y = to_pil_image(y, 'RGB')
        return y

class MotionBlurOperator(nn.Module):
    def __init__(self, kernel_size=61, intensity=0.5):
        super(MotionBlurOperator, self).__init__()
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity)

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def y_channel(self):
        return 3

    def to_pil(self, y):
        y = (y[0] + 1.0) / 2.0
        y = torch.clip(y, 0, 1)
        y = to_pil_image(y, 'RGB')
        return y

class RealWorldOperator(nn.Module):
    def __init__(self,
                 sf = 4,
                 blur_kernel_size = 21,
                 kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
                 kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                 blur_sigma = [0.2, 1.5],
                 betag_range = [0.5, 2.0],
                 betap_range = [1, 1.5],
                 sinc_prob = 0.1,
                 blur_kernel_size2 = 11,
                 kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
                 kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                 blur_sigma2 = [0.2, 1.0],
                 betag_range2 = [0.5, 2.0],
                 betap_range2 = [1, 1.5],
                 sinc_prob2 = 0.1,
                 final_sinc_prob = 0.8,
                 resize_prob = [0.2, 0.7, 0.1],
                 resize_range = [0.3, 1.5],
                 gray_noise_prob = 0.4,
                 gaussian_noise_prob = 0.5,
                 noise_range = [1, 15],
                 poisson_scale_range = [0.05, 2.0],
                 jpeg_range = [60, 95],
                 jpeg_range2= [60, 100],
                 second_blur_prob = 0.5,
                 resize_prob2 = [0.3, 0.4, 0.3],
                 resize_range2 = [0.6, 1.2],
                 gray_noise_prob2 = 0.4,
                 gaussian_noise_prob2 = 0.5,
                 noise_range2 = [1, 12],
                 poisson_scale_range2 = [0.05, 1.0],
                 ):
        super(RealWorldOperator, self).__init__()
        self.sf = sf
        self.jpeger = DiffJPEG(differentiable=True)
        self.random_add_gaussian_noise_pt = random_add_gaussian_noise_pt
        self.random_add_poisson_noise_pt = random_add_poisson_noise_pt
        self.filter2D = filter2D
        self.usm_sharpener = USMSharp()
        
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob # a list for each kernel probability
        self.blur_sigma = blur_sigma
        self.betag_range = betag_range  # betag used in generalized Gaussian blur kernels
        self.betap_range = betap_range  # betap used in plateau blur kernels
        self.sinc_prob = sinc_prob  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = blur_kernel_size2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2

        # a final sinc filter
        self.final_sinc_prob = final_sinc_prob
        self.resize_prob = resize_prob
        self.resize_range = resize_range
        self.resize_prob2 = resize_prob2
        self.resize_range2 = resize_range2

        self.gray_noise_prob = gray_noise_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        self.jpeg_range2 = jpeg_range2
        self.second_blur_prob = second_blur_prob
        self.gray_noise_prob2 = gray_noise_prob2
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.noise_range2 = noise_range2
        self.poisson_scale_range = poisson_scale_range
        self.poisson_scale_range2 = poisson_scale_range2

        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]

        for param in self.parameters():
            param.requires_grad = False

        
    def forward(self, data, skip=False, **kwargs):
        if skip == True:
            return data
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor
        
        sinc_kernel = sinc_kernel.cuda()
        kernel = torch.FloatTensor(kernel).cuda()
        kernel2 = torch.FloatTensor(kernel2).cuda()
        
        data = (data + 1.0)/2.0
        data = data.cuda()
        
        data = self.usm_sharpener(data)
        ori_h, ori_w = data.size()[2:4]

        out = filter2D(data, kernel)
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.resize_prob,
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        
        gray_noise_prob = self.gray_noise_prob
        if random.random() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.noise_range,
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        out = STEClamp.apply(out)  
        out = self.jpeger(out, quality=jpeg_p)

        if random.random() < self.second_blur_prob:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.resize_prob2,
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h / self.sf * scale),
                      int(ori_w / self.sf * scale)),
                mode=mode,
                )

        gray_noise_prob = self.gray_noise_prob2
        if random.random() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.noise_range2,
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
                )
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        out = STEClamp.apply(out)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)


        if random.random() < self.second_blur_prob:
            out = filter2D(out, kernel2)

        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.resize_prob2,
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h / self.sf * scale),
                      int(ori_w / self.sf * scale)),
                mode=mode,
                )
        
        gray_noise_prob = self.gray_noise_prob2
        if random.random() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.noise_range2,
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
                )

        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // self.sf,
                          ori_w // self.sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = STEClamp.apply(out)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = STEClamp.apply(out)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // self.sf,
                          ori_w // self.sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
        
        out = F.interpolate(
                out,
                size=(ori_h, ori_w),
                mode='bicubic',
                )

        lq = out*2 - 1.0

        return lq

    def transpose(self, data, **kwargs):
        return data

class LayoutOperator(nn.Module):

    def __init__(self):
        super(LayoutOperator, self).__init__()
        self.model = LayoutSeg.load_from_checkpoint('/NEW_EDS/JJ_Group/xutd/lsun-room/model_retrained.ckpt', backbone='resnet101')
        self.model.freeze()

    def forward(self, data, init=False):
        scores, _ = self.model(data)
        if init == True:
            return torch.argmax(scores, dim=1, keepdim=True)
        else:
            return scores

    def transpose(self, data):
        return data

if __name__ == "__main__":
    from dataset import ImageDataset
    from op_seg import visualize_result
    DATA_ROOT = "./ffhq_512"
    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    f = EdgeOperator()
    # f = AEDSegOperator()
    f = f.to(device="cuda")

    dataset = ImageDataset(root=DATA_ROOT, transform=test_transforms, return_path=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (x, x_path) in enumerate(dataloader):
        x = x.to(device="cuda")
        x.requires_grad = True
        y = f(x, mode='non-init')
        y = f.to_pil(y)
        save_image(x, 'x.png', normalize=True, value_range=(-1,1))
        y.save('y.png')
        assert(0)

