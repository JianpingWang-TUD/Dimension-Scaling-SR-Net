import torch
import torch.nn as nn
from complex_layers.complexLayers import ComplexLinear,ComplexReLU,ComplexConv1d,ComplexConvTranspose1d
from net_model.cbam import SAM

"""
DSSR-net网络模块代码
DSSR-net network module code
"""

# X升维模块 / Dimension elevation module (complex-valued convolution)
class Complex_Elevated_dimensional(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Complex_Elevated_dimensional, self).__init__()
        self.ext = ComplexConv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1)
        padding = (kernel_size-1)//2
        self.pre = torch.nn.Sequential(
            ComplexConv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=padding),
            ComplexReLU(),
            ComplexConv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=padding),
        )
    def forward(self, x):
        y = self.ext(x)
        return y+self.pre(y)
    
# X降维模块 / Dimension reduction + upsampling (complex-valued)
class Inverse_Complex_Elevated_dimensional(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,upsample):
        super(Inverse_Complex_Elevated_dimensional, self).__init__()
        padding = (kernel_size-1)//2
        self.ext = ComplexConv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        self.pre = torch.nn.Sequential(
            ComplexConv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=1, padding=padding),
            ComplexReLU(),
            ComplexConv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=1, padding=padding),
            ComplexReLU(),
        )
        # 升采样模块 / Upsampling module
        self.Tran = ComplexConvTranspose1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=upsample, padding=(kernel_size - upsample + 1) // 2, output_padding=1)
        self.relu = ComplexReLU()
    def forward(self, x):
        x = self.Tran(x)
        return x

# X降维模块 (real-valued, no model-guidance)
class Inverse_Elevated_dimensional(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,upsample):
        super(Inverse_Elevated_dimensional, self).__init__()
        padding = (kernel_size-1)//2
        self.Tran = nn.ConvTranspose1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=upsample, padding=(kernel_size - upsample + 1) // 2, output_padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.Tran(x)
        return x

# y升维模块-主要是多核FFT模块 / Dimension elevation for y (multi-kernel FFT-like)
class ComplexLinear_y(nn.Module):
    def __init__(self, signal_dim, channel_num):
        super(ComplexLinear_y, self).__init__()
        self.pre = ComplexLinear(signal_dim, channel_num * signal_dim*2)
        self.relu = ComplexReLU()
        self.channel_num = channel_num
    def forward(self, y):
        bsz = y.size(0)
        y = self.pre(y).view(bsz, self.channel_num, -1)
        return y

# 辅助变量模块 / Auxiliary function module (stacked complex convolutions)
class Auxiliary_function(nn.Module):
    def __init__(self, in_channel,kernel_size):
        super(Auxiliary_function, self).__init__()
        padding = (kernel_size-1)//2
        self.pre = torch.nn.Sequential(
            ComplexConv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=1, padding=padding),
            ComplexReLU(),
            ComplexConv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=1, padding=padding),
            ComplexReLU(),
            ComplexConv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=1, padding=padding),
            ComplexReLU(),
            ComplexConv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=1, padding=padding),
            ComplexReLU(),
        )
    def forward(self, x):
        x = self.pre(x)
        return x    

# 权重模块 / Weight learning block
class C_ConvLayer1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(C_ConvLayer1, self).__init__()
        self.conv1d = ComplexConv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, stride=stride)
    def forward(self, x):
        return self.conv1d(x)    

# Spatial-channel attention (SCA) block
class SCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCA, self).__init__()
        if channel<=16:
            reduction = channel
        self.conv_du = nn.Sequential(
                C_ConvLayer1(in_channels=2*channel, out_channels=channel // reduction, kernel_size=3, stride=1),
                ComplexReLU(),
                C_ConvLayer1(in_channels=channel // reduction, out_channels=channel, kernel_size=3, stride=1),
        )
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        y = self.sigmoid(abs(self.conv_du(x)))
        return y
    
class Weight(nn.Module):
    """
    Weighting mechanism combining input and auxiliary variable
    """
    def __init__(self, channel):
        super(Weight, self).__init__()
        self.C = C_ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.weight = SCA(channel)
    def forward(self, x, y):
        delta = self.weight(torch.cat([self.C(y), x], 1))
        return delta


# Encoder block (for denoising module)
class ConvLayer_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(ConvLayer_encoder, self).__init__()
        padding = (kernel_size-1+2*stride)//2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )
        nn.init.xavier_normal_(self.block[0].weight.data)
    def forward(self, x):
        x = self.block(x)
        x_cbam = x
        return x, x_cbam

# Decoder block (for denoising module)
class ConvLayer_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer_decoder, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )
        nn.init.xavier_normal_(self.block[0].weight.data)
    def forward(self, x):
        return self.block(x)
    
# Denoising sub-network (U-Net style with SAM attention)
class denoise_module(nn.Module):
    def __init__(self,dim):
        super(denoise_module, self).__init__()
        self.conv_en1 = ConvLayer_encoder(dim, dim, 3, 2, 3)
        self.conv_en2 = ConvLayer_encoder(dim, dim*2, 3, 2, 3)
        self.conv_en3 = ConvLayer_encoder(dim*2, dim*4, 3, 2, 3)
        self.conv_en4 = ConvLayer_encoder(dim*4, dim*8, 3, 2, 3)
        self.SAM = SAM()   # spatial attention module
        self.de4 = ConvLayer_decoder(dim*8, dim*4, 3, 2)
        self.de3 = ConvLayer_decoder(dim*8, dim*2, 3, 2)
        self.de2 = ConvLayer_decoder(dim*4, dim, 3, 2)
        self.de1 = ConvLayer_decoder(dim*2, dim, 3, 2)

    def forward(self, x):
        _input = x
        encode0, down1 = self.conv_en1(x)
        encode1, down2 = self.conv_en2(encode0)
        encode2, down3 = self.conv_en3(encode1)
        encode3, down4 = self.conv_en4(encode2)

        decode4 = self.SAM(encode3)
        decode3 = self.de4(decode4)
        decode2 = self.de3(torch.cat((down3, decode3), dim=1))
        decode1 = self.de2(torch.cat((down2, decode2), dim=1))
        decode0 = self.de1(torch.cat((down1, decode1), dim=1))
        return _input + decode0

# DSSR-net overall structure (model-guided with complex layers)
class DSSR_net(nn.Module):
    def __init__(self, dim,layer_num,signal_dim,upsample):
        super(DSSR_net, self).__init__()
        self.transform_function = Complex_Elevated_dimensional(1, dim, 3)
        self.inverse_transform_function = Inverse_Complex_Elevated_dimensional(dim, 1, 25,upsample)
        self.ComplexLinear_y_function = ComplexLinear_y(signal_dim, dim)
        self.auxiliary_function = Auxiliary_function(dim, 3)
        self.layer_num = layer_num
        self.mod_denoise = nn.ModuleList([denoise_module(dim) for _ in range(layer_num)])
        self.mod_line = nn.ModuleList([Weight(dim) for _ in range(layer_num)])

    def forward(self, y):
        inp_c = (y[:, 0, :].type(torch.complex64) + 1j * y[:, 1, :].type(torch.complex64))
        # Initialization via linear transform
        y = self.ComplexLinear_y_function(inp_c)
        z = torch.zeros_like(y).cuda()
        for i in range(self.layer_num):
            Z = self.auxiliary_function(z)
            delta = self.mod_line[i](y,Z)    
            x = torch.mul((1 - delta), y) + torch.mul(delta, Z)
            # Apply denoising and preserve phase
            z = self.mod_denoise[i](abs(x))*torch.exp(1j*torch.angle(x))
        final = self.inverse_transform_function(z)
        return abs(final.squeeze(1))
