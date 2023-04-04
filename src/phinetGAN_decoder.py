import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

    
class PointwiseConv2D(nn.Module):
    
    """ Constructor of Pointwise Convolution 
    
        Args:
            in_channels : int 
                number of input channels
                
            expansion : int  
                ratio between input/output channels
                
            debug : bool
                print some info about the operation
            
            General note: 
                input/output tensor size(NxWxH) is the same; n_channels differ:
                Cout = Cin / E, where E is the espansion factor 
            
    """
        
    def __init__(self, in_channels, expansion, debug):
        
        super(PointwiseConv2D, self).__init__()
        try:
            isinstance(expansion, int)
        except:
            raise Exception("invalid type for expansion variable; \
                            expected [int] and given [{}]".format(type(expansion))) 
        self.PW_conv = nn.Conv2d(      
            in_channels = in_channels,
            out_channels = in_channels//expansion,
            kernel_size = 1,
            padding = 0, 
            bias = False)
        self.activation = nn.ReLU6()
        self.debug = debug
        
    def forward(self, x):
        
        """ Executes Pointwise convolution Block
         
            Args:
                x : torch.Tensor
                    input tensor
                    
            Returns:
                x : torch.Tensor
                    output of pointwiseConv2D
                
        """
        if self.debug:
            print("\n====== PointwiseConv operation ======")
            print("input_shape: {}".format(x.shape))
        x = self.PW_conv(x)
        x = self.activation(x)
        if self.debug:
            print("output_shape: {}".format(x.shape))
            print(x)
        return x
        


class BilinearUpSampling(nn.Module):
    
    """ Constructor of Bilinear interpolation upsampling  
    
        Args:
            up_factor : int
                value in which scale the H,W dimensions
            
            debug : bool
                print some info about the operation
                
            mode : str 
                interpolation method for the upsampling 
                ['nearest', 'linear', 'bilinear', 'bicubic']; default = bilinear 
            
            General Note:
                the output will have the same number of channels as
                tensor in input but different weight and height: 
                W_out = scale_factor*W_in 
                H_out = scale_factor*H_in  
               
    """
    def __init__(self, up_factor, debug, mode = 'bilinear'):
        super(BilinearUpSampling, self).__init__()
        self.BL_interpol = torch.nn.Upsample(
            scale_factor = up_factor, 
            mode = mode, 
            align_corners = True)
        self.debug = debug 
        
    def forward(self, x):
        """
         Executes Bilinear upsampling
         
            Args:
                x : torch.Tenso
                    input tensor
                
            Returns:
                x : torch.Tensor 
                    output of bilinear interpolation
            
            General note:
                unsqueeze and squeeze are needed because 
                the bilinear interpolation module want 
                that also the batch dimension is specified
            
        """
        
        if self.debug:
            print("\n====== Bilinear operation (align_corners = True) ======")
            print("input_shape: {}".format(x.shape))
        x = x.unsqueeze(0)
        x = self.BL_interpol(x)
        x = x.squeeze(0) 
        if self.debug:
            print("output_shape: {}".format(x.shape))
            print(x)
        return x
    
    
    
class Conv2D(nn.Module):
    
    """ Constructor of 3x3 2D convolution 
    
        Args:
            input_channel : int 
                number of channels in input 
                
            output_channel : input 
                number of channels in output 
            
            k_size : int
                kernel size of the 2D convolution
            
            debug : bool
                print some info about the operation
        
        General Note:
            standard convolutional layer for making NN learning
            the upsampling relation to the lower layer 
        
    """
    
    def __init__(self, in_channels, out_channels, k_size, debug):
        
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(      
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = k_size,
            bias = False, 
            padding = 'same')
        self.activation = nn.ReLU6()
        self.debug = debug 
        
    def forward(self, x):
        """
         Executes convolution operation 
         
            Args:
                x : torch.Tensor 
                    input tensor
                
            Returns:
                x : torch.Tensor 
                    output of convolution
            
        """
        if self.debug:
            print("\n====== conv2D operation (PAD = True) ======")
            print("input_shape: {}".format(x.shape))
        x = self.conv(x)
        x = self.activation(x)
        if self.debug:
            print("output_shape: {}".format(x.shape))
            print(x)
        return x
        
        
        
class PhinetGanBlock(nn.Module):
    
    """ Constructor of PhiNet-GAN's block, consisting of:
    
            - Pointwise 2D convolution
            - bilinnear upsampling 
            - 3x3 2D convolution
        
        Args:
            in_shape : tuple
                Input shape of the conv block.
            expansion : float
                Expansion coefficient for the depthwise conv block
            filters : int
                Output channels of the conv2D block.
            k_size : int
                Kernel size for the 2D convolution.
            scale_factor : 
                dimension ratio for the upsampling operation 
            debug : bool
                print some info about the operation

    """
    
    def __init__(
        self, 
        in_shape, 
        expansion,
        filters, 
        k_size, 
        up_factor, 
        debug):
        
        super(PhinetGanBlock, self).__init__()
        self._layers = torch.nn.ModuleList()
        in_channels = in_shape[0]
        
        self._layers.append(PointwiseConv2D(
            in_channels = in_channels, 
            expansion = expansion, 
            debug = debug))
            
        self._layers.append(BilinearUpSampling(
            up_factor = up_factor,
            mode = 'bilinear',
            debug = debug))
        
        self._layers.append(Conv2D(
            in_channels = in_channels//expansion, 
            out_channels = filters,
            k_size = k_size,
            debug = debug))
        
        # output info
        self.C, self.H, self.W = [filters, 
                                  int(in_shape[0]*up_factor),
                                  int(in_shape[1]*up_factor)]
    
    def forward(self, x):
        """ Executes the PhinetGAN block 
         
            Args:
                x : torch.Tensor 
                    input tensor
                
            Returns:
                x : torch.Tensor 
                    output of convolution
            
        """
        for layer in self._layers:
            x = layer(x)
        return x
        
        
class phinectGanDecoder(nn.Module):
        
    def __init__(
        self,
        latent_in_shape,
        alpha,
        k_size, 
        expansion,
        up_factor,
        debug        
        ):
        '''  Constructor of PhiNet-GAN decoder architecture
        
            Args:
                latent_in_shape : tuple
                    input shape (coming from latent space)
                
                alpha : float in [0,1] 
                    paper hyper-parameter
                
                k_size : int
                    Kernel size for the 2D convolution of phinetGAN block.
                
                expansion : float
                    Expansion coefficient for the depthwise conv of phinetGAN block
                    
                up_factor : 
                    dimension ratio for the upsampling operation 
                    
                debug : bool
                    print some info about the operation
    
        '''

        assert len(latent_in_shape) == 3    # (C, H, W) 
    
        super(phinectGanDecoder, self).__init__()
        self._layers = torch.nn.ModuleList()

        # this hyperparameters are hard-coded. copied to 
        # match the paper choice 
        n_filters1 = 96
        n_filters2 = 48
        n_filters3 = 24
        n_filters4 = 3    # added here for the first time 
        
        
        #in_shape, expansion, filters,  k_size, up_factor
        
        block1 = PhinetGanBlock(latent_in_shape, 
                                    expansion, 
                                    int(n_filters1*alpha),
                                    k_size = k_size, 
                                    up_factor = up_factor,
                                    debug = debug)
        self._layers.append(block1)
        
        block2 = PhinetGanBlock((block1.C, block1.H, block1.W), 
                                    expansion, 
                                    int(n_filters2*alpha),
                                    k_size = k_size, 
                                    up_factor = up_factor,
                                    debug = debug)
        self._layers.append(block2)
        
        block3 = PhinetGanBlock((block2.C, block2.H, block2.W), 
                                    expansion, 
                                    int(n_filters3*alpha),
                                    k_size = k_size, 
                                    up_factor = up_factor,
                                    debug = debug)
        self._layers.append(block3)
        
        block4 = PhinetGanBlock((block3.C, block3.H, block3.W), 
                                    expansion, 
                                    n_filters4,   # RGB image?
                                    k_size = k_size, 
                                    up_factor = up_factor,
                                    debug = debug)
        self._layers.append(block4)
        
        
    def forward(self, x):
        """Executes PhiNet network
        Args:
            x : torch.Tensor
                input tensor
                
        return:
            x : torch.tensor
                output prediction of the decoder  
        """

        for layer in self._layers:
            x = layer(x)
        
        return x