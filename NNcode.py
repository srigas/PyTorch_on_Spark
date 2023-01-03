import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_height, input_width, conv_channels, kernels, maxpools, lin_channels, dropout):
        super(CNNModel, self).__init__()
        self.num_conv_layers = len(kernels)
        self.input_height = input_height
        self.input_width = input_width
        
        seq = []
        for i in range(self.num_conv_layers):
            seq.append(nn.Conv2d(in_channels=conv_channels[i], 
                                 out_channels=conv_channels[i+1],
                                 kernel_size=kernels[i], stride=1, padding=1))
            seq.append(nn.ReLU())
            seq.append(nn.MaxPool2d(kernel_size=maxpools[i]))
            
        # Flatten the output of the final convolution layer
        seq.append(nn.Flatten())
        
        convolutions = nn.Sequential(*seq)
        
        # Calculation of first linear layer dimensions
        # We build an empty tensor of appropriate size and let it go through
        # the above sequence, in order to calculate the output's size automatically
        first_lin = convolutions(torch.empty(1,conv_channels[0],input_height,input_width)).size(-1)
        
        self.num_lin_layers = len(lin_channels)
        for i in range(self.num_lin_layers):
            if i == self.num_lin_layers-1:
                seq.append(nn.Linear(lin_channels[i-1], lin_channels[i]))
                break
            elif i == 0:
                seq.append(nn.Linear(first_lin, lin_channels[i]))
            else:
                seq.append(nn.Linear(lin_channels[i-1], lin_channels[i]))
            seq.append(nn.ReLU())
            seq.append(nn.Dropout(dropout))
                
        self.fitter = nn.Sequential(*seq)

    def forward(self, x):
        x = x.view(-1,1,self.input_height,self.input_width)
        out = self.fitter(x)
        return out