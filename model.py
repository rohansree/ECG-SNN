
import torch
import torch.nn as nn
from norse.torch import LICell, LIFCell, LIFParameters
from norse.torch import SequentialState

class LIFNeuralNetwork(nn.Module):
    def __init__(self):
        super(LIFNeuralNetwork, self).__init__()

        # SNN parameters
        lif_params = LIFParameters(
            tau_syn_inv=torch.tensor(2.0),  
            tau_mem_inv=torch.tensor(0.05), 
            v_leak=torch.tensor(0.0),
            v_th=torch.tensor(1.0),
            v_reset=torch.tensor(0.0),
            alpha=torch.tensor(100.0),
            method='super'
        )

        #the models for the two channels are the same
        self.channel1_model = SequentialState(
            nn.Conv2d(1, 20, (5, 1), 1),      #first CNN layer
            LIFCell(lif_params),                        #first SNN layer
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(20, 50, (5, 1), 1),     #second CNN layer
            LIFCell(lif_params),                         #second SNN layer
            nn.MaxPool2d((2, 1)),
            nn.Flatten(),                     #to keep the dimensions from breaking
        )

        #the models for the two channels are the same
        self.channel2_model = SequentialState(
            nn.Conv2d(1, 20, (5, 1), 1),      
            LIFCell(lif_params),                        
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(20, 50, (5, 1), 1),    
            LIFCell(lif_params),
            nn.MaxPool2d((2, 1)),
            nn.Flatten(),                     
        )

        self.fc1 = nn.Linear(50 * 72 * 2, 800)  # adjusted to combine output of both channels
        self.fc2 = nn.Linear(800, 10)           # final output fc layer

    def forward(self, x1, x2):

        #ensures inputs are 4D tensors
        x1 = x1.unsqueeze(1).unsqueeze(-1) 
        x2 = x2.unsqueeze(1).unsqueeze(-1)

        #process each channel separately
        output1, _ = self.channel1_model(x1)
        output2, _ = self.channel2_model(x2)

        #concat outputs from both channels
        combined_output = torch.cat((output1, output2), dim=1)

        #pass combined output through the fully connected layers
        x = self.fc1(combined_output)
        x = nn.ReLU()(x)
        x = self.fc2(x) #this is the FINAL output

        return x