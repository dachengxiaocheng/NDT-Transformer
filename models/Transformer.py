import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class Transformer(nn.Module):
    def __init__(self, num_points=2000, emb_dims=1024, layer_number=3):
        super(Transformer, self).__init__()
        self.point_trans = STN3d(num_points=num_points, k=3, use_bn=False)
        
        self.conv1 = nn.Conv2d(12,  256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256*2,  emb_dims, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(emb_dims)

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu')
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layer_number)


        self.point_transform = True
        self.convariance_transform = False

    def forward(self, x):
        # print('x.size', x.size())
        x, y = torch.split(x, [3, 9], dim=3)
        batch_size, _, num_points, num_dims = x.size()

        if self.point_transform:
            # print("Using point propagation!")
            trans = self.point_trans(x)
            x = torch.matmul(torch.squeeze(x), trans)
            x = torch.unsqueeze(x, dim=1)

        if self.convariance_transform:
            # print("Using convariance propagation!")
            y = torch.squeeze(y, dim=1)
            y = y.view(batch_size, num_points, num_dims, num_dims)
            y = y.permute(1, 0, 2, 3) 

            y = torch.matmul(trans.transpose(2, 1).contiguous(), y)
            y = torch.matmul(y, trans)  

            # y = torch.matmul(trans, y) 
            # y = torch.matmul(y, trans.transpose(2, 1).contiguous())  
            
            y = y.permute(1, 0, 2, 3)  
            y = y.view(batch_size, num_points, num_dims * num_dims)  
            y = torch.unsqueeze(y, dim=1) 

        x = torch.cat((x, y), dim=3)

        x = x.permute(0, 3, 2, 1)
        x1 = F.relu(self.bn1(self.conv1(x)))

        x = torch.squeeze(x1, dim=3)
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = torch.unsqueeze(x, dim=3)

        x = torch.cat((x1, x), dim=1)
        x2 = F.relu(self.bn2(self.conv2(x)))

        return x2
