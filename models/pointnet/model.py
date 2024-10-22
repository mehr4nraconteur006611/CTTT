from __future__ import print_function
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from models.pointnet.pointmlp import pointMLPfeat, Modelfeat
from models.pointnet.DGCNN_model import DGCNN_clsfeat, knn

from models.pointnet.models import pointnet2_cls_msg


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

def help(x, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
    
    return idx
    
def get_graph_feature(y, idx):
    batch_size = y.size(0)
    num_points = y.size(2)
    
    # x = x.view(batch_size, -1, num_points)
    # if idx is None:
        # if dim9 == False:
            # idx = knn(x, k=k)   # (batch_size, num_points, k)
        # else:
            # idx = knn(x[:, 6:], k=k)
    # device = torch.device('cuda')

    # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    # idx = idx + idx_base

    # idx = idx.view(-1)
 
    _, num_dims, _ = y.size()

    y = y.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = y.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    y = y.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-y, y), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, input_transform=True,FPFH_type=False):
        super(PointNetfeat, self).__init__()
        self.input_transform = True #input_transform
        self.feature_transform = True
        
        channel=3
        self.stn = STN3d()
        if self.input_transform:
            self.stn = STN3d()
            
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv1_2 = torch.nn.Conv1d(81, 128, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv2_FPFH = torch.nn.Conv1d(128+128, 512, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv3_FPFH = torch.nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn2_FPFH = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        
        # self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.FPFH_type=FPFH_type


        
        self.bn_DGCNN = nn.BatchNorm2d(128)
        self.conv_DGCNN = nn.Sequential(nn.Conv2d(81*2, 128, kernel_size=1, bias=False),
                                   self.bn_DGCNN,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn_DGCNN2 = nn.BatchNorm2d(128)
        self.conv_DGCNN2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   self.bn_DGCNN2,
                                   nn.LeakyReLU(negative_slope=0.2))
        

    def forward(self, x):
        n_pts = x.size()[2]
        # trans = None
        # trans = self.stn(x)
        
        if self.FPFH_type:
            input2=x[:,:-3,:]
            x=x[:,-3:,:]
            # x1=x[:,-3:,:]
            # idx=help(x, k=20)
        
        
        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            trans = x.clone()
            x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = None
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        
        if self.FPFH_type:
            # x = x / torch.linalg.matrix_norm(x, 'fro').reshape(-1,1,1)
            # input2 = input2 / torch.linalg.matrix_norm(input2, 'fro').reshape(-1,1,1)
            # print(torch.linalg.norm(input2, axis=1).unsqueeze(1).shape)
            # input2 = input2 / torch.linalg.norm(input2, axis=1).unsqueeze(1).repeat(1, input2.shape[1], 1)
            # x = torch.cat((x,input2),1)
            # # print(x.shape)
    
            # input2 = F.relu(self.bn1_2(self.conv1_2(input2)))

            # print(input2.shape)
            input2 = get_graph_feature(input2, idx)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
            # print(input2.shape)
            input2 = self.conv_DGCNN(input2)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
            # print(input2.shape)
            input2 = input2.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
            # print(input2.shape)

            # print(x.shape)
            x = get_graph_feature(x, idx)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
            # print(x.shape)
            x = self.conv_DGCNN2(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
            # print(x.shape)
            x = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
            # print(x.shape)


            x = torch.cat((x,input2),1)
            # print(x.shape)

            pointfeat = x
            x = F.relu(self.bn2_FPFH(self.conv2_FPFH(x)))
            x = self.bn3(self.conv3_FPFH(x))
        
        else:            
            pointfeat = x
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))

        # pointfeat = x
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = self.bn3(self.conv3(x))


        h = x.clone()
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        
        # input2 = torch.max(input2, 2)[0]
        # print(input2.shape)
        # print(x.shape)
        # x = torch.cat((x,input2),1)
        # print(x.shape)


        if self.global_feat:
            features = x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            features = torch.cat([x, pointfeat], 1)
        # print('feature 1 ',features.shape)
        
        temp = features
        features = F.relu(self.bn4(self.fc1(features)))
        # print('feature 2 ',features.shape)

        features = F.relu(self.bn5(self.dropout(self.fc2(features))))
        # print('feature 3',features.shape)
        
        return features, trans, h, trans_feat, temp


class PointNetfeat2(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, input_transform=True,in_k=3):
        super(PointNetfeat2, self).__init__()
        self.input_transform = input_transform
        if self.input_transform:
            self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(in_k, 64, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # import torch.nn.functional as F
        n_pts = x.size()[2]
        # print(x.shape)
        input1=x[:,64:,:]
        input2=x[:,:64,:]
        # print('\nx: ',x.shape)
        # print(input1.shape)
        # print(input2.shape)

        trans = None
        # if self.input_transform:
        #     trans = self.stn(x)
        #     x = x.transpose(2, 1)
        #     x = torch.bmm(x, trans)
        #     trans = x.clone()
        #     x = x.transpose(2, 1)
        input1 = F.relu(self.bn1(self.conv1(input1)))

        trans_feat = None
        # if self.feature_transform:
        #     trans_feat = self.fstn(x)
        #     x = x.transpose(2, 1)
        #     x = torch.bmm(x, trans_feat)
        #     x = x.transpose(2, 1)


        # aa=torch.linalg.matrix_norm(input1, 'fro')
        # print(aa.shape)

        # aa=torch.linalg.norm(input1, 'fro')
        # print(aa.shape)
        
        # print('0 ',input1[0,:5,:5])
        # print('5 ',input1[5,:5,:5])
        input1 = input1 / torch.linalg.matrix_norm(input1, 'fro').reshape(-1,1,1)
        # print(input1.shape)
        # print('0 ',aa[0])
        # print('0 ',input1[0,:5,:5])
        # print('5 ',aa[5])
        # print('5 ',input1[5,:5,:5])

        input2 = input2 / torch.linalg.matrix_norm(input2, 'fro').reshape(-1,1,1)
        # print(input2.shape)

        # asddas=asd

        pointfeat = torch.cat((input1,input2),1)
        # w=0.5
        # FF = np.hstack(( (1-w) * input1, w * input2 ))
        
        # print(pointfeat.shape)
        # print(FF.shape)

        
        x = F.relu(self.bn2(self.conv2(pointfeat)))
        x = self.bn3(self.conv3(x))
        h = x.clone()
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            features1 = x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            features1 = torch.cat([x, pointfeat], 1)

        features2 = F.relu(self.bn4(self.fc1(features1)))
        features = F.relu(self.bn5(self.dropout(self.fc2(features2))))
        return features, features1, pointfeat, h


class PointNetClsFPFH(nn.Module):
    def __init__(self, k=2, feature_transform=False, last_fc=False, log_softmax=False, input_transform=False, log=True):
        super(PointNetClsFPFH, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat2(global_feat=True, feature_transform=feature_transform, input_transform=input_transform,in_k=3)
        self.fc3 = nn.Linear(256, k)
        self.last_fc = last_fc
        self.log_softmax = log_softmax
        self.feature = None
        self.log = log

    def forward(self, x):
        # print('\n')
        x, trans, trans_feat, x2 = self.feat(x)
        # print(x.shape,'\n')
        self.feature = x
        if self.last_fc:
            x = self.fc3(x)
            # print(x.shape,'\n')
        if self.log_softmax:
            if self.log:
                x = F.log_softmax(x, dim=1)
            else:
                x = F.softmax(x, dim=1)
        # print(x.shape)
        return x, trans, trans_feat

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        return self



class PointNetCls2(nn.Module):
    def __init__(self, k=2, feature_transform=False, last_fc=False, log_softmax=False, input_transform=False, log=True, FPFH_type=False):
        super(PointNetCls2, self).__init__()
        self.feature_transform = feature_transform
        self.FPFH_type=FPFH_type
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, input_transform=input_transform, FPFH_type=self.FPFH_type)
        self.fc3 = nn.Linear(256, k)
        self.last_fc = last_fc
        self.log_softmax = log_softmax
        self.feature = None
        self.log = log

    def forward(self, x):
        x, trans, trans_feat, _, _ = self.feat(x)
        self.feature = x
        # print('feature ',x.shape)

        if self.last_fc:
            x = self.fc3(x)
            # print('fc3 ',x.shape)

        if self.log_softmax:
            if self.log:
                x = F.log_softmax(x, dim=1)
            else:
                x = F.softmax(x, dim=1)
        # print('fc3 ',x.shape)

        return x, trans, trans_feat

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        return self

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
        
class PointNet2Cls(nn.Module):
    def __init__(self, k=2, feature_transform=False, last_fc=False, log_softmax=False, input_transform=False, log=True, FPFH_type=False):
        super(PointNet2Cls, self).__init__()
        self.feature_transform = feature_transform
        # self.feat = pointMLPfeat(num_classes=40)
        self.FPFH_type=FPFH_type
        num_classes=k
        # self.emb_dims=1024
        # self.k=20
        # self.dropout=0.5

        print('self.FPFH_type ',self.FPFH_type)

        self.feat = pointnet2_cls_msg.get_model(num_classes, normal_channel=self.FPFH_type)
        self.feat.apply(inplace_relu)
    
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)


        # self.fc3 = nn.Linear(256, k)
        # self.last_fc = last_fc
        # self.log_softmax = log_softmax
        # self.feature = None
        # self.log = log
        

    def forward(self, x):
        # x, trans, trans_feat, _, _ = self.feat(x)
        
        B, _, _ = x.shape
        # print('x ', x.shape)
        
        x, l3_points = self.feat(x)
        # self.feature = x
        # print('x ', x.shape)
        
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        # # print('linear3 ',x.shape)

        trans=None
        trans_feat = None
        
        return x, trans, l3_points 

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        return self


class PointNetCls_pointmlp(nn.Module):
    def __init__(self, k=2, feature_transform=False, last_fc=False, log_softmax=False, input_transform=False, log=True, FPFH_type=False):
        super(PointNetCls_pointmlp, self).__init__()
        self.feature_transform = feature_transform
        # self.feat = pointMLPfeat(num_classes=40)
        self.FPFH_type=FPFH_type

        num_classes=k
        self.feat = Modelfeat(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], FPFH_type=self.FPFH_type)
        
        self.fc3 = nn.Linear(256, k)
        self.last_fc = last_fc
        self.log_softmax = log_softmax
        self.feature = None
        self.log = log
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(last_channel, 512),
        #     nn.BatchNorm1d(512),
        #     self.act,
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     self.act,
        #     nn.Dropout(0.5),
        #     nn.Linear(256, self.class_num)
        # )
        
        

    def forward(self, x):
        # x, trans, trans_feat, _, _ = self.feat(x)
        
        x, trans, trans_feat, _, _  = self.feat(x)
        # x=self.classifier(x)
        # print('linear3 ',x.shape)
        
        if self.last_fc:
            x = self.fc3(x)
        if self.log_softmax:
            if self.log:
                x = F.log_softmax(x, dim=1)
            else:
                x = F.softmax(x, dim=1)

        # print('fc3 ',x.shape)
        
        return x, trans, trans_feat 

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        return self



class PointNetCls_DGCNN(nn.Module):
    def __init__(self, k=2, feature_transform=False, last_fc=False, log_softmax=False, input_transform=False, log=True, FPFH_type=False):
        super(PointNetCls_DGCNN, self).__init__()
        self.feature_transform = feature_transform
        # self.feat = pointMLPfeat(num_classes=40)
        self.FPFH_type=FPFH_type
         
        num_classes=k
        self.emb_dims=1024
        self.k=20
        self.dropout=0.5
        
        self.feat = DGCNN_clsfeat(emb_dims=1024, k=20, dropout=0.5, output_channels=num_classes, FPFH_type=self.FPFH_type)
        

        self.fc3 = nn.Linear(256, k)
        self.last_fc = last_fc
        self.log_softmax = log_softmax
        self.feature = None
        self.log = log
        

    def forward(self, x):
        # x, trans, trans_feat, _, _ = self.feat(x)
        
        x, trans, trans_feat, _, _  = self.feat(x)
        self.feature = x
        
        # print('linear3 ',x.shape)

        if self.last_fc:
            x = self.fc3(x)
        if self.log_softmax:
            if self.log:
                x = F.log_softmax(x, dim=1)
            else:
                x = F.softmax(x, dim=1)

        # print('fc3 ',x.shape)

        
        # return x, trans_feat 
        return x, trans, trans_feat 

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        return self


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class PointNetLwf(nn.Module):
    def __init__(self, shared_model, old_k, new_k):
        super(PointNetLwf, self).__init__()
        for param in shared_model.parameters():
            param.requires_grad = True
        self.shared_model = shared_model.feat
        fc3 = nn.Linear(256, old_k)

        self.classifiers = nn.ModuleList([
            nn.ModuleDict({
                'fc3': fc3
            }),

            nn.ModuleDict({
                'fc3': nn.Linear(256, new_k)
            })
        ])

        self.classifiers[1].apply(init_weights)

    def forward(self, x):
        x, trans, trans_feat = self.shared_model(x)

        old = self.classifiers[0].fc3(x)

        # new
        new = self.classifiers[1].fc3(x)
        return F.log_softmax(old, dim=1), F.log_softmax(new, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat, _ = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k=5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k=3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
