
import os
import sys
import torch
import numpy as np
from torch.cuda import temperature
from torch.utils.data import DataLoader
import datasets.tta_datasets as tta_datasets
import random
import datetime
import logging
import provider
import importlib
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.pointnet.model import PointNetLwf, PointNetCls2, PointNetCls_pointmlp, PointNetCls_DGCNN, PointNetClsFPFH, PointNet2Cls

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--device', action='store_true', default= device  , help='use cpu mode')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.000005, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--dataset_name', default='modelnet', help='model name [default: modelnet]')
    parser.add_argument('--tta_dataset_path',type=str, default='/content/CTTT/modelnet40_c/modelnet40_c', help='TTA dataset path')
    parser.add_argument('--model_path', type=str, default="/content/CTTT/pretrained_model/" , help='pretrined model path')
    parser.add_argument('--base_image_save_path', type=str, default='/content/CTTT/SaveImage' , help='save image path')

    parser.add_argument('--severity', default=5, help='severity for corruption dataset')
    parser.add_argument('--online', default=True, help='online training setting')
    parser.add_argument('--grad_steps', default=1, help='if we train online, we have to set this to one')
    parser.add_argument('--split', type=str, default='test', help='Data split to use: train/test/val')
    parser.add_argument('--debug', action='store_true', help='Use debug mode with a small dataset')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset during loading')
    parser.add_argument('--disable_bn_adaptation', action='store_true', help='Disable batch normalization adaptation')
    parser.add_argument('--stride_step', type=int, default=1, help='Stride step for logging or operations')
    parser.add_argument('--batch_size_tta', type=int, default=1, help='batch size in training')
    parser.add_argument('--enable_plots', action='store_true', default=True, help='plots images')
    parser.add_argument('--IWF', action='store_true', default=True, help='enable invariant weak Filtering')
    parser.add_argument('--label_refinement', action='store_true', default=True, help='enable pseudo label refinement')
    parser.add_argument('--seed', type=int, default=10, help='random seed for reproducibility')
    
    return parser.parse_args()





def load_tta_dataset(args):
    # we have 3 choices - every tta_loader returns only point and labels
    root = args.tta_dataset_path  # being lazy - 1

    if args.dataset_name == 'modelnet':
        root = '/content/CTTT/modelnet40_c/modelnet40_c'


        if args.corruption == 'clean':
            inference_dataset = tta_datasets.ModelNet_h5(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'scanobject':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'shapenetcore':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    else:
        raise NotImplementedError(f'TTA for ---- is not implemented')

    print(f'\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n')

    return tta_loader





    
def concat_same_input(points, num_repeats=8):
    """ Concatenate the same input points to form a batch where each point is repeated `num_repeats` times """
    # Repeat the points along a new batch dimension
    repeated_points = points.repeat(num_repeats, 1, 1)  # Shape: [num_repeats, 3, 1024]
    return repeated_points


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


# Function to calculate the entropy of the predictions

def pc_normalize(pc):
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
    pc = pc / m
    return pc

def  process_point_cloud(point_cloud, neighbour=5):

    point_cloud=torch.asarray(point_cloud)
    dist = torch.cdist(point_cloud, point_cloud)
    # print(dist.shape)
    # print(dist)
    # neighbors_array = torch.argsort(dist, dim=1)
    # neighbors_array=neighbors_array[:,:6]
    # print(neighbors_array.shape)
    # print(neighbors_array)


    dist= dist+((dist==0) * 100)
    neighbors_array = torch.argsort(dist, dim=1)
    neighbors_array=neighbors_array[:,:neighbour]
    # print(neighbors_array.shape)
    # print(neighbors_array)

    dist = torch.sort(dist, dim=1)[0]
    dist= dist[:,:neighbour]
    # print(dist.shape)
    # print(dist)

    distance_threshold = torch.mean(dist.min(dim=0)[0])
    # print(torch.sum(dist>distance_threshold*2, dim=1)/neighbour > 0.75)
    mask = torch.sum(dist>distance_threshold*2, dim=1)/neighbour < 0.75

    # mask = torch.ones(point_cloud.shape[0], dtype=torch.bool)
    # mask[nodes_to_remove] = False

    # Apply the mask to the tensor
    point_cloud = point_cloud[mask]
    # print(point_cloud.shape)
    # print(point_cloud1.shape)

    point_cloud = pc_normalize(point_cloud)

    return point_cloud



def kl_divergence_loss(logits1, logits2):
    """Calculate KL Divergence between two sets of logits."""
    p = F.log_softmax(logits1, dim=1)  # Log-Softmax for the first logits
    q = F.softmax(logits2, dim=1)      # Softmax for the second logits
    return F.kl_div(p, q, reduction='batchmean')  # Batch-wise KL Divergence

# Function to calculate the entropy of the predictions
def entropy_loss(predictions):
    """Calculate the entropy loss for a batch of predictions.
       Entropy is maximized when the distribution is uniform, and minimized when it's one-hot."""
    p = F.softmax(predictions, dim=1)
    log_p = F.log_softmax(predictions, dim=1)
    entropy = -(p * log_p).sum(dim=1)  # Summing over classes
    return entropy.mean()  # Mean entropy over the batch

def augment_data(points):
    """ Apply common weak augmentations to point cloud data with random augmentations. """
    # Change seed inside the function to ensure diversity in augmentations
    random.seed()  # Reset seed to system time (non-deterministic)
    torch.manual_seed(torch.randint(0, 10000, (1,)).item())

   # Random scaling
    scale = torch.FloatTensor(points.shape[0], 1, 1).uniform_(0.95, 1.05).to(points.device)
    scaled_points = points * scale

    # Random translation
    translation = torch.FloatTensor(points.shape[0], 3, 1).uniform_(-0.01, 0.01).to(points.device)
    translated_points = scaled_points + translation
 
    # Random jittering (small noise)
    jitter = torch.randn_like(translated_points) * 0.04
    jittered_points = translated_points + jitter

    # Cutout augmentation (removing random points)
    num_points_to_remove = int(0.15 * points.shape[2])  # Remove 10% of the points
    mask = torch.ones(points.shape[2], dtype=torch.bool)
    drop_indices = torch.randperm(points.shape[2])[:num_points_to_remove]
    mask[drop_indices] = False
    cutout_points = jittered_points[:, :, mask]

    # Random rotation around the z-axis (yaw) with constraints
    angle = torch.FloatTensor(1).uniform_(-0.3, 0.3).to(points.device)  # Slight rotation
    cos_val = torch.cos(angle)
    sin_val = torch.sin(angle)
    rotation_matrix = torch.tensor([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0],
        [0, 0, 1]
    ], dtype=torch.float32).to(points.device)
    rotated_points = torch.matmul(cutout_points.transpose(2, 1), rotation_matrix).transpose(2, 1)

    # Return the augmented points
    return rotated_points


def generate_weighted_pseudo_labels(stu_pred, aug_stu_pred, teacher_pred, teacher_weight=2.0, label_smoothing=0.1):
    # Apply weighting to the teacher probabilities
    weighted_teacher_pred = teacher_weight * teacher_pred

    # Calculate the weighted mean of the probabilities
    total_weight = 1 + 1 + teacher_weight
    weighted_avg_prob = (stu_pred + aug_stu_pred + weighted_teacher_pred) / total_weight

    # Apply label smoothing
    num_classes = stu_pred.size(1)
    smoothed_labels = (1 - label_smoothing) * weighted_avg_prob + (label_smoothing / num_classes)

    # Convert smoothed probabilities to class indices using argmax
    pseudo_labels_class_indices = torch.argmax(smoothed_labels, dim=1)

    return pseudo_labels_class_indices



def assign_pseudo_labels_with_confidence_logic(stu_pred, aug_stu_pred, teacher_pred, teacher_conf_threshold_high=0.85, teacher_conf_threshold_low=0.8):
    """
    Assigns pseudo-labels based on teacher confidence:
    - If teacher's confidence is > 0.9, use teacher's prediction.
    - If teacher's confidence is between 0.9 and 0.7, use the average of teacher and student.
    - If teacher's confidence is < 0.7, use the mean of teacher, student, and augmented student predictions.
    """
    
    stu_confidence, stu_labels = torch.max(F.softmax(stu_pred, dim=1), dim=1)  
    teacher_confidence, teacher_labels = torch.max(F.softmax(teacher_pred, dim=1), dim=1)  
    

    teacher_prob = F.softmax(teacher_pred, dim=1)
    student_prob = F.softmax(stu_pred, dim=1)
    aug_student_prob = F.softmax(aug_stu_pred, dim=1)


    high_conf_mask = teacher_confidence > teacher_conf_threshold_high
    mid_conf_mask = (teacher_confidence <= teacher_conf_threshold_high) & (teacher_confidence > teacher_conf_threshold_low)
    low_conf_mask = teacher_confidence <= teacher_conf_threshold_low

    # High confidence (> 0.9): Use teacher's prediction
    pseudo_labels = torch.where(high_conf_mask, teacher_labels, stu_labels)  

    # Mid confidence (between 0.7 and 0.9): Average between teacher and student
    avg_prob_mid = (teacher_prob + student_prob) / 2.0
    pseudo_labels_mid = torch.argmax(avg_prob_mid, dim=1)
    pseudo_labels = torch.where(mid_conf_mask, pseudo_labels_mid, pseudo_labels)

    # Low confidence (< 0.7): Mean of teacher, student, and augmented student
    avg_prob_low = (teacher_prob + student_prob + aug_student_prob) / 3.0
    pseudo_labels_low = torch.argmax(avg_prob_low, dim=1)
    pseudo_labels = torch.where(low_conf_mask, pseudo_labels_low, pseudo_labels)

    #(1 = Teacher, 2 = Average, 3 = Mean)
    label_source = torch.where(high_conf_mask, torch.ones_like(teacher_labels), torch.zeros_like(teacher_labels))
    label_source = torch.where(mid_conf_mask, torch.full_like(teacher_labels, 2), label_source)
    label_source = torch.where(low_conf_mask, torch.full_like(teacher_labels, 3), label_source)

    return pseudo_labels, label_source


def make_feature(teacher_model):
      # Step 1: Create empty tensors for features and probability
    featurese = torch.tensor([], dtype=torch.float32).to(args.device)
    probabilitye = torch.tensor([], dtype=torch.float32).to(args.device)

    # Step 2: Generate random point cloud data with a batch size of 32
    # num_points = 1024
    #points12 = torch.rand((1024, 1024, 3)) * 2 - 1  # Random points in the range [-1, 1]
    points12 = torch.load('points12.pt')
    points12 = points12.to(args.device)
    # Step 3: Define the batch size for processing
    # batch_size_for_processing = 64

    # Step 4: Apply the pre-trained model to each batch of size 128
    for i in range(0, 1024, 64):
        batch_points = points12[i:i + 64]
        pred_student_original1, _, trans_feat_original1 = teacher_model(batch_points.transpose(2, 1))
        # print("trans_feat_original:",trans_feat_original.shape)

        # Concatenate the results
        featurese = torch.cat((featurese, trans_feat_original1), dim=0)
        probabilitye = torch.cat((probabilitye, pred_student_original1), dim=0)

    print("Features shape:",featurese.shape)

    num_classese = pred_student_original1.shape[1]

    class_indicese = torch.argmax(probabilitye, dim=1)

    final_features = torch.zeros((num_classese, trans_feat_original1.shape[1]), dtype=torch.float32).to(args.device)
    final_probability = torch.zeros((num_classese, num_classese), dtype=torch.float32).to(args.device)
    # print("Features shape:", final_probability.shape)
    print("Features shape:", final_features.shape)

    for class_idx in range(num_classese):
        class_mask = (class_indicese == class_idx)
        if class_mask.sum() > 0:
            final_features[class_idx] = featurese[class_mask].mean(dim=0)
            final_probability[class_idx] = probabilitye[class_mask].mean(dim=0)

    # Ensure the final feature matrix has shape (40, 1024) and the probability matrix has shape (40, 40)
    # final_features = mean_features
    # final_probability = mean_probability

    # print("Final Features shape:", final_features.shape)
    # print("Final Features shape:", final_features)
    # print("Final Probability shape:", final_probability.shape)
    # print("Final Probability shape:", final_probability)
    return final_features, final_probability
    
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)


    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    corruptions = [
        'uniform', 'gaussian', 'background', 'impulse', 'upsampling',
        'distortion_rbf', 'distortion_rbf_inv', 'density', 'density_inc',
        'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar'
    ]
        
    dataset_name = args.dataset_name
    npoints = args.num_point
    num_class = args.num_category
    level = [5]





    '''MODEL LOADING'''
    loss_type = 'focal_loss'
    model = importlib.import_module('pointnet_cls')

    if args.model == 'pointnet':
        teacher_model = PointNetCls2(k=num_class, feature_transform=True, input_transform=True, log=True if loss_type != 'focal_loss' else False,FPFH_type=False)           
        student_model = teacher_model.copy()
    elif args.model == 'pointmlp':
        teacher_model = PointNetCls_pointmlp(k=num_class, feature_transform=True, input_transform=True, log=True if loss_type != 'focal_loss' else False,FPFH_type=False)
        student_model = teacher_model.copy()
        criterion = model.get_loss_pointmlp()
    elif args.model == 'dgcnn':
        teacher_model = PointNetCls_DGCNN(k=num_class, feature_transform=True, input_transform=True, log=True if loss_type != 'focal_loss' else False,FPFH_type=False)
        student_model = teacher_model.copy()
        criterion = model.get_loss_dgcnn()

    elif args.model == 'pointnet2':
        teacher_model = PointNet2Cls(k=num_class, feature_transform=True, input_transform=True, log=True if loss_type != 'focal_loss' else False,FPFH_type=False)
        student_model = teacher_model.copy()
        criterion = model.get_loss()
    elif args.model == 'pointnet_cls':
        teacher_model = model.get_model(num_class, normal_channel=args.use_normals)
        # student_model = teacher_model.copy()
        student_model = model.get_model(num_class, normal_channel=args.use_normals)
        teacher_model_2 = model.get_model(num_class, normal_channel=args.use_normals)

        criterion = model.get_loss()



    print(args.device)
    teacher_model = teacher_model.to(args.device)

    student_model = student_model.to(args.device)
    criterion = criterion.to(args.device)

    teacher_model.apply(inplace_relu)
    student_model.apply(inplace_relu)
    # Load weights into both teacher and student models
    model_path = args.model_path

    if args.model == 'pointnet':
        torch.save(teacher_model.state_dict(), model_path+"pointnet_3.pth")
        checkpoint = torch.load(model_path+"pointnet_3.pth", map_location=torch.device(args.device))
        
    elif args.model == 'pointmlp':
        torch.save(teacher_model.state_dict(), model_path+"pointmlp_3.pth")
        checkpoint = torch.load(model_path+"pointmlp_3.pth", map_location=torch.device(args.device))

    elif args.model == 'dgcnn':
        torch.save(teacher_model.state_dict(), model_path+"DGCNN.1024_3.pth")
        # checkpoint = torch.load(model_path+"DGCNN.1024.t7", map_location=torch.device(args.device))
        # checkpoint = torch.load(model_path+"DGCNN.1024.pth", map_location=torch.device(args.device))
        checkpoint = torch.load(model_path+"DGCNN.1024_3.pth", map_location=torch.device(args.device))

    elif args.model == 'pointnet2':
        torch.save(teacher_model.state_dict(), model_path+"pointnet2_3.pth")
        checkpoint = torch.load(model_path+"pointnet2_3.pth", map_location=torch.device(args.device))

    elif args.model == 'pointnet_cls':
        checkpoint = torch.load(model_path+"pointnet_cls.pth", map_location=torch.device(args.device))
    
    print('teacher_model ',teacher_model)

    if 'model_state_dict' in checkpoint:
        # model_state_dict = checkpoint['state_dict']
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        student_model.load_state_dict(checkpoint['model_state_dict'])
        
    else:
        # model_state_dict = checkpoint
        teacher_model.load_state_dict(checkpoint)
        student_model.load_state_dict(checkpoint)


    # teacher_model.load_state_dict(checkpoint['model_state_dict'])
    # student_model.load_state_dict(checkpoint['model_state_dict'])


    # Freezing Teacher model
    for param in teacher_model.parameters():
        param.requires_grad = False

    for param in student_model.parameters():
        param.requires_grad = True

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=args.learning_rate
        )
    else:
        optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9)

    '''
    import torch

    # Generate random points in the range [-1, 1]
    points12 = torch.rand((1024, 1024, 3)) * 2 - 1

    # Save the generated points to a file
    torch.save(points12, 'points12.pt')
    '''
    
    teacher_model.eval()
    final_features, final_probability=make_feature(teacher_model)
    
    total_classification_losses = {corruption: 0.0 for corruption in corruptions}
    total_consistency_losses = {corruption: 0.0 for corruption in corruptions}
    total_consistency2_losses = {corruption: 0.0 for corruption in corruptions}
    total_entropy_losses = {corruption: 0.0 for corruption in corruptions}
    loss_counts = {corruption: 0 for corruption in corruptions}  # Count the number of losses for averaging


    final_accuracy = {}

    if args.enable_plots :
        teacher_pseudo_labels ={}
        teacher_pseudo_labels_prob ={}
        weakaug_student_pseudo_labels = {}
        weakaug_student_pseudo_labels_prob = {}
        strongaug_student_pseudo_labels ={}
        strongaug_student_pseudo_labels_prob ={}
        original_labels={}
        teacher_confidance_scores={}
        weakaug_student_confidance_scores={}
        strongaug_student_confidance_scores={}


    correct_teacher_pseudo_labels = {}
    correct_student_pseudo_labels = {}
    teacher_student_equal_labels = {}

    lambda_classification = 0.6
    lambda_consistency = 0.4  
    lambda_consistency2 =0.07  

    lambda_entropy = 3.5 

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            if args.corruption == 'clean':
                continue

            tta_loader = load_tta_dataset(args)
            total_batches = len(tta_loader)

            test_pred = torch.tensor([], dtype=torch.long).to(torch.device(args.device))
            test_label = torch.tensor([], dtype=torch.long).to(torch.device(args.device))


            correct_teacher_pseudo_labels[args.corruption] = 0
            correct_student_pseudo_labels[args.corruption] = 0
            teacher_student_equal_labels[args.corruption] = 0 


            if args.enable_plots :
                teacher_pseudo_labels[args.corruption] = list()
                teacher_pseudo_labels_prob[args.corruption] = list()
                weakaug_student_pseudo_labels[args.corruption] = list()
                weakaug_student_pseudo_labels_prob[args.corruption] = list()
                strongaug_student_pseudo_labels[args.corruption] = list()
                strongaug_student_pseudo_labels_prob[args.corruption] = list()
                original_labels[args.corruption] = list()
                teacher_confidance_scores[args.corruption]=list()
                weakaug_student_confidance_scores[args.corruption]=list()
                strongaug_student_confidance_scores[args.corruption]=list()


            for idx, (data, labels) in enumerate(tta_loader):
                student_model.zero_grad()
                student_model.train()

                # Ensure batch norm layers are disabled for both models
                for m in student_model.modules():
                    if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                        m.eval()
                
                teacher_model.eval()


                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points1 = data.to(args.device)
                        points = points1.clone()
                        if args.IWF:
                            points = process_point_cloud(points1, neighbour=5).reshape(1,-1,3)
                            points = points.permute(0, 2, 1)  # Now the shape will be [1, 3, 1024]

                        points1 = points1.permute(0, 2, 1)  # Now the shape will be [1, 3, 1024]

                        labels = labels.to(args.device)
                        pred_teacher, _ ,_= teacher_model(points)
                        pseudo_labels = pred_teacher.argmax(dim=1).to(args.device)
                        pseudo_labels = pseudo_labels.long().to(args.device)
                        
                        pred_student_original, trans_feat_original,_ = student_model(points)
                        pred_student = pred_student_original.argmax(dim=1).to(args.device)
                        stu_pseudo_labels = pred_student.long().to(args.device)


                        #pseudo_labels = torch.max(pred_teacher, pred_student_original).argmax(dim=1).to(args.device)                 

                        augmented_points = augment_data(points1)

                        # Predictions for original and augmented data
                        pred_student_augmented, trans_feat_augmented, _= student_model(augmented_points)
                        pred_student_au = pred_student_augmented.argmax(dim=1).to(args.device)
                        stu_pseudo_labels_aug = pred_student_au.long().to(args.device)

                        #pseudo_labels = generate_weighted_pseudo_labels(pred_student_original, pred_student_augmented, pred_teacher).long().to(args.device)
                        if args.label_refinement:
                            pseudo_labels,_ = assign_pseudo_labels_with_confidence_logic(pred_student_original, pred_student_augmented, pred_teacher, teacher_conf_threshold_high=0.85, teacher_conf_threshold_low=0.8)

                        #classification_loss = criterion(pred_student_original, pseudo_labels, trans_feat_original)
                        classification_loss = criterion(pred_student_original, pseudo_labels,trans_feat_original)                        # Consistency loss (difference between original and augmented predictions)
                        #consistency_loss = F.mse_loss(pred_student_original, pred_student_augmented)
                        consistency_loss = criterion(pred_student_augmented, pseudo_labels ,trans_feat_augmented)
                        #consistency_loss2 = kl_divergence_loss(pred_student_original, pred_student_augmented)
                        consistency_loss2 = F.mse_loss(pred_student_original, pred_student_augmented)

                        entropy_loss_value = entropy_loss(pred_student_original)

                        total_loss = (lambda_classification*classification_loss 
                                      + lambda_consistency * consistency_loss
                                      + lambda_consistency2 * consistency_loss2
                                      + lambda_entropy * entropy_loss_value)

                        scaled_classification_loss = lambda_classification * classification_loss.item()
                        scaled_consistency_loss = lambda_consistency * consistency_loss.item()
                        scaled_consistency2_loss = lambda_consistency2 * consistency_loss2.item()
                        scaled_entropy_loss = lambda_entropy * entropy_loss_value.item()

                        # Accumulate total losses for each corruption type
                        total_classification_losses[args.corruption] += scaled_classification_loss
                        total_consistency_losses[args.corruption] += scaled_consistency_loss
                        total_consistency2_losses[args.corruption] += scaled_consistency2_loss
                        total_entropy_losses[args.corruption] += scaled_entropy_loss
                        loss_counts[args.corruption] += 1

                        # Backward pass and optimization step
                        total_loss.backward()
                        optimizer.step()
                        student_model.zero_grad()
                        optimizer.zero_grad()

                        if args.enable_plots :
                            teacher_pseudo_labels[args.corruption].append(pseudo_labels)
                            teacher_pseudo_labels_prob[args.corruption].append(pred_teacher)
                            weakaug_student_pseudo_labels[args.corruption].append(stu_pseudo_labels)
                            weakaug_student_pseudo_labels_prob[args.corruption].append(pred_student_original)
                            strongaug_student_pseudo_labels[args.corruption].append(stu_pseudo_labels_aug)
                            strongaug_student_pseudo_labels_prob[args.corruption].append(pred_student_augmented)
                            original_labels[args.corruption].append(labels)


                        if pseudo_labels == labels:
                            correct_teacher_pseudo_labels[args.corruption] += 1
                        if pred_student == labels:
                            correct_student_pseudo_labels[args.corruption] += 1
                        if pred_student == pseudo_labels:
                            teacher_student_equal_labels[args.corruption] += 1


                        '''
                        log_string(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                                  f'classification_loss {lambda_classification * classification_loss.item()},'
                                  f'consistency_loss {lambda_consistency * consistency_loss.item()},'
                                  f'consistency_loss2 {lambda_consistency2 * consistency_loss2.item()},'
                                  f'entropy_loss_value {lambda_entropy * entropy_loss_value.item()},'
                                  f'Total Loss {total_loss.item()}')
                        '''
                # Inference on the sample
                student_model.eval()
                labels = labels.to(args.device)

                target = labels[0]
                pred= pred_student[0]

                test_pred = torch.cat((test_pred, pred.unsqueeze(0)), dim=0)
                test_label = torch.cat((test_label, target.unsqueeze(0)), dim=0)
                    
                if idx % 200 == 0 and idx!=0 :
                    acc = (test_pred == test_label).sum().item() / float(len(test_label)) * 100.
                    intermediate_accuracy = (test_pred == test_label).sum().item() / float(test_label.size(0)) * 100.

                    log_string(f'Intermediate Accuracy - IDX {idx} - {intermediate_accuracy:.1f} ,'
                              f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'classification_loss {lambda_classification * classification_loss.item():.3f}, '
                              f'consistency_loss {lambda_consistency * consistency_loss.item():.3f}, '
                              f'consistency_loss2 {lambda_consistency2 * consistency_loss2.item():.3f}, '
                              f'entropy_loss_value {lambda_entropy * entropy_loss_value.item():.3f}, '
                              f'Total Loss {total_loss.item():.3f}')


            # Compute the mean for each scaled loss component after all batches are processed
            total_classification_losses[args.corruption] /= loss_counts[args.corruption]
            total_consistency_losses[args.corruption] /= loss_counts[args.corruption]
            total_consistency2_losses[args.corruption] /= loss_counts[args.corruption]
            total_entropy_losses[args.corruption] /= loss_counts[args.corruption]

            # Compute accuracy for the corruption type
            acc = (test_pred == test_label).sum().item() / float(test_label.size(0)) * 100.
            log_string(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n')
            final_accuracy[args.corruption] = acc


    log_string('------------------------------train finished -----------------------------')
    for _, args.corruption in enumerate(corruptions):
        log_string(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {final_accuracy[args.corruption]} ########\n\n')
        final_mean_accuracy = np.mean([accuracy for accuracy in final_accuracy.values()])
    log_string(f' mean accuracy {final_mean_accuracy}\n\n\n')

    log_string(f'correct teacher pseudo labels ={correct_teacher_pseudo_labels} \n')
    log_string(f'correct student pseudo labels ={correct_student_pseudo_labels} \n')
    log_string(f'teacher student equal labels ={teacher_student_equal_labels} \n')

    # Concatenate the list of probabilities into a single tensor




    if args.enable_plots :

        base_image_save_path = args.base_image_save_path

        # Check if the directory exists, if not, create it
        if not os.path.exists(base_image_save_path):
            os.makedirs(base_image_save_path)

        print(f"Directory created at {base_image_save_path}")

        import shutil
        import helper_plots

        helper_plots.plot_grouped_bar_chart(correct_teacher_pseudo_labels,correct_student_pseudo_labels,teacher_student_equal_labels,base_image_save_path, "Comparison of Pseudo Labels")
        helper_plots.plot_label_situation_function_corruption(teacher_pseudo_labels,
                                weakaug_student_pseudo_labels,
                                strongaug_student_pseudo_labels,
                                original_labels,
                                teacher_pseudo_labels_prob,
                                weakaug_student_pseudo_labels_prob,
                                strongaug_student_pseudo_labels_prob,
                                base_image_save_path, "Label condition")


        helper_plots.plot_mean_losses_and_accuracy(
                                    corruptions,
                                    total_classification_losses,
                                    total_consistency_losses,
                                    total_consistency2_losses,
                                    total_entropy_losses,
                                    final_accuracy)
                                     
        helper_plots.plot_teacher_confidence_stacked_bar(teacher_pseudo_labels_prob, base_image_save_path, "Teacher Prediction Confidence Distribution")
        helper_plots.compute_and_plot_metrics_for_corruptions(teacher_pseudo_labels_prob, weakaug_student_pseudo_labels_prob, 
                                                strongaug_student_pseudo_labels_prob, original_labels,
                                                confidence_threshold=0.7, base_image_save_path=base_image_save_path,
                                                title_prefix='Label Confidence Distribution')

        helper_plots.plot_teacher_predictions_multiple_thresholds(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path=base_image_save_path, title_prefix='Teacher Predictions by Thresholds')
        helper_plots.plot_teacher_predictions_multiple_thresholds11(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path=base_image_save_path, title_prefix='Teacher Predictions by Thresholds')
        helper_plots.plot_teacher_predictions_multiple_thresholds_with_accuracy(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path=base_image_save_path, title_prefix='Teacher Predictions by Thresholds')

        directory_to_zip = "/content/CTTT/SaveImage"
        output_filename = "/content/CTTT/SaveImage.zip"
        shutil.make_archive(output_filename.replace('.zip', ''), 'zip', directory_to_zip)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    args = parse_args() 
    set_seed(args.seed)  

    main(args)  
