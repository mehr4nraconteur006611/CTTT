
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
    parser.add_argument('--tta_dataset_path', default='/content/CTTT/modelnet40_c/modelnet40_c', help='/content/CTTT/modelnet40_c/modelnet40_c')
    parser.add_argument('--severity', default=5, help='severity for corruption dataset')
    parser.add_argument('--online', default=True, help='online training setting')
    parser.add_argument('--grad_steps', default=1, help='if we train online, we have to set this to one')
    parser.add_argument('--split', type=str, default='test', help='Data split to use: train/test/val')
    parser.add_argument('--debug', action='store_true', help='Use debug mode with a small dataset')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset during loading')
    parser.add_argument('--disable_bn_adaptation', action='store_true', help='Disable batch normalization adaptation')
    parser.add_argument('--stride_step', type=int, default=1, help='Stride step for logging or operations')
    parser.add_argument('--batch_size_tta', type=int, default=1, help='batch size in training')
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




import torch
import numpy as np
import matplotlib.pyplot as plt



def plot_teacher_predictions_multiple_thresholds_with_accuracy(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path='', title_prefix='Teacher Predictions by Thresholds'):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))  # Two plots: ax1 for the first plot, ax2 for the accuracy plot
    
    # Part 1: Stacked Bar Plot for Teacher Predictions by Thresholds (Existing Plot)
    # Define colors and threshold labels for stacked bars
    threshold_labels = [f"< {threshold}" for threshold in thresholds] + [f">= {thresholds[-1]}"]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0'][:len(threshold_labels)]
    categories = ['Total', 'Correct', 'Incorrect']
    
    # Bar chart parameters
    n_categories = len(teacher_pseudo_labels_prob.keys())  # Number of main categories (corruption types)
    n_cols_per_category = 3  # Number of columns per category (Total, Correct, Incorrect)
    bar_width = 0.25  # Width of each bar
    category_spacing = 0.4  # Extra space between categories
    
    # Calculate positions for bars
    category_width = (n_cols_per_category * bar_width) + category_spacing
    x_positions = np.arange(n_categories) * category_width

    all_categories = []
    accuracies = []  # To store accuracy for each category
    
    # Start plotting each corruption type for the first stacked bar plot
    for cat_idx, corruption_type in enumerate(teacher_pseudo_labels_prob.keys()):
        print(f"Processing corruption type: {corruption_type}")
        
        if corruption_type not in original_labels:
            print(f"Error: Corruption type '{corruption_type}' not found in original_labels")
            continue
        
        # Concatenate tensors for current corruption type (for teacher)
        teacher_pseudo_labels_prob_tensor = torch.cat(teacher_pseudo_labels_prob[corruption_type], dim=0)
        
        # Compute softmax probabilities and get max confidence scores and predicted labels for teacher
        teacher_confidance_scores, teacher_confidance_scores_args = torch.max(
            torch.softmax(teacher_pseudo_labels_prob_tensor, dim=1), dim=1)
        
        # Move everything to GPU if necessary
        teacher_confidance_scores = teacher_confidance_scores.cuda()
        teacher_confidance_scores_args = teacher_confidance_scores_args.cuda()
        
        true_labels = torch.tensor(original_labels[corruption_type]).cuda()
        
        # Prepare data structure for storing counts per threshold for each category
        total_predictions = len(true_labels)
        correct_predictions = (teacher_confidance_scores_args == true_labels).sum().item()
        incorrect_predictions = total_predictions - correct_predictions
        
        # Define arrays to store the counts for each threshold range
        correct_threshold_counts = np.zeros(len(thresholds) + 1)
        incorrect_threshold_counts = np.zeros(len(thresholds) + 1)
        total_threshold_counts = np.zeros(len(thresholds) + 1)
        
        # Function to find which threshold bin the score falls into
        def get_threshold_bin(score, thresholds):
            for i, threshold in enumerate(thresholds):
                if score < threshold:
                    return i
            return len(thresholds)
        
        # Count how many predictions fall into each threshold bin
        for i in range(total_predictions):
            bin_idx = get_threshold_bin(teacher_confidance_scores[i].item(), thresholds)
            total_threshold_counts[bin_idx] += 1
            if teacher_confidance_scores_args[i] == true_labels[i]:
                correct_threshold_counts[bin_idx] += 1
            else:
                incorrect_threshold_counts[bin_idx] += 1
        
        # Prepare the data for stacked bar plotting
        total_stack = total_threshold_counts / total_predictions * 100
        correct_stack = correct_threshold_counts / total_predictions * 100
        incorrect_stack = incorrect_threshold_counts / total_predictions * 100

        all_categories.append(corruption_type)
        
        # Plot bars for Total, Correct, Incorrect for the current corruption type
        base_x = x_positions[cat_idx]
        for col_idx, category_data in enumerate([total_stack, correct_stack, incorrect_stack]):
            bottom = 0
            x = base_x + (col_idx * bar_width)
            
            for threshold_idx in range(len(threshold_labels)):
                height = category_data[threshold_idx]
                ax1.bar(x, height, bar_width, bottom=bottom,
                       color=colors[threshold_idx], label=threshold_labels[threshold_idx] if cat_idx == 0 and col_idx == 0 else "")
                bottom += height
        
        # Calculate accuracy for the second plot
        accuracy = (correct_predictions / total_predictions) * 100
        accuracies.append(accuracy)
    
    # Customize the first plot
    ax1.set_xlabel('Categories', fontsize=14)
    ax1.set_ylabel('Percentage of Labels (%)', fontsize=14)
    ax1.set_title(f'{title_prefix} - Teacher Predictions by Thresholds', fontsize=16, fontweight='bold')
    
    # Set x-axis ticks and labels for the first plot
    category_centers = x_positions + (category_width - category_spacing) / 2
    ax1.set_xticks(category_centers)
    ax1.set_xticklabels(all_categories, rotation=45, ha='right')
    
    # Add grid and legend for the first plot
    ax1.grid(True, linestyle='--', axis='y', alpha=0.7)
    ax1.legend(fontsize=12, title="Thresholds")
    
    # Set y-axis limit for the first plot
    ax1.set_ylim(0, 100)
    
    # Part 2: Second Plot for Teacher Accuracy per Category
    ax2.bar(np.arange(len(all_categories)), accuracies, bar_width, color='blue')
    
    # Customize the second plot
    ax2.set_xlabel('Categories', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.set_title('Teacher Accuracy by Category', fontsize=16, fontweight='bold')
    
    # Set x-axis ticks and labels for the second plot
    ax2.set_xticks(np.arange(len(all_categories)))
    ax2.set_xticklabels(all_categories, rotation=45, ha='right')
    
    # Add grid to the second plot
    ax2.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if base_image_save_path:
        save_path = f"{base_image_save_path}/{title_prefix.replace(' ', '_')}_combined_plot.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved at: {save_path}")
    
    plt.show()











def plot_teacher_predictions_multiple_thresholds11(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path='', title_prefix='Teacher Predictions by Thresholds'):
    fig, ax = plt.subplots(figsize=(20, 8))  # Increase the width for 45 columns (3 per category)
    
    # Define colors and threshold labels for stacked bars
    threshold_labels = [f"< {threshold}" for threshold in thresholds] + [f">= {thresholds[-1]}"]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0'][:len(threshold_labels)]
    categories = ['Total', 'Correct', 'Incorrect']
    
    # Bar chart parameters
    n_categories = len(teacher_pseudo_labels_prob.keys())  # Number of main categories (corruption types)
    n_cols_per_category = 3  # Number of columns per category (Total, Correct, Incorrect)
    bar_width = 0.25  # Width of each bar
    category_spacing = 0.4  # Extra space between categories
    
    # Calculate positions for bars
    category_width = (n_cols_per_category * bar_width) + category_spacing
    x_positions = np.arange(n_categories) * category_width

    all_categories = []
    
    # Start plotting each corruption type
    for cat_idx, corruption_type in enumerate(teacher_pseudo_labels_prob.keys()):
        print(f"Processing corruption type: {corruption_type}")
        
        if corruption_type not in original_labels:
            print(f"Error: Corruption type '{corruption_type}' not found in original_labels")
            continue
        
        # Concatenate tensors for current corruption type (for teacher)
        teacher_pseudo_labels_prob_tensor = torch.cat(teacher_pseudo_labels_prob[corruption_type], dim=0)
        
        # Compute softmax probabilities and get max confidence scores and predicted labels for teacher
        teacher_confidance_scores, teacher_confidance_scores_args = torch.max(
            torch.softmax(teacher_pseudo_labels_prob_tensor, dim=1), dim=1)
        
        # Move everything to GPU if necessary
        teacher_confidance_scores = teacher_confidance_scores.cuda()
        teacher_confidance_scores_args = teacher_confidance_scores_args.cuda()
        
        true_labels = torch.tensor(original_labels[corruption_type]).cuda()
        
        # Prepare data structure for storing counts per threshold for each category
        total_predictions = len(true_labels)
        correct_predictions = (teacher_confidance_scores_args == true_labels).sum().item()
        incorrect_predictions = total_predictions - correct_predictions
        
        # Define arrays to store the counts for each threshold range
        correct_threshold_counts = np.zeros(len(thresholds) + 1)
        incorrect_threshold_counts = np.zeros(len(thresholds) + 1)
        total_threshold_counts = np.zeros(len(thresholds) + 1)
        
        # Function to find which threshold bin the score falls into
        def get_threshold_bin(score, thresholds):
            for i, threshold in enumerate(thresholds):
                if score < threshold:
                    return i
            return len(thresholds)
        
        # Count how many predictions fall into each threshold bin
        for i in range(total_predictions):
            bin_idx = get_threshold_bin(teacher_confidance_scores[i].item(), thresholds)
            total_threshold_counts[bin_idx] += 1
            if teacher_confidance_scores_args[i] == true_labels[i]:
                correct_threshold_counts[bin_idx] += 1
            else:
                incorrect_threshold_counts[bin_idx] += 1
        
        # Prepare the data for stacked bar plotting
        total_stack = total_threshold_counts / total_predictions * 100
        correct_stack = correct_threshold_counts / total_predictions * 100
        incorrect_stack = incorrect_threshold_counts / total_predictions * 100

        all_categories.append(corruption_type)
        
        # Plot bars for Total, Correct, Incorrect for the current corruption type
        base_x = x_positions[cat_idx]
        for col_idx, category_data in enumerate([total_stack, correct_stack, incorrect_stack]):
            bottom = 0
            x = base_x + (col_idx * bar_width)
            
            for threshold_idx in range(len(threshold_labels)):
                height = category_data[threshold_idx]
                ax.bar(x, height, bar_width, bottom=bottom,
                       color=colors[threshold_idx], label=threshold_labels[threshold_idx] if cat_idx == 0 and col_idx == 0 else "")
                bottom += height
    
    # Customize the plot
    ax.set_xlabel('Categories', fontsize=14)
    ax.set_ylabel('Percentage of Labels (%)', fontsize=14)
    ax.set_title(f'{title_prefix}', fontsize=16, fontweight='bold')
    
    # Set x-axis ticks and labels
    category_centers = x_positions + (category_width - category_spacing) / 2
    ax.set_xticks(category_centers)
    ax.set_xticklabels(all_categories, rotation=45, ha='right')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', axis='y', alpha=0.7)
    ax.legend(fontsize=12, title="Thresholds")
    
    # Set y-axis limit
    ax.set_ylim(0, 100)
    
    # Add subtle vertical lines between categories
    for x in x_positions[:-1]:
        ax.axvline(x + category_width - category_spacing / 2, color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if base_image_save_path:
        save_path = f"{base_image_save_path}/{title_prefix.replace(' ', '_')}_all_categories.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved at: {save_path}")
        print('mehran')
    plt.show()






import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_teacher_predictions_multiple_thresholds2(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path='', title_prefix='Teacher Predictions by Thresholds'):
    # Initialize arrays to accumulate overall counts across all corruption types
    overall_total_threshold_counts = np.zeros(len(thresholds) + 1)
    overall_correct_threshold_counts = np.zeros(len(thresholds) + 1)
    overall_incorrect_threshold_counts = np.zeros(len(thresholds) + 1)
    
    # Use the keys of the pseudo-labels dictionaries to find corruption types
    for corruption_type in teacher_pseudo_labels_prob.keys():
        print(f"Processing corruption type: {corruption_type}")

        if corruption_type not in original_labels:
            print(f"Error: Corruption type '{corruption_type}' not found in original_labels")
            continue

        # Step 1: Concatenate the tensors for the current corruption type (for teacher)
        teacher_pseudo_labels_prob_tensor = torch.cat(teacher_pseudo_labels_prob[corruption_type], dim=0)

        # Step 2: Compute the softmax probabilities and get max confidence scores and predicted labels for teacher
        teacher_confidance_scores, teacher_confidance_scores_args = torch.max(
            torch.softmax(teacher_pseudo_labels_prob_tensor, dim=1), dim=1)

        # Step 3: Move everything to GPU if necessary
        teacher_confidance_scores = teacher_confidance_scores.cuda()
        teacher_confidance_scores_args = teacher_confidance_scores_args.cuda()

        true_labels = torch.tensor(original_labels[corruption_type]).cuda()

        # Step 4: Prepare data structure for storing counts per threshold for each category
        total_predictions = len(true_labels)
        correct_predictions = (teacher_confidance_scores_args == true_labels).sum().item()
        incorrect_predictions = total_predictions - correct_predictions

        # Define arrays to store the counts for each threshold range
        correct_threshold_counts = np.zeros(len(thresholds) + 1)
        incorrect_threshold_counts = np.zeros(len(thresholds) + 1)
        total_threshold_counts = np.zeros(len(thresholds) + 1)

        # Function to find which threshold bin the score falls into
        def get_threshold_bin(score, thresholds):
            for i, threshold in enumerate(thresholds):
                if score < threshold:
                    return i
            return len(thresholds)

        # Step 5: Count how many predictions fall into each threshold bin
        for i in range(total_predictions):
            bin_idx = get_threshold_bin(teacher_confidance_scores[i].item(), thresholds)
            total_threshold_counts[bin_idx] += 1
            if teacher_confidance_scores_args[i] == true_labels[i]:
                correct_threshold_counts[bin_idx] += 1
            else:
                incorrect_threshold_counts[bin_idx] += 1

        # Step 6: Accumulate counts into the overall counts
        overall_total_threshold_counts += total_threshold_counts
        overall_correct_threshold_counts += correct_threshold_counts
        overall_incorrect_threshold_counts += incorrect_threshold_counts

        # Plot individual corruption type as before (optional)
        # ...

    # Step 7: Prepare overall histogram data
    overall_total_stack = overall_total_threshold_counts / overall_total_threshold_counts.sum() * 100
    overall_correct_stack = overall_correct_threshold_counts / overall_total_threshold_counts.sum() * 100
    overall_incorrect_stack = overall_incorrect_threshold_counts / overall_total_threshold_counts.sum() * 100

    # Step 8: Plot summary histogram for all corruption types
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.4
    bar_positions = np.arange(3)
    
    threshold_labels = [f"< {threshold}" for threshold in thresholds] + [f">= {thresholds[-1]}"]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336'][:len(threshold_labels)]

    overall_stacks = [overall_total_stack, overall_correct_stack, overall_incorrect_stack]
    
    bottom = np.zeros(len(bar_positions))
    for i in range(len(threshold_labels)):
        ax.bar(bar_positions, [stack[i] for stack in overall_stacks], bar_width, bottom=bottom, color=colors[i], label=threshold_labels[i])
        bottom += [stack[i] for stack in overall_stacks]

    # Step 9: Add labels, title, and axes
    ax.set_xlabel('Prediction Category', fontsize=14)
    ax.set_ylabel('Percentage of Labels (%)', fontsize=14)
    ax.set_title(f'{title_prefix} - Overall Summary', fontsize=16, fontweight='bold')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(['Total Predictions', 'Correct Predictions', 'Incorrect Predictions'], fontsize=12, rotation=45, ha='right')
    ax.legend(fontsize=12, title="Thresholds")

    # Add grid for better readability
    ax.grid(True, linestyle='--', axis='y', alpha=0.7)

    # Step 10: Save the overall plot
    overall_save_path = f"{base_image_save_path}/{title_prefix.replace(' ', '_')}_overall_summary.png"
    plt.savefig(overall_save_path, bbox_inches='tight', dpi=300)
    print(f"Overall summary plot saved at: {overall_save_path}")

    # Show the plot
    plt.tight_layout()
    plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_teacher_predictions_multiple_thresholds3(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path='', title_prefix='Teacher Predictions by Thresholds'):
    # Initialize lists to store data for all corruption types
    all_categories = []
    total_stack_all = []
    correct_stack_all = []
    incorrect_stack_all = []

    # Use the keys of the pseudo-labels dictionaries to find corruption types
    for corruption_type in teacher_pseudo_labels_prob.keys():
        print(f"Processing corruption type: {corruption_type}")

        if corruption_type not in original_labels:
            print(f"Error: Corruption type '{corruption_type}' not found in original_labels")
            continue

        # Step 1: Concatenate the tensors for the current corruption type (for teacher)
        teacher_pseudo_labels_prob_tensor = torch.cat(teacher_pseudo_labels_prob[corruption_type], dim=0)

        # Step 2: Compute the softmax probabilities and get max confidence scores and predicted labels for teacher
        teacher_confidance_scores, teacher_confidance_scores_args = torch.max(
            torch.softmax(teacher_pseudo_labels_prob_tensor, dim=1), dim=1)

        # Step 3: Move everything to GPU if necessary
        teacher_confidance_scores = teacher_confidance_scores.cuda()
        teacher_confidance_scores_args = teacher_confidance_scores_args.cuda()

        true_labels = torch.tensor(original_labels[corruption_type]).cuda()

        # Step 4: Prepare data structure for storing counts per threshold for each category
        total_predictions = len(true_labels)
        correct_predictions = (teacher_confidance_scores_args == true_labels).sum().item()
        incorrect_predictions = total_predictions - correct_predictions

        # Define arrays to store the counts for each threshold range
        correct_threshold_counts = np.zeros(len(thresholds) + 1)
        incorrect_threshold_counts = np.zeros(len(thresholds) + 1)
        total_threshold_counts = np.zeros(len(thresholds) + 1)

        # Function to find which threshold bin the score falls into
        def get_threshold_bin(score, thresholds):
            for i, threshold in enumerate(thresholds):
                if score < threshold:
                    return i
            return len(thresholds)

        # Step 5: Count how many predictions fall into each threshold bin
        for i in range(total_predictions):
            bin_idx = get_threshold_bin(teacher_confidance_scores[i].item(), thresholds)
            total_threshold_counts[bin_idx] += 1
            if teacher_confidance_scores_args[i] == true_labels[i]:
                correct_threshold_counts[bin_idx] += 1
            else:
                incorrect_threshold_counts[bin_idx] += 1

        # Step 6: Prepare the data for plotting
        total_stack = total_threshold_counts.sum()  # Summing across all thresholds
        correct_stack = correct_threshold_counts.sum()
        incorrect_stack = incorrect_threshold_counts.sum()

        # Store the values for this corruption type
        all_categories.append(corruption_type)
        total_stack_all.append(total_stack)
        correct_stack_all.append(correct_stack)
        incorrect_stack_all.append(incorrect_stack)

    # Step 7: Flatten the stacks so we can plot them as bins in the histogram
    total_stack_all_flat = np.array(total_stack_all).flatten()
    correct_stack_all_flat = np.array(correct_stack_all).flatten()
    incorrect_stack_all_flat = np.array(incorrect_stack_all).flatten()

    # Step 8: Prepare the bins and the data for the histogram
    bins = []
    heights = []
    for i, corruption_type in enumerate(all_categories):
        bins.append(f'{corruption_type} - Total')
        bins.append(f'{corruption_type} - Correct')
        bins.append(f'{corruption_type} - Incorrect')

        heights.append(total_stack_all_flat[i])
        heights.append(correct_stack_all_flat[i])
        heights.append(incorrect_stack_all_flat[i])

    # Step 9: Plot the histogram with 45 bins (3 per corruption type)
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_positions = np.arange(len(bins))  # X positions for each bar
    bar_width = 0.8

    # Plot bars
    ax.bar(bar_positions, heights, bar_width)

    # Step 10: Add labels, title, and axes
    ax.set_xlabel('Corruption Types and Prediction Categories', fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.set_title(f'{title_prefix} - 45 Bins Histogram', fontsize=16, fontweight='bold')

    # Set X-axis tick labels to show the category + prediction types
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bins, fontsize=10, rotation=90, ha='right')

    # Add grid for better readability
    ax.grid(True, linestyle='--', axis='y', alpha=0.7)

    # Step 11: Save the plot
    overall_save_path = f"{base_image_save_path}/{title_prefix.replace(' ', '_')}_45_bins_summary.png"
    plt.savefig(overall_save_path, bbox_inches='tight', dpi=300)
    print(f"Summary plot saved at: {overall_save_path}")

    # Show the plot
    plt.tight_layout()
    plt.show()


# Example usage:
# Assuming you have teacher pseudo-labels probabilities and original labels
# plot_all_categories_grouped_in_one_image(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path='/your/path', title_prefix='Teacher Predictions for All Categories')


def plot_teacher_predictions_multiple_thresholds(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path='', title_prefix='Teacher Predictions by Thresholds'):
    # Use the keys of the pseudo-labels dictionaries to find corruption types
    for corruption_type in teacher_pseudo_labels_prob.keys():
        print(f"Processing corruption type: {corruption_type}")

        if corruption_type not in original_labels:
            print(f"Error: Corruption type '{corruption_type}' not found in original_labels")
            continue

        # Step 1: Concatenate the tensors for the current corruption type (for teacher)
        teacher_pseudo_labels_prob_tensor = torch.cat(teacher_pseudo_labels_prob[corruption_type], dim=0)

        # Step 2: Compute the softmax probabilities and get max confidence scores and predicted labels for teacher
        teacher_confidance_scores, teacher_confidance_scores_args = torch.max(
            torch.softmax(teacher_pseudo_labels_prob_tensor, dim=1), dim=1)

        # Step 3: Move everything to GPU if necessary
        teacher_confidance_scores = teacher_confidance_scores.cuda()
        teacher_confidance_scores_args = teacher_confidance_scores_args.cuda()

        true_labels = torch.tensor(original_labels[corruption_type]).cuda()

        # Step 4: Prepare data structure for storing counts per threshold for each category
        total_predictions = len(true_labels)
        correct_predictions = (teacher_confidance_scores_args == true_labels).sum().item()
        incorrect_predictions = total_predictions - correct_predictions

        # Define arrays to store the counts for each threshold range
        correct_threshold_counts = np.zeros(len(thresholds) + 1)
        incorrect_threshold_counts = np.zeros(len(thresholds) + 1)
        total_threshold_counts = np.zeros(len(thresholds) + 1)

        # Function to find which threshold bin the score falls into
        def get_threshold_bin(score, thresholds):
            for i, threshold in enumerate(thresholds):
                if score < threshold:
                    return i
            return len(thresholds)

        # Step 5: Count how many predictions fall into each threshold bin
        for i in range(total_predictions):
            bin_idx = get_threshold_bin(teacher_confidance_scores[i].item(), thresholds)
            total_threshold_counts[bin_idx] += 1
            if teacher_confidance_scores_args[i] == true_labels[i]:
                correct_threshold_counts[bin_idx] += 1
            else:
                incorrect_threshold_counts[bin_idx] += 1

        # Step 6: Prepare the data for stacked bar plotting
        categories = ['Total Predictions', 'Correct Predictions', 'Incorrect Predictions']
        
        # Correct, Incorrect, and Total percentages by thresholds
        total_stack = total_threshold_counts / total_predictions * 100
        correct_stack = correct_threshold_counts / total_predictions * 100
        incorrect_stack = incorrect_threshold_counts / total_predictions * 100

        # Combine the correct, incorrect, and total stacks into a single array for plotting
        stacks = [total_stack, correct_stack, incorrect_stack]

        # Step 7: Plot stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define bar positions and width
        bar_width = 0.4
        bar_positions = np.arange(len(categories))

        # Define threshold labels for the legend
        threshold_labels = [f"< {threshold}" for threshold in thresholds] + [f">= {thresholds[-1]}"]
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0'][:len(threshold_labels)]

        # Plot stacked bars for each category (Total, Correct, Incorrect)
        bottom = np.zeros(len(bar_positions))
        for i in range(len(threshold_labels)):
            # Create bars for each threshold segment
            ax.bar(bar_positions, [stack[i] for stack in stacks], bar_width, bottom=bottom, color=colors[i], label=threshold_labels[i])
            bottom += [stack[i] for stack in stacks]  # Update the bottom to stack the bars

        # Step 8: Add labels, title, and axes
        ax.set_xlabel('Prediction Category', fontsize=14)
        ax.set_ylabel('Percentage of Labels (%)', fontsize=14)
        ax.set_title(f'{title_prefix} - {corruption_type}', fontsize=16, fontweight='bold')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(categories, fontsize=12, rotation=45, ha='right')

        # Add grid for better readability
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)

        # Step 9: Add legend for the thresholds
        ax.legend(fontsize=12, title="Thresholds")

        # Step 10: Save the plot
        save_path = f"{base_image_save_path}/{title_prefix.replace(' ', '_')}_{corruption_type}_threshold_analysis.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot for corruption '{corruption_type}' saved at: {save_path}")

        # Show the plot
        plt.tight_layout()
        plt.show()


def compute_and_plot_metrics_for_corruptions(teacher_pseudo_labels_prob, weakaug_student_pseudo_labels_prob, strongaug_student_pseudo_labels_prob, original_labels, confidence_threshold=0.7, base_image_save_path='', title_prefix='Label Confidence Distribution'):
    # Use the keys of the pseudo-labels dictionaries to find corruption types
    for corruption_type in strongaug_student_pseudo_labels_prob.keys():
        print(f"Processing corruption type: {corruption_type}")

        if corruption_type not in original_labels:
            print(f"Error: Corruption type '{corruption_type}' not found in original_labels")
            continue

        # Step 1: Concatenate the tensors for the current corruption type for all three sets (teacher, weak, strong)
        teacher_pseudo_labels_prob_tensor = torch.cat(teacher_pseudo_labels_prob[corruption_type], dim=0)
        weakaug_student_pseudo_labels_prob_tensor = torch.cat(weakaug_student_pseudo_labels_prob[corruption_type], dim=0)
        strongaug_student_pseudo_labels_prob_tensor = torch.cat(strongaug_student_pseudo_labels_prob[corruption_type], dim=0)

        # Step 2: Compute the softmax probabilities and get max confidence scores and predicted labels for all three sets
        teacher_confidance_scores, teacher_confidance_scores_args = torch.max(
            torch.softmax(teacher_pseudo_labels_prob_tensor, dim=1), dim=1)
        
        weakaug_student_confidance_scores, weakaug_student_confidance_scores_args = torch.max(
            torch.softmax(weakaug_student_pseudo_labels_prob_tensor, dim=1), dim=1)
        
        strongaug_student_confidance_scores, strongaug_student_confidance_scores_args = torch.max(
            torch.softmax(strongaug_student_pseudo_labels_prob_tensor, dim=1), dim=1)

        # Step 3: Move everything to GPU if necessary
        teacher_confidance_scores, teacher_confidance_scores_args = teacher_confidance_scores.cuda(), teacher_confidance_scores_args.cuda()
        weakaug_student_confidance_scores, weakaug_student_confidance_scores_args = weakaug_student_confidance_scores.cuda(), weakaug_student_confidance_scores_args.cuda()
        strongaug_student_confidance_scores, strongaug_student_confidance_scores_args = strongaug_student_confidance_scores.cuda(), strongaug_student_confidance_scores_args.cuda()
        
        true_labels = torch.tensor(original_labels[corruption_type]).cuda()

        # Step 4: Calculate metrics for teacher, weakly augmented, and strongly augmented students
        def calculate_metrics(predicted_labels, confidence_scores, true_labels, threshold):
            correct_predictions = (predicted_labels == true_labels).sum().item()
            above_threshold = (confidence_scores > threshold).sum().item()
            correct_and_above_threshold = ((predicted_labels == true_labels) & (confidence_scores > threshold)).sum().item()
            incorrect_and_above_threshold = ((predicted_labels != true_labels) & (confidence_scores > threshold)).sum().item()
            correct_and_below_threshold = ((predicted_labels == true_labels) & (confidence_scores <= threshold)).sum().item()
            return correct_predictions, above_threshold, correct_and_above_threshold, incorrect_and_above_threshold, correct_and_below_threshold

        # Teacher Metrics
        teacher_metrics = calculate_metrics(teacher_confidance_scores_args, teacher_confidance_scores, true_labels, confidence_threshold)

        # Weak Augmented Student Metrics
        weakaug_metrics = calculate_metrics(weakaug_student_confidance_scores_args, weakaug_student_confidance_scores, true_labels, confidence_threshold)

        # Strong Augmented Student Metrics
        strongaug_metrics = calculate_metrics(strongaug_student_confidance_scores_args, strongaug_student_confidance_scores, true_labels, confidence_threshold)

        # Step 5: Print metrics for the current corruption type
        print(f"Corruption Type: {corruption_type}")
        print(f"Teacher: {teacher_metrics}")
        print(f"Weak Augmented Student: {weakaug_metrics}")
        print(f"Strong Augmented Student: {strongaug_metrics}")

        # Step 6: Plot grouped histogram for comparison between Teacher, Weak Augmented, and Strong Augmented students
        categories = ['Correct (Total)', 'Above Threshold (Total)', 'Correct (Above Threshold)', 
                      'Incorrect (Above Threshold)', 'Correct (Below Threshold)']

        fig, ax = plt.subplots(figsize=(14, 8))

        # Define the bar positions and width
        bar_width = 0.2
        bar_positions = np.arange(len(categories))

        # Plot Teacher, Weak Augmented, and Strong Augmented bars
        teacher_bars = ax.bar(bar_positions - bar_width, teacher_metrics, bar_width, label='Teacher', color='#4CAF50')
        weakaug_bars = ax.bar(bar_positions, weakaug_metrics, bar_width, label='Weak Augmented Student', color='#2196F3')
        strongaug_bars = ax.bar(bar_positions + bar_width, strongaug_metrics, bar_width, label='Strong Augmented Student', color='#FF9800')

        # Step 7: Add labels, title, and axes
        ax.set_xlabel('Label Condition', fontsize=14)
        ax.set_ylabel('Count of Labels', fontsize=14)
        ax.set_title(f'{title_prefix} - {corruption_type}', fontsize=16, fontweight='bold')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(categories, fontsize=12, rotation=45, ha='right')

        # Step 8: Add grid for better readability
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)

        # Step 9: Add value labels on top of each bar
        def add_labels(bars):
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 5, int(yval),
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

        add_labels(teacher_bars)
        add_labels(weakaug_bars)
        add_labels(strongaug_bars)

        # Step 10: Add legend
        ax.legend(fontsize=12)

        # Save the plot
        save_path = f"{base_image_save_path}/{title_prefix.replace(' ', '_')}_{corruption_type}_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot for corruption '{corruption_type}' saved at: {save_path}")

        # Show the plot
        plt.tight_layout()
        plt.show()
# Example usage (modify as needed):
# strongaug_student_pseudo_labels_prob = {...}  # Dictionary of corruption types with corresponding predictions
# original_labels = {...}  # Dictionary of corruption types with corresponding true labels
# base_image_save_path = '/content/SaveImage'  # Path where you want to save the images
# compute_and_plot_metrics_for_corruptions(strongaug_student_pseudo_labels_prob, original_labels, confidence_threshold=0.7, base_image_save_path=base_image_save_path)


# Example usage (adjust with actual arguments)
# compute_and_plot_metrics_for_corruptions(strongaug_student_pseudo_labels_prob, original_labels, args, confidence_threshold=0.7, base_image_save_path=base_image_save_path)

























# Function to plot and save results in the specified folder
def plot_grouped_bar_chart(teacher_dict, student_dict, equal_dict, base_folder, title):
    """Plot and save a grouped bar chart comparing teacher, student, and equal labels."""
    categories = list(teacher_dict.keys())
    teacher_values = list(teacher_dict.values())
    student_values = list(student_dict.values())
    equal_values = list(equal_dict.values())

    x = np.arange(len(categories))  # the label locations
    width = 0.25  # width of the bars

    plt.figure(figsize=(12, 8))
    
    # Plotting the bars
    plt.bar(x - width, teacher_values, width, label='Correct Teacher Pseudo Labels', color='b')
    plt.bar(x, student_values, width, label='Correct Student Pseudo Labels', color='g')
    plt.bar(x + width, equal_values, width, label='Teacher Student Equal Labels', color='r')

    # Labeling
    plt.xlabel('Corruption Type')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    # Save the plot in the base folder
    file_path = os.path.join(base_folder, f"{title.replace(' ', '_')}.png")
    plt.savefig(file_path)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os






def plot_teacher_confidence_stacked_bar(teacher_pseudo_labels_prob, base_image_save_path, title="Teacher Prediction Confidence Distribution"):
    """
    This function plots a stacked bar chart showing the accumulation of predictions in different probability ranges.

    teacher_pseudo_labels_prob: Dictionary where keys are corruption types and values are lists of teacher probabilities.
    base_image_save_path: Path to save the generated plot.
    title: Title of the plot.
    """
    # Define the probability ranges (bins)
    bins = np.linspace(0, 1, 11)  # 0.0–0.1, 0.1–0.2, ..., 0.9–1.0
    
    # Initialize a list to accumulate the count of predictions for each probability range
    range_counts = np.zeros(len(bins)-1)  # We have 10 ranges (0.0–0.1, ..., 0.9–1.0)
    
    # Collect all probabilities from the teacher's predictions
    for corruption, probabilities in teacher_pseudo_labels_prob.items():
        flat_probs = np.concatenate([prob.cpu().detach().numpy() for prob in probabilities], axis=0)
        all_probs = flat_probs.flatten()  # Flatten the probabilities array
        
        # Count how many predictions fall into each range
        hist, _ = np.histogram(all_probs, bins=bins)
        
        # Accumulate the counts for each range
        range_counts += hist

    # Plot a single stacked bar representing the accumulated probability counts
    plt.figure(figsize=(10, 6))
    
    # Stacked bar: each bin will be a segment in the stacked bar
    labels = [f'{bins[i]:.1f}–{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    plt.bar(0, range_counts, width=0.5, align='center', color=plt.cm.viridis(np.linspace(0, 1, len(range_counts))),
            label=labels)

    # Customize the plot
    plt.title(title)
    plt.ylabel('Number of Predictions')
    plt.xticks([])  # No x-axis labels needed for a single stacked bar
    plt.legend(labels, title="Probability Range", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Save the plot
    save_path = f"{base_image_save_path}/{title.replace(' ', '_')}_stacked.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Stacked bar chart saved to {save_path}")
    return save_path


















def plot_label_situation_function_corruption(teacher_pseudo_labels_dict,
                                             weakaug_student_pseudo_labels_dict,
                                             strongaug_student_pseudo_labels_dict,
                                             original_labels_dict,
                                             teacher_pseudo_labels_prob_dict,
                                             weakaug_student_pseudo_labels_prob_dict,
                                             strongaug_student_pseudo_labels_prob_dict,
                                             base_image_save_path, 
                                             plot_title):
    """
    Plots the number of instances where:
    1. Weakly augmented teacher pseudo-labels match original labels.
    2. Weakly augmented student pseudo-labels match original labels.
    3. Strongly augmented student pseudo-labels match original labels.
    4. Teacher pseudo-labels match weakly augmented student pseudo-labels.
    5. Teacher pseudo-labels match strongly augmented student pseudo-labels.
    6. Strongly augmented student pseudo-labels match weakly augmented student pseudo-labels.
    For each corruption category, and one final plot aggregating all corruptions.

    Additionally, adds a plot for label value distribution for teacher, weak student, strong student, and original (correct) labels,
    and a separate table displaying counts.
    """
    
    # Ensure the save path exists
    if not os.path.exists(base_image_save_path):
        os.makedirs(base_image_save_path)

    # Initialize lists to accumulate results for the final combined plot
    total_teacher = []
    total_weak_student = []
    total_strong_student = []
    total_teacher_weak = []
    total_teacher_strong = []
    total_strong_weak = []
    
    # Helper function to handle lists of tensors
    def tensor_list_to_numpy(tensor_list):
        return np.array([t.cpu().numpy() for t in tensor_list])
     
    # Iterate over each corruption type in the dictionary
    for idx, corruption_type in enumerate(teacher_pseudo_labels_dict.keys()):
        # Convert lists of tensors to numpy arrays
        teacher_pseudo_labels = tensor_list_to_numpy(teacher_pseudo_labels_dict[corruption_type])
        weakaug_student_pseudo_labels = tensor_list_to_numpy(weakaug_student_pseudo_labels_dict[corruption_type])
        strongaug_student_pseudo_labels = tensor_list_to_numpy(strongaug_student_pseudo_labels_dict[corruption_type])
        original_labels = tensor_list_to_numpy(original_labels_dict[corruption_type])

        # Condition 1: Teacher pseudo-labels == original labels
        condition_teacher = np.sum(teacher_pseudo_labels == original_labels)

        # Condition 2: Weak augmentation student pseudo-labels == original labels
        condition_weak_student = np.sum(weakaug_student_pseudo_labels == original_labels)

        # Condition 3: Strong augmentation student pseudo-labels == original labels
        condition_strong_student = np.sum(strongaug_student_pseudo_labels == original_labels)

        # Condition 4: Teacher pseudo-labels == weakly augmented student pseudo-labels
        condition_teacher_weak = np.sum(teacher_pseudo_labels == weakaug_student_pseudo_labels)

        # Condition 5: Teacher pseudo-labels == strongly augmented student pseudo-labels
        condition_teacher_strong = np.sum(teacher_pseudo_labels == strongaug_student_pseudo_labels)

        # Condition 6: Strongly augmented student pseudo-labels == weakly augmented student pseudo-labels
        condition_strong_weak = np.sum(strongaug_student_pseudo_labels == weakaug_student_pseudo_labels)

        # Store results for final combined plot
        total_teacher.append(condition_teacher)
        total_weak_student.append(condition_weak_student)
        total_strong_student.append(condition_strong_student)
        total_teacher_weak.append(condition_teacher_weak)
        total_teacher_strong.append(condition_teacher_strong)
        total_strong_weak.append(condition_strong_weak)

        # Plotting for individual corruption type
        conditions = ['Teacher Pseudo-labels = Correct Labels', 
                      'Weak Aug Student Pseudo-labels = Correct Labels',
                      'Strong Aug Student Pseudo-labels = Correct Labels',
                      'Teacher Pseudo-labels = Weak Aug Student',
                      'Teacher Pseudo-labels = Strong Aug Student',
                      'Strong Aug Student = Weak Aug Student']

        values = [condition_teacher, condition_weak_student, condition_strong_student, 
                  condition_teacher_weak, condition_teacher_strong, condition_strong_weak]

        plt.figure(figsize=(10, 8))
        bars = plt.bar(conditions, values, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan'])
        plt.title(f"{plot_title} - {corruption_type}")
        plt.ylabel('Number of Matching Labels')
        plt.xticks(rotation=45, ha='right')

        # Add notes (the count values) above each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

        plt.tight_layout()

        # Save each corruption-specific plot
        plt.savefig(f"{base_image_save_path}/{idx}_{plot_title}_{corruption_type}.png")
        plt.close()

        # NEW PLOT: Pseudo-label distribution (stacked histogram) for each corruption type
        label_range = np.arange(0, np.max(original_labels) + 1)  # Assuming labels are integers starting from 0

        plt.figure(figsize=(12, 6))

        # Flatten the pseudo-label arrays to make them 1D
        teacher_pseudo_labels_flat = teacher_pseudo_labels.flatten()
        weakaug_student_pseudo_labels_flat = weakaug_student_pseudo_labels.flatten()
        strongaug_student_pseudo_labels_flat = strongaug_student_pseudo_labels.flatten()
        original_labels_flat = original_labels.flatten()

        # Stacked Histogram for Teacher, Weak, Strong student, and Original correct labels
        hist_data = [teacher_pseudo_labels_flat, weakaug_student_pseudo_labels_flat, strongaug_student_pseudo_labels_flat, original_labels_flat]
        colors = ['blue', 'green', 'red', 'black']
        labels = ['Teacher Pseudo-labels', 'Weak Augmented Student Pseudo-labels', 'Strong Augmented Student Pseudo-labels', 'Original Correct Labels']

        # Stacked histogram
        counts, bins, bars = plt.hist(hist_data, bins=label_range, stacked=True, color=colors, label=labels, density=True)

        plt.title(f"Label Distribution - {corruption_type}")
        plt.xlabel("Label Class (Category)")
        plt.ylabel("Proportion of Labels")
        plt.legend(loc='upper right')
        plt.tight_layout()

        # Save the pseudo-label distribution plot
        plt.savefig(f"{base_image_save_path}/{idx}_{plot_title}_Label_Distribution_{corruption_type}.png")
        plt.close()

        # NEW: Create and save the table separately
        counts_teacher, _ = np.histogram(teacher_pseudo_labels_flat, bins=label_range)
        counts_weak_student, _ = np.histogram(weakaug_student_pseudo_labels_flat, bins=label_range)
        counts_strong_student, _ = np.histogram(strongaug_student_pseudo_labels_flat, bins=label_range)
        counts_original, _ = np.histogram(original_labels_flat, bins=label_range)

        # Create DataFrame for the table
        table_data = {
            'Class Label': label_range[:-1],
            'Teacher Pseudo-labels': counts_teacher,
            'Weak Augmented Student Pseudo-labels': counts_weak_student,
            'Strong Augmented Student Pseudo-labels': counts_strong_student,
            'Original Correct Labels': counts_original
        }
        df_table = pd.DataFrame(table_data)

        # Create figure for the table only
        fig, ax_table = plt.subplots(figsize=(12, 8))  # Adjust size as needed
        ax_table.axis('tight')
        ax_table.axis('off')
        table = ax_table.table(cellText=df_table.values, colLabels=df_table.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(2, 2)  # Scale for better visibility

        # Save the table image separately
        plt.savefig(f"{base_image_save_path}/{idx}_{plot_title}_Label_Table_{corruption_type}.png", bbox_inches='tight')
        plt.close()

    # Final combined plot (aggregate all corruption types)
    plt.figure(figsize=(14, 8))
    x = np.arange(len(teacher_pseudo_labels_dict.keys()))  # Corruption types for x-axis
    width = 0.15

    bars_teacher = plt.bar(x - 2*width, total_teacher, width, label='Teacher Pseudo-labels Match Correct Labels', color='b')
    bars_weak_student = plt.bar(x - width, total_weak_student, width, label='Weak Augmented Student Pseudo-labels Match Correct Labels', color='g')
    bars_strong_student = plt.bar(x, total_strong_student, width, label='Strong Augmented Student Pseudo-labels Match Correct Labels', color='r')
    bars_teacher_weak = plt.bar(x + width, total_teacher_weak, width, label='Teacher Pseudo-labels Match Weak Augmented Student', color='purple')
    bars_teacher_strong = plt.bar(x + 2*width, total_teacher_strong, width, label='Teacher Pseudo-labels Match Strong Augmented Student', color='orange')
    bars_strong_weak = plt.bar(x + 3*width, total_strong_weak, width, label='Strong Augmented Student Matches Weak Augmented Student', color='cyan')

    # Labeling
    plt.xlabel('Corruption Type')
    plt.ylabel('Count of Matching Labels')
    plt.title(f"{plot_title} - Aggregated Over All Corruptions")
    plt.xticks(x, teacher_pseudo_labels_dict.keys(), rotation=45, ha='right')

    # Add notes (the count values) above each bar for the final combined plot
    for bars in [bars_teacher, bars_weak_student, bars_strong_student, bars_teacher_weak, bars_teacher_strong, bars_strong_weak]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    # Move the legend outside the plot area with more descriptive labels
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

    plt.tight_layout()

    # Save the final aggregated plot
    plt.savefig(f"{base_image_save_path}/{plot_title}_Final_Aggregated.png", bbox_inches='tight')
    plt.close()










    
def concat_same_input(points, num_repeats=8):
    """ Concatenate the same input points to form a batch where each point is repeated `num_repeats` times """
    # Repeat the points along a new batch dimension
    repeated_points = points.repeat(num_repeats, 1, 1)  # Shape: [num_repeats, 3, 1024]
    return repeated_points


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True



import torch.nn.functional as F

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
def generate_weighted_pseudo_labels1(stu_pred, aug_stu_pred, teacher_pred, teacher_weight=1.0):
    # Apply weighting to the teacher probabilities
    weighted_teacher_pred = teacher_weight * teacher_pred

    # Stack all predictions along a new dimension
    combined_probs = torch.stack([stu_pred, aug_stu_pred, weighted_teacher_pred], dim=0)

    # Find the maximum probability along the stacked dimension (0) and get the corresponding class
    max_probs, max_indices = torch.max(combined_probs, dim=0)

    # Use the class with the highest probability as the pseudo-labels
    pseudo_labels = torch.argmax(max_probs, dim=1)

    return pseudo_labels

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
    model = importlib.import_module(args.model)
    teacher_model = model.get_model(num_class, normal_channel=args.use_normals)
    student_model = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    criterion2 = model.get_loss()
    
    print(args.device)
    teacher_model = teacher_model.to(args.device)
    student_model = student_model.to(args.device)
    criterion = criterion.to(args.device)

    teacher_model.apply(inplace_relu)
    student_model.apply(inplace_relu)

    # Load weights into both teacher and student models
    model_path = "/content/CTTT/pretrained_model/modelnet_jt.pth" 
    checkpoint = torch.load(model_path, map_location=torch.device(args.device))
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    student_model.load_state_dict(checkpoint['model_state_dict'])

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

    final_accuracy = {}
    reliable_pseudo_labels = {}
    pseudo_label_equal_label_for_high_tereshold = {}
    correct_pseudo_labels = {}

    #plot dictionaries
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


            #plot dataset 
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
                        points = process_point_cloud(points1, neighbour=5).reshape(1,-1,3)
                        points = points.permute(0, 2, 1)  # Now the shape will be [1, 3, 1024]
                        points1 = points1.permute(0, 2, 1)  # Now the shape will be [1, 3, 1024]

                        labels = labels.to(args.device)

                        pred_teacher, _ = teacher_model(points)
                        
                        pseudo_labels = pred_teacher.argmax(dim=1).to(args.device)
                        pseudo_labels = pseudo_labels.long().cuda()
                        
                        pred_student_original, trans_feat_original = student_model(points)
                        pred_student = pred_student_original.argmax(dim=1).to(args.device)
                        stu_pseudo_labels = pred_student.long().cuda()

                        #pseudo_labels = torch.max(pred_teacher, pred_student_original).argmax(dim=1).cuda()                  

                        # augmentations (scaling, jitter, translation, rotation)
                        augmented_points = augment_data(points1)

                        # Predictions for original and augmented data
                        pred_student_augmented, trans_feat_augmented = student_model(augmented_points)
                        pred_student_au = pred_student_augmented.argmax(dim=1).to(args.device)
                        stu_pseudo_labels_aug = pred_student_au.long().cuda()


                        #pseudo_labels = generate_weighted_pseudo_labels(pred_student_original, pred_student_augmented, pred_teacher).long().cuda()

                        #classification_loss = criterion(pred_student_original, pseudo_labels, trans_feat_original)
                        classification_loss = criterion(pred_student_original, pseudo_labels,trans_feat_original)                        # Consistency loss (difference between original and augmented predictions)
                        #consistency_loss = F.mse_loss(pred_student_original, pred_student_augmented)
                        consistency_loss = criterion(pred_student_augmented, pseudo_labels ,trans_feat_augmented)
                        #consistency_loss2 = kl_divergence_loss(pred_student_original, pred_student_augmented)
                        consistency_loss2 = F.mse_loss(pred_student_original, pred_student_augmented)

                        entropy_loss_value = entropy_loss(pred_student_original)

                        # Combine classification, consistency, and entropy losses
                        total_loss = (lambda_classification*classification_loss 
                                      + lambda_consistency * consistency_loss
                                      + lambda_consistency2 * consistency_loss2
                                      + lambda_entropy * entropy_loss_value)


                        # Backward pass and optimization step
                        total_loss.backward()
                        optimizer.step()
                        student_model.zero_grad()
                        optimizer.zero_grad()

                        #dictionaries for plot function 
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
                '''
                points = data.to(args.device)
                points = points.permute(0, 2, 1)
                '''
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

            #test_pred = torch.cat(test_pred, dim=0)
            #test_label = torch.cat(test_label, dim=0)
            acc = (test_pred == test_label).sum().item() / float(test_label.size(0)) * 100.
            log_string(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n')
            final_accuracy[args.corruption] = acc


    # Print final accuracy of all corruption kinds
    log_string('------------------------------train finished -----------------------------')
    for _, args.corruption in enumerate(corruptions):
        log_string(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {final_accuracy[args.corruption]} ########\n\n')
        final_mean_accuracy = np.mean([accuracy for accuracy in final_accuracy.values()])
    log_string(f' mean accuracy {final_mean_accuracy}\n\n\n')

    log_string(f'correct teacher pseudo labels ={correct_teacher_pseudo_labels} \n')
    log_string(f'correct student pseudo labels ={correct_student_pseudo_labels} \n')
    log_string(f'teacher student equal labels ={teacher_student_equal_labels} \n')

    # Concatenate the list of probabilities into a single tensor

    base_image_save_path = '/content/CTTT/SaveImage'

    # Check if the directory exists, if not, create it
    if not os.path.exists(base_image_save_path):
        os.makedirs(base_image_save_path)

    # Now, you can proceed to save your plots in this directory
    print(f"Directory created at {base_image_save_path}")


    
    plot_grouped_bar_chart(correct_teacher_pseudo_labels,correct_student_pseudo_labels,teacher_student_equal_labels,base_image_save_path, "Comparison of Pseudo Labels")

    plot_label_situation_function_corruption(teacher_pseudo_labels,
                            weakaug_student_pseudo_labels,
                            strongaug_student_pseudo_labels,
                            original_labels,
                            teacher_pseudo_labels_prob,
                            weakaug_student_pseudo_labels_prob,
                            strongaug_student_pseudo_labels_prob,
                            base_image_save_path, "Label condition")

    plot_teacher_confidence_stacked_bar(teacher_pseudo_labels_prob, base_image_save_path, "Teacher Prediction Confidence Distribution")
    compute_and_plot_metrics_for_corruptions(teacher_pseudo_labels_prob, weakaug_student_pseudo_labels_prob, 
                                              strongaug_student_pseudo_labels_prob, original_labels,
                                              confidence_threshold=0.7, base_image_save_path=base_image_save_path,
                                              title_prefix='Label Confidence Distribution')

    
    plot_teacher_predictions_multiple_thresholds(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path=base_image_save_path, title_prefix='Teacher Predictions by Thresholds')
    
    plot_teacher_predictions_multiple_thresholds11(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path=base_image_save_path, title_prefix='Teacher Predictions by Thresholds')

    plot_teacher_predictions_multiple_thresholds_with_accuracy(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path=base_image_save_path, title_prefix='Teacher Predictions by Thresholds')

    import shutil

    # Specifying the directory to zip
    directory_to_zip = "/content/CTTT/SaveImage"
    output_filename = "/content/CTTT/SaveImage.zip"

    # Zipping the directory
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
    args = parse_args()  # Parse the arguments
    set_seed(args.seed)  # Set the random seed for reproducibility

    main(args)  # Call the main function to start the training process

