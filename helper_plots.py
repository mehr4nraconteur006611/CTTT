import os
import torch
import numpy as np
import os
import matplotlib.pyplot as plt


modelnet40_class_names = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
    'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
    'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio',
    'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand',
    'vase', 'wardrobe', 'xbox'
]

def plot_mean_losses_and_accuracy(corruptions, classification_losses, consistency_losses, consistency2_losses, entropy_losses, accuracies, base_image_save_path='/content/CTTT/SaveImage', title_prefix='Corruptions Losses and Accuracies'):
    # Ensure the directory exists
    if not os.path.exists(base_image_save_path):
        os.makedirs(base_image_save_path)
    
    categories = list(corruptions)

    mean_classification_values = list(classification_losses.values())
    mean_consistency_values = list(consistency_losses.values())
    mean_consistency2_values = list(consistency2_losses.values())
    mean_entropy_values = list(entropy_losses.values())
    accuracy_values = list(accuracies.values())
    print(accuracy_values)
    # Calculate total loss as the sum of all losses for each corruption type
    total_loss_values = [classification + consistency + consistency2 + entropy 
                         for classification, consistency, consistency2, entropy 
                         in zip(mean_classification_values, mean_consistency_values, mean_consistency2_values, mean_entropy_values)]
    print(total_loss_values)
    print(mean_entropy_values)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot the individual loss values as lines for each component (subplot 1)
    ax1.plot(categories, mean_classification_values, marker='o', label='Classification Loss', color='skyblue')
    ax1.plot(categories, mean_consistency_values, marker='o', label='Consistency Loss', color='orange')
    ax1.plot(categories, mean_consistency2_values, marker='o', label='Consistency2 Loss', color='green')
    ax1.plot(categories, mean_entropy_values, marker='o', label='Entropy Loss', color='red')
    ax1.plot(categories, total_loss_values, marker='o', label='Total Loss', color='blue', linestyle='--')  # Adding total loss

    ax1.set_xlabel('Corruption Types', fontsize=14)
    ax1.set_ylabel('Mean Loss Value', fontsize=14)
    ax1.set_title('Mean Loss for Each Component and Total Loss by Corruption Type', fontsize=16)
    ax1.legend()
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')

    # Plot the accuracy values as a line plot (subplot 2)
    ax2.plot(categories, accuracy_values, marker='o', label='Accuracy', color='purple')

    ax2.set_xlabel('Corruption Types', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.set_title('Accuracy by Corruption Type', fontsize=16)
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()

    # Save the final aggregated plot
    save_path = os.path.join(base_image_save_path, f"{title_prefix}_Final_Aggregated.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    plt.show()
    plt.close()



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
    plt.show()





def plot_teacher_predictions_multiple_thresholds(teacher_pseudo_labels_prob, original_labels, thresholds=[0.5, 0.7, 0.9], base_image_save_path='', title_prefix='Teacher Predictions by Thresholds'):
    # Use the keys of the pseudo-labels dictionaries to find corruption types
    for corruption_type in teacher_pseudo_labels_prob.keys():

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

        # Show the plot
        plt.tight_layout()
        plt.show()


def compute_and_plot_metrics_for_corruptions(teacher_pseudo_labels_prob, weakaug_student_pseudo_labels_prob, strongaug_student_pseudo_labels_prob, original_labels, confidence_threshold=0.9, base_image_save_path='', title_prefix='Label Confidence Distribution'):
    # Use the keys of the pseudo-labels dictionaries to find corruption types
    for corruption_type in strongaug_student_pseudo_labels_prob.keys():

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
            incorrect_predictions = (predicted_labels != true_labels).sum().item()
            above_threshold = (confidence_scores > threshold).sum().item()
            correct_and_above_threshold = ((predicted_labels == true_labels) & (confidence_scores > threshold)).sum().item()
            incorrect_and_above_threshold = ((predicted_labels != true_labels) & (confidence_scores > threshold)).sum().item()
            correct_and_below_threshold = ((predicted_labels == true_labels) & (confidence_scores <= threshold)).sum().item()
            incorrect_and_below_threshold = ((predicted_labels != true_labels) & (confidence_scores <= threshold)).sum().item()
            return (correct_predictions, above_threshold, correct_and_above_threshold, 
                    incorrect_and_above_threshold, correct_and_below_threshold, 
                    incorrect_and_below_threshold, incorrect_predictions)

        # Teacher Metrics
        teacher_metrics = calculate_metrics(teacher_confidance_scores_args, teacher_confidance_scores, true_labels, confidence_threshold)

        # Weak Augmented Student Metrics
        weakaug_metrics = calculate_metrics(weakaug_student_confidance_scores_args, weakaug_student_confidance_scores, true_labels, confidence_threshold)

        # Strong Augmented Student Metrics
        strongaug_metrics = calculate_metrics(strongaug_student_confidance_scores_args, strongaug_student_confidance_scores, true_labels, confidence_threshold)

        # Step 6: Plot grouped histogram for comparison between Teacher, Weak Augmented, and Strong Augmented students
        categories = ['Correct (Total)', 'Above Threshold (Total)', 'Correct (Above Threshold)', 
                      'Incorrect (Above Threshold)', 'Correct (Below Threshold)', 
                      'Incorrect (Below Threshold)', 'Incorrect (Total)']

        fig, ax = plt.subplots(figsize=(14, 8))

        # Define the bar positions and width
        bar_width = 0.23
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
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

        add_labels(teacher_bars)
        add_labels(weakaug_bars)
        add_labels(strongaug_bars)

        # Step 10: Add legend
        ax.legend(fontsize=12)

        # Save the plot
        save_path = f"{base_image_save_path}/{title_prefix.replace(' ', '_')}_{corruption_type}_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

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
        label_range = np.arange(0, len(modelnet40_class_names))  # Range for ModelNet40 classes

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

        # Stacked histogram with ModelNet40 class names on x-axis
        counts, bins, bars = plt.hist(hist_data, bins=np.arange(len(modelnet40_class_names)+1), stacked=True, color=colors, label=labels, density=True)

        # Adjust the tick positions to center the labels under each bin
        plt.xticks(ticks=(bins[:-1] + bins[1:]) / 2, labels=modelnet40_class_names, rotation=90, ha='center')  # Center the labels

        plt.title(f"Label Distribution - {corruption_type}")
        plt.xlabel("Label Class (ModelNet40 Category)")
        plt.ylabel("Proportion of Labels")
        plt.legend(loc='upper right')
        plt.tight_layout()

        # Save the pseudo-label distribution plot
        plt.savefig(f"{base_image_save_path}/{idx}_{plot_title}_Label_Distribution_{corruption_type}.png")
        plt.close()

        # NEW: Create and save the table separately
        counts_teacher, _ = np.histogram(teacher_pseudo_labels_flat, bins=np.arange(len(modelnet40_class_names)+1))
        counts_weak_student, _ = np.histogram(weakaug_student_pseudo_labels_flat, bins=np.arange(len(modelnet40_class_names)+1))
        counts_strong_student, _ = np.histogram(strongaug_student_pseudo_labels_flat, bins=np.arange(len(modelnet40_class_names)+1))
        counts_original, _ = np.histogram(original_labels_flat, bins=np.arange(len(modelnet40_class_names)+1))

        # Create DataFrame for the table with class names
        table_data = {
            'Class Label': modelnet40_class_names,  # Use class names instead of numbers
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
    plt.figure(figsize=(30, 18))
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
            plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom',fontsize=5)

    # Move the legend outside the plot area with more descriptive labels
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

    plt.tight_layout()

    # Save the final aggregated plot
    plt.savefig(f"{base_image_save_path}/{plot_title}_Final_Aggregated.png", bbox_inches='tight')
    plt.close()
