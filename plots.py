import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import itertools

SUP_SIZE = 120
TITLE_SIZE = 22

def expand_row(row):
    tasks = ['picture_task', 'id_task']
    task_correct_columns = ['picture_task_is_correct', 'id_task_is_correct']
    
    expanded_rows = [
        {**row, 'task': task, 'task_correct': row[task_correct_col]}
        for task, task_correct_col in zip(tasks, task_correct_columns)
    ]
    
    return pd.DataFrame(expanded_rows)

def calculate_success_percentage(df, correct_column, groupby):
    total_counts = df.groupby(groupby).size().reset_index(name='total')
    success_counts = df[df[correct_column] == 1].groupby(groupby).size().reset_index(name='success')
    merged_df = pd.merge(total_counts, success_counts, on=groupby, how='left')
    merged_df['success'] = merged_df['success'].fillna(0)
    merged_df['percentage'] = (merged_df['success'] / merged_df['total']) * 100
    return merged_df

def task_performance_percentage_calc(all_results_df, correct_column, noise_constants, thresholds, groupby):
    aggregated_results = []

    for noise, threshold in itertools.product(noise_constants, thresholds):
        subset_df = all_results_df[(all_results_df['noise'] == noise) & (all_results_df['threshold'] == threshold)]
        if not subset_df.empty:
            success_df = calculate_success_percentage(subset_df, correct_column, groupby)
            success_df['noise'] = noise
            success_df['threshold'] = threshold
            aggregated_results.append(success_df)

    return pd.concat(aggregated_results, ignore_index=True)

def plot_task_performance_graph(model_df_dict, noise_constants, thresholds, export_path):
    groupby = ['familiarity', 'group']
    tasks = {}
    
    for model_name, model_df in model_df_dict.items():
        tasks[f'{model_name} - Person Task'] = ('id_task_is_correct', model_df)
        tasks[f'{model_name} - Picture Task'] = ('picture_task_is_correct', model_df)

    for title, (correct_column, df) in tasks.items():
        model_name = title[:title.find(' -')]
        task = title[title.find(' -')+3:]
        model_export_path = os.path.join(export_path, model_name)

        task_df = task_performance_percentage_calc(df, correct_column, noise_constants, thresholds, groupby)
        plot_task_performance_results(task_df, title, noise_constants, thresholds, model_export_path)
        print(f'plotted {task} performance graph for {model_name}')
    
def plot_tasks_comparison(model_df_dict, noise_constants, thresholds, export_path):
    cols_needed = ['group', 'noise', 'threshold', 'familiarity', 'picture_task_is_correct', 'id_task_is_correct']
    groups = ['Same', 'Diff', 'Unseen']

    tasks = {}

    for model_name, model_df in model_df_dict.items():
        tasks[model_name] = model_df[cols_needed]
    
    for g in groups:
        for model_name, df in tasks.items():
            model_export_path = os.path.join(export_path, model_name)
            expanded_df = pd.concat([expand_row(row) for _, row in df.iterrows()], ignore_index=True)
            plot_results_per_group(expanded_df, f'{g}_{model_name}', noise_constants, thresholds, g, model_export_path)
            print(f'plotted {g} comaprison graph for {model_name}')

def plot_task_performance_results(aggregated_results_df, title, noise_constants, thresholds, export_path):
    t_min = min(thresholds)
    t_max = max(thresholds)
    n_min = min(noise_constants)
    n_max = max(noise_constants)

    n_rows = len(noise_constants)
    n_cols = len(thresholds)
    
    fig_width = 6 * n_cols
    fig_height = 4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharey=True)

    handles = []
    labels = []
    legend_drawn = False

    for i, noise in enumerate(noise_constants):
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            subset_df = aggregated_results_df[
                (aggregated_results_df['noise'] == noise) & 
                (aggregated_results_df['threshold'] == threshold)
            ]
            
            sns.barplot(
                ax=ax, 
                x='group', 
                y='percentage', 
                hue='familiarity', 
                data=subset_df,
                dodge=True, 
                palette={'familiar': 'orange', 'unfamiliar': 'blue'}
            )
            ax.set_title(f'Noise: {noise}, Threshold: {threshold}', fontsize = TITLE_SIZE)
            ax.set_xlabel('Group')
            if j == 0:
                ax.set_ylabel('Percentage of Correct Predictions')
            else:
                ax.set_ylabel('')
            
            if not legend_drawn:
                handles, labels = ax.get_legend_handles_labels()
                legend_drawn = True
            ax.legend_.remove()

    if handles and labels:
        fig.legend(handles, labels, loc='upper left', fontsize=40)

    plt.suptitle(title, fontsize=SUP_SIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(f"{export_path}/task perf_thershold={t_min}-{t_max}_noise={n_min}-{n_max}_{title}.png")
    plt.close()

def plot_results_per_group(df, title, noise_constants, thresholds, group, export_path):
    t_min = min(thresholds)
    t_max = max(thresholds)
    n_min = min(noise_constants)
    n_max = max(noise_constants)    
    
    n_rows = len(noise_constants)
    n_cols = len(thresholds)
    fig_width = 6 * n_cols
    fig_height = 4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharey=True)

    handles = []
    labels = []
    legend_drawn = False
    
    for i, noise in enumerate(noise_constants):
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            subset_df = df[
                (df['noise'] == noise) & 
                (df['threshold'] == threshold) & 
                (df['group'] == group)
            ]
            
            sns.barplot(
                ax=ax,
                x='task',
                y='task_correct',
                hue='familiarity',
                data=subset_df,
                palette={'familiar': 'orange', 'unfamiliar': 'blue'},
                dodge=True
            )
            ax.set_ylim(0, 1)
            ax.set_title(f'Noise: {noise}, Threshold: {threshold}', fontsize = TITLE_SIZE)
            if j == 0:
                ax.set_ylabel('Task Correctness Percentage')
            else:
                ax.set_ylabel('')
            if not legend_drawn:
                handles, labels = ax.get_legend_handles_labels()
                legend_drawn = True
            ax.legend_.remove()
            
    if handles and labels:
        fig.legend(handles, labels, loc='upper left', fontsize=40)
    
    plt.suptitle(f"{title} - Group: {group}", fontsize=SUP_SIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{export_path}/task comp_thershold={t_min}-{t_max}_noise={n_min}-{n_max}_{title}.png")
    plt.close()

def plot_task_performance_by_conditions(
    performance_df, model_name, noise_threshold_pairs, export_path, title="Task Performance"
):
    """
    Plots a superplot where each threshold has its own bar plot (subplot),
    divided into groups and tasks, visualizing the noise levels.

    Args:
        performance_df (pd.DataFrame): DataFrame containing the performance results.
        model_name (str): Name of the model to filter results for.
        noise_threshold_pairs (list of tuples): List of (noise, threshold) pairs to plot.
        export_path (str): Path to save the plot.
        title (str): Title of the plot.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    
    thresholds = [x/100 for x in range(12)]
    

    num_thresholds = len(thresholds)
    
    n_cols = 4  
    n_rows = math.ceil(num_thresholds / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True)
    axes = axes.flatten() if num_thresholds > 1 else [axes]

    

    for idx, threshold in enumerate(thresholds):
        ax = axes[idx]

        subset = performance_df[performance_df['threshold'] == threshold]
        if subset.empty:
            print(f"No data available for threshold={threshold}")
            continue

        
        melted = pd.melt(
            subset,
            id_vars=['group', 'familiarity', 'noise'],
            value_vars=['picture_task_is_correct', 'id_task_is_correct'],
            var_name='Task',
            value_name='Correct'
        )

        
        task_mapping = {
            'picture_task_is_correct': 'Picture Task',
            'id_task_is_correct': 'Person Task'
        }
        melted['Task'] = melted['Task'].map(task_mapping)

        
        melted['Task-Familiarity'] = melted['Task'] + ' - ' + melted['familiarity']

        
        melted['Condition'] = melted['group'] + ' - ' + melted['Task-Familiarity']

        
        grouped = melted.groupby(['Condition', 'noise'])['Correct'].mean().reset_index()

        
        sns.barplot(
            data=grouped,
            x='Condition',
            y='Correct',
            hue='noise',
            errorbar=None,
            ax=ax
        )

        ax.set_title(f"Threshold={threshold}", fontsize=14)
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)

        ax_legend = ax.get_legend()
        if ax_legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

    
    if handles and labels:
        fig.legend(
            handles,
            labels,
            title='Noise Level',
            fontsize=10,
            title_fontsize=12,
            loc='upper right'
        )

    
    for idx in range(len(thresholds), len(axes)):
        fig.delaxes(axes[idx])

    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"{title} for Model: {model_name}", fontsize=16)

    
    plot_path = os.path.join(export_path, f"{title.replace(' ', '_')}_model={model_name}.png")
    plt.savefig(plot_path)
    plt.close()

def plot_difference_heatmap(df, l):
    """
    Plot a heatmap showing the difference in accuracy between two models.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing accuracy data for the models.
    """
    
    df['noise'] = pd.to_numeric(df['noise'], errors='coerce')
    df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce')
    df = df[df['familiarity'] == 'familiar']
    df = df[(round(df['noise'] * 1000) % 50 == 0) & 
            ((round(df['threshold']*1000) % 50 == 0) | 
             (round(df['threshold']*1000) == 75) |
             (round(df['threshold']*1000) == 125))]
    
    modelC_fc7 = df[df['Model-Layer'] == f'modelC_{l}']
    modelB_fc7 = df[df['Model-Layer'] == f'modelB_{l}']

    
    merged_df = pd.merge(modelC_fc7, modelB_fc7, on=['group', 'threshold', 'noise'], suffixes=('_A', '_B'))

    
    merged_df['accuracy_diff'] = merged_df['id_task_is_correct_A'] - merged_df['id_task_is_correct_B']
    
    groups = merged_df['group'].unique()
    for g in groups:
        g_df = merged_df.copy()
        g_df = g_df[g_df['group'] == g].copy()
        heatmap_data = g_df.pivot_table(index='noise', columns='threshold', values='accuracy_diff')
        
        plt.figure(figsize=(11, 8))

        ax = sns.heatmap(heatmap_data, cmap='coolwarm', center=0, annot=None, fmt='.2f', 
                    cbar_kws={'label': 'Accuracy Difference (%)'})
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)  
        cbar.set_label('Accuracy Difference (%)', fontsize=18)  
        plt.gca().invert_yaxis()
        plt.title(f'Difference in Accuracy between Models C and Model B, Layer {l}', fontsize=18)
        plt.xlabel('Threshold', fontsize = 18)
        plt.ylabel('Noise Level', fontsize = 18)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(rotation=0 , fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), f'diff heatmap_{l}_{g}.png'))
        plt.close()

def calculate_accuracy(df):
    accuracy_df = df.groupby(['Model-Layer', 'threshold', 'noise', 'group', 'familiarity']).agg(
        person_accuracy=('id_task_is_correct', 'mean'),
        image_accuracy=('picture_task_is_correct', 'mean')
    ).reset_index()
    
    accuracy_df['person_accuracy'] = accuracy_df['person_accuracy'] * 100
    accuracy_df['image_accuracy'] = accuracy_df['image_accuracy'] * 100

    return accuracy_df

def plot_accuracy_vs_threshold(model, group, accuracy_melted):
    plt.rc('font', family='Times New Roman')

    subset = accuracy_melted[
        (accuracy_melted['Model-Layer'] == model) & 
        (accuracy_melted['group'] == group)
    ].copy()

    subset['threshold'] = pd.to_numeric(subset['threshold'], errors='coerce')
    subset['accuracy'] = pd.to_numeric(subset['accuracy'], errors='coerce')
    subset = subset.dropna(subset=['threshold', 'accuracy'])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 10))  

    palette = sns.color_palette("viridis", n_colors=len(subset['familiarity'].unique()))

    sns.lineplot(
        data=subset,
        x='threshold',
        y='accuracy',
        hue='familiarity',       
        style='task',           
        markers=True,
        dashes=['', (2,2)],
        palette=palette,
        linewidth=2.5,
        markersize=10,
    )

    plt.title(f'Accuracy vs. Threshold Level for {model} - Group {group}', fontsize=24)
    plt.xlabel('Threshold Level', fontsize=24)
    plt.ylabel('Accuracy (%)', fontsize=24)

    ax = plt.gca()
    major_tick_spacing = x_ticks
    minor_tick_spacing = x_ticks / 2 
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    
    major_y_spacing = 10  
    minor_y_spacing = 2   
    ax.yaxis.set_major_locator(MultipleLocator(major_y_spacing))
    
    ax.tick_params(axis='x', which='major', labelsize=14, length=10, width=1.5)
    ax.tick_params(axis='y', which='major', labelsize=14, length=10, width=1.5)
   
    plt.xticks(rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    
    plt.legend(
            bbox_to_anchor=(1.05, 1), 
            frameon=True,
            loc='lower right', 
            fontsize=16)

    ax.set_xlim(-(x_ticks/2), t_max+(x_ticks/2))
    plt.tight_layout()

    save_dir = os.path.join(export ,model)
    os.makedirs(save_dir, exist_ok=True)  
 
    filename = f'{group}-accuracy_vs_threshold.png'
  
    plt.savefig(os.path.join(export, model, filename), dpi=300)
    plt.close()

    print(f"Plot saved successfully at: {os.path.join(save_dir, filename)}")

def generate_impact_of_noise_on_accuracy(df):
    """
    Generate and plot the impact of noise levels on accuracy for both tasks and models.

    Parameters:
    - df (pd.DataFrame): DataFrame containing experimental data.
    """
    accuracy_df = calculate_accuracy(df)  
    accuracy_melted = accuracy_df.melt(
        id_vars=['Model-Layer', 'threshold', 'noise', 'group','familiarity'],
        value_vars=['person_accuracy', 'image_accuracy'],
        var_name='task',
        value_name='accuracy'
    )
    
    accuracy_melted['task'] = accuracy_melted['task'].map({
        'person_accuracy': 'Person Recognition',
        'image_accuracy': 'Image Recognition'
    })

    models = ['modelC_fc7', 'modelC_conv5']
    groups = accuracy_melted['group'].unique()

    for m in models:
        for g in groups:
            plot_accuracy_vs_threshold(m, g, accuracy_melted)

df = pd.read_csv('/home/new_storage/experiments/face_memory_task/code results/n_0-100-2.5/combined_modelC_conv5-modelC_fc7-modelB_conv5-modelB_fc7.csv')
t_max = 1
export = os.path.join(os.getcwd(), 'final_vis', f'0-{t_max}_half')
x_ticks = 0.1
generate_impact_of_noise_on_accuracy(df)

data_file_path = '/home/new_storage/experiments/face_memory_task/code results/n_0-100-2.5/combined_modelC_conv5-modelC_fc7-modelB_conv5-modelB_fc7.csv'
df = pd.read_csv(data_file_path)

plot_difference_heatmap(df, 'fc7')
plot_difference_heatmap(df, 'conv5')
