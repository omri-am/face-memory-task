import os
import random
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

num_gpus = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Number of GPUs available: {num_gpus}")
print(f"Using device: {device}")

def clear_console():
    """Clears the console."""
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)

### Helper Functions ###

def open_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 256)),
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

def add_noise(dic, std_dev):
    for path in dic.keys():
        a_tensor = dic[path]
        noise_tensor = torch.randn_like(a_tensor) * float(std_dev)
        dic[path]= a_tensor+noise_tensor

def create_long_term_memory(model, base_dir, sub_folder, save_path):
    ltm = dict()
    train_dir = os.path.join(base_dir, sub_folder, 'train')
    if os.path.exists(train_dir):
        identities = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        for identity in identities:
            identity_dir = os.path.join(train_dir, identity)
            images = [f for f in os.listdir(identity_dir) if os.path.isfile(os.path.join(identity_dir, f))]

            imgs_tensors = list()
            for image in images:
                image_path = os.path.join(identity_dir, image)
                processed_image = open_image(image_path)
                tensor = model.get_output(processed_image)
                imgs_tensors.append(tensor)
            ltm[identity] = torch.stack(imgs_tensors).mean(dim=0)
    else:
        raise Exception(f"Path does not exist: {train_dir}")

    torch.save(ltm, save_path)
    return ltm

def load_long_term_memory(save_path):
    identities_memory = torch.load(save_path)
    return identities_memory

def find_closest(report_tensor, short_term_memory, long_term_memory):
    st_closest_img = None
    st_closest_identity = None
    st_closest_distance = 1

    lt_closest_identity = None
    lt_closest_distance = 1

    memory_origin = 0
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for key, from_mem in short_term_memory.items():
        distance = 1-cos(report_tensor, from_mem)
        if distance < st_closest_distance:
            st_closest_img = key
            st_closest_identity = os.path.basename(os.path.dirname(st_closest_img))
            st_closest_distance = distance

    for key, from_mem in long_term_memory.items():
        distance = 1-cos(report_tensor, from_mem)
        if distance < lt_closest_distance:
            lt_closest_identity = key
            lt_closest_distance = distance

    try:
        st_closest_distance = st_closest_distance.detach().item()
        lt_closest_distance = lt_closest_distance.detach().item()
    except:
        pass

    return st_closest_img, st_closest_identity, st_closest_distance, lt_closest_identity, lt_closest_distance

### Data Preperation ###

def process_folders(base_dir, sub_folders):
    df = pd.DataFrame(columns=['identity', 'familiarity', 'group', 'stage_a_img_path', 'stage_b_img_path'])

    for sub_folder in sub_folders:
        test_dir = os.path.join(base_dir, sub_folder, 'test')
        familiarity = "familiar" if sub_folder[18] == 'C' else "unfamiliar"
        if os.path.exists(test_dir):
            identities = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            random.shuffle(identities)

            same_group = identities[:33]
            diff_group = identities[33:66]
            unknown_group = identities[66:99]

            for identity in same_group:
                df = process_same_group(test_dir, identity, df, familiarity)
            for identity in diff_group:
                df = process_diff_group(test_dir, identity, df, familiarity)
            for identity in unknown_group:
                df = process_unknown_group(test_dir, identity, df, familiarity)
    return df

def process_same_group(base_dir, identity, df, familiarity):
    identity_dir = os.path.join(base_dir, identity)
    images = [f for f in os.listdir(identity_dir) if os.path.isfile(os.path.join(identity_dir, f))]
    if images:
        image = random.choice(images)
        image_path = os.path.join(identity_dir, image)
        new_row = pd.DataFrame({'identity': [identity], 'familiarity': [familiarity], 'group': ['same'], 'stage_a_img_path': [image_path], 'stage_b_img_path': [image_path]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df

def process_diff_group(base_dir, identity, df, familiarity):
    identity_dir = os.path.join(base_dir, identity)
    images = [f for f in os.listdir(identity_dir) if os.path.isfile(os.path.join(identity_dir, f))]
    if len(images) > 1:
        image_a, image_b = random.sample(images, 2)
        image_a_path = os.path.join(identity_dir, image_a)
        image_b_path = os.path.join(identity_dir, image_b)
        new_row = pd.DataFrame({'identity': [identity], 'familiarity': [familiarity], 'group': ['diff'], 'stage_a_img_path': [image_a_path], 'stage_b_img_path': [image_b_path]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df

def process_unknown_group(base_dir, identity, df, familiarity):
    identity_dir = os.path.join(base_dir, identity)
    images = [f for f in os.listdir(identity_dir) if os.path.isfile(os.path.join(identity_dir, f))]
    if images:
        image = random.choice(images)
        image_path = os.path.join(identity_dir, image)
        new_row = pd.DataFrame({'identity': [identity], 'familiarity': [familiarity], 'group': ['unknown'], 'stage_a_img_path': [None], 'stage_b_img_path': [image_path]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df

### Study Stage (Create Short Term Memory) ###

def study_stage(model, grouped_images_df, noise):
    memory = dict()
    for img_path in grouped_images_df["stage_a_img_path"]:
        if img_path:
            processed_image = open_image(img_path)
            tensor = model.get_output(processed_image)
            memory[img_path] = tensor
            add_noise(memory, noise)
    return memory

### Report Stage ###

def report_stage_process_record(record, nnModel, st_memory, lt_memory):
    img_path = record.get('stage_b_img_path', record.get('image_path'))
    processed_image = open_image(img_path)
    tensor = nnModel.get_output(processed_image)
    stm_closest_img_path, stm_closest_identity, stm_closest_distance, ltm_closest_identity, ltm_closest_distance = find_closest(tensor, st_memory, lt_memory)

    same_image = int(img_path == stm_closest_img_path)
    stm_same_id = int(record['identity'] == stm_closest_identity)
    ltm_same_id = int(record['identity'] == ltm_closest_identity)

    return {
        'familiarity': record['familiarity'],
        'group': record['group'],
        'report_stage_img': img_path,
        'report_stage_identity': record['identity'],
        'stm_closest_img': stm_closest_img_path,
        'stm_closest_identity': stm_closest_identity,
        'stm_distance': stm_closest_distance,
        'ltm_closest_identity': ltm_closest_identity,
        'ltm_distance': ltm_closest_distance,
        'same_image': same_image,
        'stm_same_id': stm_same_id,
        'ltm_same_id': ltm_same_id
    }

def report_stage(nnModel, noise, st_memory, lt_memory, grouped_images_df):
    closest_columns = {'familiarity': 'str',
                        'group': 'str',
                        'report_stage_img': 'str',
                        'report_stage_identity': 'str',
                        'stm_closest_img': 'str',
                        'stm_closest_identity': 'str',
                        'stm_distance': 'float',
                        'ltm_closest_identity': 'str',
                        'ltm_distance': 'float',
                        'same_image': 'int',
                        'stm_same_id': 'int',
                        'ltm_same_id': 'int'}
    closest_df = pd.DataFrame({col: pd.Series(dtype=f'{dtype}') for col, dtype in closest_columns.items()})
    closest_df = pd.DataFrame(columns=closest_columns)
    for _, record in grouped_images_df.iterrows():
        new_row = report_stage_process_record(record, nnModel, st_memory, lt_memory)
        closest_df = pd.concat([closest_df, pd.DataFrame([new_row])], ignore_index=True)
        add_noise(st_memory, noise)
    return closest_df

def single_experiment_by_noise(nnModel, grouped_images_df, long_term_memory, path_to_save, noise):
    short_term_memory = study_stage(nnModel, grouped_images_df, noise)
    # torch.save(short_term_memory, os.path.join(path_to_save, f'{nnModel.name}_shortTermMemory_noise={noise}.pth'))

    performance = report_stage(nnModel, noise, short_term_memory, long_term_memory, grouped_images_df)
    performance_df = pd.DataFrame(performance)
    # performance_df.to_csv(os.path.join(path_to_save, f'{nnModel.name}_performance_noise={noise}.csv'))
    return performance_df

def simulate_different_noises(nnModel, long_term_memory, grouped_images_df, noise_constants):
    path = f'{os.getcwd()}/code results/{nnModel.name}-{date.today()}'
    try:
        os.mkdir(path)
    except:
        pass

    results = []
    i = 1
    n_min = min(noise_constants)
    n_max = max(noise_constants)

    for noise in noise_constants:
        performance_df = single_experiment_by_noise(nnModel, grouped_images_df, long_term_memory, path, noise)
        performance_df['noise'] = noise
        results.append(performance_df)
        # clear_console()
        print(f"Simulating Noise = {noise}...")
        i += 1

    all_results_df = pd.concat(results, ignore_index=True)
    all_results_df.to_csv(os.path.join(path, f'{nnModel.name}_noise={n_min}-{n_max}_performance.csv'))
    return all_results_df

### Performance Evaluation Stage ###

def eval_performance_for_thershold(perf_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    res_df = perf_df.copy()

    res_df['img_task_is_correct'] = 0
    res_df['stm_id_task_is_correct'] = 0
    res_df['ltm_id_task_is_correct'] = 0
    res_df['id_task_is_correct'] = 0
    res_df['threshold'] = threshold

    same_group = res_df['group'] == 'same'
    diff_group = res_df['group'] == 'diff'
    unknown_group = res_df['group'] == 'unknown'

    # Conditions for 'same' group
    res_df.loc[same_group & (res_df['stm_distance'] < threshold), 'img_task_is_correct'] = res_df['same_image']
    res_df.loc[same_group & (res_df['stm_distance'] < threshold), 'stm_id_task_is_correct'] = res_df['stm_same_id']
    res_df.loc[same_group & (res_df['ltm_distance'] < threshold), 'ltm_id_task_is_correct'] = res_df['ltm_same_id']
    res_df.loc[same_group & ((res_df['stm_distance'] < threshold) | (res_df['ltm_distance'] < threshold)), 'id_task_is_correct'] = 1

    # Conditions for 'diff' group
    res_df.loc[diff_group & (res_df['stm_distance'] > threshold), 'img_task_is_correct'] = 1
    res_df.loc[diff_group & (res_df['stm_distance'] < threshold), 'stm_id_task_is_correct'] = res_df['stm_same_id']
    res_df.loc[diff_group & (res_df['ltm_distance'] < threshold), 'ltm_id_task_is_correct'] = res_df['ltm_same_id']
    res_df.loc[diff_group & ((res_df['stm_distance'] < threshold) | (res_df['ltm_distance'] < threshold)), 'id_task_is_correct'] = 1

    # Conditions for 'unknown' group
    res_df.loc[unknown_group & (res_df['stm_distance'] > threshold), 'img_task_is_correct'] = 1
    res_df.loc[unknown_group & (res_df['stm_distance'] > threshold), 'stm_id_task_is_correct'] = 1
    res_df.loc[unknown_group & (res_df['ltm_distance'] > threshold), 'ltm_id_task_is_correct'] = 1
    res_df.loc[unknown_group & ((res_df['stm_distance'] > threshold) & (res_df['ltm_distance'] > threshold)), 'id_task_is_correct'] = 1

    return res_df

def evaluate_multiple_thresholds(perf_df: pd.DataFrame, thresholds: list) -> pd.DataFrame:
    results = []
    for threshold in thresholds:
        result_df = eval_performance_for_thershold(perf_df, threshold)
        results.append(result_df)
    combined_results = pd.concat(results, ignore_index=True)
    return combined_results

### One Function To Rule Them All ###

def run_all_for_model(model, base_dir, sub_folders, long_term_memory, noise_constants, thresholds):
    grouped_images_df = process_folders(base_dir, sub_folders)
    all_noises_df = simulate_different_noises(model, long_term_memory, grouped_images_df, noise_constants)
    noises_thersholds_combined_df = evaluate_multiple_thresholds(all_noises_df, thresholds)
    return noises_thersholds_combined_df


### Visualization ###

def calculate_success_percentage(df, correct_column):
    total_counts = df.groupby(['familiarity', 'group']).size().reset_index(name='total')
    success_counts = df[df[correct_column] == 1].groupby(['familiarity', 'group']).size().reset_index(name='success')
    merged_df = pd.merge(total_counts, success_counts, on=['familiarity', 'group'], how='left')
    merged_df['success'] = merged_df['success'].fillna(0)
    merged_df['percentage'] = (merged_df['success'] / merged_df['total']) * 100
    return merged_df

def apply_calc(all_results_df, correct_column, noise_constants, thresholds):
    aggregated_results = []

    for noise, threshold in itertools.product(noise_constants, thresholds):
        subset_df = all_results_df[(all_results_df['noise'] == noise) & (all_results_df['threshold'] == threshold)]
        if not subset_df.empty:
            success_df = calculate_success_percentage(subset_df, correct_column)
            success_df['noise'] = noise
            success_df['threshold'] = threshold
            aggregated_results.append(success_df)

    # Concatenate all aggregated results
    return pd.concat(aggregated_results, ignore_index=True)

def plot_results(aggregated_results_df, title, noise_constants, thresholds):
    t_min = min(thresholds)
    t_max = max(thresholds)
    n_min = min(noise_constants)
    n_max = max(noise_constants)

    n_rows = len(noise_constants)
    n_cols = len(thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(50, 50), sharey=True)

    handles = []
    labels = []

    for i, noise in enumerate(noise_constants):
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            subset_df = aggregated_results_df[(aggregated_results_df['noise'] == noise) & (aggregated_results_df['threshold'] == threshold)]

            if not subset_df.empty:
                bar_plot = sns.barplot(ax=ax, x='group', y='percentage', hue='familiarity', data=subset_df,
                                       dodge=True, palette={ 'familiar': 'orange', 'unfamiliar': 'blue'})
                ax.set_title(f'Noise: {noise}, Threshold: {threshold}')
                ax.set_xlabel('Group')
                if j == 0:
                    ax.set_ylabel('Percentage of Correct Predictions')
                if i == 0 and j == n_cols - 1:
                    handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove()

    fig.legend(handles, labels, loc='upper left', fontsize=18)
    plt.suptitle(title+f"\n\n\n", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    plt.savefig(f"{os.getcwd()}/code results/thershold={t_min}-{t_max}_noise={n_min}-{n_max}_{title}.png")
    plt.show()