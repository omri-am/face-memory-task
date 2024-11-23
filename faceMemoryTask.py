import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

### Helper Functions ###

def open_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 256)), # 224, 224
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

def create_long_term_memory(model, base_dir, sub_folder, export_path):
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
                output_dict = model.get_output(processed_image)
                tensor = output_dict[next(iter(output_dict.keys()))]
                imgs_tensors.append(tensor)
            ltm[identity] = torch.stack(imgs_tensors).mean(dim=0)
    else:
        raise Exception(f"Path does not exist: {train_dir}")
    
    ltm_path = os.path.join(export_path, f'{model.model_name}_longTermMemory.pth')
    torch.save(ltm, ltm_path)
    return ltm

def find_closest(report_tensor, short_term_memory, long_term_memory):
    st_closest_img = None
    st_closest_identity = None
    st_closest_distance = 1

    lt_closest_identity = None
    lt_closest_distance = 1

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
            Unseen_group = identities[66:99]

            for identity in same_group:
                df = process_same_group(test_dir, identity, df, familiarity)
            for identity in diff_group:
                df = process_diff_group(test_dir, identity, df, familiarity)
            for identity in Unseen_group:
                df = process_Unseen_group(test_dir, identity, df, familiarity)
    return df

def process_same_group(base_dir, identity, df, familiarity):
    identity_dir = os.path.join(base_dir, identity)
    images = [f for f in os.listdir(identity_dir) if os.path.isfile(os.path.join(identity_dir, f))]
    if images:
        image = random.choice(images)
        image_path = os.path.join(identity_dir, image)
        new_row = pd.DataFrame({'identity': [identity], 'familiarity': [familiarity], 'group': ['Same'], 'stage_a_img_path': [image_path], 'stage_b_img_path': [image_path]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df

def process_diff_group(base_dir, identity, df, familiarity):
    identity_dir = os.path.join(base_dir, identity)
    images = [f for f in os.listdir(identity_dir) if os.path.isfile(os.path.join(identity_dir, f))]
    if len(images) > 1:
        image_a, image_b = random.sample(images, 2)
        image_a_path = os.path.join(identity_dir, image_a)
        image_b_path = os.path.join(identity_dir, image_b)
        new_row = pd.DataFrame({'identity': [identity], 'familiarity': [familiarity], 'group': ['Diff'], 'stage_a_img_path': [image_a_path], 'stage_b_img_path': [image_b_path]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df

def process_Unseen_group(base_dir, identity, df, familiarity):
    identity_dir = os.path.join(base_dir, identity)
    images = [f for f in os.listdir(identity_dir) if os.path.isfile(os.path.join(identity_dir, f))]
    if images:
        image = random.choice(images)
        image_path = os.path.join(identity_dir, image)
        new_row = pd.DataFrame({'identity': [identity], 'familiarity': [familiarity], 'group': ['Unseen'], 'stage_a_img_path': [None], 'stage_b_img_path': [image_path]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df

### Study Stage (Create Short Term Memory) ###

def study_stage(model, grouped_images_df, noise):
    memory = dict()
    for img_path in grouped_images_df["stage_a_img_path"]:
        if img_path:
            processed_image = open_image(img_path)
            output_dict = model.get_output(processed_image)
            tensor = output_dict[next(iter(output_dict.keys()))]
            memory[img_path] = tensor
            add_noise(memory, noise)
    return memory

### Report Stage ###

def report_stage_process_record(record, nnModel, st_memory, lt_memory):
    img_path = record.get('stage_b_img_path', record.get('image_path'))
    processed_image = open_image(img_path)
    output_dict = nnModel.get_output(processed_image)
    tensor = output_dict[next(iter(output_dict.keys()))]
    stm_closest_img_path, stm_closest_identity, stm_closest_distance, ltm_closest_identity, ltm_closest_distance = find_closest(tensor, st_memory, lt_memory)

    same_image = int(os.path.normpath(img_path) == os.path.normpath(stm_closest_img_path)) if stm_closest_img_path else 0
    stm_same_id = int(record['identity'] == stm_closest_identity) if stm_closest_identity else 0
    ltm_same_id = int(record['identity'] == ltm_closest_identity) if ltm_closest_identity else 0

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

def simulate_different_noises(nnModel, long_term_memory, grouped_images_df, noise_constants, export_path):
    results = []
    i = 1
    n_min = min(noise_constants)
    n_max = max(noise_constants)

    for noise in noise_constants:
        performance_df = single_experiment_by_noise(nnModel, grouped_images_df, long_term_memory, export_path, noise)
        performance_df['noise'] = noise
        results.append(performance_df)
        print(f"{nnModel.model_name}, simulating noise = {noise}...")
        i += 1

    all_results_df = pd.concat(results, ignore_index=True)
    all_results_df.to_csv(os.path.join(export_path, f'{nnModel.model_name}_noise={n_min}-{n_max}_performance.csv'))
    return all_results_df

### Performance Evaluation Stage ###

def eval_performance_for_threshold(perf_df, threshold):
    res_df = perf_df.copy()

    res_df['picture_task_is_correct'] = 0
    res_df['stm_id_task_is_correct'] = 0
    res_df['ltm_id_task_is_correct'] = 0
    res_df['id_task_is_correct'] = 0
    res_df['threshold'] = threshold

    same_group = res_df['group'] == 'Same'
    diff_group = res_df['group'] == 'Diff'
    Unseen_group = res_df['group'] == 'Unseen'

    # --- Same Group ---
    # Image Task Success: Recognized same image in STM
    mask = same_group & (res_df['stm_distance'] < threshold)
    res_df.loc[mask, 'picture_task_is_correct'] = res_df.loc[mask, 'same_image']

    # STM Identity Task Success: Recognized correct identity in STM
    mask = same_group & (res_df['stm_distance'] < threshold)
    res_df.loc[mask, 'stm_id_task_is_correct'] = res_df.loc[mask, 'stm_same_id']

    # LTM Identity Task Success: Recognized correct identity in LTM
    mask = same_group & (res_df['ltm_distance'] < threshold)
    res_df.loc[mask, 'ltm_id_task_is_correct'] = res_df.loc[mask, 'ltm_same_id']

    # Overall Identity Task Success: Correct identity recognized in STM or LTM
    mask = same_group & (
        ((res_df['stm_distance'] < threshold) & (res_df['stm_same_id'] == 1)) |
        ((res_df['ltm_distance'] < threshold) & (res_df['ltm_same_id'] == 1))
    )
    res_df.loc[mask, 'id_task_is_correct'] = 1

    # --- Different Group ---
    # Image Task Success: Model didn't recognize the image
    mask = diff_group & (res_df['stm_distance'] > threshold) & (res_df['ltm_distance'] > threshold)
    res_df.loc[mask, 'picture_task_is_correct'] = 1

    # STM Identity Task Success: Recognized correct identity in STM
    mask = diff_group & (res_df['stm_distance'] < threshold)
    res_df.loc[mask, 'stm_id_task_is_correct'] = res_df.loc[mask, 'stm_same_id']

    # LTM Identity Task Success: Recognized correct identity in LTM
    mask = diff_group & (res_df['ltm_distance'] < threshold)
    res_df.loc[mask, 'ltm_id_task_is_correct'] = res_df.loc[mask, 'ltm_same_id']

    # Overall Identity Task Success: Correct identity recognized in STM or LTM
    mask = diff_group & (
        ((res_df['stm_distance'] < threshold) & (res_df['stm_same_id'] == 1)) |
        ((res_df['ltm_distance'] < threshold) & (res_df['ltm_same_id'] == 1))
    )
    res_df.loc[mask, 'id_task_is_correct'] = 1

    # --- Unseen Group ---
    # Image Task Success: Model didn't recognize the image
    mask = Unseen_group & (res_df['stm_distance'] > threshold) & (res_df['ltm_distance'] > threshold)
    res_df.loc[mask, 'picture_task_is_correct'] = 1

    # STM Identity Task Success: Model didn't recognize the identity in STM
    mask = Unseen_group & (res_df['stm_distance'] > threshold)
    res_df.loc[mask, 'stm_id_task_is_correct'] = 1  # Success if not recognized

    # LTM Identity Task Success: Model didn't recognize the identity in LTM
    mask = Unseen_group & (res_df['ltm_distance'] > threshold)
    res_df.loc[mask, 'ltm_id_task_is_correct'] = 1  # Success if not recognized

    # Overall Identity Task Success: Identity not recognized in both STM and LTM
    mask = Unseen_group & (res_df['stm_distance'] > threshold) & (res_df['ltm_distance'] > threshold)
    res_df.loc[mask, 'id_task_is_correct'] = 1

    return res_df

# def evaluate_multiple_thresholds(perf_df, thresholds):
#     results = []
#     for threshold in thresholds:
#         result_df = eval_performance_for_threshold(perf_df, threshold)
#         results.append(result_df)
#     combined_results = pd.concat(results, ignore_index=True)
#     return combined_results

def evaluate_multiple_thresholds(perf_df, thresholds, export_path):
    results = []
    conditions_results = []  # To store results of familiar vs unfamiliar checks

    for threshold in thresholds:
        result_df = eval_performance_for_threshold(perf_df, threshold)
        results.append(result_df)

        # Perform checks for each noise-threshold combination
        unique_noises = result_df['noise'].unique()
        for noise in unique_noises:
            conditions = check_familiar_vs_unfamiliar(result_df, noise, threshold)
            conditions['noise'] = noise
            conditions['threshold'] = threshold
            conditions_results.append(conditions)

    combined_results = pd.concat(results, ignore_index=True)
    # Optionally, save or return the conditions results as a DataFrame
    conditions_df = pd.DataFrame(conditions_results)
    conditions_df.to_csv(os.path.join(export_path, "familiar_vs_unfamiliar_conditions.csv"), index=False)

    return combined_results

def check_familiar_vs_unfamiliar(performance_df, noise, threshold):
    """
    Checks whether for the given noise and threshold:
    - In the person task, 'familiar' scores better than 'unfamiliar'.
    - In the identity task, 'familiar' scores worse than 'unfamiliar'.
    
    Args:
        performance_df (pd.DataFrame): The dataframe containing performance results.
        noise (float): The noise value to filter the dataframe.
        threshold (float): The threshold value to filter the dataframe.
    
    Returns:
        dict: A dictionary indicating if conditions are met for each task.
    """
    # Filter dataframe for the specific noise and threshold
    subset = performance_df[(performance_df['noise'] == noise) & (performance_df['threshold'] == threshold)]
    
    results = {}
    
    # Group by familiarity and calculate mean scores
    familiarity_grouped = subset.groupby('familiarity')[['picture_task_is_correct', 'id_task_is_correct']].mean()
    
    if 'familiar' in familiarity_grouped.index and 'unfamiliar' in familiarity_grouped.index:
        # Check conditions for the person task
        person_task_condition = familiarity_grouped.loc['familiar', 'picture_task_is_correct'] > \
                                familiarity_grouped.loc['unfamiliar', 'picture_task_is_correct']
        results['person_task_condition'] = person_task_condition

        # Check conditions for the identity task
        identity_task_condition = familiarity_grouped.loc['familiar', 'id_task_is_correct'] < \
                                  familiarity_grouped.loc['unfamiliar', 'id_task_is_correct']
        results['identity_task_condition'] = identity_task_condition
    else:
        # If either 'familiar' or 'unfamiliar' group is missing
        results['person_task_condition'] = None
        results['identity_task_condition'] = None

    return results


### One Function To Rule Them All ###

def run_all_for_model(model, base_dir, sub_folders, long_term_memory, noise_constants, thresholds, export_path):
    grouped_images_df = process_folders(base_dir, sub_folders)
    all_noises_df = simulate_different_noises(model, long_term_memory, grouped_images_df, noise_constants, export_path)
    noises_thresholds_combined_df = evaluate_multiple_thresholds(all_noises_df, thresholds, export_path)
    
    n_min = min(noise_constants)
    n_max = max(noise_constants)
    t_min = min(thresholds)
    t_max = max(thresholds)
    file_name = f'{model.model_name}_threshold={t_min}-{t_max}_noise={n_min}-{n_max}_results.csv'
    noises_thresholds_combined_df.to_csv(os.path.join(export_path, file_name))

    return noises_thresholds_combined_df