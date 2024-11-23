from faceMemoryTask import *
from plots import *
import sys
import os
from datetime import date

benchmark_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ng_benchmark'))
sys.path.insert(0, benchmark_utils_path)
from models import Vgg16Model

def expand_row(row):
    tasks = ['picture_task', 'id_task']
    task_correct_columns = ['picture_task_is_correct', 'id_task_is_correct']
    
    expanded_rows = [
        {**row, 'task': task, 'task_correct': row[task_correct_col]}
        for task, task_correct_col in zip(tasks, task_correct_columns)
    ]
    
    return pd.DataFrame(expanded_rows)

def load_or_create_long_term_memory(model, export_path, base_dir, sub_folder):
    ltm_path = os.path.join(export_path, f'{model.model_name}_longTermMemory.pth')
    if os.path.exists(ltm_path):
        identities_memory = torch.load(ltm_path)
        return identities_memory
    else:
        return create_long_term_memory(model, base_dir, sub_folder, export_path)

def load_or_create_model_performance_file(model, base_dir, sub_folders, ltm, noises, thresholds, model_export_path):
    n_min = min(noises)
    n_max = max(noises)
    t_min = min(thresholds)
    t_max = max(thresholds)
    performance_path = os.path.join(model_export_path, f'{model.model_name}_noise={n_min}-{n_max}_performance.csv')
    if os.path.exists(performance_path):
        print("found performance file, reading...")
        df = pd.read_csv(performance_path)
        res_df = evaluate_multiple_thresholds(df, thresholds, model_export_path)
        file_name = f'{model.model_name}_threshold={t_min}-{t_max}_noise={n_min}-{n_max}_results.csv'
        res_df.to_csv(os.path.join(model_export_path, file_name))
        return res_df
    else:
        print("no performance file found")
        df = run_all_for_model(model, base_dir, sub_folders, ltm, noises, thresholds, model_export_path)
        return df

def run_model_performance(model, base_dir, sub_folders, long_term_memory, noise_constants, thresholds, export_path):
    return run_all_for_model(model, base_dir, sub_folders, long_term_memory, noise_constants, thresholds, export_path)

def save_results(df, model_name, thresholds, noise_constants, model_export_path):
    t_min = min(thresholds)
    t_max = max(thresholds)
    n_min = min(noise_constants)
    n_max = max(noise_constants)

    df.to_csv(f'{model_export_path}/{model_name}_threshold={t_min}-{t_max}_noise={n_min}-{n_max}_results.csv')
    return df

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

def save_combined(model_df_dict, export_path):
    for model_name, model_df in model_df_dict.items():
        model_df['Model-Layer'] = model_name
    
    combined_df = pd.concat(list(model_df_dict.values()), ignore_index=True)
    all_models_names = '-'.join(model_df_dict.keys())
    combined_df.to_csv(os.path.join(export_path, f'combined_{all_models_names}.csv'))

def run_models(models, noises, thresholds, export_path, plot = True):

    model_df_dict = {}
    for model in models:
        base_dir = '/home/new_storage/experiments/face_memory_task/cogn_test_datasets/phase_perc_size/'
        results_path = '/home/new_storage/experiments/face_memory_task/code results/'
        sub_folders = [
            'pretraining_fixed_C_{_train_3A 220, _val_3A 50, _test_3A 50}',
            'pretraining_fixed_A_{_train_3A 220, _val_3A 50, _test_3A 50}'
        ]
        model_export_path = os.path.join(export_path, model.model_name)
        os.makedirs(model_export_path, exist_ok=True)

        ltm = load_or_create_long_term_memory(model, results_path, base_dir, sub_folders[0])
        results_df = load_or_create_model_performance_file(model, base_dir, sub_folders, ltm, noises, thresholds, model_export_path)
        
        model_df_dict[model.model_name] = results_df

        if plot:
            list_pairs = [(n, t) for n in noises for t in thresholds]
            plot_task_performance_by_conditions(model_df_dict[model.model_name], model.model_name, list_pairs, model_export_path)
    
    save_combined(model_df_dict, export_path)
    if plot:
        plot_task_performance_graph(model_df_dict, noises, thresholds, export_path)
        plot_tasks_comparison(model_df_dict, noises, thresholds, export_path)

def main():
    dir = f'{os.getcwd()}/code results/{date.today()}/'
    os.makedirs(dir, exist_ok=True)

    noises = [x / 200.0 for x in range(0, 201, 5)]
    thresholds = [x / 200.0 for x in range(0, 201, 5)]

    modelC_conv5 = Vgg16Model(model_name='modelC_conv5', 
                       weights_file_path='/home/new_storage/experiments/face_memory_task/models/finetunedC_model_9.pth', 
                       layers_to_extract='avgpool')
    modelC_fc7 = Vgg16Model(model_name='modelC_fc7', 
                       weights_file_path='/home/new_storage/experiments/face_memory_task/models/finetunedC_model_9.pth', 
                       layers_to_extract='classifier.5')
    modelB_conv5 = Vgg16Model(model_name='modelB_conv5', 
                       weights_file_path='/home/new_storage/experiments/face_memory_task/models/finetunedB_model_9.pth', 
                       layers_to_extract='avgpool')
    modelB_fc7 = Vgg16Model(model_name='modelB_fc7', 
                       weights_file_path='/home/new_storage/experiments/face_memory_task/models/finetunedB_model_9.pth', 
                       layers_to_extract='classifier.5')

    run_models(models = [
                        modelC_conv5, 
                        modelC_fc7, 
                        modelB_conv5,
                        modelB_fc7
                        ],
               noises = noises,
               thresholds = thresholds,
               export_path = dir,
               plot = False
               )

if __name__ == '__main__':
    main()