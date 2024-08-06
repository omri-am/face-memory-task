from faceMemoryTask import *
import sys
import os
import torch.multiprocessing as mp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgg16Model import *

def run_performance(noise_constants, thresholds):
    trained = Vgg16Model(name="trained", 
                         weights_path=f"{os.getcwd()}/models/finetunedC_model_9.pth", 
                         extract_layer=34)
    pretrained = Vgg16Model(name="pretrained", 
                            weights_path=f"{os.getcwd()}/models/face_trained_vgg16_119.pth",
                            extract_layer=34)

    if num_gpus > 1:
        trained = nn.DataParallel(trained)
        pretrained = nn.DataParallel(pretrained)

    base_dir = f"{os.getcwd()}/cogn_test_datasets/phase_perc_size/"

    sub_folders = [
        "pretraining_fixed_C_{_train_3A 220, _val_3A 50, _test_3A 50}",
        "pretraining_fixed_A_{_train_3A 220, _val_3A 50, _test_3A 50}"]

    trained_long_term_memory = load_long_term_memory(f"{os.getcwd()}/code results/trained_longTermMemory.pth")
    pretrained_long_term_memory = load_long_term_memory(f"{os.getcwd()}/code results/pretrained_longTermMemory.pth")
    # trained_long_term_memory = create_long_term_memory(trained, base_dir, sub_folders[0], f"{os.getcwd()}/code results/trained_longTermMemory.pth")
    # pretrained_long_term_memory = create_long_term_memory(pretrained, base_dir, sub_folders[0], f"{os.getcwd()}/code results/pretrained_longTermMemory.pth")

    trained_results_df = run_all_for_model(trained, base_dir, sub_folders, trained_long_term_memory, noise_constants, thresholds)
    pretrained_results_df = run_all_for_model(pretrained, base_dir, sub_folders, pretrained_long_term_memory, noise_constants, thresholds)

    trained_results_df.to_csv(f"{os.getcwd()}/code results/trained-{date.today()}/trained_results_df.csv")
    pretrained_results_df.to_csv(f"{os.getcwd()}/code results/pretrained-{date.today()}/pretrained_results_df.csv")

    return trained_results_df, pretrained_results_df

def run_ploting(trained_results_df, pretrained_results_df, noise_constants, thresholds):

    # Model C
    modelC_img_task = apply_calc(trained_results_df, "img_task_is_correct", noise_constants, thresholds)
    # modelC_stm_id_task = apply_calc(trained_results_df, "stm_id_task_is_correct", noise_constants, thresholds)
    # modelC_ltm_id_task = apply_calc(trained_results_df, "ltm_id_task_is_correct", noise_constants, thresholds)
    modelC_id_task = apply_calc(trained_results_df, "id_task_is_correct", noise_constants, thresholds)

    # Pretrained Model
    pretrainedModel_img_task = apply_calc(pretrained_results_df, "img_task_is_correct", noise_constants, thresholds)
    # pretrainedModel_stm_id_task = apply_calc(pretrained_results_df, "stm_id_task_is_correct", noise_constants, thresholds)
    # pretrainedModel_ltm_id_task = apply_calc(pretrained_results_df, "ltm_id_task_is_correct", noise_constants, thresholds)
    pretrainedModel_id_task = apply_calc(pretrained_results_df, "id_task_is_correct", noise_constants, thresholds)

    all_tasks_dfs = {"Trained (Model C) - Image Task": modelC_img_task,
            # "Trained (Model C) - Short Term Memory, Identity Task": modelC_stm_id_task,
            # "Trained (Model C) - Long Term Memory, Identity Task": modelC_ltm_id_task,
            "Trained (Model C) - Identity Task": modelC_id_task,
            "Pretrained - Image Task": pretrainedModel_img_task,
            # "Pretrained - Short Term Memory, Identity Task": pretrainedModel_stm_id_task,
            # "Pretrained - Long Term Memory, Identity Task": pretrainedModel_ltm_id_task,
            "Pretrained - Identity Task": pretrainedModel_id_task}
    
    for title, df in all_tasks_dfs.items():
        plot_results(df, title, noise_constants, thresholds) 


def main():
    noise_constants = [x / 100.0 for x in range(20, 51, 2)]
    thresholds = [x / 100.0 for x in range(30, 61, 2)]

    trained_results_df, pretrained_results_df = run_performance(noise_constants, thresholds)
    run_ploting(trained_results_df, pretrained_results_df, noise_constants, thresholds)

if __name__ == "__main__":
    main()