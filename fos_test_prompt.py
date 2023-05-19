""" Prompt Testing for FosGPT
- This test how responsive the model is to prompts

TODO:

[] Use multiple GPUs
[] Add more prompts
[] Add more metrics
[] Generalize or take prompter as input
[] Use fire to make this a command line tool

"""

import torch
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize


# Flower Domains
from mario_gpt.flower_level import FLOWER_LEVEL
from mario_gpt.flower_metric import count_flowers, calculate_crookedness_score
from mario_gpt.flower_dataset import MarioDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tile_gen.fos.fos_wfc import load_wfc
from tile_gen.fos.converter import Converter
from mario_gpt.fos_prompter import FosPrompter

import json
import fire

# -- Utils ---------------------------------------------------------------- -- #

def generate_level(mario_lm, prompts, height, img_length):
    generated_level = mario_lm.sample(
        prompts=prompts,
        num_steps=(height*img_length),
        temperature=1.0,
        use_tqdm=True
    )
    layout = generated_level.level_tensor.reshape((height, img_length))
    layout[0][0] = 40 # set the starting tile to be the empty tile TODO: do this in sampling
    return layout

# Convert numpy datatypes into native Python types
def convert_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(x) for x in obj]
    return obj

def test_prompt(model_path="FosGPT_wfc1", n_iterations=16):
    # -- Load WFC --------------------------------------------------------- -- #
    # WFC
    config_file= '/home/gaiera/Code/NZB/tile_gen/config/wfc.yaml'
    express, evaluate, p = load_wfc(config_file)
    height, img_length = p['dim']

    # Converter
    converter = Converter(load_path='/home/gaiera/Code/NZB/tile_gen/data/converter.json')

    # Model
    model_path = "FosGPT_wfc1"
    prompter = FosPrompter()
    mario_lm = MarioLM(lm_path=model_path, tokenizer_path=model_path, prompter=prompter, context_len=980)
    device = torch.device('cuda')
    mario_lm = mario_lm.to(device)


    # -- Prep Prompt Tests ---------------------------------------------------- -- #
    # Prompts
    prompt_open_area = ["no", "a little", "some", "a lot of"]
    prompt_clearance = ["no", "a little", "some", "a lot of"]
    #prompt_open_area = ["no", "a little", "some", "a lot of"]
    #prompt_clearance = ["no"]

    # Initialize dictionaries to store scores for each iteration
    open_area_scores_iter = {prompt: [] for prompt in prompt_open_area}
    clearance_scores_iter = {prompt: [] for prompt in prompt_clearance}
    # Initialize dictionary to store layouts for each combination of prompts
    named_layouts_iter = {f"{prompt_o} open_area, {prompt_c} clearance": [] for prompt_o in prompt_open_area for prompt_c in prompt_clearance}


    total_iterations = len(prompt_open_area) * len(prompt_clearance) * n_iterations
    pbar = tqdm(total=total_iterations)

    # -- Generate test layouts ------------------------------------------ ------ -- #
    # Generate and Test
    for prompt_o in prompt_open_area:
        for prompt_c in prompt_clearance:
            
            for _ in range(n_iterations):
                # Generate
                prompts = [f"{prompt_o} open_area, {prompt_c} clearance"]
                generated_layout = generate_level(mario_lm, prompts, height=height, img_length=img_length)

                # Test
                named_layout = converter.id2named(generated_layout)
                hash_layout = converter.convert('named', 'hash', named_layout)
                performance = evaluate(hash_layout)
                                    
                # Append scores to lists
                open_area_scores_iter[prompt_o].append(performance['area_open'])
                clearance_scores_iter[prompt_c].append(performance['clearance'])
                
                # Append layout to list
                named_layouts_iter[f"{prompt_o} open_area, {prompt_c} clearance"].append(named_layout)
                #named_layouts_iter = {f"{prompt_o} open_area, {prompt_c} clearance": [] for prompt_o in prompt_open_area for prompt_c in prompt_clearance}

                
                pbar.update(1)
    pbar.close()

    # Convert the scores and layouts into a dictionary
    results_dict = {
        "open_area_scores": convert_types(open_area_scores_iter),
        "clearance_scores": convert_types(clearance_scores_iter),
        "named_layouts": convert_types(named_layouts_iter),
    }

    # Save scores and layouts to a single JSON file for later analysis
    with open("test_prompt.json", "w") as f:
        json.dump(results_dict, f)




# Fire up the main function
if __name__ == '__main__':
    fire.Fire(test_prompt)

# # Display Box Plots
# fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# # Flower Scores
# flower_data = [np.array(flower_scores_iter)[j, :, :].flatten() for j in range(n_cols)]
# bp = ax[0].boxplot(flower_data)
# ax[0].set_title("Number of Flowers")
# ax[0].set_xlabel("Straightness")
# ax[0].set_xticks(range(1, n_cols + 1))
# ax[0].set_xticklabels(prompt_flower)
# ax[0].set_ylabel("Flowers")

# # Straightness Scores
# straight_data = [np.array(straight_scores_iter)[:, j, :].flatten() for j in range(n_cols)]
# bp = ax[1].boxplot(straight_data)
# ax[1].set_title("Straightness")
# ax[1].set_xlabel("Straightness")
# ax[1].set_xticks(range(1, n_cols + 1))
# ax[1].set_xticklabels(prompt_straight)
# ax[1].set_ylabel("Straightness Score")

# plt.savefig("test_prompt.png")