import torch
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize
from mario_gpt.flower_level import FLOWER_LEVEL

from mario_gpt.flower_metric import count_flowers, calculate_crookedness_score


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


n_iterations = 2
img_length = 5  # Set this to the desired image length

### --- ###

def token_to_rgb(token_array, token_dict, colors=None):
    # Create a reverse dictionary mapping token values to keys
    reverse_token_dict = {int(v.item()): k for k, v in token_dict.items()}

    # Use ascending integers as colors if not provided
    if colors is None:
        colors = [
            [0, 170, 0],       
            [185, 122, 87],      
            [255, 242, 0],     
            [191, 232, 242],   
        ]        

    # Create a dictionary mapping token values to colors
    color_dict = {token_value: color for token_value, color in zip(reverse_token_dict.keys(), colors)}

    # Function to map token values to colors or red if not in token_dict
    def map_to_color(token_value):
        return color_dict.get(token_value, [255, 0, 0])

    # Create a 3D array of RGB colors based on the token_array and color_dict using nested list comprehension
    rgb_array = np.array([[[map_to_color(token_value) for token_value in row] for row in token_array]])

    return rgb_array

def generated_to_rgb(generated_level):
    A = generated_level.level_tensor
    rot_img = np.rot90(A.reshape(img_length,14))
    return token_to_rgb(rot_img, dataset.token_dict)[0]

# -- Load model
mario_lm = MarioLM(lm_path="FlowerGPT", tokenizer_path="FlowerGPT")
dataset = MarioDataset(mario_lm.tokenizer, level_string=FLOWER_LEVEL) # for token conversion

# -- Generate test levels
# Prompts
prompt_flower = ["no", "few", "some", "many"]
prompt_straight = ["not", "kinda", "very", "totally"]

n_rows = len(prompt_flower)
n_cols = len(prompt_straight)

# Initialize lists to store scores for each iteration
flower_scores_iter = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
straight_scores_iter = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

total_iterations = n_rows * n_cols * n_iterations
pbar = tqdm(total=total_iterations)
# Generate and Test
for i in range(n_rows):
    for j in range(n_cols):
        flower_scores_iter[i][j] = []
        straight_scores_iter[i][j] = []
        
        for _ in range(n_iterations):
            # Generate
            prompts = [f"{prompt_flower[i]} flowers, {prompt_straight[j]} straight"]
            generated_level = mario_lm.sample(
                prompts=prompts,
                num_steps=14*img_length,
                temperature=1.0,
                use_tqdm=False
            )
            # Test
            image = generated_to_rgb(generated_level)
            n_flowers = count_flowers(image)
            straightness = calculate_crookedness_score(image)

            # Append scores to lists
            flower_scores_iter[i][j].append(n_flowers)
            straight_scores_iter[i][j].append(straightness)
            
            pbar.update(1)
pbar.close()

# Save score lists to single pickle file for later analysis
import pickle
with open("test_prompt.pkl", "wb") as f:
    pickle.dump((flower_scores_iter, straight_scores_iter), f)



# Display Box Plots
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# Flower Scores
flower_data = [np.array(flower_scores_iter)[:, j, :].flatten() for j in range(n_cols)]
bp = ax[0].boxplot(flower_data)
ax[0].set_title("Number of Flowers")
ax[0].set_xlabel("Straightness")
ax[0].set_xticks(range(1, n_cols + 1))
ax[0].set_xticklabels(prompt_flower)
ax[0].set_ylabel("Flowers")

# Straightness Scores
straight_data = [np.array(straight_scores_iter)[:, j, :].flatten() for j in range(n_cols)]
bp = ax[1].boxplot(straight_data)
ax[1].set_title("Straightness")
ax[1].set_xlabel("Straightness")
ax[1].set_xticks(range(1, n_cols + 1))
ax[1].set_xticklabels(prompt_straight)
ax[1].set_ylabel("Straightness Score")

plt.savefig("test_prompt.png")