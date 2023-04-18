import torch
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize
from mario_gpt.flower_level import FLOWER_LEVEL


import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

img_length = 16

# Load Model and Dataset
mario_lm = MarioLM(lm_path="FlowerGPT", tokenizer_path="FlowerGPT")
dataset = MarioDataset(mario_lm.tokenizer, level_string=FLOWER_LEVEL) # for token conversion

# View dataset statistics / test prompter
prompter = mario_lm.prompter
d = prompter.dataset_statistics(dataset)
print(d)

fig, ax = plt.subplots()
prompter.vis_hexbin(d, ax=ax)
# save figure to file
fig.savefig('out.png', dpi=300)




print("Done")
