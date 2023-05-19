import torch
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize
from mario_gpt.fos_dataset import FosDataset
from mario_gpt.fos_prompter import FosPrompter
import json

BASE = "distilgpt2"
NAME = "FosGPT_wfc1"

# Load Performance into FosPrompter
datset_path = '/home/gaiera/Code/NZB/tile_gen/data/fos_wfc.json'
with open(datset_path, 'r') as f:
     raw = json.load(f)
     performance_lookup = raw['performance']
prompter = FosPrompter(performance_lookup=performance_lookup)

# Init GPT Model
mario_lm = MarioLM(lm_path=BASE, tokenizer_path=BASE, prompter=prompter, context_len=980)

# Load Dataset
dataset = FosDataset(mario_lm.tokenizer, level_string=datset_path)
print("Dataset length:", len(dataset)) 

# Setup Training
config = TrainingConfig(save_iteration=10000, output_dir=NAME)
trainer = MarioGPTTrainer(mario_lm, dataset, config)

# Train
n_train = 20000
trainer.train(n_train, batch_size=1)

# Save Model and Tokenizer
mario_lm.lm.save_pretrained(NAME)
dataset.tokenizer.save_pretrained(NAME)

print("done")