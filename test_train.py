import torch
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize
from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS

# Flower Domain
from mario_gpt.flower_level import FLOWER_LEVEL
from mario_gpt.flower_prompter import FlowerPrompter
from mario_gpt.flower_dataset import MarioDataset


BASE = "distilgpt2"
NAME = "FlowerGPT_wfc2"

mario_lm = MarioLM(lm_path=BASE, tokenizer_path=BASE)
#mario_lm = MarioLM(lm_path=BASE, tokenizer_path=BASE)

# Load Dataset
dataset = MarioDataset(mario_lm.tokenizer, level_string='flowers_dataset.json')

# Explicitly add dict for token dict from dataset to mario_lm tokenizer
mario_lm.prompter.token_dict = dataset.token_dict


print("Dataset length:", len(dataset)) 

# Setup Training
config = TrainingConfig(save_iteration=500, output_dir=NAME)
trainer = MarioGPTTrainer(mario_lm, dataset, config)

# Train
n_train = 5000
trainer.train(n_train, batch_size=1)

# Save Model and Tokenizer
mario_lm.lm.save_pretrained(NAME)
dataset.tokenizer.save_pretrained(NAME)


#---
# load model and generate levels
#---
# mario_lm = MarioLM(lm_path="FlowerGPT/iteration_9", tokenizer_path="FlowerGPT")
# prompts = [" "]
# generated_level = mario_lm.sample(
#     prompts=prompts,
#     num_steps=140,
#     temperature=2.0,
#     use_tqdm=True
# )
# A = generated_level.level_tensor
# view_level(A, mario_lm.tokenizer)
print("done")