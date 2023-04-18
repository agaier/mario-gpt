import torch
from mario_gpt import MarioDataset, MarioLM
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize

# Load Model
mario_lm = MarioLM()

# Set Device
device = torch.device('cuda')
mario_lm = mario_lm.to(device)


# Generate Levels
prompts = ["many pipes, many enemies, some blocks, high elevation"]

generated_level = mario_lm.sample(
    prompts=prompts,
    num_steps=140,
    temperature=2.0,
    use_tqdm=True,
    height=4
)

# View Tiles
A = generated_level.level_tensor
txt = view_level(A, mario_lm.tokenizer)
output = '\n'.join(txt)
print(output)

print("Done")