import torch
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize
from mario_gpt.flower_level import FLOWER_LEVEL
from mario_gpt.flower_dataset import MarioDataset

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection  # Add this import

import numpy as np
import matplotlib.pyplot as plt

import json
import pickle

# Visualization
def plot_histograms(data, bin_borders):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    prepared_data = prepare_data(data, bin_borders)
    for i, (category, values) in enumerate(prepared_data.items()):
        plot_histogram(axes[i], category, values)
    plt.show()

def prepare_data(data, bin_borders):
    prepared_data = {}
    for category, bins in bin_borders.items():
        extended_bins = np.concatenate(([bins[0] - 1], bins, [bins[-1] + 1]))
        counts, _ = np.histogram(data[category], bins=extended_bins)
        prepared_data[category] = counts
    return prepared_data

def plot_histogram(ax, category, values):
    ax.bar(range(len(values)), values)
    ax.set_title(f"{category.capitalize()} Histogram")
    ax.set_xlabel("Bins")
    ax.set_ylabel("Counts")
    ax.set_xticks(range(len(values)))

def plot_hexbin(df, ax=None):
    # Create the hexbin plot with Blue color scale and outlined hexagons
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.get_figure()
    
    gridsize = (7,7)
    hb = ax.hexbin(df['straight_count'], df['flower_count'], gridsize=gridsize, cmap='cividis', edgecolor='gray', linewidth=0.5)
    ax.set_xlabel('Straightness')
    ax.set_ylabel('Flower Count')
    cb = fig.colorbar(hb)
    cb.set_label('Count')

    # Add hatching to empty hexagons
    patches = []
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            if hb.get_array()[i*gridsize[1] + j] == 0:
                patches.append(Polygon(hb.get_offsets()[i*10 + j] + hb.get_paths()[0].vertices, True))
    p = PatchCollection(patches, facecolor='white', edgecolor='gray', hatch='///', linewidth=0.5)
    ax.add_collection(p)


# Get data
new_run = False
if new_run:
    #Load Model and Dataset
    mario_lm = MarioLM(lm_path="FlowerGPT", tokenizer_path="FlowerGPT")
    dataset = MarioDataset(mario_lm.tokenizer, level_string='flowers_dataset.json')

    # View dataset statistics / test prompter
    prompter = mario_lm.prompter
    data = prompter.dataset_statistics(dataset)

    # Save data to pkl file
    with open('data.pkl', 'wb') as fp:
        pickle.dump(data, fp)
else:

    # Load data from pkl file
    with open('data.pkl', 'rb') as fp:
        data = pickle.load(fp)

# Visualize
bin_borders = {'flower_count': np.array([3., 5., 7.]), 'straight_count': np.array([240., 250., 260.])}
plot_histograms(data, bin_borders)
plt.savefig('threshold_histogram.png', dpi=300)


#print(d)

# Save result to file





# Visualize


fig, ax = plt.subplots()
plot_hexbin(data, ax=ax)
# # save figure to file
fig.savefig('hexbin_histogram.png', dpi=300)



print("Done")
