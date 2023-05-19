from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats
from transformers import pipeline

from mario_gpt.dataset import MarioDataset
from mario_gpt.utils import view_level

from mario_gpt.flower_metric import count_flowers, calculate_crookedness_score, map_array_to_colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection  # Add this import

def view_flowers(level: str):
    plt.imshow(level)
    plt.savefig("out.png")
    plt.show()

# Threshold values
STATISTICS = {
    "flowers": np.array([3.0, 5.0, 7.0]),
    "straight": np.array([240.0, 250.0, 260.0]),
}

FEATURE_EXTRACTION_MODEL = "facebook/bart-base"


class FlowerPrompter:
    def __init__(
        self,
        level_tokenizer,
        prompter_model: str = FEATURE_EXTRACTION_MODEL,
        use_raw_counts: bool = False,
        statistics: Optional[Dict[str, Any]] = None,
        height: int = 14, # ! added height
    ):
        self.prompter_model = prompter_model
        self.feature_extraction = pipeline(
            "feature-extraction",
            model=prompter_model,
            tokenizer=prompter_model,
            framework="pt",
        )
        self.level_tokenizer = level_tokenizer
        self.use_raw_counts = use_raw_counts
        self.statistics = statistics
        if statistics is None:
            self.statistics = STATISTICS
       
        self.height = height
        #self.token_dict = None
        #self.token_dict2 = self.level_tokenizer.get_vocab()

    @property
    def flower_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["flowers"]
        keywords = ["no", "few", "some", "many"]
        return thresholds, keywords
    
    @property
    def straight_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["straight"]
        keywords = ["not", "kinda", "very", "totally"]
        return thresholds, keywords
    
    def count_flowers(self, flattened_level: str) -> int:
        # Convert to 2D level
        level = np.array(list(flattened_level)).reshape(self.height, -1)
        rgb_level = map_array_to_colors(level)
        
        return count_flowers(rgb_level)
    
    def get_straightness(self, flattened_level: str) -> int:
        # Convert to 2D level
        level = np.array(list(flattened_level)).reshape(self.height, -1)
        rgb_level = map_array_to_colors(level)        
        return calculate_crookedness_score(rgb_level)

    def _flatten_level(self, string_level: List[str]) -> str:
        return "".join(string_level)

    def flowers_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_flowers(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.flower_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} flowers", keyword
    
    def straight_prompt(self, flattened_level: str, level: str) -> str:
        count = self.get_straightness(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.straight_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} straightness", keyword
    
    def output_hidden(self, prompt: str, device: torch.device = torch.device("cpu")):
        # Reducing along the first dimension to get a 768 dimensional array
        return (
            self.feature_extraction(prompt, return_tensors="pt")[0]
            .mean(0)
            .to(device)
            .view(1, -1)
        )

    def dataset_statistics(self, dataset: MarioDataset):
        flower_counts = []
        straight_counts = []
        for i in range(len(dataset)):
            level, _ = dataset[i]
            # Level is right side up flowers
            # -- is expecting the flattened level version tilted
            #  plt.imshow(level.reshape(50,14));plt.savefig('out.png') <-- should be turned clock 90 degrees
            str_level = self._flatten_level(view_level(level, dataset.tokenizer))

            flower_count = self.count_flowers(str_level)
            straight_count = self.get_straightness(str_level)

            flower_counts.append(flower_count)
            straight_counts.append(straight_count)
        d = {"flower": {}, "straight": {}}

        # Summary stats for thresholds
        d["flower"] = stats.mstats.mquantiles(flower_counts, [0.33, 0.66, 0.95])
        d["straight"] = stats.mstats.mquantiles(straight_counts, [0.33, 0.66, 0.95])

        # All stats for analysis
        d["flower_count"] = flower_counts
        d["straight_count"] = straight_counts
        
        return d

    # Hexbin histogram of flower counts and straightness
    def vis_hexbin(self, df, ax=None):
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

    def __call__(
        self, level: torch.Tensor = None, sample_prompt: bool = False
    ) -> Union[str, torch.Tensor]:
        device: torch.device = torch.device("cpu")
        if not sample_prompt:
            if level is None:
                raise ValueError("Level must be provided if sample_prompt is not true!")
            str_level = view_level(level, self.level_tokenizer)
            flattened_level = self._flatten_level(str_level)

            flowers_prompt, _ = self.flowers_prompt(flattened_level, str_level)
            straight_prompt, _ = self.straight_prompt(flattened_level, str_level)

            device = level.device
        else:
            str_level = None
            flowers_prompt = random.choice(["no", "few", "some", "many"]) + " flowers"
            straight_prompt = random.choice(["not", "kinda", "very", "totally"]) + " straight"

        prompt_dict = {
            "flowers": flowers_prompt,
            "straight": straight_prompt,
        }
        prompt = f"{flowers_prompt}, {straight_prompt}"
        hidden = self.output_hidden(prompt, device=device)
        return prompt, hidden, prompt_dict, str_level
    