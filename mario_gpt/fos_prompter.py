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
    "open area": np.array([169.0, 197.0, 218.0]),
    "clearance": np.array([825.0, 929.0, 1054.0]),
}

FEATURE_EXTRACTION_MODEL = "facebook/bart-base"


class FosPrompter:
    def __init__(
        self,
        performance_lookup,
        prompter_model: str = FEATURE_EXTRACTION_MODEL,
        statistics: Optional[Dict[str, Any]] = None,
    ):
        self.prompter_model = prompter_model
        self.feature_extraction = pipeline(
            "feature-extraction",
            model=prompter_model,
            tokenizer=prompter_model,
            framework="pt",
        )
        self.statistics = statistics
        if statistics is None:
            self.statistics = STATISTICS
       
        self.performance_lookup = performance_lookup

        #self.token_dict = None
        #self.token_dict2 = self.level_tokenizer.get_vocab()

    @property
    def open_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["open area"]
        keywords = ["no", "a little", "some", "a lot of"]
        return thresholds, keywords
    
    @property
    def clearance_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["straight"]
        keywords = ["no", "a little", "some", "a lot of"]
        return thresholds, keywords
    
    def area_open_prompt(self, idx) -> str:
        value = self.performance_lookup['area_open'][idx]
        thresholds, keywords = self.open_thresholds
        threshold = np.digitize(value, thresholds, right=True)
        keyword = keywords[threshold]
        return f"{keyword} open area", keyword

    def clearance_prompt(self, idx) -> str:
        value = self.performance_lookup['clearance'][idx]
        thresholds, keywords = self.clearance_thresholds
        threshold = np.digitize(value, thresholds, right=True)
        keyword = keywords[threshold]
        return f"{keyword} clearance", keyword        

    def output_hidden(self, prompt: str, device: torch.device = torch.device("cpu")):
        # Reducing along the first dimension to get a 768 dimensional array
        return (
            self.feature_extraction(prompt, return_tensors="pt")[0]
            .mean(0)
            .to(device)
            .view(1, -1)
        )


    def __call__(
        self, idx, sample_prompt: bool = False
    ) -> Union[str, torch.Tensor]:
        device: torch.device = torch.device("cpu")
        if not sample_prompt:
            if idx is None:
                raise ValueError("Level must be provided if sample_prompt is not true!")
            area_open_prompt, _ = self.area_open_prompt(idx)
            clearance_prompt, _ = self.area_open_prompt(idx)

        else:
            area_open_prompt = random.choice(["no", "a little", "some", "a lot of"]) + " open area"
            clearance_prompt = random.choice(["no", "a little", "some", "a lot of"]) + " clearance"

        prompt_dict = {
            "area_open": area_open_prompt,
            "clearance": clearance_prompt,
        }
        prompt = f"{area_open_prompt}, {clearance_prompt}"
        hidden = self.output_hidden(prompt, device=device)
        return prompt, hidden, prompt_dict
    