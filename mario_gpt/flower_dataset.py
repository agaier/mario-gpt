from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS
import json

DEFAULT_MODEL = "distilgpt2"


def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))


def flip_and_transpose(arr: np.array, flip_first: bool = False):
    if arr.shape[-1] > 1:
        if flip_first:
            return np.flip(arr, -1).transpose()
        return np.flip(arr.transpose(), -1)
    return arr

def flip_and_transpose(arr: np.array, flip_first: bool = False):
    if arr.shape[-1] > 1:
        if flip_first:
            return np.flip(arr, -1).transpose()
        return np.flip(arr.transpose(), -1)
    return arr

def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]


def characterize(str_lists):
    return [list(s) for s in str_lists]


class MarioDataset(Dataset):
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        level_string: Optional[str] = None,
        context_len: int = 700,
        height: int = 14,
        remove_start_end_tokens: bool = False,
        sample_all_indices: bool = False,
    ):
        # Import level string from json
        with open(level_string) as json_file:
            level_list = json.load(json_file)
            level_string = ''.join(level_list)


        self.character_set = set(level_string)
        if "\n" in self.character_set:
            self.character_set.remove("\n")
        self.vocab_size = len(self.character_set)
        self.sample_all_indices = sample_all_indices

        # -- Train Tokenizer --
        def get_training_corpus():
            yield list(level_string)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

        self.tokenizer = tokenizer
        if getattr(tokenizer, "train_new_from_iterator", None) is not None:
            self.tokenizer = self.tokenizer.train_new_from_iterator(
                get_training_corpus(), 52000
            )
        elif getattr(tokenizer, "train_from_iterator", None) is not None:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
            self.tokenizer = self.tokenizer.train_new_from_iterator(
                get_training_corpus(), self.vocab_size
            )
        self.context_len = context_len
        self.height = height

        # -- Convert Level to Tensors --
        """ Leftover from original code all of the tokens are used to 
        construct the token dictionary"""
        x, self.str_arr = self.convert_level_to_tensor(level_string.split("\n"))
        self.input_ids = x["input_ids"].squeeze()
        self.unique_tokens, self.unique_counts = self.input_ids.unique(
            return_counts=True
        )
        self.weighted_unique_counts = (
            1.0 / self.unique_counts / torch.sum(self.unique_counts)
        )

        # -- Convert each level in list to tensor --
        level_tensors = []
        for level in level_list:
            x, str_arr = self.convert_level_to_tensor(level.split("\n")[:-1])
            level_tensor = x["input_ids"].squeeze()
            if remove_start_end_tokens:
                level_tensor = level_tensor[1:-1]
            level_tensors.append(level_tensor)
        self.level_tensors = torch.stack(level_tensors)
        self.attention_masks = x["attention_mask"].squeeze()

        # -- Create token dictionary --
        self.token_dict = {}
        string_tokens = list(self.tokenizer.decode(self.unique_tokens))
        for int_token, string_token in zip(self.unique_tokens, string_tokens):
            self.token_dict[string_token] = int_token


    def convert_level_to_tensor(self, level: List[str]):
        str_arr = flip_and_transpose(np.array(characterize(level)))
        str_arr = "".join(join_list_of_list(str_arr))

        x = self.tokenizer(str_arr, return_tensors="pt")
        return x, str_arr

    def __len__(self):
        return len(self.level_tensors)

    def __getitem__(self, idx):
        return self.level_tensors[idx], self.attention_masks
    
    def generate_indices(self):
        out = []
        for idx in range(self.input_ids.shape[0] - self.context_len):
            if idx % self.height == 0 or self.sample_all_indices:
                arange = torch.arange(idx, idx + self.context_len)
                out.append(arange)
        return torch.stack(out)

    def sample_indices(self, batch_size):
        out = []
        for _ in range(batch_size):
            start_idx = np.random.randint(0, self.__len__() - self.context_len)
            indices = torch.arange(start_idx, start_idx + self.context_len)
            out.append(indices)
        return torch.stack(out)

    def __str__(self):
        str_list = characterize(self.tokenizer.batch_decode(self.x["input_ids"]))
        string = "\n".join(
            join_list_of_list(flip_and_transpose(np.array(str_list), True))
        )
        return string
