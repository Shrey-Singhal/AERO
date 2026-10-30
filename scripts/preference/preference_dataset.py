# preference_dataset.py
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "right", value: int = 0) -> torch.Tensor:
    """
    Function for padding the sequences.
    """
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (0, pad_len) if side == "right" else (pad_len, 0)
        padded.append(F.pad(seq, padding, value=value))
    return torch.stack(padded, dim=0)

@dataclass
class TokenizedPreferenceSample:
    chosen_ids: torch.Tensor
    chosen_mask: torch.Tensor
    rejected_ids: torch.Tensor
    rejected_mask: torch.Tensor
    extras: Dict[str, Any]


class PreferenceDataset(Dataset):
    """
    Handle logic of turning raw dictionary data into tokenized tensors for policy training.
    """

    def __init__(
        self,
        buffer: List[Dict[str, Any]],
        tokenizer: Callable,
        prompt_max_length: int,
        generate_max_length: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_max_length = int(prompt_max_length)
        self.generate_max_length = int(generate_max_length)

        self.too_long_prompt_count = 0
        self.total_getitem_count = 0

        self.tokenized_samples: List[TokenizedPreferenceSample] = []
        for item in buffer:
            self.tokenized_samples.append(self._tokenize_sample(item))

    def __len__(self) -> int:
        return len(self.tokenized_samples)

    def _tokenize_sample(self, sample: Dict[str, Any]) -> TokenizedPreferenceSample:
        self.total_getitem_count += 1

        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]
        reference_response = sample.get("reference_response", "")
        has_reference_response = bool(
            sample.get("has_reference_response", "reference_response" in sample)
        )

        prompt_tok = self.tokenizer(
            prompt,
            max_length=self.prompt_max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        prompt_ids_len = int(prompt_tok["attention_mask"].int().sum().item())

        loss_mask = bool(sample.get("loss_mask", True))
        if prompt_ids_len >= self.prompt_max_length - 2 and loss_mask:
            self.too_long_prompt_count += 1
            logging.warning("Masking samples with too long prompts.")
            loss_mask = True

        max_len = self.prompt_max_length + self.generate_max_length

        # Concat prompt + response (strip trailing newlines for clean tokenization)
        chosen_text = (prompt + chosen).rstrip("\n")
        rejected_text = (prompt + rejected).rstrip("\n")
        reference_text = (
            (prompt + reference_response).rstrip("\n")
            if has_reference_response
            else ""
        )

        # Tokenize without adding special tokens (we'll handle EOS manually)
        chosen_tok = self.tokenizer(
            chosen_text,
            max_length=max_len - 1,  # Reserve space for EOS
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        rejected_tok = self.tokenizer(
            rejected_text,
            max_length=max_len - 1,  # Reserve space for EOS
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        reference_tok = None
        if has_reference_response:
            reference_tok = self.tokenizer(
                reference_text,
                max_length=max_len - 1,  # Reserve space for EOS
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

        # Append EOS token ID directly (works for all tokenizers including Llama)
        eos_id = self.tokenizer.eos_token_id
        chosen_tok["input_ids"] = torch.cat([
            chosen_tok["input_ids"],
            torch.tensor([[eos_id]])
        ], dim=1)
        chosen_tok["attention_mask"] = torch.cat([
            chosen_tok["attention_mask"],
            torch.tensor([[1]])
        ], dim=1)

        rejected_tok["input_ids"] = torch.cat([
            rejected_tok["input_ids"],
            torch.tensor([[eos_id]])
        ], dim=1)
        rejected_tok["attention_mask"] = torch.cat([
            rejected_tok["attention_mask"],
            torch.tensor([[1]])
        ], dim=1)

        if reference_tok is not None:
            reference_tok["input_ids"] = torch.cat([
                reference_tok["input_ids"],
                torch.tensor([[eos_id]])
            ], dim=1)
            reference_tok["attention_mask"] = torch.cat([
                reference_tok["attention_mask"],
                torch.tensor([[1]])
            ], dim=1)

        extras = {
            "prompt_ids_lens": prompt_ids_len,
            "loss_masks": loss_mask,
            "prompt_text": prompt,
            "chosen_text": chosen,
            "rejected_text": rejected,
        }
        if reference_tok is not None:
            extras["reference_ids"] = reference_tok["input_ids"]
            extras["reference_masks"] = reference_tok["attention_mask"]
            extras["reference_prompt_ids_lens"] = prompt_ids_len
            extras["reference_text"] = reference_response

        return TokenizedPreferenceSample(
            chosen_ids=chosen_tok["input_ids"],
            chosen_mask=chosen_tok["attention_mask"],
            rejected_ids=rejected_tok["input_ids"],
            rejected_mask=rejected_tok["attention_mask"],
            extras=extras,
        )

    def __getitem__(self, idx: int):
        sample = self.tokenized_samples[idx]
        return (
            sample.chosen_ids,
            sample.chosen_mask,
            sample.rejected_ids,
            sample.rejected_mask,
            sample.extras,
        )

    def collate_fn(self, item_list):
        """
        Turns a list of tokenized samples into a single, batched, and padded tensor package ready for preference optimisation training.
        """
        chosen_ids, chosen_masks = [], []
        rejected_ids, rejected_masks = [], []
        reference_ids, reference_masks = [], []
        extras = {
            "prompt_ids_lens": [],
            "loss_masks": [],
            "prompt_text": [],
            "chosen_text": [],
            "rejected_text": [],
        }
        has_reference = False

        for c_ids, c_m, r_ids, r_m, ex in item_list:
            chosen_ids.append(c_ids.squeeze(0))
            chosen_masks.append(c_m.squeeze(0))
            rejected_ids.append(r_ids.squeeze(0))
            rejected_masks.append(r_m.squeeze(0))

            extras["prompt_ids_lens"].append(int(ex["prompt_ids_lens"]))
            extras["loss_masks"].append(bool(ex["loss_masks"]))
            extras["prompt_text"].append(ex["prompt_text"])
            extras["chosen_text"].append(ex["chosen_text"])
            extras["rejected_text"].append(ex["rejected_text"])

            if "reference_ids" in ex:
                has_reference = True
                reference_ids.append(ex["reference_ids"].squeeze(0))
                reference_masks.append(ex["reference_masks"].squeeze(0))
                extras.setdefault("reference_prompt_ids_lens", []).append(int(ex["reference_prompt_ids_lens"]))
                extras.setdefault("reference_text", []).append(ex["reference_text"])

        padding_side = "right"
        pad_id = int(self.tokenizer.pad_token_id)

        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=pad_id)
        chosen_masks = zero_pad_sequences([m.to(torch.long) for m in chosen_masks], side=padding_side, value=0).bool()
        rejected_ids = zero_pad_sequences(rejected_ids, side=padding_side, value=pad_id)
        rejected_masks = zero_pad_sequences([m.to(torch.long) for m in rejected_masks], side=padding_side, value=0).bool()

        extras["prompt_ids_lens"] = torch.tensor(extras["prompt_ids_lens"], dtype=torch.long)
        extras["loss_masks"] = torch.tensor(extras["loss_masks"], dtype=torch.bool)

        if has_reference:
            extras["reference_ids"] = zero_pad_sequences(reference_ids, side=padding_side, value=pad_id)
            extras["reference_masks"] = zero_pad_sequences(
                [m.to(torch.long) for m in reference_masks],
                side=padding_side,
                value=0,
            ).bool()
            extras["reference_prompt_ids_lens"] = torch.tensor(
                extras["reference_prompt_ids_lens"],
                dtype=torch.long,
            )

        return chosen_ids, chosen_masks, rejected_ids, rejected_masks, extras


class TextPreferenceDataset(Dataset):
    """
    Simple text-only dataset for ERM training.
    This is used for ERM training because the backbone (pairrm) has its own tokenizer,
    """

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @staticmethod
    def collate_fn(batch):
        return {
            "prompt": [x["prompt"] for x in batch],
            "chosen": [x["chosen"] for x in batch],
            "rejected": [x["rejected"] for x in batch],
        }
