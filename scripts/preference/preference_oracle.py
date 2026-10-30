from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from configs.config import AeroConfig
from scripts.hf_auth import hf_token_kwargs
from scripts.dataset_utils import (
    PromptInput,
    prompt_and_response_to_messages,
)

class Oracle:
    def __init__(self, config: AeroConfig):
        self.config = config
        self.device = self.config.device_oracle

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.oracle,
            **hf_token_kwargs(),
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = None
        if "cuda" in self.device:
            device_index = int(self.device.split(":")[-1]) if ":" in self.device else 0
            device_map = {"": device_index}   # force the whole model onto that GPU

        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": device_map,
            "num_labels": 1,
            "trust_remote_code": True,
            **hf_token_kwargs(),
        }
        if getattr(self.config, "use_flash_attention_2", False):
            model_kwargs["use_flash_attention_2"] = True

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.oracle,
                **model_kwargs,
            )
        except TypeError:
            model_kwargs.pop("use_flash_attention_2", None)
            if getattr(self.config, "use_flash_attention_2", False):
                model_kwargs["attn_implementation"] = "flash_attention_2"
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.oracle,
                    **model_kwargs,
                )
            except TypeError:
                model_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.oracle,
                    **model_kwargs,
                )

        if device_map is None:
            # CPU / MPS fallback
            self.model.to(self.device)
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def move_to(self, device: str) -> None:
        """Move oracle model to a new device (e.g., 'cpu' or 'cuda:1')."""
        self.device = device
        self.model.to(device)
    
    def get_scores(self, prompts: List[PromptInput], responses: List[str], batch_size: int):
        all_scores = []
        batch_size = max(1, int(batch_size))
        i = 0

        with tqdm(total=len(prompts), desc="Oracle Scoring") as pbar:
            while i < len(prompts):
                cur_bs = min(batch_size, len(prompts) - i)
                batch_prompts = prompts[i : i + cur_bs]
                batch_responses = responses[i : i + cur_bs]
                batch_msgs = [
                    prompt_and_response_to_messages(p, r)
                    for p, r in zip(batch_prompts, batch_responses)
                ]
                rendered = self.tokenizer.apply_chat_template(batch_msgs, tokenize=False)

                try:
                    inputs = self.tokenizer(
                        rendered,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                    ).to(self.device)

                    with torch.no_grad():
                        logits = self.model(**inputs).logits.view(-1).float()
                        all_scores.append(logits)
                    i += cur_bs
                    pbar.update(cur_bs)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                    err_msg = str(err).lower()
                    is_oom = "out of memory" in err_msg
                    if not is_oom or cur_bs <= 1:
                        raise
                    next_bs = max(1, cur_bs // 2)
                    print(f"[Oracle] WARNING: OOM at batch_size={cur_bs}, retrying with {next_bs}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_size = next_bs

        return torch.cat(all_scores)
    
    def compare(self, prompts, responses_A, responses_B, batch_size=32):
        """
        Returns probability P(A > B).
        """
        combined_scores = self.get_scores(
            list(prompts) + list(prompts),
            list(responses_A) + list(responses_B),
            batch_size,
        )
        split = len(prompts)
        scores_A = combined_scores[:split]
        scores_B = combined_scores[split:]

        # Calculate Win Probability: sigmoid(Score_A - Score_B)
        probs = torch.sigmoid(scores_A - scores_B)
        return probs
