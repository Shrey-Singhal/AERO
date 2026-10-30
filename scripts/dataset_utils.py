from typing import Dict, List, Tuple, Optional, Union

# Instruction prefix for instruct models on TL;DR task
TLDR_INSTRUCTION = "Write a concise TL;DR summary for the following Reddit post.\n\n"

PromptInput = Union[str, List[Dict[str, str]]]

def _normalize_chat_history(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Normalize mixed-source chat histories into a strict user/assistant alternation.

    We fold system text into the next user turn, merge adjacent same-role turns,
    and drop any leading assistant-only prefix.
    """
    normalized: List[Dict[str, str]] = []
    pending_system: List[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        content = content.strip()

        if role == "system":
            pending_system.append(content)
            continue

        if role not in {"user", "assistant"}:
            pending_system.append(content)
            continue

        if role == "user" and pending_system:
            content = "\n\n".join(pending_system + [content])
            pending_system = []

        if not normalized:
            if role != "user":
                # Can't start an instruction-style generation history with assistant.
                continue
            normalized.append({"role": role, "content": content})
            continue

        if normalized[-1]["role"] == role:
            normalized[-1]["content"] = (
                f"{normalized[-1]['content']}\n\n{content}".strip()
            )
        else:
            normalized.append({"role": role, "content": content})

    if pending_system:
        if normalized and normalized[0]["role"] == "user":
            normalized[0]["content"] = (
                f"{' '.join(pending_system).strip()}\n\n{normalized[0]['content']}"
            ).strip()
        elif pending_system:
            normalized = [{"role": "user", "content": " ".join(pending_system).strip()}] + normalized

    return normalized


def _flatten_chat_messages(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> Optional[str]:
    parts: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            prefix = "Human"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = role.capitalize() or "User"
        parts.append(f"{prefix}: {content.strip()}")

    if add_generation_prompt:
        parts.append("Assistant:")

    text = "\n\n".join(parts).strip()
    return text or None


def normalize_prompt_messages(messages: object) -> Optional[List[Dict[str, str]]]:
    """
    Coerce a prompt history into normalized user/assistant messages.

    Returns None when the input is not a list-like chat history.
    """
    if not isinstance(messages, list):
        return None

    cleaned: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})

    cleaned = _normalize_chat_history(cleaned)
    return cleaned or None


def flatten_prompt_input(
    prompt: PromptInput,
    add_generation_prompt: bool = False,
) -> str:
    """
    Convert a prompt input into the flat text form used by string-based scorers.
    """
    if isinstance(prompt, str):
        return prompt

    messages = normalize_prompt_messages(prompt)
    if not messages:
        return ""
    return _flatten_chat_messages(messages, add_generation_prompt=add_generation_prompt) or ""


def prompt_and_response_to_messages(
    prompt: PromptInput,
    response: str,
) -> List[Dict[str, str]]:
    """
    Build a full conversation for scorers from a prompt history plus a response.
    """
    messages = normalize_prompt_messages(prompt)
    if messages is None:
        prompt_text = str(prompt).strip()
        messages = [{"role": "user", "content": prompt_text}] if prompt_text else []

    if isinstance(response, str) and response.strip():
        messages = [*messages, {"role": "assistant", "content": response.strip()}]
    return messages


def _split_chat_prompt_and_response(
    conversation: object,
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    cleaned = normalize_prompt_messages(conversation)

    if not cleaned:
        return None, None

    if cleaned[-1]["role"] == "assistant":
        return cleaned[:-1], cleaned[-1]["content"]
    return cleaned, None


def extract_prompt(row: Dict[str, object], columns: List[str], prompt_data: str) -> Tuple[Optional[str], str]:
    """
    Finds the prompt text across different dataset formats.
    """
    prompt_data_lower = prompt_data.lower()

    if "tldr" in prompt_data_lower:
        # "prompt" column (Standard TL;DR)
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt, "prompt"

    elif "ultrafeedback" in prompt_data_lower:
        # UltraFeedback: prompt is in 'prompt' column or chosen[0]["content"]
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt, "prompt"
        # Fallback: extract from chosen conversation
        chosen = row.get("chosen")
        if isinstance(chosen, list) and len(chosen) > 0:
            content = chosen[0].get("content")
            if isinstance(content, str) and content.strip():
                return content, "chosen"

    return None, "none"


def extract_ultrafeedback_response(row: Dict[str, object], key: str = "chosen") -> Optional[str]:
    """
    Extract the assistant response from UltraFeedback format.

    UltraFeedback format: chosen/rejected is a list of messages like:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    conversation = row.get(key)
    if not isinstance(conversation, list):
        return None

    # Find the assistant message (usually the second one)
    for msg in conversation:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content

    return None


def extract_prompt_messages(
    row: Dict[str, object],
    columns: List[str],
    prompt_data: str,
) -> Optional[List[Dict[str, str]]]:
    """
    Return a structured message history when the dataset stores multi-turn chat.
    The history excludes the final assistant response.
    """
    prompt_data_lower = prompt_data.lower()

    if "ultrafeedback" in prompt_data_lower:
        prompt_messages, _ = _split_chat_prompt_and_response(row.get("chosen"))
        return prompt_messages

    return None


def wrap_prompt_for_instruct(prompt: str, prompt_data: str, is_instruct_model: bool) -> str:
    """
    Wrap prompt with task instruction for instruct-tuned models.

    Pythia models were SFT'd specifically on the "Post... TL;DR:" format,
    but Llama/instruct models need explicit instructions.

    UltraFeedback prompts are already complete instructions, so no wrapping needed.
    """
    if not is_instruct_model:
        return prompt

    prompt_data_lower = prompt_data.lower()

    if "tldr" in prompt_data_lower:
        return TLDR_INSTRUCTION + prompt

    # UltraFeedback: prompts are already complete, no wrapping needed
    if "ultrafeedback" in prompt_data_lower:
        return prompt

    # Other datasets: return as-is
    return prompt


def build_prompt_records(
    dataset,
    tokenizer,
    max_length: int,
    apply_chat_template: bool,
    data_string: str,
    is_instruct_model: bool = False,
) -> List[Dict[str, object]]:
    """
    Build aligned raw/processed prompt records and drop prompts that exceed
    max_length after formatting. No truncation is applied.

    `raw_prompt` is the original user-facing prompt used for logging/results.
    `processed_prompt` is what the policy sees for generation / DPO.
    `score_prompt` preserves structured histories for oracle / RM scoring.
    """
    prompt_records = []
    columns = list(dataset.column_names)
    skipped = 0
    use_chat_template = bool(apply_chat_template)

    for i in range(len(dataset)):
        raw_prompt, _ = extract_prompt(dataset[i], columns, data_string)
        if not raw_prompt:
            continue
        prompt_messages = extract_prompt_messages(dataset[i], columns, data_string)

        # Wrap with instruction for instruct models (e.g., "Summarize..." for TL;DR)
        if prompt_messages:
            if use_chat_template:
                processed_prompt = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                processed_prompt = _flatten_chat_messages(
                    prompt_messages,
                    add_generation_prompt=True,
                )
        else:
            processed_prompt = wrap_prompt_for_instruct(raw_prompt, data_string, is_instruct_model)

            if use_chat_template:
                processed_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": processed_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

        tokens = tokenizer.encode(processed_prompt, add_special_tokens=False)
        if len(tokens) > max_length:
            skipped += 1
            continue

        normalized_raw_prompt = raw_prompt.strip()

        prompt_records.append(
            {
                "raw_prompt": normalized_raw_prompt,
                "processed_prompt": processed_prompt.strip(),
                "score_prompt": prompt_messages if prompt_messages else normalized_raw_prompt,
            }
        )

    print(
        f"[Dataset] Filtered {len(prompt_records)} prompts. "
        f"Dropped {skipped} too-long samples."
    )
    return prompt_records
