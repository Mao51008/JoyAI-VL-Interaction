from __future__ import annotations

import torch

from training.omni.projector_stage2.generate_vision_distillation import (
    SHORT_ANSWER_INSTRUCTION,
    build_teacher_prompt,
    full_sequence_tokens,
    generate_teacher_response,
)


class _Tokenizer:
    eos_token_id = 9


class _Processor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        del messages
        del kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def batch_decode(self, rows, **kwargs):
        del kwargs
        return [" concise answer " if rows[0] == [4, 5] else ""]


class _Model:
    def __init__(self, generated_tokens: list[int]) -> None:
        self.generated_tokens = generated_tokens

    def generate(self, input_ids, **kwargs):
        del kwargs
        return torch.tensor([input_ids[0].tolist() + self.generated_tokens])


def test_build_teacher_prompt_preserves_question_under_fixed_instruction() -> None:
    prompt = build_teacher_prompt("What is the signal color?")

    assert prompt.startswith(SHORT_ANSWER_INSTRUCTION)
    assert prompt.endswith("Question: What is the signal color?")


def test_generation_records_eos_and_excludes_it_from_answer() -> None:
    result = generate_teacher_response(
        _Model([4, 5, 9]), _Processor(), "/image.jpg", "Question", "cpu", 256
    )

    assert result.answer == "concise answer"
    assert result.prompt_tokens == 3
    assert result.generated_tokens == 3
    assert result.answer_tokens == 2
    assert result.finished_by_eos is True


def test_generation_without_eos_is_marked_for_rejection() -> None:
    result = generate_teacher_response(
        _Model([4, 5]), _Processor(), "/image.jpg", "Question", "cpu", 256
    )

    assert result.finished_by_eos is False
    assert result.answer == "concise answer"


def test_full_sequence_tokens_uses_processor_training_template() -> None:
    assert full_sequence_tokens(_Processor(), "/image.jpg", "Question", "Answer") == 3
