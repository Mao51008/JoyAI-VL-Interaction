"""Answer-token-only KL loss for the optional Stage2.1 text-teacher ablation."""

from __future__ import annotations

from typing import Any


def freeze_text_teacher(model: Any) -> Any:
    """Freeze/eval the separately instantiated text-only teacher before its forward pass."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def answer_token_kl(student_logits: Any, student_labels: Any, teacher_logits: Any, teacher_labels: Any, weight: float) -> Any:
    """KL(student || frozen teacher), aligned by identical assistant target tokens only."""
    import torch
    import torch.nn.functional as functional

    if weight < 0:
        raise ValueError("text teacher KL weight must be non-negative")
    if weight == 0:
        return student_logits.new_zeros(())
    student_targets, teacher_targets = student_labels[:, 1:], teacher_labels[:, 1:]
    student_mask, teacher_mask = student_targets.ne(-100), teacher_targets.ne(-100)
    if student_mask.sum().item() != teacher_mask.sum().item():
        raise ValueError("student and teacher assistant target counts differ")
    if not torch.equal(student_targets[student_mask], teacher_targets[teacher_mask]):
        raise ValueError("student and teacher assistant targets differ")
    student = student_logits[:, :-1][student_mask]
    teacher = teacher_logits[:, :-1][teacher_mask].detach()
    if student.shape != teacher.shape or student.numel() == 0:
        raise ValueError("invalid or empty answer-token logits for teacher KL")
    return functional.kl_div(functional.log_softmax(student.float(), dim=-1), functional.softmax(teacher.float(), dim=-1), reduction="batchmean").to(student_logits.dtype) * weight
