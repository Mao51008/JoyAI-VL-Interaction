import pytest

torch = pytest.importorskip("torch")

from training.omni.projector_stage2.text_teacher_kl import answer_token_kl, freeze_text_teacher
from training.omni.projector_stage2.train import _loss_with_optional_text_teacher_kl


def test_answer_token_kl_uses_only_assistant_tokens_and_teacher_has_no_gradient():
    student = torch.randn(1, 4, 5, requires_grad=True)
    teacher = torch.randn(1, 4, 5, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3]])
    answer_token_kl(student, labels, teacher, labels, 0.25).backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_freeze_text_teacher_disables_all_parameters():
    teacher = freeze_text_teacher(torch.nn.Linear(2, 2))
    assert not teacher.training
    assert not any(parameter.requires_grad for parameter in teacher.parameters())


def test_zero_weight_is_noop_and_mismatched_targets_rejected():
    student = torch.randn(1, 3, 4)
    labels = torch.tensor([[-100, 1, 2]])
    assert answer_token_kl(student, labels, student, labels, 0.0).item() == 0
    batch = type("Batch", (), {"labels": labels, "teacher_labels": labels})()
    assert _loss_with_optional_text_teacher_kl({"loss": torch.tensor(1.5)}, batch, 0.0).item() == 1.5
    with pytest.raises(ValueError, match="targets differ"):
        answer_token_kl(student, labels, student, torch.tensor([[-100, 1, 3]]), 1.0)
