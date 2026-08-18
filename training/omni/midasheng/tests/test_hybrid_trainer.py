from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")

from training.omni.midasheng.hybrid_trainer import HybridConfig, HybridTrainer


class Encoder(torch.nn.Module):
    def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.0)); self.no_sync_calls = 0
    def no_sync(self):
        class C:
            def __enter__(inner): self.no_sync_calls += 1
            def __exit__(inner, *args): return False
        return C()
    def forward(self, waves, lengths): return (waves * self.weight).float(), torch.ones_like(waves, dtype=torch.bool)

class Core(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.audio_projector = torch.nn.Parameter(torch.tensor(1.0)); self.lora_b = torch.nn.Parameter(torch.tensor(1.0)); self.backward_calls = self.step_calls = 0
    def forward(self, batch, features, mask):
        assert features.dtype == torch.bfloat16
        return (features.float() * self.audio_projector * self.lora_b).sum()
    def backward(self, loss): self.backward_calls += 1; loss.backward()
    def step(self):
        self.step_calls += 1
        with torch.no_grad():
            for parameter in self.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-1e-3)
    def save_checkpoint(self, *args, **kwargs): pass
    def load_checkpoint(self, *args, **kwargs): pass


class Scheduler:
    def __init__(self): self.steps = 0
    def step(self): self.steps += 1
    def state_dict(self): return {"steps": self.steps}
    def load_state_dict(self, state): self.steps = state["steps"]

def test_eight_microbatches_use_one_boundary_and_keep_encoder_gradient():
    encoder, core = Encoder(), Core(); optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    scheduler = Scheduler()
    trainer = HybridTrainer(encoder, core, optimizer, HybridConfig(8), scheduler)
    batch = SimpleNamespace(audio_features=torch.ones(1), audio_attention_mask=torch.ones(1, dtype=torch.long))
    trainer.run_update([batch] * 8)
    assert core.backward_calls == 8 and core.step_calls == 1
    assert encoder.no_sync_calls == 7 and trainer.global_step == 1 and scheduler.steps == 1
    assert encoder.weight.grad is not None
