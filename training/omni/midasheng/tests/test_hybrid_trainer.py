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

class Core:
    def __init__(self): self.scale = torch.nn.Parameter(torch.tensor(1.0)); self.backward_calls = self.step_calls = 0
    def __call__(self, batch, features, mask): assert features.dtype == torch.bfloat16; return (features.float() * self.scale).sum()
    def backward(self, loss): self.backward_calls += 1; loss.backward()
    def step(self): self.step_calls += 1
    def save_checkpoint(self, *args, **kwargs): pass
    def load_checkpoint(self, *args, **kwargs): pass

def test_eight_microbatches_use_one_boundary_and_keep_encoder_gradient():
    encoder, core = Encoder(), Core(); optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    trainer = HybridTrainer(encoder, core, optimizer, HybridConfig(8))
    batch = SimpleNamespace(waveforms=torch.ones(1), lengths=torch.ones(1, dtype=torch.long))
    trainer.run_update([batch] * 8)
    assert core.backward_calls == 8 and core.step_calls == 1
    assert encoder.no_sync_calls == 7 and trainer.global_step == 1
    assert encoder.weight.grad is not None
