import copy
import importlib.util
import unittest
from types import SimpleNamespace


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "projector stage-one tests require PyTorch")
class ProjectorDependencyTest(unittest.TestCase):
    def test_pytorch_training_contracts(self) -> None:
        if not TORCH_AVAILABLE:
            return

        import torch
        from torch import nn

        from training.omni.projector import (
            AudioProjector,
            AudioProjectorConfig,
            assert_projector_gradients,
            parameter_report,
            trainable_parameters,
        )
        from training.omni.stage1_model import (
            ProjectorStage1Model,
            replace_audio_placeholders,
        )

        class TinyAudioEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(3, 4)

            def forward(self, values, attention_mask=None):
                return SimpleNamespace(last_hidden_state=self.linear(values))

        class TinyLanguageModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.head = nn.Linear(6, 5)

            def forward(self, inputs_embeds, attention_mask, labels):
                logits = self.head(inputs_embeds)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                return SimpleNamespace(loss=loss, logits=logits)

        text = torch.zeros(1, 4, 2)
        audio = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        replaced = replace_audio_placeholders(
            text,
            audio,
            torch.tensor([[False, True, True, False]]),
            torch.tensor([[True, True]]),
        )
        self.assertTrue(torch.equal(replaced[0, 1:3], audio[0]))
        with self.assertRaisesRegex(ValueError, "placeholders"):
            replace_audio_placeholders(
                text,
                audio,
                torch.tensor([[False, True, False, False]]),
                torch.tensor([[True, True]]),
            )

        torch.manual_seed(7)
        model = ProjectorStage1Model(
            TinyAudioEncoder(),
            TinyLanguageModel(),
            AudioProjector(AudioProjectorConfig(input_size=4, output_size=6)),
        )
        before_encoder = copy.deepcopy(model.audio_encoder.state_dict())
        before_language_model = copy.deepcopy(model.language_model.state_dict())
        before_projector = copy.deepcopy(model.audio_projector.state_dict())
        report = parameter_report(model)
        self.assertTrue(report.trainable_names)
        self.assertTrue(
            all(name.startswith("audio_projector.") for name in report.trainable_names)
        )

        optimizer = torch.optim.AdamW(trainable_parameters(model), lr=0.05)
        result = model(
            audio_inputs=torch.randn(1, 2, 3),
            audio_attention_mask=torch.tensor([[True, True]]),
            text_embeddings=torch.zeros(1, 4, 6),
            audio_placeholder_mask=torch.tensor([[False, True, True, False]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            labels=torch.tensor([[-100, -100, 1, 2]]),
        )
        result.loss.backward()
        assert_projector_gradients(model, model.audio_projector)
        optimizer.step()

        for name, value in model.audio_encoder.state_dict().items():
            self.assertTrue(torch.equal(before_encoder[name], value))
        for name, value in model.language_model.state_dict().items():
            self.assertTrue(torch.equal(before_language_model[name], value))
        self.assertTrue(
            any(
                not torch.equal(before_projector[name], value)
                for name, value in model.audio_projector.state_dict().items()
            )
        )


if __name__ == "__main__":
    unittest.main()
