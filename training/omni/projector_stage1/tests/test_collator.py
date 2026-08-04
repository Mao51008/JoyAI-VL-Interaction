import unittest
from types import SimpleNamespace

from training.omni.schema import OmniSample
from training.omni.projector_stage1.collator import JoyAIStage1TokenLayout, build_sample_sequence
from training.omni.projector_stage1.data import IGNORE_INDEX


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = -1
    _special = {"<|vision_start|>": 10, "<|vision_pad|>": 11, "<|vision_end|>": 12}

    def convert_tokens_to_ids(self, token):
        return self._special.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[ord(char) % 97 + 20 for char in text])


class Stage1CollatorTest(unittest.TestCase):
    def test_only_transcript_tokens_have_labels(self) -> None:
        sample = OmniSample.from_dict({
            "schema_version": "omni-training-v1", "sample_id": "one", "duration_ms": 1000,
            "provenance": {"dataset": "test", "version": "1", "source_uri": "local://test",
                           "license_name": "CC0", "license_tier": "redistributable",
                           "allows_training": True, "allows_modification": True,
                           "allows_redistribution": True, "allows_commercial_use": True},
            "audio": [], "video": [],
            "text": [{"text": "audio request", "timestamp_ms": 0, "channel": "user_text", "auxiliary": False}],
            "targets": [],
            "metadata": {"assistant_target_text": "word"},
        })
        tokenizer = _Tokenizer()
        sequence = build_sample_sequence(
            sample, tokenizer=tokenizer, layout=JoyAIStage1TokenLayout.from_tokenizer(tokenizer),
            audio_token_count=3,
        )
        self.assertEqual(sum(sequence.audio_placeholder_mask), 3)
        self.assertTrue(all(label == IGNORE_INDEX for label in sequence.labels[:sequence.target_start]))
        self.assertTrue(all(label != IGNORE_INDEX for label in sequence.labels[sequence.target_start:]))


if __name__ == "__main__":
    unittest.main()
