import unittest

from training.omni.stage1_data import IGNORE_INDEX, build_stage1_sequence, pad_sequences


class Stage1DataTest(unittest.TestCase):
    def test_audio_span_and_target_mask(self) -> None:
        sequence = build_stage1_sequence(
            context_ids=[10, 11],
            target_ids=[30, 31],
            audio_token_count=3,
            audio_start_id=20,
            audio_placeholder_id=21,
            audio_end_id=22,
            assistant_prefix_ids=[25],
            eos_id=32,
        )
        self.assertEqual(sequence.input_ids, (10, 11, 20, 21, 21, 21, 22, 25, 30, 31, 32))
        self.assertEqual(sequence.labels[:8], (IGNORE_INDEX,) * 8)
        self.assertEqual(sequence.labels[8:], (30, 31, 32))
        self.assertEqual(sum(sequence.audio_placeholder_mask), 3)

    def test_padding_keeps_ignore_index(self) -> None:
        first = build_stage1_sequence(
            context_ids=[], target_ids=[7], audio_token_count=1,
            audio_start_id=1, audio_placeholder_id=2, audio_end_id=3,
        )
        second = build_stage1_sequence(
            context_ids=[9], target_ids=[8, 7], audio_token_count=2,
            audio_start_id=1, audio_placeholder_id=2, audio_end_id=3,
        )
        batch = pad_sequences([first, second], pad_token_id=0)
        self.assertEqual(batch["attention_mask"], [[1] * 4 + [0] * 3, [1] * 7])
        self.assertEqual(batch["labels"][0][-3:], [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX])

    def test_empty_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_ids"):
            build_stage1_sequence(
                context_ids=[], target_ids=[], audio_token_count=1,
                audio_start_id=1, audio_placeholder_id=2, audio_end_id=3,
            )


if __name__ == "__main__":
    unittest.main()
