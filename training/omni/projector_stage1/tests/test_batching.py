import unittest

from training.omni.projector_stage1.batching import (
    distribute_batches,
    padded_attention_cost,
    plan_length_aware_batches,
    shuffled_rank_batches,
)


class Stage1BatchingTest(unittest.TestCase):
    def test_long_samples_become_single_batches_without_being_dropped(self) -> None:
        tokens = {"long": 900, "medium": 500, "short-a": 100, "short-b": 90}
        batches = plan_length_aware_batches(
            list(tokens), tokens, max_batch_size=2, max_attention_cost=500_000
        )
        self.assertEqual([item for batch in batches for item in batch], ["long", "medium", "short-a", "short-b"])
        self.assertEqual(batches, [["long"], ["medium", "short-a"], ["short-b"]])
        self.assertEqual(padded_attention_cost([500, 100]), 500_000)

    def test_distributed_ranks_get_the_same_step_count(self) -> None:
        distributed = distribute_batches([["a"], ["b"], ["c"], ["d"], ["e"]], world_size=2)
        self.assertEqual(len(distributed[0]), len(distributed[1]))
        self.assertEqual({item for rank in distributed for batch in rank for item in batch}, {"a", "b", "c", "d", "e"})

    def test_epoch_shuffle_is_reproducible_and_changes_order(self) -> None:
        batches = [["a"], ["b"], ["c"], ["d"]]
        first = shuffled_rank_batches(batches, world_size=1, rank=0, seed=7, epoch=0)
        self.assertEqual(first, shuffled_rank_batches(batches, world_size=1, rank=0, seed=7, epoch=0))
        self.assertNotEqual(first, shuffled_rank_batches(batches, world_size=1, rank=0, seed=7, epoch=1))


if __name__ == "__main__":
    unittest.main()
