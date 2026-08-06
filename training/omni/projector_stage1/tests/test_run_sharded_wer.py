import json
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training.omni.projector_stage1.run_sharded_wer import parse_progress_line, run_sharded


class FakeChild:
    def __init__(self, status=0):
        self.status = status
        self.signals = []
        self.poll_count = 0
        self.wait_count = 0

    def poll(self):
        self.poll_count += 1
        return None if self.poll_count < 3 else self.status

    def wait(self):
        self.wait_count += 1
        self.status = self.status
        return self.status

    def send_signal(self, signum):
        self.signals.append(signum)
        self.status = 143


class RunShardedWerTest(unittest.TestCase):
    def _args(self, root: Path, *, gpus=(0, 1, 2, 3)):
        return SimpleNamespace(
            manifest=root / "manifest.jsonl",
            feature_dir=root / "features",
            checkpoint=root / "best.pt",
            joyai_model="joyai",
            audio_ablation="none",
            zero_feature_dir=None,
            seed=3407,
            max_new_tokens=256,
            gpus=list(gpus),
            output_dir=root / "run",
            merged_output=root / "merged.json",
            environment={},
        )

    def test_progress_parser_extracts_completed_elapsed_and_eta(self):
        parsed = parse_progress_line(
            "shard 3/4:  25%|██▌| 44/174 [02:26<07:38, 3.53s/sample]"
        )
        self.assertEqual(parsed, {
            "shard": 3, "completed": 44, "total": 174,
            "elapsed": "02:26", "eta": "07:38",
        })

    def test_rejects_duplicate_gpu_ids_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root, gpus=(0, 0, 1, 2))
            with self.assertRaises(ValueError): run_sharded(args)
            args = self._args(root)
            args.output_dir.mkdir()
            with self.assertRaises(FileExistsError): run_sharded(args)

    def test_success_maps_gpu_to_shard_and_merges_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            children = []
            launches = []

            def popen(command, **kwargs):
                child = FakeChild()
                children.append(child)
                launches.append((command, kwargs["env"]))
                Path(command[command.index("--output") + 1]).write_text("{}", encoding="utf-8")
                return child

            def merge(paths, output):
                self.assertEqual([p.name for p in paths], [f"shard-{i}.json" for i in range(4)])
                output.write_text(json.dumps({"samples": 4}), encoding="utf-8")

            result = run_sharded(args, popen_factory=popen, merge_runner=merge, sleep_fn=lambda _: None)
            self.assertEqual(result["samples"], 4)
            self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for _, env in launches], ["0", "1", "2", "3"])
            self.assertTrue(all("PATH" in env for _, env in launches))
            self.assertEqual([command[command.index("--shard-index") + 1] for command, _ in launches], ["0", "1", "2", "3"])

    def test_child_environment_inherits_sentinel_without_mutating_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            args.environment = {"RUNNER_SENTINEL": "kept"}
            parent_value = __import__("os").environ.get("RUNNER_SENTINEL")
            environments = []

            def popen(_command, **kwargs):
                environments.append(kwargs["env"])
                Path(_command[_command.index("--output") + 1]).write_text("{}", encoding="utf-8")
                return FakeChild()

            run_sharded(args, popen_factory=popen, merge_runner=lambda paths, output: output.write_text("{}"),
                        sleep_fn=lambda _: None)
            self.assertEqual([env["RUNNER_SENTINEL"] for env in environments], ["kept"] * 4)
            self.assertEqual(__import__("os").environ.get("RUNNER_SENTINEL"), parent_value)

    def test_failure_waits_for_all_children_and_does_not_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            children = []

            def popen(_command, **_kwargs):
                child = FakeChild(status=1 if len(children) == 1 else 0)
                children.append(child)
                return child

            merged = False

            def merge(_paths, _output):
                nonlocal merged
                merged = True

            with self.assertRaises(RuntimeError):
                run_sharded(args, popen_factory=popen, merge_runner=merge, sleep_fn=lambda _: None)
            self.assertFalse(merged)
            self.assertTrue(all(child.poll_count >= 2 for child in children))

    def test_signal_handler_stops_only_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            children = []
            captured = {}
            signal_calls = []

            def popen(_command, **_kwargs):
                child = FakeChild()
                children.append(child)
                return child

            def fake_signal(sig, handler):
                signal_calls.append((sig, handler))
                captured[sig] = handler
                return signal.SIG_DFL

            calls = 0

            def interrupt_once(_seconds):
                nonlocal calls
                calls += 1
                if calls == 1:
                    captured[signal.SIGTERM](signal.SIGTERM, None)

            with patch("training.omni.projector_stage1.run_sharded_wer.signal.signal", side_effect=fake_signal):
                with self.assertRaises(RuntimeError):
                    run_sharded(args, popen_factory=popen, merge_runner=lambda *_: None,
                                sleep_fn=interrupt_once)
            self.assertEqual([child.signals for child in children], [[signal.SIGTERM]] * 4)
            self.assertTrue(all(child.wait_count >= 1 for child in children))
            registered_count = 3 if hasattr(signal, "SIGHUP") else 2
            registered = [sig for sig, _handler in signal_calls[:registered_count]]
            self.assertEqual([sig for sig, _handler in signal_calls[-registered_count:]], registered)

    def test_merged_output_appearing_during_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            original = "existing"
            merged = False

            def appear(_seconds):
                args.merged_output.write_text(original, encoding="utf-8")

            def merge(_paths, _output):
                nonlocal merged
                merged = True

            with self.assertRaises(FileExistsError):
                run_sharded(args, popen_factory=lambda *_args, **_kwargs: FakeChild(),
                            merge_runner=merge, sleep_fn=appear)
            self.assertFalse(merged)
            self.assertEqual(args.merged_output.read_text(encoding="utf-8"), original)

    def test_missing_shard_output_does_not_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            merged = False

            def merge(_paths, _output):
                nonlocal merged
                merged = True

            with self.assertRaisesRegex(RuntimeError, "missing shard outputs"):
                run_sharded(args, popen_factory=lambda *_args, **_kwargs: FakeChild(),
                            merge_runner=merge, sleep_fn=lambda _: None)
            self.assertFalse(merged)


if __name__ == "__main__":
    unittest.main()
