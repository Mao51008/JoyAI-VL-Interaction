"""Turn changing ASR hypotheses into a monotonic stable prefix."""

from collections import deque


def common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        limit = min(len(prefix), len(value))
        index = 0
        while index < limit and prefix[index] == value[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


class StablePrefixTracker:
    def __init__(self, observations: int = 2):
        if observations < 2:
            raise ValueError("observations must be at least 2")
        self.observations = observations
        self._history: deque[str] = deque(maxlen=observations)
        self.stable = ""

    def update(self, hypothesis: str) -> tuple[str, str]:
        hypothesis = hypothesis.strip()
        self._history.append(hypothesis)
        if len(self._history) == self.observations:
            candidate = common_prefix(list(self._history))
            if candidate.startswith(self.stable):
                self.stable = candidate
        unstable = hypothesis.removeprefix(self.stable)
        return self.stable, unstable

    def reset(self) -> None:
        self._history.clear()
        self.stable = ""
