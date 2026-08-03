from pathlib import Path


INDEX_HTML = (
    Path(__file__).parents[1] / "src" / "joy_interaction_webui" / "static" / "index.html"
)


def test_continuous_asr_final_submits_prompt_and_interrupts_tts() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "function interruptTtsForUserInput()" in html
    assert "['speech_partial', 'speech_stable'].includes(kind)" in html
    assert "kind === 'speech_final'" in html
    assert "!metadata.likely_tts_echo" in html
    assert "promptText.value = transcript" in html
    assert "stopTtsPlayback({ sendStop: true, status: 'interrupted' })" in html
    assert "function isMeaningfulContinuousAsrText(text)" in html
    assert "^[嗯啊呃唔哦噢额哼诶欸]+$" in html
    assert "function isDuplicateContinuousAsrFinal" in html
