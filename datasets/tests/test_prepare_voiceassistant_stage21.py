import json
from datasets.prepare_voiceassistant_stage21 import select

def _raw(answer="answer"):
 return {"messages":[{"role":"system","content":"s"},{"role":"user","content":"<|audio|>"},{"role":"assistant","content":"<|audio|> "+answer}],"audios":["user/a.wav","assistant/a.wav"]}
def test_select_keeps_only_remaining_train_and_excludes_known_conflict(tmp_path, monkeypatch):
 raw=tmp_path/'raw.jsonl'; existing=tmp_path/'existing.jsonl'
 lines=[_raw(),_raw(),_raw()]
 raw.write_text('\n'.join(json.dumps(x) for x in lines)+'\n')
 existing.write_text(json.dumps({"sample_id":"voiceassistant:0000000"})+'\n')
 monkeypatch.setattr("datasets.prepare_voiceassistant_stage21._split",lambda sample:"train")
 selected,audit=select(raw,existing,1)
 assert len(selected)==1 and all("question" not in x for x in selected)
 assert audit["rejected"]["already_in_existing_20000"]==1
