import json
from training.omni.projector_stage2.audit_vision_teacher_manifest import audit
def test_audit_counts_completed_outputs(tmp_path):
 source=tmp_path/'s';out=tmp_path/'o';rej=tmp_path/'r'
 row={"sample_id":"x","teacher_response":"Short answer.","provenance":{"generation":{"finished_by_eos":True,"generated_tokens":3},"sequence":{"full_tokens":10}}}
 source.write_text(json.dumps({"sample_id":"x"})+'\n');out.write_text(json.dumps(row)+'\n');rej.write_text('')
 result=audit([source],[out],[rej]);assert result["kept"]==1 and result["natural_eos"]==1
