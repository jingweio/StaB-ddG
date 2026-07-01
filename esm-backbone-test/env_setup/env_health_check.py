import sys, torch, transformers, esm, peft, biotite, numpy, scipy, gemmi
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "| cuda_avail", torch.cuda.is_available(), "| cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("esm         ", esm.__version__, "->", esm.__file__)
print("peft        ", peft.__version__)
print("biotite     ", biotite.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__, "| gemmi", gemmi.__version__)
# scorer-side API imports (what our esm3_stab scorer uses)
from esm.pretrained import ESM3_sm_open_v0
from esm.utils.encoding import tokenize_sequence
from esm.models.esm3 import ESM3, ESMOutput
from peft import LoraConfig, inject_adapter_in_model
print("scorer-side imports OK (ESM3_sm_open_v0 / tokenize_sequence / ESMOutput / peft.inject_adapter_in_model)")
print("HEALTH_OK")
