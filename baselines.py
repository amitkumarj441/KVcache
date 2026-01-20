import transformers

from h2o_base import H2OAttention, H2OCache
from kvc_main import OursAttention, OursCache
#from rkv_base import RKVAttention, RKVCache - ToDo
#from vatp_base import VATPAttention, VATPCache - ToDo
#from asink import SinkAttention, SinkCache - ToDo

def init_h2o():
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = H2OAttention.forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = H2OAttention.forward
    return H2OCache

def init_kvc():
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = OursAttention.forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = OursAttention.forward
    return OursCache
