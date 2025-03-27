import torch
from vllm.config import VllmConfig
import vllm.envs as envs
from vllm.logger import init_logger
from vllm import _custom_ops as ops

logger = init_logger(__name__)

class kv_helper(object):

    def __init__(self, config: VllmConfig):
        self.is_deepseek_mla = config.model_config.is_deepseek_mla
        self.use_mla_opt = not envs.VLLM_MLA_DISABLE
        self.tp_size = config.parallel_config.tensor_parallel_size

    def get_model_args(self, 
                        model_executable: torch.nn.Module):

        self.model_config = model_executable.model.config
        self.model_executable = model_executable
        num_heads = int(self.model_config.num_key_value_heads / self.tp_size)
        hidden_size = self.model_config.hidden_size
        num_attention_heads = self.model_config.num_attention_heads

        # Deepseek's MLA (Multi-head Latent Attention) uses two different
        # kv_cache shapes based on whether VLLM_MLA_DISABLE is set to 0.
        # When VLLM_MLA_DISABLE=0 (default), forward absorb is applied,
        # resulting in a kv_cache shape of [num_blks, blk_size, 1,
        # kv_lora_rank + qk_rope_head_dim].
        # When VLLM_MLA_DISABLE=1, standard FA is used instead, leading
        # to a kv_cache shape of [2, num_blks, blk_size,
        # num_key_value_heads / tp, qk_nope_head_dim + qk_rope_head_dim].
        # For more details, see vllm/attention/backends/mla/common.py.
        if self.is_deepseek_mla and self.use_mla_opt:
            head_size = self.model_config.kv_lora_rank + \
                self.model_config.qk_rope_head_dim
            num_heads = 1
        elif self.is_deepseek_mla and not self.use_mla_opt:
            head_size = self.model_config.qk_nope_head_dim + \
                self.model_config.qk_rope_head_dim
        else:
            head_size = getattr(self.model_config, "head_dim",
                                int(hidden_size // num_attention_heads))

        return num_heads, head_size

    def get_key_value_cache(self, kv_cache, num_heads, head_size):
        if self.is_deepseek_mla and self.use_mla_opt:
            key_cache = kv_cache.reshape(-1, num_heads, head_size)
            value_cache = kv_cache.reshape(-1, num_heads, head_size)
        else:
            key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
            value_cache = kv_cache[1].reshape(-1, num_heads, head_size)
        return key_cache, value_cache

    def put_kv_to_cache_from_mooncake(self, model_executable: torch.nn.Module,
                                      remote_kv, layer, layer_id, kv_cache, slot_mapping, start_pos, end_pos):

        self.model_config = model_executable.model.config

        if self.is_deepseek_mla and self.use_mla_opt:

            remote_k, remote_v = remote_kv[0][layer_id], remote_kv[1][layer_id]
            layer.self_attn.attn = layer.self_attn.mla_attn
            remote_k_c_normed_k_pe = remote_k.squeeze(1)
            logger.info(f"bill-dbg: remote_k_c_normed_k_pe.shape: {remote_k_c_normed_k_pe.shape}")
            remote_k_c_normed = remote_k_c_normed_k_pe[:, :self.model_config.kv_lora_rank]
            remote_k_pe = remote_k_c_normed_k_pe[:, self.model_config.kv_lora_rank:]
            ops.concat_and_cache_mla(
                remote_k_c_normed.to(kv_cache.device),
                remote_k_pe.to(kv_cache.device),
                kv_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
            )
        else:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            remote_k, remote_v = remote_kv[0][layer_id], remote_kv[1][
            layer_id]
            logger.info(f"bill-dbg: remote_k.shape: {remote_k.shape}")
            ops.reshape_and_cache_flash(
                remote_k.to(
                    key_cache.device),
                remote_v.to(
                    value_cache.device),
                key_cache,
                value_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
                layer.self_attn.attn._v_scale,
            )
    
    def put_kv_to_cache_from_simple(self, model_executable: torch.nn.Module,
                                    keys, values, start_layer, layer, layer_id, kv_cache, slot_mapping, start_pos, end_pos):

        self.model_config = model_executable.model.config

        if self.is_deepseek_mla and self.use_mla_opt:
            layer.self_attn.attn = layer.self_attn.mla_attn
            k_c_normed_k_pe = keys[
                layer_id - start_layer].to(
                    kv_cache.device).squeeze(1)
            k_c_normed = k_c_normed_k_pe[:, :self.model_config.kv_lora_rank]
            k_pe = k_c_normed_k_pe[:, self.model_config.kv_lora_rank:]
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe,
                kv_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
            )
        else:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            ops.reshape_and_cache_flash(
                keys[layer_id - start_layer].to(
                    key_cache.device),
                values[layer_id - start_layer].to(
                    value_cache.device),
                key_cache,
                value_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
                layer.self_attn.attn._v_scale,
            )