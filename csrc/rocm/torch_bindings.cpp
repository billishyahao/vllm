#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, Tensor! out_c, int rows_per_block) -> "
      "()");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);
  rocm_ops.def(
      "LLMM_Silu(Tensor in_a, Tensor in_b, Tensor! out_c, int rows_per_block) "
      "-> ()");
  rocm_ops.impl("LLMM_Silu", torch::kCUDA, &LLMM_Silu);

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor context_lens, int block_size,"
      "                int max_context_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                int partition_size) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);
  rocm_ops.def(
      "wvSpltK(Tensor in_a, Tensor in_b, Tensor! out_c, int N_in,"
      "        int CuCount) -> ()");
  rocm_ops.impl("wvSpltK", torch::kCUDA, &wvSpltK);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
