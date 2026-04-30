import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        dtype = x.dtype
        x = x.to(dtype=torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x.to(dtype=dtype)
        x = x * self.weight
        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        rope_theta: float,
        max_seq_len: int,
        max_batch_size: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_size = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.device = "cpu"

        self.q_proj = nn.Linear(hidden_size, self.head_size * num_heads, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.head_size * num_key_value_heads, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.head_size * num_key_value_heads, bias=False
        )
        self.o_proj = nn.Linear(self.head_size * num_heads, hidden_size, bias=False)

        self.kv_cache = {
            "k": torch.zeros(
                max_batch_size, num_key_value_heads, max_seq_len, self.head_size
            ),
            "v": torch.zeros(
                max_batch_size, num_key_value_heads, max_seq_len, self.head_size
            ),
        }

        self.freqs_cis = self.precompute_freqs_cis()

    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        # Values obtained from grid search
        scale_factor = 8
        low_freq_factor = 1
        high_freq_factor = 4
        old_context_len = 8192  # original llama3 length

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * torch.pi / freqs
        new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
        smooth = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        return torch.where(
            (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
            (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
            new_freqs,
        )

    def precompute_freqs_cis(self, end: int = 200, use_scaled=False):
        freqs = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_size, 2)[: (self.head_size // 2)].float()
                / self.head_size
            )
        )
        t = torch.arange(end, device=self.device, dtype=torch.float32)
        if use_scaled:
            freqs = self.apply_scaling(freqs)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        start_pos: int,
    ):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.freqs_cis[start_pos : start_pos + xq.shape[1]]
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        use_kv_cache: bool,
    ):
        B, T, _ = x.shape

        q = self.q_proj(x)
        v = self.v_proj(x)
        k = self.k_proj(x)

        q = q.view(B, T, self.num_heads, self.head_size)
        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        if use_kv_cache:
            q, k = self.apply_rotary_emb(q, k, start_pos)
        else:
            q, k = self.apply_rotary_emb(q, k, 0)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if use_kv_cache:
            self.kv_cache["k"][:B, :, start_pos : start_pos + T] = k
            self.kv_cache["v"][:B, :, start_pos : start_pos + T] = v
            k = self.kv_cache["k"][:B, :, : start_pos + T]
            v = self.kv_cache["v"][:B, :, : start_pos + T]

        k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, 1)
        v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, 1)

        h = (q @ k.transpose(2, 3)) / math.sqrt(self.head_size)

        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)

        h = h + mask.unsqueeze(0).unsqueeze(0)

        h = torch.softmax(h, dim=-1) @ v
        h = self.o_proj(
            h.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size)
        )

        return h


class Layer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_seq_len: int,
        max_batch_size: int,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = SelfAttention(
            hidden_size,
            num_heads,
            num_key_value_heads,
            rope_theta,
            max_seq_len,
            max_batch_size,
        )
        self.mlp = MultiLayerPerceptron(hidden_size, intermediate_size)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        embeds: torch.Tensor,
        start_pos: int,
        use_kv_cache: bool,
    ):
        residual = embeds
        embeds = self.input_layernorm(embeds)
        embeds = self.self_attn(embeds, start_pos, use_kv_cache)
        embeds = residual + embeds
        residual = embeds
        embeds = self.post_attention_layernorm(embeds)
        embeds = self.mlp(embeds)
        embeds = residual + embeds
        return embeds


class Llama3(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        eos_token: list[int],
        rms_norm_eps: float,
        rope_theta: float,
        max_seq_len: int,
        max_batch_size: int,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.eos_tokens = eos_token
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                Layer(
                    hidden_size,
                    num_heads,
                    num_key_value_heads,
                    intermediate_size,
                    rms_norm_eps,
                    rope_theta,
                    max_seq_len,
                    max_batch_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        embeds: torch.Tensor,
        start_pos: int,
        use_kv_cache: bool,
    ):
        for layer in self.layers:
            embeds = layer(embeds, start_pos, use_kv_cache)
        embeds = self.norm(embeds)
        return embeds

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 20,
        use_kv_cache: bool = True,
    ):
        print(input_ids.shape)
        assert input_ids.shape[0] <= self.max_batch_size, (
            input_ids.shape[0],
            self.max_batch_size,
        )
        embeds = self.embed_tokens(input_ids)
        B, T, _ = embeds.shape
        eos_reached = torch.zeros(B, dtype=torch.bool)
        generated_tokens = [[] for _ in range(B)]
        start_pos = 0

        prompt_lens = attention_mask.sum(dim=-1)

        start_pos = 0
        end_pos = prompt_lens.min().item()
        for _ in range(max_new_tokens):
            if use_kv_cache:
                hidden_states = self.forward(
                    embeds[:, start_pos:end_pos],
                    start_pos,
                    use_kv_cache=True,
                )
                start_pos = end_pos
                end_pos += 1
            else:
                hidden_states = self.forward(
                    embeds[:, start_pos:end_pos], start_pos, use_kv_cache=False
                )
                end_pos += 1
            logits = self.lm_head(hidden_states[:, -1, :])
            next_token = torch.argmax(logits, dim=-1)

            print("next_token", next_token)

            for b in range(B):
                if next_token[b] in self.eos_tokens and end_pos > prompt_lens[b]:
                    eos_reached[b] = True

            if all(eos_reached):
                break

            for b in range(B):
                if not eos_reached[b] and end_pos > prompt_lens[b]:
                    generated_tokens[b].append(next_token[b])

            if end_pos > embeds.shape[1]:
                embeds = torch.cat(
                    [embeds, self.embed_tokens(next_token.unsqueeze(1))], dim=1
                )
            else:
                for b in range(B):
                    if end_pos > prompt_lens[b]:
                        embeds[b, end_pos - 1] = self.embed_tokens(next_token[b])

        print("generated_tokens", generated_tokens)

        return generated_tokens

    @staticmethod
    def from_pretrained(
        model_path: Path,
        max_batch_size: int = 2,
        max_seq_len: int = 2048,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        config = json.load(open(model_path / "config.json"))

        with torch.device(device):
            torch.set_default_dtype(dtype)
            model = Llama3(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_hidden_layers"],
                num_heads=config["num_attention_heads"],
                num_key_value_heads=config["num_key_value_heads"],
                intermediate_size=config["intermediate_size"],
                eos_token=config["eos_token_id"],
                rms_norm_eps=config["rms_norm_eps"],
                rope_theta=config["rope_theta"],
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
            )

        state_dict = {}
        for shard_path in sorted(model_path.glob("model-*-of-*.safetensors")):
            for key, value in load_file(shard_path, device=device).items():
                # HF checkpoints keep the decoder under "model."; this class does not.
                state_dict[key.removeprefix("model.")] = value

        model.load_state_dict(state_dict)

        return model
