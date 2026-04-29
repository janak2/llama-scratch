import json
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
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_size = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.rope_theta = rope_theta
        self.device = "cpu"

        self.q_proj = nn.Linear(hidden_size, self.head_size * num_heads, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.head_size * num_key_value_heads, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.head_size * num_key_value_heads, bias=False
        )
        self.o_proj = nn.Linear(self.head_size * num_heads, hidden_size, bias=False)

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
    ):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.freqs_cis[: xq.shape[1]]
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        B, T, _ = x.shape

        q = self.q_proj(x)
        v = self.v_proj(x)
        k = self.k_proj(x)

        q = q.view(B, T, self.num_heads, self.head_size)
        k = k.view(B, T, self.num_key_value_heads, self.head_size)

        q, k = self.apply_rotary_emb(q, k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(
            self.num_heads // self.num_key_value_heads, 1
        )

        v = (
            v.view(B, T, self.num_key_value_heads, self.head_size)
            .transpose(1, 2)
            .repeat_interleave(self.num_heads // self.num_key_value_heads, 1)
        )

        h = torch.softmax(q @ k.transpose(2, 3), dim=-1) @ v
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
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = SelfAttention(
            hidden_size, num_heads, num_key_value_heads, rope_theta
        )
        self.mlp = MultiLayerPerceptron(hidden_size, intermediate_size)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(self, embeds: torch.Tensor, attention_mask: torch.Tensor):
        residual = embeds
        embeds = self.input_layernorm(embeds)
        embeds = self.self_attn(embeds, attention_mask)
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
    ):
        super().__init__()
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
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, embeds: torch.Tensor, attention_mask: torch.Tensor):
        for layer in self.layers:
            embeds = layer(embeds, attention_mask)
        embeds = self.norm(embeds)
        return embeds

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 10,
    ):
        print(input_ids.shape)
        embeds = self.embed_tokens(input_ids)
        generated_tokens = []
        for _ in range(max_new_tokens):
            hidden_states = self.forward(embeds, attention_mask)
            logits = self.lm_head(hidden_states[:, -1, :])
            next_token = torch.argmax(logits, dim=-1)

            print("next_token", next_token)

            if next_token in self.eos_tokens:
                print("EOS token found")
                break

            generated_tokens.append(next_token)

            embeds = torch.cat(
                [embeds, self.embed_tokens(next_token.unsqueeze(1))], dim=1
            )

        print("generated_tokens", generated_tokens)

        return generated_tokens

    @staticmethod
    def from_pretrained(model_path: Path):
        config = json.load(open(model_path / "config.json"))

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
        )

        state_dict = {}
        for shard_path in sorted(model_path.glob("model-*-of-*.safetensors")):
            for key, value in load_file(shard_path, device="cpu").items():
                # HF checkpoints keep the decoder under "model."; this class does not.
                state_dict[key.removeprefix("model.")] = value

        model.load_state_dict(state_dict)

        return model
