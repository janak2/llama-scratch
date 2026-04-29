import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        dtype = x.dtype
        x = x.to(dtype=torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keep_dim=True) + self.eps)
        x = x.to(dtype=dtype)
        x = x * self.weight
        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int):
        super().__init__()
        head_size = hidden_size // num_heads
        self.q_proj = nn.Linear(head_size * num_heads, hidden_size, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, head_size * num_key_value_heads, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, head_size * num_key_value_heads, bias=False
        )
        self.o_proj = nn.Linear(hidden_size, head_size * num_heads, bias=False)


class Layer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = SelfAttention(hidden_size, num_heads, num_key_value_heads)
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
        max_new_tokens: int = 100,
    ):
        embeds = self.embed_tokens(input_ids)
        for _ in range(max_new_tokens):
            hidden_states = self.forward(embeds, attention_mask)
            logits = self.lm_head(hidden_states)
            next_token = torch.argmax(logits, dim=-1)

            if next_token in self.eos_tokens:
                break

            embeds = torch.cat(
                [embeds, self.embed_tokens(next_token.unsqueeze(1))], dim=-1
            )

        return input_ids

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
        )

        state_dict = {}
        for shard_path in sorted(model_path.glob("model-*-of-*.safetensors")):
            for key, value in load_file(shard_path, device="cpu").items():
                # HF checkpoints keep the decoder under "model."; this class does not.
                state_dict[key.removeprefix("model.")] = value

        print(state_dict.keys())
        model.load_state_dict(state_dict)

        return model
