"""Model description = the *upper* layer of the simulator.

A model is a list of layer groups; each group repeats `repeat` times and contains
blocks. Blocks are high-level (mla / gqa / mlp / moe / norm) or raw (gemm /
elementwise). You can hand-write this YAML for any model ("here are the sizes of
every Kimi layer") or import a HuggingFace config.json:

    name: kimi-k2
    hidden: 7168
    vocab: 163840
    layers:
      - name: dense
        repeat: 1
        blocks:
          - {type: norm}
          - {type: mla, heads: 64, q_lora_rank: 1536, kv_lora_rank: 512,
             qk_nope: 128, qk_rope: 64, v_head: 128}
          - {type: norm}
          - {type: mlp, d_ff: 18432}
      - name: moe
        repeat: 60
        blocks:
          - ...
          - {type: moe, experts: 384, topk: 8, d_ff: 2048, shared_experts: 1}

Raw GEMM block: {type: gemm, name: my_proj, N: 4096, K: 7168, shard: col|row|none,
                 tokens: attn|moe, weight: attn|mlp|expert}
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PRESET_DIR = Path(__file__).parent / "presets"


@dataclass
class LayerGroup:
    name: str
    repeat: int
    blocks: list[dict[str, Any]]


@dataclass
class ModelSpec:
    name: str
    hidden: int
    vocab: int
    layers: list[LayerGroup] = field(default_factory=list)
    tie_embeddings: bool = False

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, name_or_path: str) -> "ModelSpec":
        p = Path(name_or_path)
        if not p.exists():
            for ext in (".yaml", ".json"):
                if (PRESET_DIR / f"{name_or_path}{ext}").exists():
                    p = PRESET_DIR / f"{name_or_path}{ext}"
                    break
        text = p.read_text()
        raw = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
        if "layers" not in raw:                 # looks like a HF config.json
            return cls.from_hf_config(raw)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelSpec":
        raw = copy.deepcopy(raw)
        layers = [LayerGroup(l["name"], int(l.get("repeat", 1)), l["blocks"]) for l in raw["layers"]]
        return cls(raw["name"], int(raw["hidden"]), int(raw["vocab"]), layers, bool(raw.get("tie_embeddings", False)))

    def to_dict(self) -> dict:
        return dict(name=self.name, hidden=self.hidden, vocab=self.vocab, tie_embeddings=self.tie_embeddings,
                    layers=[dict(name=l.name, repeat=l.repeat, blocks=l.blocks) for l in self.layers])

    def dump_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    # ------------------------------------------------------------------ HF import
    @classmethod
    def from_hf_config(cls, cfg: dict) -> "ModelSpec":
        cfg = cfg.get("text_config", cfg)           # multimodal wrappers (e.g. Kimi-K2.5)
        mt = cfg.get("model_type", "")
        H, V = cfg["hidden_size"], cfg["vocab_size"]
        L = cfg["num_hidden_layers"]
        name = cfg.get("_name_or_path") or mt
        if "q_lora_rank" in cfg or mt.startswith("deepseek") or mt.startswith("kimi"):
            attn = dict(type="mla", heads=cfg["num_attention_heads"], q_lora_rank=cfg.get("q_lora_rank") or 0,
                        kv_lora_rank=cfg["kv_lora_rank"], qk_nope=cfg["qk_nope_head_dim"],
                        qk_rope=cfg["qk_rope_head_dim"], v_head=cfg["v_head_dim"])
        else:
            nh = cfg["num_attention_heads"]
            hd = cfg.get("head_dim") or H // nh
            kvh = cfg.get("num_key_value_heads") or nh
            attn = dict(type="mha", heads=nh, head_dim=hd) if kvh == nh else \
                dict(type="gqa", heads=nh, kv_heads=kvh, head_dim=hd)
            if mt.startswith(("qwen3", "gemma3")):
                attn["qk_norm"] = True
            if cfg.get("partial_rotary_factor"):
                attn["rope_dim"] = int(hd * cfg["partial_rotary_factor"])
            if cfg.get("use_sliding_window") and cfg.get("sliding_window"):
                attn["sliding_window"] = cfg["sliding_window"]
        n_exp = cfg.get("n_routed_experts") or cfg.get("num_experts") or cfg.get("num_local_experts") or 0
        dense_first = cfg.get("first_k_dense_replace", 0 if n_exp else L)
        groups = []
        if dense_first:
            groups.append(LayerGroup("dense", dense_first, [
                {"type": "norm"}, attn, {"type": "norm"}, {"type": "mlp", "d_ff": cfg["intermediate_size"]}]))
        if n_exp and L - dense_first > 0:
            moe = dict(type="moe", experts=n_exp, topk=cfg.get("num_experts_per_tok", 8),
                       d_ff=cfg.get("moe_intermediate_size", cfg["intermediate_size"]),
                       shared_experts=cfg.get("n_shared_experts", 0) or 0)
            groups.append(LayerGroup("moe", L - dense_first, [{"type": "norm"}, attn, {"type": "norm"}, moe]))
        return cls(name, H, V, groups, bool(cfg.get("tie_word_embeddings", False)))
