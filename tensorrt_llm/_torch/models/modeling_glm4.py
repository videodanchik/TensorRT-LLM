import re

from transformers import PretrainedConfig

import torch
from .modeling_deepseekv3 import DeepseekV3DecoderLayer
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.embedding import Embedding
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType


class Glm4MoeDecoderLayer(DeepseekV3DecoderLayer):
    def __init__(
            self,
            model_config: ModelConfig[PretrainedConfig],
            layer_idx: int,
            aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        ):
        super().__init__(
            model_config=model_config, layer_idx=layer_idx, aux_stream_dict=aux_stream_dict,
        )


class Glm4MoeModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        self.layers = torch.nn.ModuleList([
            DeepseekV3DecoderLayer(model_config, layer_idx, self.aux_stream_dict)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        spec_metadata: SpecMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None

        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
            )

        return hidden_states


@register_auto_model("Glm4MoeForCausalLM")
class Glm4MoeForCausalLM(DecoderModelForCausalLM[Glm4MoeModel, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        model_config.quant_config.group_size = 64
        super().__init__(
            Glm4MoeModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: dict[str, torch.Tensor]):
        for name, module in self.model.named_modules():
            if name == "embed_tokens":
                if len(tuple(module.named_parameters())) > 0:
                    module.load_weights(weights=[{"weight": weights["model.embed_tokens.weight"]}])

            elif re.fullmatch(r"layers\.\d+", name):
                for n, p in module.named_parameters():
                    if n in (
                        "input_layernorm.weight", "post_attention_layernorm.weight",
                    ):
                        p.data.copy_(weights[f"model.{name}.{n}"][:])

                # Use first_k_dense_replace from config instead of hardcoding
                layer_idx = int(name.split('.')[1])
                first_k_dense = getattr(self.config, 'first_k_dense_replace', 0)
                if layer_idx < first_k_dense:
                    if any("mlp." in n for n, _ in module.named_parameters()):
                        module.mlp.gate_up_proj.load_weights(
                            weights=[
                                {"weight": weights[f"model.{name}.mlp.gate_proj.weight"]},
                                {"weight": weights[f"model.{name}.mlp.up_proj.weight"]},
                            ]
                        )

                        module.mlp.down_proj.load_weights(
                            weights=[{"weight": weights[f"model.{name}.mlp.down_proj.weight"]}]
                        )
                else:

                    if any("mlp.experts." in n for n, _ in module.named_parameters()):
                        experts_prefix = f"model.{name}.mlp.experts."
                        module.mlp.experts.load_weights(
                            weights=[
                                {
                                    weight_name.replace(experts_prefix, "")\
                                                .replace("gate_proj", "w1")\
                                                .replace("down_proj", "w2")\
                                                .replace("up_proj", "w3"): weights[weight_name]
                                    for weight_name in weights if weight_name.startswith(experts_prefix)
                                },
                            ]
                        )

                    if any("mlp.gate." in n for n, _ in module.named_parameters()):
                        module.mlp.gate.load_weights(
                            weights=[
                                {
                                    "weight": weights[f"model.{name}.mlp.gate.weight"],
                                    "e_score_correction_bias": weights[f"model.{name}.mlp.gate.e_score_correction_bias"],
                                },
                            ]
                        )

                    if any("mlp.shared_experts." in n for n, _ in module.named_parameters()):
                        module.mlp.shared_experts.gate_up_proj.load_weights(
                            weights=[
                                {"weight": weights[f"model.{name}.mlp.shared_experts.gate_proj.weight"]},
                                {"weight": weights[f"model.{name}.mlp.shared_experts.up_proj.weight"]},
                            ]
                        )

                        module.mlp.shared_experts.down_proj.load_weights(
                            weights=[{"weight": weights[f"model.{name}.mlp.shared_experts.down_proj.weight"]}]
                        )

                if any("self_attn." in n for n, _ in module.named_parameters()):

                    # Only load q_norm/k_norm if use_qk_norm is enabled
                    if hasattr(self.config, 'use_qk_norm') and self.config.use_qk_norm:
                        for n, p in module.self_attn.named_parameters():
                            if n in ("q_norm.weight", "k_norm.weight"):
                                p.data.copy_(weights[f"model.{name}.self_attn.{n}"][:])

                    module.self_attn.qkv_proj.load_weights(
                        weights=[
                            {
                                "weight": weights[f"model.{name}.self_attn.q_proj.weight"],
                                "bias": weights[f"model.{name}.self_attn.q_proj.bias"],
                            },
                            {
                                "weight": weights[f"model.{name}.self_attn.k_proj.weight"],
                                "bias": weights[f"model.{name}.self_attn.k_proj.bias"],
                            },
                            {
                                "weight": weights[f"model.{name}.self_attn.v_proj.weight"],
                                "bias": weights[f"model.{name}.self_attn.v_proj.bias"],
                            },
                        ]
                    )

                    module.self_attn.o_proj.load_weights(
                        weights=[{"weight": weights[f"model.{name}.self_attn.o_proj.weight"]}]
                    )

            elif name == "norm":
                for n, p in module.named_parameters():
                    p.data.copy_(weights[f"model.{name}.{n}"][:])

        if len(tuple(self.lm_head.named_parameters())) > 0:
            self.lm_head.load_weights(weights=[{"weight": weights[f"lm_head.weight"]}])

    def post_load_weights(self):
        for idx, layer in enumerate(self.model.layers):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[idx + 1].input_layernorm
