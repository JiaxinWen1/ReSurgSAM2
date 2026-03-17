import torch
from torch import nn
from typing import Tuple, List
import torch.nn.functional as F
import math
from functools import partial
from typing import Tuple, Type

import torch
from torch import nn, Tensor

from sam2.modeling.sam.mamba_block import MambaLayer
from sam2.modeling.sam.transformer import Attention, MLP
from sam2.modeling.sam2_utils import MLPDropout
from timm.models.layers import trunc_normal_


class CrossModalFusionModule(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        image_embedding_size: Tuple[int, int],
        use_feature_level: Tuple[int, ...] = (2),  # the last layer
        depth: int=3,
        bimamba: bool=False,
        use_sp_bimamba: bool=False,
        use_dwconv: bool=False,
        dropout: float=0.2,
        use_mamba_attn: bool=True,
        num_temp_pos_embed: int=3,
        pad_sequence: bool=False,
        num_ref_frames: int=3,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.image_embedding_size = image_embedding_size
        self.num_temp_pos_embed = num_temp_pos_embed
        self.pad_sequence = pad_sequence
        self.num_ref_frames = num_ref_frames

        self.transformer = TwoWayTokenTransformer(
            depth=depth,
            embedding_dim=self.transformer_dim,
            mlp_dim=2048,
            num_heads=8,
            attention_downsample_rate=2,
            use_mamba_before_cross_attn=use_mamba_attn,
            drop_path_rate=0.2,
            bimamba=bimamba,
            sp_bimamba=use_sp_bimamba,
            use_dwconv=use_dwconv,
            dropout=dropout,
        )
        print("CrossModalFusionModule depth=", depth)
        self.cls_token = nn.Embedding(1, transformer_dim)
        self.use_feature_level = use_feature_level
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_temp_pos_embed, transformer_dim))
        self.use_dwconv = use_dwconv

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        image_embeddings: List[torch.Tensor],  # (H*W, B, C)
        image_pe: List[torch.Tensor],  # (H*W, B, C)
        text_embeddings: torch.Tensor,  # (B, N, C)
        feat_sizes: List[Tuple[int, int]],
        previous_ref_feats_list: List[List],
        previous_ref_pos_embeds_list: List[List],
        return_intermediate=False,
    ):
        output_tokens = self.cls_token.weight
        output_tokens = output_tokens.unsqueeze(0).expand(
            text_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, text_embeddings), dim=1)

        # read out the current level of features and organize them
        h, w = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        b, c = image_embeddings[-1].shape[1:]
        f = len(previous_ref_feats_list) + 1

        # current frame
        keys = image_embeddings[-1]
        keys_pe = image_pe[-1]
        keys_tmp_embed = self.temporal_pos_embed[:, 0:1, :].repeat(image_embeddings[-1].size(0), 1, 1)

        # previous frame
        previous_feat_list = []
        previous_pos_embed_list = []
        previous_tmp_embed_list = []

        pad_num = 0
        if self.pad_sequence:
            pad_num = self.num_ref_frames - len(previous_ref_feats_list) - 1
            f = self.num_ref_frames
            for j in range(pad_num):
                previous_feat_list.append(keys)
                previous_pos_embed_list.append(keys_pe)
                if self.num_temp_pos_embed == 3:
                    time_index = 1 if j == 0 else 2
                else:
                    time_index = pad_num + j + 1
                tmp_embed = self.temporal_pos_embed[:, time_index:time_index + 1, :].repeat(image_embeddings[-1].size(0), 1, 1)
                previous_tmp_embed_list.append(tmp_embed)

        if len(previous_ref_feats_list) != 0:
            for j in range(len(previous_ref_feats_list)):
                previous_feat_list.append(previous_ref_feats_list[j][-1])
                previous_pos_embed_list.append(previous_ref_pos_embeds_list[j][-1])
                if self.num_temp_pos_embed == 3:
                    time_index = 1 if j == 0 else 2
                else:
                    time_index = pad_num + j + 1
                tmp_embed = self.temporal_pos_embed[:, time_index:time_index+1, :].repeat(image_embeddings[-1].size(0), 1, 1)
                previous_tmp_embed_list.append(tmp_embed)

        keys = torch.cat([keys, *previous_feat_list], dim=0)
        keys_pe = torch.cat([keys_pe, *previous_pos_embed_list], dim=0)
        keys_tmp_embed = torch.cat([keys_tmp_embed, *previous_tmp_embed_list], dim=0)

        keys_pe = keys_pe + keys_tmp_embed
        # hs (B, N, C), src (B, H*W, C)
        hs, src = self.transformer(keys, keys_pe, tokens, vol_sizes=(f, h, w))

        image_embeddings = src  # (B, H*W, C)
        text_embeddings = hs  # (B, N, C)
        cls_tokens = text_embeddings[:, :1, :]

        image_embeddings = image_embeddings.reshape(-1, f, h*w, c).permute(1, 2, 0, 3)

        if not return_intermediate:
            image_embeddings = image_embeddings[0]  # (f, h*w, b, c)

        return image_embeddings, cls_tokens


class TwoWayTokenTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        use_mamba_before_cross_attn: bool = False,
        drop_path_rate: float = 0.2,
        bimamba: bool = False,
        sp_bimamba: bool = False,
        use_dwconv: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.use_dwconv = use_dwconv
        self.layers = nn.ModuleList()

        mamba_depths = [2 for i in range(depth)]
        self.drop_path_rate = drop_path_rate
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(mamba_depths))]
        for i in range(depth):
            self.layers.append(
                TwoWayTokenAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    use_mamba_before_cross_attn=use_mamba_before_cross_attn,
                    temporal_drop_rates=dp_rates[i * 2: (i + 1) * 2],
                    bimamba=bimamba,
                    sp_bimamba=sp_bimamba,
                    use_dwconv=use_dwconv,
                    dropout=dropout,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        prompt_embedding: Tensor,
        vol_sizes: Tuple[int, int, int]=None,
    ) -> Tuple[Tensor, Tensor]:
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.permute(1, 0, 2)
        image_pe = image_pe.permute(1, 0, 2)

        # Prepare queries
        queries = prompt_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=prompt_embedding,
                key_pe=image_pe,
                vol_sizes=vol_sizes,
            )

        # Apply the final attention layer from the points to the image
        q = queries + prompt_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayTokenAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        use_mamba_before_cross_attn: bool = False,
        temporal_drop_rates: List[float] = None,
        bimamba: bool = False,
        sp_bimamba: bool = False,
        use_dwconv: bool = False,
        dropout: float = 0.2,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPDropout(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation, dropout=dropout
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        # =========================================================
        # 改动1：Gated Side-Adapter（T2V 门控）
        # 用名词token的最大激活而非全局CLS生成空间权重图
        # 压缩到64维再做门控，减少计算量
        # =========================================================
        self.compress = nn.Conv2d(embedding_dim, 64, kernel_size=1)           # 压缩：256→64
        self.noun_gate_proj = nn.Linear(embedding_dim, 64)                    # 文本名词token投影到64维
        self.expand = nn.Conv2d(64, embedding_dim, kernel_size=1)             # 扩张：64→256
        self.gate_norm = nn.LayerNorm(embedding_dim)                          # 残差后norm
        # 初始化 expand 为零，确保训练初期gate不影响原始特征
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

        # =========================================================
        # 改动2：V2T 置信度加权（遮挡鲁棒性）
        # 用空间gate权重对视觉token加权，降低遮挡区域噪声影响
        # =========================================================
        self.visibility_proj = nn.Linear(embedding_dim, embedding_dim)       # 可见性权重投影
        self.visibility_norm = nn.LayerNorm(embedding_dim)                    # 加权后norm
        # 初始化为单位变换，确保训练初期行为与原版一致
        nn.init.eye_(self.visibility_proj.weight)
        nn.init.zeros_(self.visibility_proj.bias)
        # =========================================================

        self.skip_first_layer_pe = skip_first_layer_pe
        self.use_mamba_before_cross_attn = use_mamba_before_cross_attn
        self.use_dwconv = use_dwconv
        if self.use_mamba_before_cross_attn:
            self.mamba_layers = nn.ModuleList()
            for rate in temporal_drop_rates:
                self.mamba_layers.append(
                    MambaLayer(
                        dim=embedding_dim,
                        drop_path=rate,
                        bimamba=bimamba,
                        sp_bimamba=sp_bimamba,
                        use_dwconv=use_dwconv,
                    )
                )
            print("TwoWayTokenAttentionBlock use_mamba_before_cross_attn")

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor,
        vol_sizes: Tuple[int, int, int]=None,
    ) -> Tuple[Tensor, Tensor]:

        # ----------------------------------------------------------
        # Self attention block（原版不变）
        # ----------------------------------------------------------
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        if self.use_mamba_before_cross_attn:
            for layer in self.mamba_layers:
                keys = layer(keys, vol_sizes)

        # ----------------------------------------------------------
        # T2V Cross attention（原版不变）
        # ----------------------------------------------------------
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block（原版不变）
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # ----------------------------------------------------------
        # 改动1：Gated Side-Adapter
        # 只作用于当前帧的视觉特征（cur_keys），历史帧不动
        # ----------------------------------------------------------
        gate_map = None  # 用于改动2复用
        if vol_sizes is not None:
            f, h, w = vol_sizes
            B, N_all, C = keys.shape

            # 取当前帧视觉特征
            cur_keys = keys[:, :h*w, :]                                       # (B, H*W, C)
            rest_keys = keys[:, h*w:, :]                                      # (B, rest, C)

            # 文本侧：取名词token的最大激活（比CLS更有判别性）
            # queries[:,0] 是CLS token，[:,1:] 是文本序列token
            text_tokens = queries[:, 1:, :]                                   # (B, N_text, C)
            # max over token维度，取最显著的token作为判别特征
            noun_feat = text_tokens.max(dim=1).values                         # (B, C)
            noun_gate = self.noun_gate_proj(noun_feat)                        # (B, 64)

            # 视觉侧：压缩到64维空间做门控
            cur_keys_2d = cur_keys.permute(0, 2, 1).reshape(B, C, h, w)      # (B, C, H, W)
            cur_compressed = self.compress(cur_keys_2d)                       # (B, 64, H, W)

            # 计算空间权重图：文本gate与压缩视觉特征的逐通道点积
            # noun_gate: (B, 64) → (B, 64, 1, 1)
            gate_map = torch.sigmoid(
                (cur_compressed * noun_gate.view(B, 64, 1, 1)).sum(dim=1, keepdim=True)
            )                                                                  # (B, 1, H, W)

            # 双向门控：高激活增强，低激活抑制（不归零保留30%）
            # gate_map=1 → ×1.0（完全保留），gate_map=0 → ×0.3（抑制但不消除）
            gated_compressed = cur_compressed * (0.3 + 0.7 * gate_map)       # (B, 64, H, W)

            # 扩张回256维，残差连接（expand初始化为零，训练初期=原始特征）
            gated_expanded = self.expand(gated_compressed)                    # (B, C, H, W)
            gated_expanded = gated_expanded.reshape(B, C, h*w).permute(0, 2, 1)  # (B, H*W, C)
            cur_keys_out = cur_keys + gated_expanded                          # 残差连接
            cur_keys_out = self.gate_norm(cur_keys_out)

            keys = torch.cat([cur_keys_out, rest_keys], dim=1)                # (B, N_all, C)

        # ----------------------------------------------------------
        # 改动2：V2T 置信度加权
        # 用 gate_map 对当前帧视觉token加权，遮挡区域自动降权
        # 历史帧保持原样（它们本身就是"干净帧"）
        # ----------------------------------------------------------
        if vol_sizes is not None and gate_map is not None:
            f, h, w = vol_sizes
            B, N_all, C = keys.shape

            # gate_map: (B, 1, H, W) → 可见性权重 [0.3, 1.0]
            visibility = (0.3 + 0.7 * gate_map)                              # (B, 1, H, W)
            visibility = visibility.reshape(B, h*w, 1)                       # (B, H*W, 1)

            # 当前帧加权，历史帧权重为1（不变）
            rest_len = N_all - h*w
            ones = torch.ones(B, rest_len, 1, device=keys.device)
            visibility_full = torch.cat([visibility, ones], dim=1)           # (B, N_all, 1)

            # 加权后做线性变换 + 残差（visibility_proj初始化为单位阵，训练初期=原版）
            keys_weighted = keys * visibility_full
            keys_proj = self.visibility_proj(keys_weighted)
            keys_for_v2t = self.visibility_norm(keys + keys_proj)            # 残差保证稳定
        else:
            keys_for_v2t = keys

        # ----------------------------------------------------------
        # V2T Cross attention（结构与原版一致，输入换为加权后的keys）
        # ----------------------------------------------------------
        q = queries + query_pe
        k = keys_for_v2t + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out                                                # 残差仍基于原始keys
        keys = self.norm4(keys)

        return queries, keys