from __future__ import annotations

import pytest
import torch

from pytorch_distributed_training.models import (
    TCNClassifier,
    VisionTransformer,
    create_temporal_model,
)
from pytorch_distributed_training.models.temporal_network import TemporalBlock, TemporalTransformer
from pytorch_distributed_training.models.vision_transformer import Attention


def test_small_vision_transformer_forward_and_attention() -> None:
    model = VisionTransformer(
        img_size=16,
        patch_size=4,
        num_classes=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
    )
    inputs = torch.randn(2, 3, 16, 16)

    assert model(inputs).shape == (2, 3)
    assert model.get_last_selfattention(inputs).shape == (2, 4, 17, 17)

    empty_model = VisionTransformer(
        img_size=16,
        patch_size=4,
        num_classes=3,
        embed_dim=32,
        depth=0,
        num_heads=4,
    )
    with pytest.raises(RuntimeError, match="at least one"):
        empty_model.get_last_selfattention(inputs)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 3, 17, 19))
    with pytest.raises(ValueError, match="img_size"):
        VisionTransformer(img_size=17, patch_size=4, embed_dim=32, depth=1, num_heads=4)
    with pytest.raises(ValueError, match="depth"):
        VisionTransformer(img_size=16, patch_size=4, embed_dim=32, depth=-1, num_heads=4)
    with pytest.raises(ValueError, match="mlp_ratio"):
        VisionTransformer(
            img_size=16,
            patch_size=4,
            embed_dim=32,
            depth=1,
            num_heads=4,
            mlp_ratio=0,
        )

    zero_scale_attention = Attention(dim=4, num_heads=2, qk_scale=0)
    assert zero_scale_attention.scale == 0


def test_temporal_classifier_forward() -> None:
    model = TCNClassifier(
        input_size=6,
        num_classes=3,
        num_channels=[8, 8],
        dropout=0,
    )
    assert model(torch.randn(2, 12, 6)).shape == (2, 3)
    assert model.tcn.network[0].conv1.weight.detach().std() < 0.02


def test_temporal_model_honors_dropout_and_rejects_unsupported_stride() -> None:
    model = TemporalTransformer(
        input_size=3,
        hidden_size=8,
        num_classes=2,
        num_heads=2,
        tcn_channels=[4],
        dropout=0,
    )

    assert model.tcn_encoder.tcn.network[0].dropout1.p == 0
    with pytest.raises(ValueError, match="stride=1"):
        TemporalBlock(3, 4, kernel_size=3, stride=2, dilation=1, padding=2)
    with pytest.raises(ValueError, match="num_layers"):
        TemporalTransformer(3, 8, 2, num_heads=2, num_layers=-1, tcn_channels=[4])


def test_seq2seq_factory_forward() -> None:
    model = create_temporal_model(
        "seq2seq_tcn",
        input_size=4,
        num_classes=2,
        output_size=1,
        hidden_size=8,
        num_channels=[8],
        dropout=0,
    )
    assert model(torch.randn(2, 10, 4)).shape == (2, 10, 1)

    default_output = create_temporal_model(
        "seq2seq_tcn",
        input_size=4,
        num_classes=3,
        hidden_size=8,
        num_channels=[8],
        dropout=0,
    )
    assert default_output(torch.randn(2, 10, 4)).shape == (2, 10, 3)

    with pytest.raises(TypeError, match="kernel_sze"):
        create_temporal_model("tcn_classifier", 4, 2, kernel_sze=3)
