"""Temporal convolutional and attention models for sequence data."""

from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Remove padding from temporal convolution"""

    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Temporal Convolutional Block"""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super(TemporalBlock, self).__init__()
        if stride != 1:
            raise ValueError("TemporalBlock supports only stride=1")

        conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        nn.init.normal_(conv1.weight, 0, 0.01)
        self.conv1 = weight_norm(conv1)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        nn.init.normal_(conv2.weight, 0, 0.01)
        self.conv2 = weight_norm(conv2)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network"""

    def __init__(
        self, num_inputs: int, num_channels: List[int], kernel_size: int = 2, dropout: float = 0.2
    ):
        super(TemporalConvNet, self).__init__()
        if not num_channels:
            raise ValueError("num_channels must contain at least one value")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be greater than zero")
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNEncoder(nn.Module):
    """TCN Encoder for sequence encoding"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super(TCNEncoder, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        y = self.tcn(x)  # (batch, num_channels[-1], seq_len)
        y = y.transpose(1, 2)  # (batch, seq_len, num_channels[-1])
        return self.linear(y)  # (batch, seq_len, hidden_size)


class TCNDecoder(nn.Module):
    """TCN Decoder for sequence generation"""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super(TCNDecoder, self).__init__()
        reversed_channels = num_channels[::-1]
        self.tcn = TemporalConvNet(hidden_size, reversed_channels, kernel_size, dropout)
        self.linear = nn.Linear(reversed_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        x = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        y = self.tcn(x)  # (batch, num_channels[-1], seq_len)
        y = y.transpose(1, 2)  # (batch, seq_len, num_channels[-1])
        return self.linear(y)  # (batch, seq_len, output_size)


class Seq2SeqTCN(nn.Module):
    """Sequence-to-Sequence TCN for time series prediction"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super(Seq2SeqTCN, self).__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        self.encoder = TCNEncoder(input_size, hidden_size, num_channels, kernel_size, dropout)
        self.decoder = TCNDecoder(hidden_size, output_size, num_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TCNClassifier(nn.Module):
    """TCN for sequence classification"""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super(TCNClassifier, self).__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        y = self.tcn(x)  # (batch, num_channels[-1], seq_len)

        # Global average pooling across time dimension
        y = torch.mean(y, dim=2)  # (batch, num_channels[-1])
        y = self.dropout(y)
        return self.linear(y)  # (batch, num_classes)


class AttentionLayer(nn.Module):
    """Multi-head attention layer"""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super(AttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output


class TemporalTransformer(nn.Module):
    """Transformer-based temporal model combining TCN and attention"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 2,
        tcn_channels: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super(TemporalTransformer, self).__init__()
        if type(num_layers) is not int or num_layers < 0:
            raise ValueError("num_layers must be a non-negative integer")
        if tcn_channels is None:
            tcn_channels = [64, 64]

        # TCN encoder
        self.tcn_encoder = TCNEncoder(input_size, hidden_size, tcn_channels, dropout=dropout)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode with TCN
        encoded = self.tcn_encoder(x)  # (batch, seq_len, hidden_size)

        # Apply transformer layers
        for layer in self.transformer_layers:
            encoded = layer(encoded)

        # Global average pooling
        pooled = torch.mean(encoded, dim=1)  # (batch, hidden_size)
        pooled = self.dropout(pooled)

        # Classification
        return self.classifier(pooled)  # (batch, num_classes)


def create_temporal_model(
    model_type: str, input_size: int, num_classes: int, **kwargs
) -> nn.Module:
    """Factory function for temporal models"""

    if model_type == "tcn_classifier":
        unknown = set(kwargs) - {"num_channels", "kernel_size", "dropout"}
        if unknown:
            raise TypeError(f"unexpected options for {model_type}: {', '.join(sorted(unknown))}")
        return TCNClassifier(
            input_size=input_size,
            num_classes=num_classes,
            num_channels=kwargs.get("num_channels", [64, 64, 64]),
            kernel_size=kwargs.get("kernel_size", 2),
            dropout=kwargs.get("dropout", 0.2),
        )

    elif model_type == "seq2seq_tcn":
        unknown = set(kwargs) - {
            "output_size",
            "hidden_size",
            "num_channels",
            "kernel_size",
            "dropout",
        }
        if unknown:
            raise TypeError(f"unexpected options for {model_type}: {', '.join(sorted(unknown))}")
        return Seq2SeqTCN(
            input_size=input_size,
            output_size=kwargs.get("output_size", num_classes),
            hidden_size=kwargs.get("hidden_size", 64),
            num_channels=kwargs.get("num_channels", [64, 64, 64]),
            kernel_size=kwargs.get("kernel_size", 2),
            dropout=kwargs.get("dropout", 0.2),
        )

    elif model_type == "temporal_transformer":
        unknown = set(kwargs) - {
            "hidden_size",
            "num_heads",
            "num_layers",
            "tcn_channels",
            "dropout",
        }
        if unknown:
            raise TypeError(f"unexpected options for {model_type}: {', '.join(sorted(unknown))}")
        return TemporalTransformer(
            input_size=input_size,
            hidden_size=kwargs.get("hidden_size", 128),
            num_classes=num_classes,
            num_heads=kwargs.get("num_heads", 8),
            num_layers=kwargs.get("num_layers", 2),
            tcn_channels=kwargs.get("tcn_channels", [64, 64]),
            dropout=kwargs.get("dropout", 0.1),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
