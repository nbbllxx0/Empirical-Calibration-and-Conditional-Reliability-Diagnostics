from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from bearing_dt.physics import normalized_first_passage_rul


class ContextEncoder(nn.Module):
    def __init__(self, context_dim: int, hidden: int) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        if self.context_dim > 0:
            self.net = nn.Sequential(nn.Linear(self.context_dim, max(1, hidden // 2)), nn.SiLU(), nn.Linear(max(1, hidden // 2), hidden))
            self.bias = None
        else:
            self.net = None
            self.bias = nn.Parameter(torch.zeros(hidden))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if self.net is not None:
            return self.net(context.float())
        return self.bias.unsqueeze(0).expand(context.shape[0], -1)


class FeatureRegressor(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 64, context_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + context_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = torch.cat([batch["features"].float(), batch["context"].float()], dim=1)
        rul = F.softplus(self.net(x)).squeeze(-1)
        return {"rul": rul}


class CNNRegressor(nn.Module):
    def __init__(self, channels: int, feature_dim: int, hidden: int = 64, context_dim: int = 3) -> None:
        super().__init__()
        self.signal = nn.Sequential(
            nn.Conv1d(channels, hidden // 2, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Linear(hidden + feature_dim + context_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sig = self.signal(batch["signal"].float()).squeeze(-1)
        x = torch.cat([sig, batch["features"].float(), batch["context"].float()], dim=1)
        return {"rul": F.softplus(self.head(x)).squeeze(-1)}


class TCNRegressor(CNNRegressor):
    def __init__(self, channels: int, feature_dim: int, hidden: int = 64, context_dim: int = 3) -> None:
        super().__init__(channels, feature_dim, hidden, context_dim)
        self.signal = nn.Sequential(
            nn.Conv1d(channels, hidden // 2, kernel_size=5, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )


class LSTMRegressor(nn.Module):
    def __init__(self, channels: int, feature_dim: int, hidden: int = 64, context_dim: int = 3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + feature_dim + context_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        seq = batch["signal"].float().transpose(1, 2)
        _, (h, _) = self.lstm(seq)
        x = torch.cat([h[-1], batch["features"].float(), batch["context"].float()], dim=1)
        return {"rul": F.softplus(self.head(x)).squeeze(-1)}


class TransformerRegressor(nn.Module):
    def __init__(self, channels: int, feature_dim: int, hidden: int = 64, heads: int = 4, context_dim: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(channels, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(hidden + feature_dim + context_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        seq = batch["signal"].float().transpose(1, 2)
        encoded = self.encoder(self.proj(seq)).mean(dim=1)
        x = torch.cat([encoded, batch["features"].float(), batch["context"].float()], dim=1)
        return {"rul": F.softplus(self.head(x)).squeeze(-1)}


class AttnPINNRegressor(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 64, context_dim: int = 3) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Sigmoid())
        self.net = nn.Sequential(nn.Linear(feature_dim + context_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.rul_head = nn.Linear(hidden, 1)
        self.damage_head = nn.Linear(hidden, 1)
        self.rate_head = nn.Linear(hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = batch["features"].float()
        x = torch.cat([features * self.attn(features), batch["context"].float()], dim=1)
        z = self.net(x)
        return {
            "rul": F.softplus(self.rul_head(z)).squeeze(-1),
            "damage": torch.sigmoid(self.damage_head(z)).squeeze(-1),
            "damage_rate": F.softplus(self.rate_head(z)).squeeze(-1),
        }


class BearingDigitalTwin(nn.Module):
    def __init__(self, channels: int, feature_dim: int, hidden: int = 96, context_dim: int = 3) -> None:
        super().__init__()
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(channels, hidden // 3, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.Conv1d(hidden // 3, hidden // 2, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.feature_encoder = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Dropout(0.1))
        self.context_encoder = ContextEncoder(context_dim, hidden)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.SiLU())
        self.damage_head = nn.Linear(hidden, 1)
        self.rate_head = nn.Linear(hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sig = self.signal_encoder(batch["signal"].float()).squeeze(-1)
        feat = self.feature_encoder(batch["features"].float())
        ctx = self.context_encoder(batch["context"].float())
        z = self.fusion(torch.cat([sig, feat, ctx], dim=1))
        damage = torch.sigmoid(self.damage_head(z)).squeeze(-1)
        rate = F.softplus(self.rate_head(z)).squeeze(-1)
        rul = normalized_first_passage_rul(damage, rate)
        return {"rul": rul, "damage": damage, "damage_rate": rate, "embedding": z}


class FusedDirectRegressor(nn.Module):
    """Same fused encoders as the latent-state model, but with a direct RUL head."""

    def __init__(self, channels: int, feature_dim: int, hidden: int = 96, context_dim: int = 3) -> None:
        super().__init__()
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(channels, hidden // 3, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.Conv1d(hidden // 3, hidden // 2, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.feature_encoder = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Dropout(0.1))
        self.context_encoder = ContextEncoder(context_dim, hidden)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.SiLU())
        self.rul_head = nn.Linear(hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sig = self.signal_encoder(batch["signal"].float()).squeeze(-1)
        feat = self.feature_encoder(batch["features"].float())
        ctx = self.context_encoder(batch["context"].float())
        z = self.fusion(torch.cat([sig, feat, ctx], dim=1))
        return {"rul": F.softplus(self.rul_head(z)).squeeze(-1), "embedding": z}


def build_model(model_type: str, channels: int, feature_dim: int, hidden: int = 64, context_dim: int = 3) -> nn.Module:
    key = model_type.lower()
    if key in {"feature_mlp", "mlp"}:
        return FeatureRegressor(feature_dim, hidden, context_dim)
    if key == "cnn":
        return CNNRegressor(channels, feature_dim, hidden, context_dim)
    if key == "tcn":
        return TCNRegressor(channels, feature_dim, hidden, context_dim)
    if key == "lstm":
        return LSTMRegressor(channels, feature_dim, hidden, context_dim)
    if key == "transformer":
        return TransformerRegressor(channels, feature_dim, hidden, context_dim=context_dim)
    if key in {"attnpinn", "attn_pinn"}:
        return AttnPINNRegressor(feature_dim, hidden, context_dim)
    if key in {"digital_twin", "bearing_digital_twin"}:
        return BearingDigitalTwin(channels, feature_dim, hidden, context_dim)
    if key in {"fused_direct", "latent_direct_head", "direct_fused"}:
        return FusedDirectRegressor(channels, feature_dim, hidden, context_dim)
    raise ValueError(f"Unsupported torch model type: {model_type}")
