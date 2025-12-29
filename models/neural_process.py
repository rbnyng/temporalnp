import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# Numerical stability constraints for variance and standard deviation predictions
# These values prevent numerical instability (overflow/underflow) during training

# Log-variance bounds (used in Decoder)
# log_var represents log(variance), so variance = exp(log_var)
LOG_VAR_MIN = -7.0
LOG_VAR_MAX = 7.0

# Log-sigma bounds (used in LatentEncoder)
# log_sigma represents log(std), so std = exp(log_sigma)
LOG_SIGMA_MIN = -10.0
LOG_SIGMA_MAX = 2.0


class EmbeddingEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 3,
        in_channels: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        super().__init__()

        # CNN to process spatial structure of embedding patch
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        # res connection for first block
        self.residual_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = embeddings.permute(0, 3, 1, 2)

        identity = self.residual_proj(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)) + identity)

        x = F.relu(self.bn3(self.conv3(x)) + x)

        # Global pooling
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)

        return x


class ContextEncoder(nn.Module):
    """
    Encode context points (coordinates + embedding features + AGBD).

    Args:
        coord_dim: Dimension of coordinate vector. Use 2 for spatial-only (lon, lat),
                  or 5 for spatiotemporal (lon, lat, sin_doy, cos_doy, norm_time).
        embedding_dim: Dimension of embedding features from EmbeddingEncoder.
        hidden_dim: Hidden layer dimension.
        output_dim: Output representation dimension.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        super().__init__()
        self.coord_dim = coord_dim

        input_dim = coord_dim + embedding_dim + 1  # coords + embedding + agbd

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        coords: torch.Tensor,
        embedding_features: torch.Tensor,
        agbd: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([coords, embedding_features, agbd], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))

        identity = x
        x = F.relu(self.ln2(self.fc2(x)) + identity)
        x = F.relu(self.ln3(self.fc3(x)) + x)
        x = self.fc_out(x)

        return x


class Decoder(nn.Module):
    """
    Decode query points to AGBD predictions.

    Args:
        coord_dim: Dimension of coordinate vector. Use 2 for spatial-only,
                  or 5 for spatiotemporal.
        embedding_dim: Dimension of embedding features.
        context_dim: Dimension of aggregated context representation.
        hidden_dim: Hidden layer dimension.
        output_uncertainty: Whether to output uncertainty (log variance).
    """

    def __init__(
        self,
        coord_dim: int = 2,
        embedding_dim: int = 128,
        context_dim: int = 128,
        hidden_dim: int = 256,
        output_uncertainty: bool = True
    ):
        super().__init__()
        self.coord_dim = coord_dim

        self.output_uncertainty = output_uncertainty
        input_dim = coord_dim + embedding_dim + context_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.mean_head = nn.Linear(hidden_dim, 1)

        if output_uncertainty:
            self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        query_coords: torch.Tensor,
        query_embedding_features: torch.Tensor,
        context_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.cat([query_coords, query_embedding_features, context_repr], dim=-1)

        x = F.relu(self.ln1(self.fc1(x)))

        # layers with residual
        identity = x
        x = F.relu(self.ln2(self.fc2(x)) + identity)

        x = F.relu(self.ln3(self.fc3(x)) + x)

        mean = self.mean_head(x)

        if self.output_uncertainty:
            log_var = self.log_var_head(x)
            # clamp log_var to prevent numerical instability
            log_var = torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
            return mean, log_var
        else:
            return mean, None


class AttentionAggregator(nn.Module):

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_repr: torch.Tensor,
        context_repr: torch.Tensor
    ) -> torch.Tensor:
        query = query_repr.unsqueeze(0)  # (1, n_query, dim)
        context = context_repr.unsqueeze(0)  # (1, n_context, dim)

        # Apply cross-attention
        attended, _ = self.attention(query, context, context)

        # Apply dropout and residual connection
        attended = self.dropout(attended)
        output = self.norm(attended.squeeze(0) + query_repr)

        return output


class LatentEncoder(nn.Module):
    """
    Encode context representations into latent distribution (stochastic path).

    This encoder outputs (mu, log_sigma) where log_sigma is the
    logarithm of the STANDARD DEVIATION (not variance).
    Convention: log_sigma = log(std), so sigma = exp(log_sigma)
    """

    def __init__(
        self,
        context_repr_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128
    ):
        super().__init__()

        self.fc1 = nn.Linear(context_repr_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Separate heads for mean and log-variance
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mean pool context
        pooled = context_repr.mean(dim=0, keepdim=True)

        # relu hidden layers
        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))

        # pred distribution parameters
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)

        # clamp log_sigma for numerical stability
        log_sigma = torch.clamp(log_sigma, min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)

        return mu, log_sigma


class GEDINeuralProcess(nn.Module):
    """
    Neural Process for GEDI AGB interpolation with foundation model embeddings.

    Architecture modes:
    - 'deterministic': Only deterministic attention path (original implementation)
    - 'latent': Only latent stochastic path (global context)
    - 'anp': Full Attentive Neural Process (both paths)
    - 'cnp': Conditional Neural Process (mean pooling, no attention/latent)

    Components:
    1. Encode embedding patches to feature vectors
    2. Encode context points (coord + embedding feature + agbd)
    3. Deterministic path: Query-specific attention aggregation (optional)
    4. Latent path: Global stochastic latent variable (optional)
    5. Decode query points to AGBD predictions with uncertainty

    Spatiotemporal support:
    - Set coord_dim=2 for spatial-only (lon, lat)
    - Set coord_dim=5 for spatiotemporal (lon, lat, sin_doy, cos_doy, norm_time)
    """

    def __init__(
        self,
        patch_size: int = 3,
        embedding_channels: int = 128,
        embedding_feature_dim: int = 128,
        context_repr_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        output_uncertainty: bool = True,
        architecture_mode: str = 'deterministic',
        num_attention_heads: int = 4,
        coord_dim: int = 2
    ):
        super().__init__()

        assert architecture_mode in ['deterministic', 'latent', 'anp', 'cnp'], \
            f"Invalid architecture_mode: {architecture_mode}"

        self.output_uncertainty = output_uncertainty
        self.architecture_mode = architecture_mode
        self.latent_dim = latent_dim
        self.coord_dim = coord_dim

        # which components to use
        self.use_attention = architecture_mode in ['deterministic', 'anp']
        self.use_latent = architecture_mode in ['latent', 'anp']

        # embedding encoder (shared for context and query)
        self.embedding_encoder = EmbeddingEncoder(
            patch_size=patch_size,
            in_channels=embedding_channels,
            hidden_dim=hidden_dim,
            output_dim=embedding_feature_dim
        )

        # context encoder
        self.context_encoder = ContextEncoder(
            coord_dim=coord_dim,
            embedding_dim=embedding_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=context_repr_dim
        )

        # attention aggregator (deterministic path)
        if self.use_attention:
            self.attention_aggregator = AttentionAggregator(
                dim=context_repr_dim,
                num_heads=num_attention_heads
            )
            # query projection for attention (coord + embedding -> context_repr_dim)
            self.query_proj = nn.Linear(coord_dim + embedding_feature_dim, context_repr_dim)

        # latent encoder (stochastic path)
        if self.use_latent:
            self.latent_encoder = LatentEncoder(
                context_repr_dim=context_repr_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim
            )

        # decoder
        # context dim depends on which paths are active
        decoder_context_dim = 0
        if self.use_attention or architecture_mode == 'cnp':
            decoder_context_dim += context_repr_dim
        if self.use_latent:
            decoder_context_dim += latent_dim

        self.decoder = Decoder(
            coord_dim=coord_dim,
            embedding_dim=embedding_feature_dim,
            context_dim=decoder_context_dim,
            hidden_dim=hidden_dim,
            output_uncertainty=output_uncertainty
        )

    def forward(
        self,
        context_coords: torch.Tensor,
        context_embeddings: torch.Tensor,
        context_agbd: torch.Tensor,
        query_coords: torch.Tensor,
        query_embeddings: torch.Tensor,
        query_agbd: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            context_coords: (n_context, 2)
            context_embeddings: (n_context, patch_size, patch_size, channels)
            context_agbd: (n_context, 1)
            query_coords: (n_query, 2)
            query_embeddings: (n_query, patch_size, patch_size, channels)
            query_agbd: (n_query, 1) or None (only needed during training for ANP)
            training: Whether in training mode (affects latent sampling)

        Returns:
            (predicted_agbd, log_variance, z_mu_context, z_log_sigma_context, z_mu_all, z_log_sigma_all)
            - predicted_agbd: (n_query, 1)
            - log_variance: (n_query, 1) or None
            - z_mu_context: (1, latent_dim) or None - q(z|C) distribution mean
            - z_log_sigma_context: (1, latent_dim) or None - q(z|C) distribution log std
            - z_mu_all: (1, latent_dim) or None - p(z|C,T) distribution mean (training only)
            - z_log_sigma_all: (1, latent_dim) or None - p(z|C,T) distribution log std (training only)
        """
        # encode embeddings
        context_emb_features = self.embedding_encoder(context_embeddings)
        query_emb_features = self.embedding_encoder(query_embeddings)

        # encode context points
        context_repr = self.context_encoder(
            context_coords,
            context_emb_features,
            context_agbd
        )

        z_mu_context, z_log_sigma_context = None, None
        z_mu_all, z_log_sigma_all = None, None
        context_components = []

        # Deterministic path (attention or mean pooling)
        if self.use_attention:
            # query specific attention aggregation
            query_repr = torch.cat([query_coords, query_emb_features], dim=-1)
            query_repr_projected = self.query_proj(query_repr)
            aggregated_context = self.attention_aggregator(
                query_repr_projected,
                context_repr
            )
            context_components.append(aggregated_context)
        elif self.architecture_mode == 'cnp':
            # Mean pooling for CNP
            aggregated_context = context_repr.mean(dim=0, keepdim=True)
            aggregated_context = aggregated_context.expand(query_coords.shape[0], -1)
            context_components.append(aggregated_context)

        # Latent path (stochastic)
        if self.use_latent:
            # Always encode q(z|C) - context only distribution
            z_mu_context, z_log_sigma_context = self.latent_encoder(context_repr)

            # During training, also encode p(z|C,T) - full distribution with targets
            if training and query_agbd is not None:
                # Encode query/target points
                query_repr_full = self.context_encoder(
                    query_coords,
                    query_emb_features,
                    query_agbd
                )
                # Combine context and target representations
                all_repr = torch.cat([context_repr, query_repr_full], dim=0)
                # Encode combined representation
                z_mu_all, z_log_sigma_all = self.latent_encoder(all_repr)

                # Sample from p(z|C,T) during training
                epsilon = torch.randn_like(z_mu_all, device=z_mu_all.device, dtype=z_mu_all.dtype)
                z = z_mu_all + epsilon * torch.exp(z_log_sigma_all)
            else:
                # Use q(z|C) during inference or when query_agbd not provided
                if training:
                    epsilon = torch.randn_like(z_mu_context, device=z_mu_context.device, dtype=z_mu_context.dtype)
                    z = z_mu_context + epsilon * torch.exp(z_log_sigma_context)
                else:
                    z = z_mu_context

            # Expand latent to match query batch size
            z_expanded = z.expand(query_coords.shape[0], -1)
            context_components.append(z_expanded)

        if len(context_components) > 0:
            combined_context = torch.cat(context_components, dim=-1)
        else:
            raise ValueError(f"No context components generated for mode: {self.architecture_mode}")

        pred_mean, pred_log_var = self.decoder(
            query_coords,
            query_emb_features,
            combined_context
        )

        return pred_mean, pred_log_var, z_mu_context, z_log_sigma_context, z_mu_all, z_log_sigma_all

    def predict(
        self,
        context_coords: torch.Tensor,
        context_embeddings: torch.Tensor,
        context_agbd: torch.Tensor,
        query_coords: torch.Tensor,
        query_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_mean, pred_log_var, _, _, _, _ = self.forward(
            context_coords,
            context_embeddings,
            context_agbd,
            query_coords,
            query_embeddings,
            query_agbd=None,
            training=False
        )

        if pred_log_var is not None:
            pred_std = torch.exp(0.5 * pred_log_var)
        else:
            pred_std = torch.zeros_like(pred_mean)

        return pred_mean, pred_std


def kl_divergence_gaussian(
    mu_q: torch.Tensor,
    log_sigma_q: torch.Tensor,
    mu_p: Optional[torch.Tensor] = None,
    log_sigma_p: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute KL divergence between two Gaussians.

    If mu_p and log_sigma_p are provided:
        KL[p||q] where p ~ N(mu_p, exp(log_sigma_p)^2) and q ~ N(mu_q, exp(log_sigma_q)^2)
    Otherwise:
        KL[q||N(0,1)] where q ~ N(mu_q, exp(log_sigma_q)^2)

    Args:
        mu_q: Mean of q distribution
        log_sigma_q: Log std of q distribution
        mu_p: Mean of p distribution (optional)
        log_sigma_p: Log std of p distribution (optional)

    Returns:
        KL divergence scalar
    """
    if mu_p is not None and log_sigma_p is not None:
        # KL[p||q] = E_p[log p(z) - log q(z)]
        # For Gaussians: KL[N(mu_p, sigma_p^2) || N(mu_q, sigma_q^2)]
        # = log(sigma_q/sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2) / (2*sigma_q^2) - 1/2
        # Since we have log_sigma = log(sigma), this becomes:
        # = (log_sigma_q - log_sigma_p) + (exp(2*log_sigma_p) + (mu_p - mu_q)^2) / (2*exp(2*log_sigma_q)) - 1/2

        var_p = torch.exp(2 * log_sigma_p)
        var_q = torch.exp(2 * log_sigma_q)

        kl = 0.5 * torch.sum(
            (log_sigma_q - log_sigma_p) * 2 +  # log(var_q / var_p)
            (var_p + (mu_p - mu_q) ** 2) / var_q - 1,
            dim=-1
        )
    else:
        # KL[q||N(0,1)] - original behavior for backwards compatibility
        kl = 0.5 * torch.sum(
            torch.exp(2 * log_sigma_q) + mu_q ** 2 - 1 - 2 * log_sigma_q,
            dim=-1
        )

    return kl.mean()


def neural_process_loss(
    pred_mean: torch.Tensor,
    pred_log_var: Optional[torch.Tensor],
    target: torch.Tensor,
    z_mu_context: Optional[torch.Tensor] = None,
    z_log_sigma_context: Optional[torch.Tensor] = None,
    z_mu_all: Optional[torch.Tensor] = None,
    z_log_sigma_all: Optional[torch.Tensor] = None,
    kl_weight: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    Compute neural process loss.

    Args:
        pred_mean: Predicted mean
        pred_log_var: Predicted log variance (optional)
        target: Target values
        z_mu_context: Mean of q(z|C) - context only (optional)
        z_log_sigma_context: Log std of q(z|C) - context only (optional)
        z_mu_all: Mean of p(z|C,T) - context + target (optional)
        z_log_sigma_all: Log std of p(z|C,T) - context + target (optional)
        kl_weight: Weight for KL term

    Returns:
        (total_loss, loss_dict)
    """

    if pred_log_var is not None:
        # Gaussian NLL
        nll = 0.5 * (
            pred_log_var +
            torch.exp(-pred_log_var) * (target - pred_mean) ** 2
        )
    else:
        # MSE loss
        nll = (target - pred_mean) ** 2

    nll = nll.mean()

    # KL divergence if latent path is used
    kl = torch.tensor(0.0, device=pred_mean.device)
    if z_mu_context is not None and z_log_sigma_context is not None:
        if z_mu_all is not None and z_log_sigma_all is not None:
            # ANP: KL[p(z|C,T) || q(z|C)]
            kl = kl_divergence_gaussian(
                z_mu_context, z_log_sigma_context,
                z_mu_all, z_log_sigma_all
            )
        else:
            # Latent-only or CNP fallback: KL[q(z|C) || N(0,1)]
            kl = kl_divergence_gaussian(z_mu_context, z_log_sigma_context)

    # loss
    total_loss = nll + kl_weight * kl

    # loss components for logging
    loss_dict = {
        'total': total_loss.item(),
        'nll': nll.item(),
        'kl': kl.item()
    }

    return total_loss, loss_dict
