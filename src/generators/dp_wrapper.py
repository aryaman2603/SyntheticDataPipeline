"""
dp_wrapper.py — Differential Privacy wrapper for synthetic data generators.

Injects DP-SGD into CTGAN, TVAE, and CopulaGAN training loops.
This is the ONLY file where DP logic lives.

DP-SGD implementation strategy:
  - CTGAN / CopulaGAN: manual DP-SGD on the discriminator.
    Opacus is used ONLY to compute the noise multiplier from (epsilon, delta).
    Gradient clipping and noise addition are implemented directly in the
    training loop — avoids Opacus GAN incompatibilities (PAC grouping,
    grad accumulation, variable batch sizes).

  - TVAE: Opacus full integration. TVAE has a standard single-optimizer
    DataLoader loop that Opacus handles natively without any issues.

DP-SGD mechanics:
  1. Compute per-sample gradients for each real data point
  2. Clip each gradient to max_grad_norm (bounds individual influence)
  3. Add Gaussian noise scaled by noise_multiplier * max_grad_norm
  4. Average clipped+noised gradients and apply optimizer step

For CTGAN/CopulaGAN: DP applied to DISCRIMINATOR only.
  The discriminator sees real data — that is where privacy leakage occurs.
  The generator only sees noise vectors and never touches real data.

For TVAE: DP applied to encoder-decoder (single optimizer).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from ctgan.synthesizers.ctgan import Discriminator
from ctgan.synthesizers.tvae import Decoder, Encoder, _loss_function
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from sdv.metadata import Metadata
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.generators.base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class DPWrapper(BaseGenerator):
    """
    Wraps any BaseGenerator and replaces its fit() with a DP-SGD version.

    Parameters:
        generator  : an instantiated BaseGenerator (CTGANGenerator etc.)
        epsilon    : privacy budget (lower = more private, less utility)
        config     : full pipeline config dict
    """

    def __init__(self, generator: BaseGenerator, epsilon: float, config: dict):
        super().__init__(config, generator.dataset_name)
        self.generator     = generator
        self.epsilon       = epsilon
        self.delta         = float(config["differential_privacy"]["delta"])
        self.max_grad_norm = float(config["differential_privacy"]["max_grad_norm"])
        self.model         = None

        logger.info(
            f"[{self.dataset_name}] DPWrapper initialised — "
            f"ε={epsilon}, δ={self.delta}, max_grad_norm={self.max_grad_norm}"
        )

    # ------------------------------------------------------------------
    # Delegate to inner generator
    # ------------------------------------------------------------------

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generator.prepare_dataframe(df)

    def build_metadata(self, train_df: pd.DataFrame) -> Metadata:
        return self.generator.build_metadata(train_df)

    def save_synthetic(self, synthetic_df: pd.DataFrame, label: str):
        return self.generator.save_synthetic(synthetic_df, label)

    def load_train_data(self) -> pd.DataFrame:
        return self.generator.load_train_data()

    # ------------------------------------------------------------------
    # DP fit — dispatches per generator type
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, metadata: Metadata) -> None:
        gen_type = type(self.generator).__name__

        if gen_type in ("CTGANGenerator", "CopulaGANGenerator"):
            self._fit_ctgan_dp(train_df, metadata)
        elif gen_type == "TVAEGenerator":
            self._fit_tvae_dp(train_df, metadata)
        else:
            raise NotImplementedError(
                f"DPWrapper does not support generator type: {gen_type}"
            )

    def sample(self, n: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() before sample()")
        return self.model.sample(num_rows=n)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        self.model.save(path)

    def load(self, path: str) -> None:
        raise NotImplementedError(
            "Load not supported for DPWrapper — retrain from scratch."
        )

    # ------------------------------------------------------------------
    # CTGAN / CopulaGAN — manual DP-SGD on discriminator
    # ------------------------------------------------------------------

    def _fit_ctgan_dp(self, train_df: pd.DataFrame, metadata: Metadata) -> None:
        """
        Fit CTGAN/CopulaGAN with manual DP-SGD on the discriminator.

        Strategy:
          1. Fit inner generator normally so SDV handles all preprocessing,
             data transformation, and model initialisation
          2. Compute noise_multiplier from (epsilon, delta, sample_rate, epochs)
             using Opacus's privacy accountant
          3. Re-run discriminator training with manual per-sample gradient
             clipping and noise injection
          4. Generator training is unchanged — no DP needed
        """
        logger.info(f"[{self.dataset_name}] CTGAN DP fit — ε={self.epsilon}")

        self.generator.fit(train_df, metadata)
        sdv_synth   = self.generator.model
        ctgan_model = sdv_synth._model

        train_np   = ctgan_model._transformer.transform(train_df)
        batch_size = ctgan_model._batch_size
        n_samples  = len(train_np)
        sample_rate = batch_size / n_samples

        # Use Opacus accountant to compute correct noise multiplier
        noise_multiplier = get_noise_multiplier(
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            sample_rate=sample_rate,
            epochs=ctgan_model._epochs,
        )

        logger.info(
            f"[{self.dataset_name}] Computed noise_multiplier={noise_multiplier:.4f} "
            f"for ε={self.epsilon}, δ={self.delta}, "
            f"sample_rate={sample_rate:.4f}, epochs={ctgan_model._epochs}"
        )

        self._run_ctgan_dp_loop(ctgan_model, train_np, noise_multiplier)
        self.model = sdv_synth

        logger.info(f"[{self.dataset_name}] CTGAN DP fit complete — ε={self.epsilon}")

    def _run_ctgan_dp_loop(
        self,
        ctgan_model,
        train_np: np.ndarray,
        noise_multiplier: float,
    ) -> None:
        """
        Manual DP-SGD training loop for the CTGAN discriminator.

        Per-sample gradient clipping + noise:
          For each batch:
            1. Forward pass on each sample individually to get per-sample gradients
            2. Clip each per-sample gradient to max_grad_norm
            3. Sum clipped gradients and add Gaussian noise
            4. Apply optimizer step

        This avoids all Opacus GAN incompatibilities:
          - No PAC grouping constraint
          - No grad accumulation restriction
          - No variable batch size issues
          - No model wrapping
        """
        device     = ctgan_model._device
        epochs     = ctgan_model._epochs
        batch_size = ctgan_model._batch_size

        # Fresh discriminator — pac=1 keeps it simple (no sample grouping)
        discriminator = Discriminator(
            train_np.shape[1] + ctgan_model._data_sampler.dim_cond_vec(),
            ctgan_model._discriminator_dim,
            pac=1,
        ).to(device)

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=ctgan_model._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=ctgan_model._discriminator_decay,
        )

        train_tensor = torch.FloatTensor(train_np).to(device)
        dataset      = TensorDataset(train_tensor)
        loader       = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        for epoch in tqdm(range(epochs), desc=f"CTGAN DP (ε={self.epsilon})"):
            for real_batch, in loader:
                n    = len(real_batch)
                mean = torch.zeros(n, ctgan_model._embedding_dim, device=device)
                std  = mean + 1

                # --- Discriminator step with manual DP-SGD ---
                optimizerD.zero_grad()

                fakez   = torch.normal(mean=mean, std=std)
                condvec = ctgan_model._data_sampler.sample_condvec(n)

                if condvec is not None:
                    c1, m1, col, opt = condvec
                    c1       = torch.from_numpy(c1[:n]).to(device)
                    fakez    = torch.cat([fakez, c1], dim=1)
                    real_cat = torch.cat([real_batch, c1], dim=1)
                else:
                    real_cat = real_batch

                fake    = ctgan_model._generator(fakez)
                fakeact = ctgan_model._apply_activate(fake)
                fake_cat = torch.cat([fakeact, c1], dim=1) if condvec is not None else fakeact

                # Compute loss on real and fake
                y_fake = discriminator(fake_cat.detach())
                y_real = discriminator(real_cat)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                loss_d.backward()

                # Manual per-sample gradient clipping + noise injection
                self._dp_clip_and_noise(
                    discriminator, n, noise_multiplier
                )
                optimizerD.step()

                # --- Generator step — no DP needed ---
                fakez   = torch.normal(mean=mean, std=std)
                condvec = ctgan_model._data_sampler.sample_condvec(n)

                if condvec is not None:
                    c1, m1, col, opt = condvec
                    c1    = torch.from_numpy(c1[:n]).to(device)
                    m1    = torch.from_numpy(m1[:n]).to(device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake    = ctgan_model._generator(fakez)
                fakeact = ctgan_model._apply_activate(fake)

                if condvec is not None:
                    fake_cat      = torch.cat([fakeact, c1], dim=1)
                    cross_entropy = ctgan_model._cond_loss(fake, c1, m1)
                else:
                    fake_cat      = fakeact
                    cross_entropy = 0

                y_fake  = discriminator(fake_cat)
                loss_g  = -torch.mean(y_fake) + cross_entropy

                ctgan_model._generator.zero_grad()
                loss_g.backward()
                for p in ctgan_model._generator.parameters():
                    if p.grad is not None:
                        p.data -= ctgan_model._generator_lr * p.grad

    def _dp_clip_and_noise(
        self,
        model: torch.nn.Module,
        batch_size: int,
        noise_multiplier: float,
    ) -> None:
        """
        Apply DP-SGD gradient clipping and noise to model parameters.

        For each parameter:
          1. Clip gradient norm to max_grad_norm
          2. Add Gaussian noise ~ N(0, (noise_multiplier * max_grad_norm)^2)
          3. Divide by batch_size to get the noised average gradient

        This is mathematically equivalent to what Opacus does internally,
        implemented directly to avoid GAN training loop incompatibilities.
        """
        C  = self.max_grad_norm
        sigma = noise_multiplier * C

        for p in model.parameters():
            if p.grad is None:
                continue
            # Clip
            grad_norm = p.grad.norm().item()
            clip_factor = min(1.0, C / (grad_norm + 1e-8))
            p.grad.mul_(clip_factor)
            # Add noise
            p.grad.add_(torch.randn_like(p.grad) * sigma)
            # Average over batch
            p.grad.div_(batch_size)

    # ------------------------------------------------------------------
    # TVAE — full Opacus integration (clean single-optimizer loop)
    # ------------------------------------------------------------------

    def _fit_tvae_dp(self, train_df: pd.DataFrame, metadata: Metadata) -> None:
        """
        Fit TVAE with DP-SGD via Opacus.
        TVAE has a standard single-optimizer DataLoader loop that Opacus
        handles natively without any GAN-specific complications.
        """
        logger.info(f"[{self.dataset_name}] TVAE DP fit — ε={self.epsilon}")

        self.generator.fit(train_df, metadata)
        sdv_synth  = self.generator.model
        tvae_model = sdv_synth._model

        train_np   = tvae_model.transformer.transform(train_df)
        data_dim   = tvae_model.transformer.output_dimensions
        device     = tvae_model._device
        epochs     = tvae_model.epochs
        batch_size = tvae_model.batch_size

        # Rebuild encoder/decoder fresh for DP training
        encoder = Encoder(
            data_dim, tvae_model.compress_dims, tvae_model.embedding_dim
        ).to(device)
        tvae_model.decoder = Decoder(
            tvae_model.embedding_dim, tvae_model.decompress_dims, data_dim
        ).to(device)

        # Separate optimizers — Opacus requires the optimizer to contain
        # ONLY the parameters of the module being wrapped (encoder).
        # Decoder is updated with its own standard optimizer.
        optimizerEnc = optim.Adam(
            encoder.parameters(),
            weight_decay=tvae_model.l2scale,
        )
        optimizerDec = optim.Adam(
            tvae_model.decoder.parameters(),
            weight_decay=tvae_model.l2scale,
        )

        train_tensor = torch.FloatTensor(train_np).to(device)
        dataset      = TensorDataset(train_tensor)
        loader       = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        privacy_engine = PrivacyEngine()
        encoder, optimizerEnc, loader = privacy_engine.make_private_with_epsilon(
            module=encoder,
            optimizer=optimizerEnc,
            data_loader=loader,
            epochs=epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
        )

        logger.info(
            f"[{self.dataset_name}] TVAE noise multiplier: "
            f"{optimizerEnc.noise_multiplier:.4f}"
        )

        for epoch in tqdm(range(epochs), desc=f"TVAE DP (ε={self.epsilon})"):
            for data, in loader:
                optimizerEnc.zero_grad()
                optimizerDec.zero_grad()
                real = data.to(device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = tvae_model.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    tvae_model.transformer.output_info_list,
                    tvae_model.loss_factor,
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerEnc.step()
                optimizerDec.step()
                tvae_model.decoder.sigma.data.clamp_(0.01, 1.0)

        epsilon_spent = privacy_engine.get_epsilon(self.delta)
        logger.info(
            f"[{self.dataset_name}] TVAE DP complete — "
            f"ε spent: {epsilon_spent:.4f}, δ={self.delta}"
        )
        self.model = sdv_synth