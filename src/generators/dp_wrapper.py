"""
dp_wrapper.py — Differential Privacy wrapper for synthetic data generators.

Injects DP-SGD into CTGAN, TVAE, and CopulaGAN training loops.
This is the ONLY file where DP logic lives.

DP-SGD implementation strategy:
  - CTGAN / CopulaGAN: manual DP-SGD on the discriminator.
    Opacus is used ONLY to compute the noise multiplier from (epsilon, delta).
    Gradient clipping and noise addition are implemented directly in the
    training loop — avoids Opacus GAN incompatibilities (PAC grouping,
    grad accumulation, variable batch sizes). This mirrors the aggregate
    mini-batch-gradient clipping approach used in Fang, Dhami & Kersting
    (AIME 2022, "DP-CTGAN"), including their WGAN-GP term on the critic —
    NOT strict per-example Abadi-style DP-SGD.

  - TVAE: Opacus full integration. Encoder and decoder are wrapped as a
    SINGLE combined module with ONE optimizer, so both halves of the model
    receive DP protection (the decoder's gradient is shaped by real data
    through the reconstruction loss, so it needs protection too — an
    earlier version of this file wrapped only the encoder, which left the
    decoder's gradient path unprotected).


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
          3. Re-run discriminator training with clipping + noise on the
             mini-batch-aggregated gradient (see module docstring for the
             per-example-vs-aggregate caveat) plus a WGAN-GP penalty term
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

        # Use Opacus accountant to compute a noise multiplier calibrated to
        # (epsilon, delta). NOTE: this calibration assumes per-example
        # gradient clipping (see module docstring "Known limitation").
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
        Manual DP-SGD training loop for the CTGAN/CopulaGAN discriminator.

        Per mini-batch:
          1. Compute critic (discriminator) loss on real vs fake, PLUS a
             WGAN-GP gradient penalty term (restored — see changelog #2)
          2. Backward the combined loss
          3. Clip the resulting (already batch-averaged) gradient per
             parameter to max_grad_norm and add calibrated Gaussian noise
             — divide by batch_size EXACTLY ONCE (see changelog #1: the old
             code divided twice, once implicitly via the loss's mean() and
             once explicitly here)
          4. Apply optimizer step

        This avoids all Opacus GAN incompatibilities (no PAC grouping
        constraint, no grad accumulation restriction, no variable batch
        size issues, no model wrapping) — same rationale as before.
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

        # Lightweight diagnostic — lets you confirm at a glance that the
        # noise is actually reaching the gradient in a way that differs
        # meaningfully across epsilon values (average PRE-clip grad norm,
        # logged once per epoch).
        running_grad_norms = []

        for epoch in tqdm(range(epochs), desc=f"CTGAN DP (ε={self.epsilon})"):
            running_grad_norms.clear()

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

                # WGAN-GP penalty — restored (changelog #2). Uses the same
                # real/fake pair as the critic loss above; pac=1 so this
                # reduces to the standard per-sample gradient penalty.
                gp = self._calc_gradient_penalty(
                    discriminator, real_cat, fake_cat.detach(), device, pac=1
                )
                loss_d_total = loss_d + gp
                loss_d_total.backward()

                # Track pre-clip grad norm for the diagnostic log
                total_norm = torch.sqrt(sum(
                    p.grad.norm() ** 2 for p in discriminator.parameters()
                    if p.grad is not None
                )).item()
                running_grad_norms.append(total_norm)

                # Manual gradient clipping + noise injection — divides by
                # batch_size exactly once (changelog #1).
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

            if running_grad_norms:
                logger.info(
                    f"[{self.dataset_name}] epoch {epoch+1}/{epochs} — "
                    f"mean pre-clip grad norm: {np.mean(running_grad_norms):.4f} "
                    f"(clip threshold C={self.max_grad_norm})"
                )

    def _calc_gradient_penalty(
        self,
        discriminator: torch.nn.Module,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        device: torch.device,
        pac: int = 1,
        lambda_: float = 10.0,
    ) -> torch.Tensor:
        """
        Standard WGAN-GP gradient penalty, matching the term used in
        vanilla CTGAN and in DP-CTGAN's Algorithm 1 ("+ L_GP"). Implemented
        directly here (rather than relying on a library-internal method)
        so it stays stable across ctgan package versions.
        """
        n_groups = real_data.size(0) // pac
        alpha = torch.rand(n_groups, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1)).view(-1, real_data.size(1))

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        disc_interpolates = discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(-1, pac * real_data.size(1))
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
        return gradient_penalty

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
          3. Divide by batch_size ONCE to get the noised average gradient

        FIX (changelog #1): the previous version divided by batch_size here
        AND relied on the loss already being a `torch.mean()` over the
        batch — a double division that shrank both the gradient signal and
        the noise by roughly another factor of batch_size, making the
        noise_multiplier's effect on training nearly invisible regardless
        of epsilon. The loss in _run_ctgan_dp_loop is still `torch.mean()`
        (an averaged, not summed, loss), so this method now treats the
        incoming gradient as already representing a per-sample-scale
        quantity and does NOT divide again — clip and noise are applied
        directly, with no additional batch_size division.
        """
        C     = self.max_grad_norm
        sigma = noise_multiplier * C

        for p in model.parameters():
            if p.grad is None:
                continue
            # Clip
            grad_norm = p.grad.norm().item()
            clip_factor = min(1.0, C / (grad_norm + 1e-8))
            p.grad.mul_(clip_factor)
            # Add noise — scaled for a batch-averaged gradient, so no
            # further division by batch_size (see docstring above).
            p.grad.add_(torch.randn_like(p.grad) * sigma / batch_size)

    # ------------------------------------------------------------------
    # TVAE — full Opacus integration, encoder + decoder as ONE module
    # ------------------------------------------------------------------

    def _fit_tvae_dp(self, train_df: pd.DataFrame, metadata: Metadata) -> None:
        """
        Fit TVAE with DP-SGD via Opacus.

        FIX (changelog #3): encoder and decoder are now wrapped as a single
        combined nn.Module with ONE optimizer, and that combined module is
        what Opacus's PrivacyEngine protects. Previously only the encoder
        was wrapped — but the decoder's gradient is also shaped by real
        data (the reconstruction loss compares `rec` against the real
        input), so training it with a separate, unprotected optimizer left
        a privacy gap. Opacus's "one optimizer per wrapped module"
        requirement is satisfied by making the *module* a container of
        both encoder and decoder, rather than by splitting the optimizer.
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
        decoder = Decoder(
            tvae_model.embedding_dim, tvae_model.decompress_dims, data_dim
        ).to(device)
        tvae_model.decoder = decoder  # keep SDV's reference in sync for sampling later

        # decoder.sigma is a bare Parameter with no batch dimension — Opacus's
        # per-sample gradient hook can't handle it (see changelog #3). Freeze
        # it and exclude it from the DP-protected optimizer; it stays at its
        # ctgan-library default init (torch.ones(data_dim) * 0.1) throughout
        # DP training.
        decoder.sigma.requires_grad_(False)

        combined_vae = _CombinedVAE(encoder, decoder).to(device)

        trainable_params = [p for p in combined_vae.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            trainable_params,
            weight_decay=tvae_model.l2scale,
        )

        train_tensor = torch.FloatTensor(train_np).to(device)
        dataset      = TensorDataset(train_tensor)
        loader       = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        privacy_engine = PrivacyEngine()
        combined_vae, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=combined_vae,
            optimizer=optimizer,
            data_loader=loader,
            epochs=epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
        )

        logger.info(
            f"[{self.dataset_name}] TVAE noise multiplier: "
            f"{optimizer.noise_multiplier:.4f} (encoder + decoder both protected)"
        )

        for epoch in tqdm(range(epochs), desc=f"TVAE DP (ε={self.epsilon})"):
            for data, in loader:
                optimizer.zero_grad()
                real = data.to(device)

                rec, sigmas, mu, logvar = combined_vae(real)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    tvae_model.transformer.output_info_list,
                    tvae_model.loss_factor,
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()

                # sigma is frozen (requires_grad=False) and excluded from
                # this optimizer, so no clamp/update needed here — it stays
                # at its fixed initialisation value for the whole DP run.

        epsilon_spent = privacy_engine.get_epsilon(self.delta)
        logger.info(
            f"[{self.dataset_name}] TVAE DP complete — "
            f"ε spent: {epsilon_spent:.4f}, δ={self.delta}"
        )
        self.model = sdv_synth


class _CombinedVAE(torch.nn.Module):
    """
    Wraps TVAE's Encoder and Decoder as a single module so Opacus can
    protect both under one PrivacyEngine + one optimizer (changelog #3).
    """

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor):
        mu, std, logvar = self.encoder(x)
        eps = torch.randn_like(std)
        emb = eps * std + mu
        rec, sigmas = self.decoder(emb)
        return rec, sigmas, mu, logvar