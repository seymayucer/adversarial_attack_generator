from typing import Tuple

import numpy as np
import random
import torch
import torch.nn as nn


from .base_attack import BaseAttack


class TGR(BaseAttack):
    """Implements Transferable Global Representations (TGR) attack."""

    def __init__(
        self,
        model_name: str,
        sample_num_batches: int = 130,
        steps: int = 10,
        epsilon: float = 16 / 255,
        target: bool = False,
        decay: float = 1.0,
    ) -> None:
        """Initialize TGR attack.

        Args:
            model_name: Name of the model to attack
            sample_num_batches: Number of batch samples
            steps: Number of attack iterations
            epsilon: Maximum perturbation magnitude
            target: If True, perform targeted attack
            decay: Momentum decay factor
        """
        super().__init__("TGR", model_name, target)

        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay

        # Image processing parameters
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((self.image_size / self.crop_length) ** 2)

        if self.sample_num_batches > self.max_num_batches:
            raise ValueError(f"sample_num_batches must be <= {self.max_num_batches}")

        self._register_model()

        self.params = {
            "attack_name": "TGR",
            "model_name": model_name,
            "steps": steps,
            "epsilon": epsilon,
            "sample_num_batches": sample_num_batches,
            "decay": decay,
        }

    def _register_model(self) -> None:
        """Register model-specific attention handling."""

        def attn_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            if self.model_name in ["visformer_small", "pit_b_224"]:
                B, C, H, W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, C, H * W)

                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_H = max_all // H
                max_all_W = max_all % H
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_H = min_all // H
                min_all_W = min_all % H

                out_grad[:, range(C), max_all_H, :] = 0.0
                out_grad[:, range(C), :, max_all_W] = 0.0
                out_grad[:, range(C), min_all_H, :] = 0.0
                out_grad[:, range(C), :, min_all_W] = 0.0

            elif self.model_name in ["cait_s24_224"]:
                B, H, W, C = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, H * W, C)

                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                max_all_H = max_all // H
                max_all_W = max_all % H
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
                min_all_H = min_all // H
                min_all_W = min_all % H

                out_grad[:, max_all_H, :, range(C)] = 0.0
                out_grad[:, :, max_all_W, range(C)] = 0.0
                out_grad[:, min_all_H, :, range(C)] = 0.0
                out_grad[:, :, min_all_W, range(C)] = 0.0

            return (out_grad,)

    def _generate_samples_for_interactions(
        self, perturbations: torch.Tensor, seed: int
    ) -> torch.Tensor:
        """Generate interaction samples for attack.

        Args:
            perturbations: Current perturbations
            seed: Random seed for sampling

        Returns:
            Perturbations with noise mask applied
        """
        add_noise_mask = torch.zeros_like(perturbations)
        grid_num_axis = int(self.image_size / self.crop_length)

        # Unrepeatable sampling
        ids = list(range(self.max_num_batches))
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[: self.sample_num_batches])

        rows, cols = ids // grid_num_axis, ids % grid_num_axis

        for r, c in zip(rows, cols):
            add_noise_mask[
                :,
                :,
                r * self.crop_length : (r + 1) * self.crop_length,
                c * self.crop_length : (c + 1) * self.crop_length,
            ] = 1

        return perturbations * add_noise_mask

    def forward(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Generate adversarial examples using TGR attack.

        Args:
            inputs: Input images to attack
            labels: Target labels

        Returns:
            Tuple of (adversarial examples, final loss)
        """
        loss_fn = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(inputs)

        # Denormalize inputs for perturbation calculation
        unnorm_inputs = self._mul_std_add_mean(inputs)
        perturbations = torch.zeros_like(unnorm_inputs)
        perturbations.requires_grad_()

        for i in range(self.steps):
            # Uncomment for patch-out version:
            # add_perturbation = self._generate_samples_for_interactions(perturbations, i)
            # outputs = self.model(self._sub_mean_div_std(unnorm_inputs + add_perturbation))

            # Regular version:
            outputs = self.model(self._sub_mean_div_std(unnorm_inputs + perturbations))

            cost = self.loss_flag * loss_fn(outputs, labels)
            cost.backward()

            grad = perturbations.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
            grad += momentum * self.decay
            momentum = grad

            # Use base class method for perturbation update
            perturbations.data = self._update_perturbations(
                perturbations.data, grad, self.step_size
            )

            # Clip to valid image range
            perturbations.data = (
                torch.clamp(unnorm_inputs.data + perturbations.data, 0.0, 1.0)
                - unnorm_inputs.data
            )

            perturbations.grad.data.zero_()

        # Return normalized adversarial examples
        return (
            self._sub_mean_div_std(unnorm_inputs + perturbations.data).detach(),
            cost.item(),
        )

    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Wrapper for forward method."""
        return self.forward(inputs, labels)
