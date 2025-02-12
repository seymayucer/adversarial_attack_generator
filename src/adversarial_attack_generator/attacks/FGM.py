from typing import Tuple, Optional

import torch
import torch.nn as nn

from .base_attack import BaseAttack


class FGM(BaseAttack):
    """Implements Fast Gradient Method attack with iterative checking."""

    def __init__(
        self,
        model_name: str,
        epsilon: float = 0.1,
        eps_step: float = 0.001,
        target: bool = False,
        clip_values: Optional[Tuple[float, float]] = None,
        max_iter: int = 500,
    ) -> None:
        """Initialize FGM attack.

        Args:
            model_name: Name of the model to attack
            epsilon: Maximum perturbation magnitude
            eps_step: Step size for each iteration
            target: If True, perform targeted attack
            clip_values: Optional (min, max) values for pixel range
            max_iter: Maximum number of iterations
        """
        super().__init__("FGM", model_name, target)

        if clip_values is not None and (
            len(clip_values) != 2 or clip_values[0] >= clip_values[1]
        ):
            raise ValueError("clip_values must be None or (min, max) with min < max")

        self.epsilon = epsilon
        self.eps_step = eps_step
        self.clip_values = clip_values
        self.max_iter = max_iter
        self.loss_fn = nn.CrossEntropyLoss()

        # Store attack parameters for analysis
        self.params = {
            "attack_name": "Fast Gradient Method",
            "model_name": model_name,
            "max_iter": max_iter,
            "epsilon": epsilon,
            "eps_step": eps_step,
            "clip_values": clip_values,
        }

    def _clip_values(self, x_adv: torch.Tensor) -> torch.Tensor:
        """Clip values to valid pixel range.

        Args:
            x_adv: Adversarial examples to clip

        Returns:
            Clipped adversarial examples
        """
        if self.clip_values is not None:
            clip_min, clip_max = self.clip_values
            return torch.clamp(x_adv, min=clip_min, max=clip_max)
        return torch.clamp(x_adv, min=0, max=1)

    def _check_attack_success(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> bool:
        """Check if attack has succeeded based on target mode.

        Args:
            predictions: Model predictions
            labels: Target labels

        Returns:
            True if attack succeeded, False otherwise
        """
        if self.target:
            return torch.all(predictions == labels)
        return torch.any(predictions != labels)

    def _normalize_gradient(self, grad: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Normalize gradient based on infinity norm.

        Args:
            grad: Input gradient
            batch_size: Batch size for reshaping

        Returns:
            Normalized gradient
        """
        grad_norm = torch.linalg.norm(
            grad.reshape(batch_size, -1), ord=float("inf"), dim=1
        )
        scale = self.epsilon / (grad_norm + 1e-12)
        scale = torch.minimum(scale, torch.ones_like(scale))
        return grad * scale.reshape(-1, 1, 1, 1)

    def forward(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Generate adversarial examples using FGM attack.

        Args:
            inputs: Input images to attack
            labels: Target labels

        Returns:
            Tuple of (adversarial examples, final loss)
        """
        batch_size = inputs.shape[0]
        x_adv = inputs.clone().detach()

        for iteration in range(self.max_iter):
            x_adv.requires_grad = True
            outputs = self.model(x_adv)
            predictions = outputs.argmax(dim=1)
            loss = self.loss_fn(outputs, labels)

            # Check if attack succeeded
            if self._check_attack_success(predictions, labels):
                return self._clip_values(x_adv).detach(), loss.item()

            # Continue attack if target not reached
            cost = self.loss_flag * loss
            cost.backward()

            # Update adversarial examples
            grad = x_adv.grad.data
            normalized_grad = self._normalize_gradient(grad, batch_size)
            x_adv = x_adv.detach() + normalized_grad
            x_adv = self._clip_values(x_adv)

        # Return final result if max iterations reached
        with torch.no_grad():
            outputs = self.model(x_adv)
            final_loss = self.loss_fn(outputs, labels)

        return x_adv.detach(), final_loss.item()

    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Wrapper for forward method."""
        return self.forward(inputs, labels)
