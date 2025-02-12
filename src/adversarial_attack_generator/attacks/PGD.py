from .base_attack import BaseAttack
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PGD(BaseAttack):
    """
    This class implements the Projected Gradient Descent (PGD) attack with iterative checking.
    """

    def __init__(
        self,
        model_name: str,
        epsilon: float = 0.1,
        eps_step: float = 0.002,
        target: bool = False,
        clip_values: Optional[Tuple[float, float]] = None,
        max_iter: int = 500,
        random_start: bool = False,  # Added random start
    ):
        """
        Create a :class:`.PGD` instance.

        :param model_name: The name of the model to attack.
        :param epsilon: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size for each iteration.
        :param target: Should the attack target one specific class? (If True, performs a targeted attack).
        :param clip_values: Tuple of min/max values for clipping the adversarial example.
        :param max_iter: Maximum number of iterations to perform.
        """
        super(PGD, self).__init__("PGD", model_name, target)
        self.epsilon = epsilon
        self.eps_step = eps_step  # Step size for each iteration
        self.clip_values = clip_values
        self.loss_fn = nn.CrossEntropyLoss()
        self.max_iter = max_iter
        self.random_start = random_start
        self.params = {
            "attack_name": "Projected Gradient Descent",
            "model_name": model_name,
            "max_iter": max_iter,
            "eps": epsilon,
            "eps_step": eps_step,
            "random_start": random_start,
        }
        if clip_values is not None:
            assert len(clip_values) == 2 and clip_values[0] < clip_values[1]

    def forward(self, inps, labels):
        """
        Generate adversarial examples using the Projected Gradient Descent (PGD) attack with iterative checking.

        :param inps: Sample input with shape as expected by the model.
        :param labels: Target values (class labels) indices of shape (nb_samples,).
        :return: Adversarial examples.
        """

        # Ensure that the model is in train mode
        self.model.train()

        # Initialize adversarial examples
        if self.random_start:
            x_adv = inps.clone().detach()
            noise = torch.rand_like(x_adv)
            noise = (noise - 0.5) * 2 * self.epsilon
            if self.clip_values is not None:
                clip_min, clip_max = self.clip_values
                x_adv = torch.clamp(x_adv + noise, clip_min, clip_max)
            else:
                x_adv = torch.clamp(x_adv + noise, 0, 1)

        else:
            x_adv = inps.clone().detach()

        batch_size = x_adv.shape[0]

        # Iteratively refine the adversarial example
        for i in range(self.max_iter):
            x_adv.requires_grad = True
            outputs = self.model(x_adv)
            preds = torch.argmax(outputs, dim=1)

            loss = self.loss_fn(outputs, labels)

            # Check if attack succeeded
            if self.target:
                if preds == labels:
                    x_adv = torch.clamp(x_adv, 0, 1)
                    return x_adv.detach(), loss.item()
            else:
                if preds != labels:
                    x_adv = torch.clamp(x_adv, 0, 1)
                    return x_adv.detach(), loss.item()

            # If target not reached, continue attack

            cost = self.loss_flag * loss
            cost.backward()

            grad = x_adv.grad.data
            grad_norm = torch.linalg.norm(
                grad.reshape(batch_size, -1), ord=float("inf"), dim=1
            )
            scale = self.epsilon / (grad_norm + 1e-12)  # Correctly use epsilon
            scale = torch.minimum(scale, torch.ones_like(scale))
            grad *= scale.reshape(batch_size, 1, 1, 1)

            # Update adversarial example
            x_adv = x_adv + grad

            # Project back into the l_inf ball
            noise = torch.clamp(x_adv - inps, -self.epsilon, self.epsilon)
            x_adv = inps + noise

            # Apply clip values
            if self.clip_values is not None:
                clip_min, clip_max = self.clip_values
                x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)
            else:
                x_adv = torch.clamp(x_adv, min=0, max=1)

            x_adv = x_adv.detach()
            self.model.zero_grad()  # Clear gradients

        # Return last result if max iterations reached
        self.model.eval()  # Restore eval mode
        with torch.no_grad():
            outputs = self.model(x_adv)
            loss = self.loss_fn(outputs, labels)
        return x_adv.detach(), loss.item()

    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Wrapper for forward method."""
        return self.forward(inputs, labels)
