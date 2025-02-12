from .base_attack import BaseAttack
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional


class CW(BaseAttack):
    """
    Implements the Carlini & Wagner L2 attack.
    """

    def __init__(
        self,
        model_name: str,
        target: bool = False,
        clip_values: Optional[Tuple[float, float]] = None,
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        binary_search_steps: int = 9,
        initial_const: float = 0.01,
        confidence: float = 0.0,  # Added confidence parameter
        abort_early: bool = True,  # Added abort_early parameter
    ):
        """
        Create a :class:`.CarliniL2Method` instance.

        :param model_name: The name of the model to attack.
        :param target: Should the attack target one specific class? (If True, performs a targeted attack).
        :param clip_values: Tuple of min/max values for clipping the adversarial example.
        :param max_iter: Maximum number of iterations.
        :param learning_rate: Learning rate for optimization.
        :param binary_search_steps: Number of binary search steps to find the optimal constant c.
        :param initial_const: The initial trade-off constant c to use.
        :param confidence: Confidence of adversarial examples: a higher value produces samples that are farther away,
                           but more strongly classified as adversarial.
        :param abort_early: Abort early if the optimization is not making progress.
        """
        super().__init__(
            model_name=model_name, attack_name="CW", target=target
        )  # Corrected super().__init__ call
        self.clip_values = clip_values
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.confidence = confidence  # Store confidence parameter
        self.abort_early = abort_early  # Store abort_early parameter
        self.params = {
            "attack_name": "CarliniL2Method",
            "model_name": model_name,
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "binary_search_steps": binary_search_steps,
            "initial_const": initial_const,
        }
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")  # Use reduction="none"
        if clip_values is not None:
            assert len(clip_values) == 2 and clip_values[0] < clip_values[1]

    def _l2_loss(self, x, x_adv):
        """Calculates the L2 distortion."""
        return torch.sum((x - x_adv) ** 2, dim=(1, 2, 3))

    def _cw_loss(self, outputs, labels, const, targeted, confidence):
        """Calculates the Carlini-Wagner loss function."""
        target_one_hot = torch.zeros_like(outputs)
        target_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        real = torch.sum(target_one_hot * outputs, dim=1)
        other = torch.max(
            (1 - target_one_hot) * outputs - (target_one_hot * 1e6), dim=1
        )[0]

        if targeted:
            loss_1 = torch.maximum(torch.zeros_like(other), other - real + confidence)
        else:
            loss_1 = torch.maximum(torch.zeros_like(other), real - other + confidence)

        return const * loss_1

    def forward(self, inps, labels):
        """
        Generate adversarial examples using the Carlini & Wagner L2 attack.

        :param inps: Sample input with shape as expected by the model.
        :param labels: Target values (class labels) indices of shape (nb_samples,).  If targeted=True, these
                       are the target labels. If targeted=False, these are the true labels.
        :return: Adversarial examples.
        """

        # Ensure that the model is in evaluation mode
        self.model.eval()

        batch_size = inps.size(0)
        targeted = self.target

        # Define constant values for binary search
        lower_bounds = torch.zeros(batch_size, device=inps.device)
        upper_bounds = torch.ones(batch_size, device=inps.device) * 1e10
        const = torch.ones(batch_size, device=inps.device) * self.initial_const

        # Best adversarial examples and corresponding loss values
        best_adv_images = inps.clone().detach()
        best_l2 = torch.ones(batch_size, device=inps.device) * 1e10
        best_scores = -torch.ones(batch_size, device=inps.device) * 1e10

        # Perform binary search for the trade-off constant c
        for binary_search_step in range(self.binary_search_steps):
            # Initialize adversarial examples and set requires_grad
            w = torch.zeros_like(inps, requires_grad=True).to(
                inps.device
            )  # Ensure correct device
            optimizer = optim.Adam([w], lr=self.learning_rate)

            # Optimize adversarial examples
            for iteration in range(self.max_iter):
                # Map w to an adversarial example
                x_adv = torch.tanh(w)
                if self.clip_values is not None:
                    clip_min, clip_max = self.clip_values
                    x_adv = (x_adv * (clip_max - clip_min) / 2) + (
                        clip_max + clip_min
                    ) / 2
                    x_adv = torch.min(
                        torch.max(x_adv, torch.tensor(clip_min, device=x_adv.device)),
                        torch.tensor(clip_max, device=x_adv.device),
                    )  # Clip values correctly
                else:
                    x_adv = (x_adv / 2) + 0.5  # map from tanh to [0, 1]

                # Check if attack succeeded EARLY
                with torch.no_grad():  # Avoid tracking gradients during this check
                    outputs = self.model(x_adv)  # Perform prediction
                    preds = torch.argmax(outputs, dim=1)
                    if self.target:  # Targeted attack
                        success = torch.equal(preds, labels)
                    else:  # Untargeted attack
                        success = not torch.equal(preds, labels)

                    if success:
                        x_adv = torch.clamp(x_adv, 0, 1)  # Ensure final clamping
                        return x_adv.detach(), 0.0  # Return early with loss 0.0

                # Calculate the loss
                outputs = self.model(x_adv)
                loss_cw = self._cw_loss(
                    outputs, labels, const, targeted, self.confidence
                )
                loss_l2 = self._l2_loss(inps, x_adv)
                loss = torch.mean(loss_l2 + loss_cw)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Abort early if the optimization is not making progress
                if self.abort_early and iteration % (self.max_iter // 10) == 0:
                    with torch.no_grad():
                        adv_preds = self.model(x_adv).detach()
                        scores = torch.max(adv_preds, dim=1)[0]
                        l2s = self._l2_loss(inps, x_adv)

                        if targeted:
                            active_indices = (
                                (torch.argmax(adv_preds, dim=1) == labels)
                                .nonzero()
                                .flatten()
                            )
                        else:
                            active_indices = (
                                (torch.argmax(adv_preds, dim=1) != labels)
                                .nonzero()
                                .flatten()
                            )
                        # Check if the active indices loss changed (early stopping)
                        current_scores = scores[active_indices]
                        current_l2s = l2s[active_indices]
                        if len(current_l2s) > 0:
                            new_best_score = torch.min(current_scores)
                            if new_best_score <= best_scores.mean():
                                break

            # Evaluate and update best adversarial examples
            with torch.no_grad():
                adv_preds = self.model(x_adv).detach()
                scores = torch.max(adv_preds, dim=1)[0]
                l2s = self._l2_loss(inps, x_adv)

                if targeted:
                    are_adversarial = torch.argmax(adv_preds, dim=1) == labels
                else:
                    are_adversarial = torch.argmax(adv_preds, dim=1) != labels

                improved = (l2s < best_l2) * are_adversarial

                best_l2[improved] = l2s[improved]
                best_scores[improved] = scores[improved]
                best_adv_images[improved] = x_adv[improved]

            # Adjust the constant c for the next binary search step
            upper_bounds[improved] = torch.minimum(
                upper_bounds[improved], const[improved]
            )
            lower_bounds[~improved] = torch.maximum(
                lower_bounds[~improved], const[~improved]
            )
            const[are_adversarial == 0] = (
                lower_bounds[are_adversarial == 0] + upper_bounds[are_adversarial == 0]
            ) / 2
            const[upper_bounds == lower_bounds] = 0

        # Calculate the overall adversarial loss
        cost_item = torch.mean(best_l2).item()
        # cost_item = loss.item()

        return best_adv_images.detach(), cost_item
