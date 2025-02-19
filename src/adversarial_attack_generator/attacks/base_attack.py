from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch.nn import Module
from torchvision.models import WeightsEnum
import logging

logger = logging.getLogger(__name__)
# Mapping of model names to their corresponding model functions and weights
MODEL_MAPPING: Dict[str, Tuple] = {
    "vit_h_14": (models.vit_h_14, models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1),
    "regnet_y_128gf": (
        models.regnet_y_128gf,
        models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1,
    ),
    "efficientnet_v2_l": (
        models.efficientnet_v2_l,
        models.EfficientNet_V2_L_Weights.IMAGENET1K_V1,
    ),
    "convnext_large": (
        models.convnext_large,
        models.ConvNeXt_Large_Weights.IMAGENET1K_V1,
    ),
    "swin_v2_b": (models.swin_v2_b, models.Swin_V2_B_Weights.IMAGENET1K_V1),
    "maxvit_t": (models.maxvit_t, models.MaxVit_T_Weights.IMAGENET1K_V1),
    "resnext101_64x4d": (
        models.resnext101_64x4d,
        models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1,
    ),
    "vit_l_16": (models.vit_l_16, models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1),
    "resnet_v2_50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
}


def get_model(model_name: str) -> Tuple[Module, WeightsEnum]:
    """Load a pretrained model and its weights."""
    if model_name not in MODEL_MAPPING:
        raise ValueError(
            f"Model {model_name} not supported. Available: {list(MODEL_MAPPING.keys())}"
        )

    model_fn, weights = MODEL_MAPPING[model_name]
    model = model_fn(weights=weights)

    if "inception" in model_name:
        model.aux_logits = False

    return model, weights


class BaseAttack:
    """Base class for implementing adversarial attacks on image classifiers."""

    def __init__(self, attack_name: str, model_name: str, target: bool):
        self.attack_name = attack_name
        self.model_name = model_name
        self.target = target
        self.loss_flag = -1 if self.target else 1
        self.params: Optional[Dict] = None

        # Initialize model and preprocessor
        self.model, self.weights = get_model(self.model_name)
        self.image_precessor = self.weights.transforms()
        self.model.eval()

    def get_prediction(self, image: torch.Tensor) -> Tuple[str, int, float]:
        """Get model prediction for an image."""
        with torch.no_grad():
            prediction = self.model(image).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            category_name = self.weights.meta["categories"][class_id]
            return category_name, class_id, score

    def _mul_std_add_mean(self, inputs: torch.Tensor) -> torch.Tensor:
        """Denormalize image by multiplying by std and adding mean."""
        dtype = inputs.dtype
        mean = torch.as_tensor(self.image_precessor.mean, dtype=dtype)
        std = torch.as_tensor(self.image_precessor.std, dtype=dtype)
        return inputs.mul_(std[:, None, None]).add_(mean[:, None, None])

    def _sub_mean_div_std(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize image by subtracting mean and dividing by std."""
        dtype = inputs.dtype
        mean = torch.as_tensor(self.image_precessor.mean, dtype=dtype)
        std = torch.as_tensor(self.image_precessor.std, dtype=dtype)
        return (inputs - mean[:, None, None]) / std[:, None, None]

    def _save_images(
        self,
        orig_image: torch.Tensor,
        adv_image: torch.Tensor,
        filename: str,
        output_dir: str,
        orig_conf: float,
        adv_conf: float,
        target_class: int,
    ) -> None:
        """Save original, adversarial, and noise images with analysis."""
        # Setup output directory
        filename = Path(filename).stem
        sample_dir = Path(output_dir) / filename / self.attack_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Denormalize images
        unnorm_adv = self._mul_std_add_mean(adv_image)
        unnorm_orig = self._mul_std_add_mean(orig_image)

        # Save original image
        orig_img = self._prepare_image_for_saving(unnorm_orig)
        Image.fromarray(orig_img).save(sample_dir / f"{filename}_original.png")

        # Save adversarial image
        adv_img = self._prepare_image_for_saving(unnorm_adv)
        Image.fromarray(adv_img).save(sample_dir / f"{filename}_adversarial.png")

        # Modified noise visualization
        noise = unnorm_adv - unnorm_orig
        # Scale noise to better visualize small perturbations
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = self._prepare_image_for_saving(noise)
        Image.fromarray(noise).save(sample_dir / f"{filename}_noise.png")

        # Save analysis
        self._save_analysis(
            sample_dir / f"{filename}_analysis.txt", orig_conf, adv_conf, target_class
        )
        logger.info(f"Results saved to: {sample_dir}")

    def _prepare_image_for_saving(
        self, image: torch.Tensor, normalize: bool = False
    ) -> np.ndarray:
        """Prepare image tensor for saving as PNG."""
        img = image.squeeze(0).permute(1, 2, 0)
        if normalize:
            img = img / img.max()
        img = torch.clamp(img, 0, 1)
        return (img.detach().cpu().numpy() * 255).astype(np.uint8)

    def _save_analysis(
        self, filepath: Path, orig_conf: float, adv_conf: float, target_class: int
    ) -> None:
        """Save attack analysis to text file."""
        with open(filepath, "w") as f:
            f.write("Attack Parameters:\n")
            if self.params:
                for param, value in self.params.items():
                    f.write(f"{param}: {value}\n")
            f.write(f"\nResult for {self.attack_name} attack\n")
            f.write(f"Original Class Confidence: {orig_conf:.4f}\n")
            f.write(f"Adversarial Class Confidence: {adv_conf:.4f}\n")
            if self.target:
                f.write(
                    f"Target Category: {self.weights.meta["categories"][target_class]}\n"
                )
                f.write(f"Target Class: {target_class} \n\n")
            f.write(f"Output files saved in {filepath.parent}.\n")

    def _update_inputs(
        self, inputs: torch.Tensor, grad: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update input images with gradient step."""
        unnorm_inputs = self._mul_std_add_mean(inputs.clone().detach())
        unnorm_inputs = unnorm_inputs + step_size * grad.sign()
        unnorm_inputs = torch.clamp(unnorm_inputs, min=0, max=1).detach()
        return self._sub_mean_div_std(unnorm_inputs)

    def _update_perturbations(
        self, perturbations: torch.Tensor, grad: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update perturbations with gradient step."""
        perturbations = perturbations + step_size * grad.sign()
        return torch.clamp(perturbations, -self.epsilon, self.epsilon)

    def _get_perturbations(
        self, clean_inputs: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Calculate perturbations between clean and adversarial images."""
        clean_unnorm = self._mul_std_add_mean(clean_inputs.clone().detach())
        adv_unnorm = self._mul_std_add_mean(inputs.clone().detach())
        return adv_unnorm - clean_unnorm

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Generate adversarial examples."""
        return self.forward(*args, **kwargs)
