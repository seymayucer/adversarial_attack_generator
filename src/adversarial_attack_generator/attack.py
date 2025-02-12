from pathlib import Path
import torch
from PIL import Image
from typing import Dict, Tuple
import logging

from adversarial_attack_generator.attacks import ATTACK_MAPPING
from adversarial_attack_generator.models import SUPPORTED_MODELS

logger = logging.getLogger(__name__)


def generate_adversarial(
    image_path: str,
    model_name: str,
    attack_name: str,
    output_path: str,
    target_class: int = None,
    attack_params: Dict = None,
) -> Tuple[torch.Tensor, Dict]:
    """Generate adversarial example for given image using specified model and attack.

    Args:
        image_path (str):  Path to input image
        model_name (str):  Name of model to attack
        attack_name (str):  Name of attack method
        output_path (str):  Path to save results
        target_class (int, optional):  Target class index for targeted attacks.
        Defaults to None.
        attack_params (Dict, optional):  Dictionary of attack-specific parameters.
        Defaults to None.

    Raises:
        ValueError: If model_name is not supported

    Returns:
        Tuple[torch.Tensor, Dict]: Adversarial image tensor and attack results
    """

    # Input validation
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model {model_name} not supported. Choose from {SUPPORTED_MODELS}"
        )
    if attack_name not in ATTACK_MAPPING:
        raise ValueError(
            f"Attack {attack_name} not supported. Choose from {ATTACK_MAPPING.keys()}"
        )

    # Initialize attack
    attack_cls = ATTACK_MAPPING[attack_name]
    attack = attack_cls(model_name, target=target_class is not None)

    # Load and preprocess image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")
    image_tensor = attack.image_precessor(image).unsqueeze(0).to(device)
    image_tensor_not_changed = image_tensor.clone().detach()
    # Get original prediction
    orig_class_name, orig_class_id, orig_conf = attack.get_prediction(image_tensor)
    logger.info(f"Original prediction: {orig_class_name} (conf: {orig_conf:.4f})")

    # Generate adversarial example
    logger.info(f"Generating adversarial example using {attack_name} attack")
    target = torch.tensor([target_class or orig_class_id]).to(device)
    adv_image, loss_info = attack(image_tensor, target, **attack_params or {})

    # Get adversarial prediction
    adv_class_name, adv_class_id, adv_conf = attack.get_prediction(adv_image)
    logger.info(
        f"Adversarial prediction: {adv_class_name} (conf: {adv_conf:.4f})/ class_id: {adv_class_id}"
    )

    # Save results
    filename = Path(image_path).name
    attack._save_images(
        image_tensor_not_changed,
        adv_image,
        filename,
        output_path,
        orig_conf,
        adv_conf,
        target_class,
    )

    results = {
        "original": {
            "class_name": orig_class_name,
            "class_id": orig_class_id,
            "confidence": orig_conf,
        },
        "adversarial": {
            "class_name": adv_class_name,
            "class_id": adv_class_id,
            "confidence": adv_conf,
        },
        "attack_info": loss_info,
    }

    return adv_image, results
