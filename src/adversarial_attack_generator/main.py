import argparse
import logging
from pathlib import Path
from adversarial_attack_generator.attack import generate_adversarial
from adversarial_attack_generator.models import SUPPORTED_MODELS
from adversarial_attack_generator.attacks import ATTACK_MAPPING

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Generate adversarial examples to fool deep learning models using various attack methods"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help=f"Name of the target model to attack (e.g., {SUPPORTED_MODELS})",
        metavar="MODEL",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to the input image file to generate adversarial example from",
        metavar="INPUT",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory path where the adversarial example and results will be saved",
        metavar="OUTPUT",
    )
    parser.add_argument(
        "--attack_method",
        type=str,
        required=True,
        choices=list(ATTACK_MAPPING.keys()),
        help=f"Type of adversarial attack to perform (e.g., {list(ATTACK_MAPPING.keys())})",
        metavar="ATTACK",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        help="Target class index from ImageNet (0-999) for targeted attacks. If not specified, untargeted attack will be performed",
        metavar="CLASS",
    )

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input_image)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")

    # Ensure output directory exists
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Generate adversarial example with specified parameters
        logger.info(
            f"Starting adversarial attack with {args.attack_method} on model {args.model_name}"
        )
        adv_image, results = generate_adversarial(
            args.input_image,
            args.model_name,
            args.attack_method,
            args.output_path,
            args.target_class,
        )

        # Log attack results
        logger.info(
            f"Attack completed successfully:\n"
            f"Original class: {results['original']['class_name']} ({results['original']['confidence']:.4f})\n"
            f"Adversarial class: {results['adversarial']['class_name']} ({results['adversarial']['confidence']:.4f})"
        )

    except Exception as e:
        logger.error(f"Error generating adversarial example: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
