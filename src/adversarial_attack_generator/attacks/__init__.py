from .TGR import TGR
from .FGM import FGM
from .PGD import PGD
from .CW import CW
from .base_attack import BaseAttack

__all__ = ["TGR", "BaseAttack", "FGM", "PGD", "CW"]
ATTACK_MAPPING = {"TGR": TGR, "FGM": FGM, "PGD": PGD, "CW": CW}
