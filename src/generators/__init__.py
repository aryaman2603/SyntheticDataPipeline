from src.generators.ctgan_generator import CTGANGenerator
from src.generators.tvae_generator import TVAEGenerator
from src.generators.copulagan_generator import CopulaGANGenerator

GENERATOR_MAP = {
    "ctgan":     CTGANGenerator,
    "tvae":      TVAEGenerator,
    "copulagan": CopulaGANGenerator,
}