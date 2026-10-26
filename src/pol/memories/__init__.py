from .base import REGISTRY, MemoryProducer, StepObs, make_memory, memory_bytes, project50  # noqa: F401
from . import simple, learned, pol  # noqa: F401  (registers variants)

# CheckFIFO is CheckVLA with fifo_occluded=True (set in configs/pol/variants/CheckFIFO.yaml).
REGISTRY["CheckFIFO"] = REGISTRY["CheckVLA"]
