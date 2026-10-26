"""Padding and TBPTT boundaries: never invent a new episode at segment starts."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    burn_in: bool
    reset: bool
    detach_before: bool


def segments(length, burn_in=0, tbptt=16):
    if length < 1 or burn_in < 0 or burn_in >= length or tbptt < 1:
        raise ValueError("Invalid sequence schedule")
    if burn_in:
        yield Segment(0, burn_in, True, True, False)
    for start in range(burn_in, length, tbptt):
        yield Segment(start, min(start + tbptt, length), False, start == 0, start > 0)


def pad_sequences(items, burn_in=0):
    """Tensors [T,...] -> [B,T,...], valid padding and separate loss masks."""
    if not items or any(len(x) == 0 for x in items) or burn_in < 0:
        raise ValueError("Empty sequence batch")
    length = max(map(len, items))
    shape = (len(items), length, *items[0].shape[1:])
    result = items[0].new_zeros(shape)
    valid = torch.zeros(shape[:2], device=result.device, dtype=torch.bool)
    for i, x in enumerate(items):
        result[i, : len(x)] = x
        valid[i, : len(x)] = True
    loss_mask = valid.clone()
    loss_mask[:, :burn_in] = False
    return result, valid, loss_mask
