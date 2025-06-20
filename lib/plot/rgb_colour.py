from dataclasses import dataclass


@dataclass(frozen=True)
class RgbColour:
    r: int
    g: int
    b: int