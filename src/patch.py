from dataclasses import dataclass

@dataclass
class Patch:
    x: int
    y: int
    color: str = "green"
    countdown: int = 0