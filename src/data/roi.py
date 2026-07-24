from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def validate(self, image_width: int, image_height: int) -> None:
        if min(self.x, self.y) < 0 or min(self.width, self.height) <= 0:
            raise ValueError(f"Invalid ROI: {self}")
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError(f"ROI {self} is outside image size {image_width}x{image_height}")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Roi":
        try:
            return cls(*(int(value[key]) for key in ("x", "y", "width", "height")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ROI configuration: {value}") from exc

