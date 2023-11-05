from typing import TypedDict
import drawsvg as draw
import numpy as np


class RawDrawing(TypedDict):
    drawing: list[tuple[list[int], list[int]]]


Line = list[tuple[int, int]]

Stroke5 = tuple[float, float, int, int, int]


class Drawing:
    lines: list[Line]

    def __init__(self, raw_drawing: RawDrawing):
        self.lines = []
        for xs, ys in raw_drawing["drawing"]:
            self.lines.append([(x, y) for x, y in zip(xs, ys)])

    def render(self):
        d = draw.Drawing(256, 256, origin="top-left")
        d.append(draw.Rectangle(0, 0, 256, 256, fill="#eee"))
        for line in self.lines:
            if len(line) == 0:
                continue
            d.append(
                draw.Lines(
                    *merge_coordinates(line),
                    close=False,
                    stroke="black",
                    stroke_width=2,
                    fill="transparent",
                )
            )
        return d

    def to_stroke5(self) -> list[Stroke5]:
        """Implements the stroke-5 format used in quick draw."""
        drawing = []
        ref = self.lines[0][0]
        for line in self.lines:
            if len(drawing) > 0:
                dx, dy, *_ = drawing[-1]
                drawing[-1] = (dx, dy, 0, 1, 0)
            for x, y in line:
                x_ref, y_ref = ref
                dx, dy = x - x_ref, y - y_ref
                drawing.append((dx, dy, 1, 0, 0))
                ref = (x, y)
        dx, dy, *_ = drawing[-1]
        drawing[-1] = (dx, dy, 0, 0, 1)
        return drawing

    @staticmethod
    def from_stroke5(strokes: list[Stroke5]) -> "Drawing":
        drawing = Drawing.__new__(Drawing)
        lines = []
        x, y = 0, 0
        line = []
        for dx, dy, a, b, c in strokes:
            x += dx
            y += dy
            line.append((x, y))
            if b == 1 and len(line) > 0:
                lines.append(line)
                line = []
            if c == 1:
                break
        if len(line) > 0:
            lines.append(line)
        # Rescale.
        (mx, my), (Mx, My) = np.min(np.concatenate(lines), axis=0), np.max(
            np.concatenate(lines), axis=0
        )
        scale_factor = max(Mx - mx, My - my)
        drawing.lines = [
            [
                (
                    int(
                        256
                        * (
                            (x - mx) / scale_factor
                            + 1 / 2
                            - (Mx - mx) / (2 * scale_factor)
                        )
                    ),
                    int(
                        256
                        * (
                            (y - my) / scale_factor
                            + 1 / 2
                            - (My - my) / (2 * scale_factor)
                        )
                    ),
                )
                for (x, y) in line
            ]
            for line in lines
        ]
        return drawing


def merge_coordinates(line: Line) -> list[int]:
    coordinates = []
    for x, y in line:
        coordinates.append(x)
        coordinates.append(y)
    return coordinates
