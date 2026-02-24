
import numpy as np
from manimlib import *
from manimlib.utils.space_ops import rotation_between_vectors


class Arrow3d(Group):
    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        thickness=0.02,
        tip_height=0.3,
        tip_radius=0.08,
        color=WHITE,
        **kwargs,
    ):
        super().__init__(**kwargs)

        start = np.array(start, dtype=float)
        end   = np.array(end,   dtype=float)

        direction = end - start
        length    = np.linalg.norm(direction)
        unit      = direction / length

        shaft_length = max(length - tip_height, 1e-6)

        # Shaft using ManimGL's built-in Line3D
        shaft = Line3D(start, start + unit * shaft_length, width=thickness)
        shaft.set_color(color)

        # Tip — ManimGL's Cone uses `radius`, not `base_radius`
        tip = Cone(radius=tip_radius, height=tip_height)
        tip.set_color(color)

        # Cone is born pointing up (+Z). Rotate to face `unit`.
        if not np.allclose(unit, OUT):
            if np.allclose(unit, -OUT):
                rot = rotation_matrix(PI, RIGHT)
            else:
                rot = rotation_between_vectors(OUT, unit)
            tip.apply_matrix(rot)

        # Place cone so its base meets the shaft tip
        tip.move_to(start + unit * (shaft_length + tip_height / 2))

        self.add(shaft, tip)
