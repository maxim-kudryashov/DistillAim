# Ported from https://github.com/eeveemara/frc-fire-control
"""
ShotParameters and ShotResult dataclasses.
"""

from dataclasses import dataclass


@dataclass
class ShotParameters:
    """Ballistic parameters at a single distance:
    - RPM
    - Hood angle
    - ToF"""
    distance: float       # meters
    rpm: float
    hood_angle: float     # degrees from horizontal
    time_of_flight: float  # seconds

    @staticmethod
    def lerp(a: "ShotParameters", b: "ShotParameters", t: float) -> "ShotParameters":
        """Linear interpolation between two ShotParameters at fraction t in [0,1]."""
        def _l(x: float, y: float) -> float:
            return x + t * (y - x)
        return ShotParameters(
            distance=_l(a.distance, b.distance),
            rpm=_l(a.rpm, b.rpm),
            hood_angle=_l(a.hood_angle, b.hood_angle),
            time_of_flight=_l(a.time_of_flight, b.time_of_flight),
        )

# Zero ShotParameters (0 distance, 0 rpm, 0 hood angle, 0 tof)
ShotParameters.ZERO = ShotParameters(distance=0.0, rpm=0.0, hood_angle=0.0, time_of_flight=0.0)


@dataclass
class ShotResult:
    """Output of ShotCalculator.calculate()."""
    rpm: float
    hood_angle: float          # degrees from horizontal
    turret_angle_delta: float  # degrees, positive = CCW in field frame
    confidence: float          # 0–100
