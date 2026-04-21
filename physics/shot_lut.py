# Ported from https://github.com/eeveemara/frc-fire-control
"""
Distance-keyed lookup table for RPM, hood angle, and TOF.
Uses sorted list + bisect for interpolation.
"""

import bisect
from typing import List

from .shot_parameters import ShotParameters


class ShotLUT:
    """Distance-keyed lookup table with linear interpolation and clamped extrapolation."""

    def __init__(self) -> None:
        """Create an empty ShotLUT.

        Returns:
            None
        """
        self._keys: List[float] = []
        self._entries: List[ShotParameters] = []

    def put(
            self,
            distance_m: float,
            params_or_rpm: "ShotParameters | float",
            hood_angle: float | None = None,
            time_of_flight: float | None = None,
    ) -> None:
        """Insert or overwrite an entry.

        Parameters:
            distance_m: Distance in meters (key).
            params_or_rpm: ShotParameters instance or an RPM value (float/int).
            hood_angle: Hood angle in degrees if providing an RPM.
            time_of_flight: Time of flight in seconds if providing an RPM.

        Returns:
            None
        """
        if isinstance(params_or_rpm, (int, float)):
            from .shot_parameters import ShotParameters  # avoid circular at top-level
            params = ShotParameters(
                rpm=float(params_or_rpm),
                hood_angle=hood_angle,
                time_of_flight=time_of_flight,
            )
        else:
            params = params_or_rpm
        idx = bisect.bisect_left(self._keys, distance_m)
        if idx < len(self._keys) and abs(self._keys[idx] - distance_m) < 1e-9:
            self._entries[idx] = params
        else:
            self._keys.insert(idx, distance_m)
            self._entries.insert(idx, params)

    def get(self, distance_m: float) -> ShotParameters:
        """Return interpolated ShotParameters for a distance.

        Parameters:
            distance_m: Distance in meters.

        Returns:
            Interpolated ShotParameters (clamped to edges). If empty, returns ShotParameters.ZERO.
        """
        if not self._keys:
            return ShotParameters.ZERO

        keys = self._keys
        entries = self._entries

        if distance_m <= keys[0]:
            return entries[0]
        if distance_m >= keys[-1]:
            return entries[-1]

        hi_idx = bisect.bisect_right(keys, distance_m)
        lo_idx = hi_idx - 1

        lo_key = keys[lo_idx]
        hi_key = keys[hi_idx]
        span = hi_key - lo_key
        if span < 1e-12:
            return entries[lo_idx]
        t = (distance_m - lo_key) / span
        return ShotParameters.lerp(entries[lo_idx], entries[hi_idx], t)

    def get_rpm(self, distance_m: float) -> float:
        """Get interpolated RPM at distance_m.

        Parameters:
            distance_m: Distance in meters.

        Returns:
            RPM as float.
        """
        return self.get(distance_m).rpm

    def get_angle(self, distance_m: float) -> float:
        """Get interpolated hood angle at distance_m.

        Parameters:
            distance_m: Distance in meters.

        Returns:
            Hood angle in degrees.
        """
        return self.get(distance_m).hood_angle

    def get_tof(self, distance_m: float) -> float:
        """Get interpolated time-of-flight at distance_m.

        Parameters:
            distance_m: Distance in meters.

        Returns:
            Time of flight in seconds.
        """
        return self.get(distance_m).time_of_flight

    def size(self) -> int:
        """Return number of entries in the LUT.

        Returns:
            Number of stored entries.
        """
        return len(self._keys)

    def clear(self) -> None:
        """Remove all entries from the LUT.

        Returns:
            None
        """
        self._keys.clear()
        self._entries.clear()

    def all_entries(self) -> List[ShotParameters]:
        """Return a shallow copy of all stored ShotParameters.

        Returns:
            List of ShotParameters.
        """
        return list(self._entries)


def load_shot_lut() -> "ShotLUT":
    """Create and return the default 91-point ShotLUT using reference SimParameters.

    Returns:
        A populated ShotLUT instance.
    """
    # Import here to avoid circular import at module level
    from .projectile_simulator import ProjectileSimulator, SimParameters

    params = SimParameters(
        ball_mass_kg=0.215,
        ball_diameter_m=0.1501,
        drag_coeff=0.47,
        magnus_coeff=0.2,
        air_density=1.225,
        exit_height_m=0.43,
        wheel_diameter_m=0.1016,
        target_height_m=1.83,
        slip_factor=0.6,
        fixed_launch_angle_deg=45.0,
        dt=0.001,
        rpm_min=1500.0,
        rpm_max=6000.0,
        binary_search_iters=25,
        max_sim_time=5.0,
    )
    sim = ProjectileSimulator(params)
    lut_entries = sim.generate_shot_table()
    lut = ShotLUT()
    for sp in lut_entries:
        lut.put(sp.distance, sp)
    return lut
