# Ported from https://github.com/eeveemara/frc-fire-control
"""
RK4 projectile physics with aerodynamic drag and Magnus lift.
All internal computations in SI units (meters, seconds, radians).
"""

import math
from dataclasses import dataclass
from typing import List

from .shot_parameters import ShotParameters


@dataclass
class SimParameters:
    """Physical parameters for the shooter and ball used by the simulator.

    Attributes:
        ball_mass_kg: Ball mass in kilograms.
        ball_diameter_m: Ball diameter in meters.
        drag_coeff: Dimensionless drag coefficient.
        magnus_coeff: Dimensionless Magnus lift coefficient.
        air_density: Air density in kg/m^3.
        exit_height_m: Launch height from the floor in meters.
        wheel_diameter_m: Flywheel diameter in meters.
        target_height_m: Target height from the floor in meters.
        slip_factor: Fraction (0–1) of wheel tangential speed imparted to the ball.
        fixed_launch_angle_deg: Fixed launch angle in degrees (from horizontal).
        dt: Integration timestep in seconds.
        rpm_min: Minimum RPM for binary search.
        rpm_max: Maximum RPM for binary search.
        binary_search_iters: Iterations for binary search.
        max_sim_time: Maximum simulation time in seconds.
    """
    ball_mass_kg: float         # kg
    ball_diameter_m: float      # m
    drag_coeff: float           # dimensionless (smooth sphere ~0.47)
    magnus_coeff: float         # dimensionless
    air_density: float          # kg/m^3
    exit_height_m: float        # launch height from floor, m
    wheel_diameter_m: float     # flywheel diameter, m
    target_height_m: float      # target height from floor, m
    slip_factor: float          # 0=no grip, 1=perfect
    fixed_launch_angle_deg: float  # degrees from horizontal
    dt: float                   # integration timestep, s
    rpm_min: float              # minimum RPM for binary search
    rpm_max: float              # maximum RPM for binary search
    binary_search_iters: int    # iterations for binary search
    max_sim_time: float         # maximum simulation time, s


@dataclass
class TrajectoryResult:
    z_at_target: float
    tof: float
    reached_target: bool
    max_height: float
    apex_x: float


@dataclass
class LUTEntry:
    distance_m: float
    rpm: float
    tof: float
    reachable: bool


class ProjectileSimulator:
    """Simulate projectile motion with drag and Magnus lift and produce shot tables.

    Instantiate with SimParameters and optional magnus_sign (+1 for topspin, -1 for backspin).
    """

    def __init__(self, params: SimParameters, magnus_sign: float = 1.0) -> None:
        """Initialize simulator constants.

        Parameters:
            params: Simulation parameters (SimParameters).
            magnus_sign: +1 for topspin (upward lift), -1 for backspin.

        Returns:
            None
        """
        self.params = params
        # +1 for topspin (upward lift, default), -1 for backspin (downward push)
        self.magnus_sign = magnus_sign

        area = math.pi * (params.ball_diameter_m / 2.0) ** 2
        self.k_drag = (params.air_density * params.drag_coeff * area) / (2.0 * params.ball_mass_kg)
        self.k_magnus = (params.air_density * params.magnus_coeff * area) / (2.0 * params.ball_mass_kg)

    def exit_velocity(self, rpm: float) -> float:
        """Convert shooter RPM to ball exit velocity.

        Parameters:
            rpm: Wheel rotational speed in revolutions per minute.

        Returns:
            Exit speed of the ball in meters per second.
        """
        p = self.params
        return p.slip_factor * rpm * math.pi * p.wheel_diameter_m / 60.0

    def simulate(self, rpm: float, target_distance_m: float,
                 launch_angle_deg: float | None = None) -> TrajectoryResult:
        """Simulate a 2D trajectory using RK4 integration.

        Parameters:
            rpm: Launcher wheel RPM.
            target_distance_m: Horizontal distance to the target in meters.
            launch_angle_deg: Launch angle in degrees; if None uses fixed_launch_angle_deg.

        Returns:
            TrajectoryResult with height at target, time of flight, reachability and diagnostics.
        """
        if launch_angle_deg is None:
            launch_angle_deg = self.params.fixed_launch_angle_deg

        p = self.params
        v0 = self.exit_velocity(rpm)
        launch_rad = math.radians(launch_angle_deg)
        vx = v0 * math.cos(launch_rad)
        vz = v0 * math.sin(launch_rad)

        x = 0.0
        z = p.exit_height_m
        dt = p.dt
        max_height = z
        apex_x = 0.0
        t = 0.0
        max_time = p.max_sim_time

        while t < max_time:
            prev_x = x
            prev_z = z

            # RK4 step
            state = [x, z, vx, vz]
            k1 = self._derivatives(state)
            s2 = _add_scaled(state, k1, dt / 2.0)
            k2 = self._derivatives(s2)
            s3 = _add_scaled(state, k2, dt / 2.0)
            k3 = self._derivatives(s3)
            s4 = _add_scaled(state, k3, dt)
            k4 = self._derivatives(s4)

            x += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            z += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            vx += dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            vz += dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
            t += dt

            if z > max_height:
                max_height = z
                apex_x = x

            # Check if we've passed or reached the target distance
            if x >= target_distance_m:
                # Linear interpolation using saved pre-step state
                frac = (target_distance_m - prev_x) / (x - prev_x)
                z_at_target = prev_z + frac * (z - prev_z)
                tof_at_target = t - dt + frac * dt
                return TrajectoryResult(
                    z_at_target=z_at_target,
                    tof=tof_at_target,
                    reached_target=True,
                    max_height=max_height,
                    apex_x=apex_x,
                )

            if z < 0.0:
                return TrajectoryResult(
                    z_at_target=0.0, tof=t, reached_target=False,
                    max_height=max_height, apex_x=apex_x,
                )

        # Timed out without reaching target
        return TrajectoryResult(
            z_at_target=0.0, tof=max_time, reached_target=False,
            max_height=max_height, apex_x=apex_x,
        )

    def _derivatives(self, state: list) -> list:
        """Compute derivatives for RK4 integration.

        Parameters:
            state: [x, z, vx, vz]

        Returns:
            List [dx/dt, dz/dt, dvx/dt, dvz/dt].
        """
        svx = state[2]
        svz = state[3]
        speed = math.hypot(svx, svz)

        ax = -self.k_drag * speed * svx
        az = -9.81 - self.k_drag * speed * svz + self.magnus_sign * self.k_magnus * speed * speed

        return [svx, svz, ax, az]

    def find_rpm_for_distance(self, distance_m: float,
                               launch_angle_deg: float | None = None) -> LUTEntry:
        """Binary-search RPM to reach target height at a given distance.

        Parameters:
            distance_m: Horizontal distance to target in meters.
            launch_angle_deg: Optional launch angle in degrees; uses fixed angle if None.

        Returns:
            LUTEntry with rpm, time-of-flight and reachability flag.
        """
        if launch_angle_deg is None:
            launch_angle_deg = self.params.fixed_launch_angle_deg

        p = self.params
        height_tolerance = 0.02  # 2 cm

        lo = p.rpm_min
        hi = p.rpm_max

        # Quick feasibility check: can max RPM even reach this distance?
        max_check = self.simulate(hi, distance_m, launch_angle_deg)
        if not max_check.reached_target:
            return LUTEntry(distance_m=distance_m, rpm=0.0, tof=0.0, reachable=False)

        best_rpm = hi
        best_tof = max_check.tof
        best_error = abs(max_check.z_at_target - p.target_height_m)

        for _ in range(p.binary_search_iters):
            mid = (lo + hi) / 2.0
            result = self.simulate(mid, distance_m, launch_angle_deg)

            if not result.reached_target:
                lo = mid
                continue

            error = result.z_at_target - p.target_height_m
            abs_error = abs(error)

            if abs_error < best_error:
                best_rpm = mid
                best_tof = result.tof
                best_error = abs_error

            if abs_error < height_tolerance:
                return LUTEntry(distance_m=distance_m, rpm=mid, tof=result.tof, reachable=True)

            if error > 0:
                # Ball too high, reduce RPM
                hi = mid
            else:
                # Ball too low, increase RPM
                lo = mid

        # Return best found even if not perfectly converged
        return LUTEntry(
            distance_m=distance_m,
            rpm=best_rpm,
            tof=best_tof,
            reachable=best_error < 0.10,
        )

    def generate_lut(self,
                     min_dist_m: float = 0.50,
                     max_dist_m: float = 5.00,
                     step_m: float = 0.05) -> List[LUTEntry]:
        """Generate a lookup table of RPMs for distances.

        Parameters:
            min_dist_m: Minimum distance in meters.
            max_dist_m: Maximum distance in meters.
            step_m: Step between distances in meters.

        Returns:
            List of LUTEntry objects (one per distance step).
        """
        entries: List[LUTEntry] = []
        distance = min_dist_m
        while distance <= max_dist_m + step_m * 0.01:
            # Round to avoid floating-point drift
            distance = round(distance * 100.0) / 100.0
            entry = self.find_rpm_for_distance(distance)
            entries.append(entry)
            distance += step_m
        return entries

    def generate_variable_angle_shot_table(
            self,
            min_angle_deg: float,
            max_angle_deg: float,
            angle_step_deg: float,
            min_dist_m: float = 0.50,
            max_dist_m: float = 5.00,
            step_m: float = 0.05,
    ) -> List[ShotParameters]:
        """Find best RPM and angle per distance by sweeping angles.

        Parameters:
            min_angle_deg: Minimum launch angle in degrees.
            max_angle_deg: Maximum launch angle in degrees.
            angle_step_deg: Angle increment in degrees.
            min_dist_m: Minimum distance in meters.
            max_dist_m: Maximum distance in meters.
            step_m: Distance step in meters.

        Returns:
            List of ShotParameters for reachable shots (best rpm and hood angle per distance).
        """
        result = []
        distance = min_dist_m
        while distance <= max_dist_m + step_m * 0.01:
            distance = round(distance * 100.0) / 100.0
            best_rpm, best_angle, best_tof = float("inf"), 0.0, 0.0
            found = False
            angle = min_angle_deg
            while angle <= max_angle_deg + 0.001:
                entry = self.find_rpm_for_distance(distance, angle)
                if entry.reachable and entry.rpm < best_rpm:
                    best_rpm, best_angle, best_tof = entry.rpm, angle, entry.tof
                    found = True
                angle += angle_step_deg
            if found:
                result.append(ShotParameters(
                    distance=distance, rpm=best_rpm,
                    hood_angle=best_angle, time_of_flight=best_tof,
                ))
            distance += step_m
        return result

    def generate_shot_table(
            self,
            n_points: int = 91,
            min_dist_m: float = 0.50,
            max_dist_m: float = 5.00,
    ) -> List[ShotParameters]:
        """Produce a fixed-angle LUT of shots across a distance range.
        Fixed-hood LUT sampled at n_points evenly-spaced distances from
        min_dist_m to max_dist_m. Returns reachable entries only — the
        returned list may be shorter than n_points if the shooter cannot
        reach some distances within the configured RPM range.

        For adjustable-hood robots, use generate_variable_angle_shot_table() instead.
        Parameters:
            n_points: Number of entries to produce.
            min_dist_m: Minimum distance in meters.
            max_dist_m: Maximum distance in meters.

        Returns:
            List of ShotParameters for reachable shots using the fixed launch angle.
        """

        # Note: sampling from n_points, but returns reachable entries only. Return list may be less than n_points.
        step = (max_dist_m - min_dist_m) / (n_points - 1)
        entries = self.generate_lut(step_m=step)
        result: List[ShotParameters] = []
        for e in entries:
            if e.reachable:
                result.append(ShotParameters(
                    distance=e.distance_m,
                    rpm=e.rpm,
                    hood_angle=self.params.fixed_launch_angle_deg,
                    time_of_flight=e.tof,
                ))
        return result

    def get_k_drag(self) -> float:
        """Return computed drag constant k_drag."""
        return self.k_drag

    def get_k_magnus(self) -> float:
        """Return computed Magnus constant k_magnus."""
        return self.k_magnus

    @staticmethod
    def rpm_to_exit_velocity(rpm: float, wheel_diameter_m: float, slip_factor: float) -> float:
        """Convert RPM to exit velocity.

        Parameters:
            rpm: Rotational speed in RPM.
            wheel_diameter_m: Wheel diameter in meters.
            slip_factor: Fraction (0–1) of wheel speed transferred to ball.

        Returns:
            Exit velocity in m/s.
        """
        return slip_factor * rpm * math.pi * wheel_diameter_m / 60.0

    @staticmethod
    def exit_velocity_to_rpm(exit_vel_mps: float, wheel_diameter_m: float, slip_factor: float) -> float:
        """Convert exit velocity to required RPM.

        Parameters:
            exit_vel_mps: Desired exit velocity in m/s.
            wheel_diameter_m: Wheel diameter in meters.
            slip_factor: Fraction (0–1) of wheel speed transferred to ball.

        Returns:
            Required RPM.
        """
        return exit_vel_mps * 60.0 / (slip_factor * math.pi * wheel_diameter_m)


def _add_scaled(base: list, delta: list, scale: float) -> list:
    """Return base + delta * scale elementwise.

    Parameters:
        base: Base vector.
        delta: Delta vector.
        scale: Scalar multiplier.

    Returns:
        New list with scaled addition.
    """
    return [base[i] + delta[i] * scale for i in range(len(base))]
