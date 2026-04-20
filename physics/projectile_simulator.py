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
    """Physical parameters of the robot shooter and ball."""
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
    """
    Simulates a ball flying through the air with drag and Magnus lift.
    Uses RK4 integration in the vertical plane (x, z).
    For each distance, binary-searches RPM until the ball arrives at the target height.
    Generates the 91-point lookup table from 0.50m to 5.00m.
    """

    def __init__(self, params: SimParameters, magnus_sign: float = 1.0) -> None:
        self.params = params
        # +1 for topspin (upward lift, default), -1 for backspin (downward push)
        self.magnus_sign = magnus_sign

        area = math.pi * (params.ball_diameter_m / 2.0) ** 2
        self.k_drag = (params.air_density * params.drag_coeff * area) / (2.0 * params.ball_mass_kg)
        self.k_magnus = (params.air_density * params.magnus_coeff * area) / (2.0 * params.ball_mass_kg)

    def exit_velocity(self, rpm: float) -> float:
        """
        RPM to ball exit speed. slipFactor (0–1) = how much surface speed transfers to ball (linear relationship).
        """
        p = self.params
        return p.slip_factor * rpm * math.pi * p.wheel_diameter_m / 60.0

    def simulate(self, rpm: float, target_distance_m: float,
                 launch_angle_deg: float | None = None) -> TrajectoryResult:
        """
        Simulate trajectory with RK4 integration.
        Returns TrajectoryResult with z height at target, TOF, and diagnostics.
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
        """
        state = [x, z, vx, vz]
        ax = -kDrag * |v| * vx
        az = -g - kDrag * |v| * vz + kMagnus * |v|^2
        """
        svx = state[2]
        svz = state[3]
        speed = math.hypot(svx, svz)

        ax = -self.k_drag * speed * svx
        az = -9.81 - self.k_drag * speed * svz + self.magnus_sign * self.k_magnus * speed * speed

        return [svx, svz, ax, az]

    def find_rpm_for_distance(self, distance_m: float,
                               launch_angle_deg: float | None = None) -> LUTEntry:
        """Binary search for the RPM at a specific launch angle that puts the ball at target height."""
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
        """Generate the full lookup table. Default: 0.50m to 5.00m in 5cm steps (91 entries)."""
        entries: List[LUTEntry] = []
        distance = min_dist_m
        while distance <= max_dist_m + step_m * 0.01:
            # Round to avoid floating-point drift
            distance = round(distance * 100.0) / 100.0
            entry = self.find_rpm_for_distance(distance)
            entries.append(entry)
            distance += step_m
        return entries

    def generate_shot_table(self, n_points: int = 91) -> List[ShotParameters]:
        """
        Produces the 91-point static LUT across the legal shooting range (0.50m–5.00m).
        Returns reachable entries only, with the fixed angle baked in.
        """
        entries = self.generate_lut()
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
        return self.k_drag

    def get_k_magnus(self) -> float:
        return self.k_magnus


def _add_scaled(base: list, delta: list, scale: float) -> list:
    return [base[i] + delta[i] * scale for i in range(len(base))]
