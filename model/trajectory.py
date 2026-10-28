"""The shared Hyperion rigid-body trajectory model.

Run ``run_trajectory()`` for the selected ascent, or pass an edited configuration.
All internal calculations use SI units and radians. Position and velocity use
Earth-centered inertial (ECI) axes; forces and angular rates use body axes.
Body +x points from engine to nose. The quaternion [x, y, z, w] rotates body
vectors into ECI. Stage and fairing separation are discrete state changes
between RK45 integrations.
"""

from pathlib import Path
import copy
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "inputs"


def load_config(path=None):
    """Read vehicle.toml and flight.toml as a fresh editable input dictionary.

    Aerodynamic file paths in the configuration are relative to the repository
    root unless an absolute path is supplied. Explicit JSON run records remain
    supported. The saved inputs are never modified by a run.
    """
    from .inputs import load_config as read_inputs

    return read_inputs(path)


G0 = 9.80665  # Standard gravity, m/s²; converts Isp and acceleration in g.
MU = 3.986004418e14  # Earth's gravitational parameter, m³/s².
RE = 6371000.0  # Spherical Earth radius, m.
OMEGA = np.array([0.0, 0.0, 7.292115e-5])  # Earth rotation in ECI, rad/s.


def unit(v):
    """Normalize a vector; leave a zero vector at zero."""
    return v / max(np.linalg.norm(v), 1e-15)


def matrix(q):
    """Return the body-to-ECI rotation for an [x, y, z, w] quaternion."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_derivative(q, w):
    """Quaternion rate from body-axis angular velocity ``w`` in rad/s."""
    return 0.5 * np.r_[q[3] * w + np.cross(q[:3], w), -np.dot(q[:3], w)]


def local_frame(r):
    """Return east, north and radial-up unit vectors at ECI position ``r``."""
    up = unit(r)
    east = unit(np.cross([0.0, 0.0, 1.0], up))
    north = np.cross(up, east)
    return east, north, up


class Atmosphere:
    """US Standard Atmosphere 1976 lower layers; explicit upper extension.

    Uses geopotential height, hydrostatic pressure and ideal gas. Above 86 km
    a continuous isothermal exponential tail is a SCREENING approximation.
    Wind is a configurable deterministic profile, not a launch-day forecast.
    """

    def __init__(self, c):
        self.c = c
        self.h = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852.0])
        self.lapse = np.array([-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002])
        self.tb = [288.15]
        self.pb = [101325.0]
        for i, lapse_rate in enumerate(self.lapse):
            layer_height = self.h[i + 1] - self.h[i]
            base_temperature = self.tb[-1]
            self.tb.append(base_temperature + lapse_rate * layer_height)
            self.pb.append(
                self.pb[-1]
                * (
                    np.exp(-G0 * layer_height / (287.05287 * base_temperature))
                    if lapse_rate == 0
                    else (
                        (base_temperature + lapse_rate * layer_height)
                        / base_temperature
                    )
                    ** (-G0 / (287.05287 * lapse_rate))
                )
            )

    def sample(self, h):
        """Return temperature, pressure, density, sound speed and viscosity.

        ``h`` is geometric altitude in metres. Returned units are K, Pa, kg/m³,
        m/s and Pa·s, respectively. Layer equations use geopotential altitude.
        """
        geometric_height = max(0.0, h)
        geopotential_height = 6356766 * geometric_height / (6356766 + geometric_height)
        if geopotential_height <= self.h[-1]:
            layer = min(
                6,
                max(0, np.searchsorted(self.h, geopotential_height, side="right") - 1),
            )
            lapse_rate = self.lapse[layer]
            base_temperature = self.tb[layer]
            height_in_layer = geopotential_height - self.h[layer]
            temperature = base_temperature + lapse_rate * height_in_layer
            pressure = self.pb[layer] * (
                np.exp(-G0 * height_in_layer / (287.05287 * base_temperature))
                if lapse_rate == 0
                else (temperature / base_temperature)
                ** (-G0 / (287.05287 * lapse_rate))
            )
        else:
            temperature = self.tb[-1]
            pressure = self.pb[-1] * np.exp(
                -G0 * (geopotential_height - self.h[-1]) / (287.05287 * temperature)
            )
        temperature += self.c.get("temperature_offset_K", 0.0)
        rho = pressure / (287.05287 * temperature) * self.c.get("density_scale", 1.0)
        sound_speed = np.sqrt(1.4 * 287.05287 * temperature)
        viscosity = (
            1.716e-5
            * (temperature / 273.15) ** 1.5
            * (273.15 + 110.4)
            / (temperature + 110.4)
        )
        return temperature, pressure, rho, sound_speed, viscosity

    def wind(self, h, t, r):
        """Return the wind vector in ECI (m/s), including the optional gust."""
        config = self.c
        east, north, _ = local_frame(r)
        speed = (
            config["wind_reference_m_s"]
            * (max(h, 0.0) / config["wind_reference_height_m"])
            ** config["wind_exponent"]
        )
        speed = min(
            speed,
            config["wind_reference_m_s"]
            * (config["wind_cap_height_m"] / config["wind_reference_height_m"])
            ** config["wind_exponent"],
        )
        # Taper above the user-selected atmosphere region; avoid permanent wind in space.
        speed *= np.clip(
            (config["wind_zero_height_m"] - h)
            / max(1.0, config["wind_zero_height_m"] - config["wind_taper_height_m"]),
            0,
            1,
        )
        direction = np.deg2rad(config["wind_toward_deg_east_of_north"])
        gust = config.get("gust_amplitude_m_s", 0.0) * np.sin(
            2 * np.pi * t / config.get("gust_period_s", 12.0)
        )
        return (speed + gust) * np.cos(direction) * north + (speed + gust) * np.sin(
            direction
        ) * east


class AeroTable:
    """Interpolate each Mach curve, then interpolate between AoA curves.

    Missing coverage uses the nearest endpoint and sets a flag in the result.
    The same full-stack table is reused after separation as an explicit proxy.
    """

    def __init__(self, path, c):
        self.c = c
        table = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
        self.alpha = np.unique(table["alpha_deg"])
        self.cols = {}
        self.mins = []
        self.maxs = []
        for angle in self.alpha:
            rows = table[table["alpha_deg"] == angle]
            rows = np.sort(rows, order="mach")
            self.cols[angle] = rows
            self.mins.append(rows["mach"].min())
            self.maxs.append(rows["mach"].max())
        self.mlo = max(self.mins)
        self.mhi = min(self.maxs)

    def sample(self, mach, alpha, powered):
        """Return (axial coefficient, normal coefficient, CP metres, range flag).

        ``alpha`` is in radians; CP is measured back from the nose. ``powered``
        chooses the engine-on or engine-off axial coefficient column.
        """
        alpha_deg = abs(np.rad2deg(alpha))
        outside_table = mach < self.mlo or mach > self.mhi or alpha_deg > self.alpha[-1]
        # Endpoint holding is always flagged in output; never claimed as interpolation.
        lookup_mach = np.clip(mach, self.mlo, self.mhi)
        lookup_alpha = np.clip(alpha_deg, self.alpha[0], self.alpha[-1])
        coefficients = []
        for column in ["ca_on" if powered else "ca_off", "cn", "cp_from_nose_m"]:
            values_at_each_alpha = [
                np.interp(
                    lookup_mach, self.cols[angle]["mach"], self.cols[angle][column]
                )
                for angle in self.alpha
            ]
            coefficients.append(
                float(np.interp(lookup_alpha, self.alpha, values_at_each_alpha))
            )
        # CN must vanish at zero AoA for the symmetric rigid-body extension.
        if alpha_deg < 1e-10:
            coefficients[1] = 0.0
        return *coefficients, outside_table


class Vehicle:
    """Mass, axial centre of mass and principal inertia from simple components."""

    def __init__(self, c):
        self.c = c
        self.s = c["stages"]
        self.offset = [0.0, self.s[0]["length_m"]]

    def properties(self, props, attached, fairing):
        """Return mass (kg), CM station (m) and [roll, pitch, yaw] inertia (kg·m²).

        Stations remain measured from the original vehicle base after separation.
        ``props`` holds the two remaining propellant inventories; ``attached``
        and ``fairing`` select the components that still travel with the payload.
        """
        components = []
        radius = self.c["diameter_m"] / 2
        for i, stage in enumerate(self.s):
            if i == 0 and not attached:
                continue
            stage_base = self.offset[i]
            dry_length = stage["length_m"] - (
                self.c["fairing_length_m"] if i == 1 else 0.0
            )
            components.append(
                (
                    stage["dry_mass_kg"],
                    stage_base + 0.5 * dry_length,
                    dry_length,
                    radius,
                )
            )
            remaining_fraction = np.clip(props[i] / stage["propellant_kg"], 0.0, 1.0)
            for tank in stage["tanks"]:
                # Equivalent constant-area liquid column drains downward. Slosh omitted.
                liquid_height = tank["height_m"] * remaining_fraction
                components.append(
                    (
                        tank["loaded_mass_kg"] * remaining_fraction,
                        stage_base + tank["bottom_m"] + 0.5 * liquid_height,
                        liquid_height,
                        radius,
                    )
                )
        base = self.offset[1] + self.s[1]["length_m"] - self.c["fairing_length_m"]
        components.append(
            (
                self.c["payload_kg"],
                base + self.c["payload_cg_above_fairing_base_m"],
                self.c["payload_length_m"],
                self.c["payload_radius_m"],
            )
        )
        if fairing:
            components.append(
                (
                    self.c["fairing_kg"],
                    base + self.c["fairing_cg_above_base_m"],
                    self.c["fairing_length_m"],
                    radius,
                )
            )
        total = sum(x[0] for x in components)
        cg = sum(m * x for m, x, L, r in components) / total
        # Equivalent uniform solid cylinders + parallel-axis theorem.
        roll_inertia = sum(0.5 * m * r * r for m, x, L, r in components)
        transverse_inertia = sum(
            m * (3 * r * r + L * L) / 12 + m * (x - cg) ** 2
            for m, x, L, r in components
        )
        return (
            total,
            cg,
            np.array([roll_inertia, transverse_inertia, transverse_inertia]),
        )


class Simulation:
    """One fresh 6DOF ascent run, with phase events and a 22-component state.

    ``root`` optionally overrides the base for relative aerodynamic-data paths.
    The configuration is copied so a run cannot modify caller-owned inputs.
    Create a new Simulation for each run; this object retains its sampled rows.

    State layout (zero-based slices):
        0:3   ECI position, m
        3:6   ECI velocity, m/s
        6:10  body-to-ECI quaternion [x, y, z, w]
        10:13 body angular rate, rad/s
        13:15 stage 1 and stage 2 propellant, kg
        15:17 gimbal tilt toward body y and z, rad
        17:22 accumulated ideal delta-v, steering, gimbal, drag and gravity
              terms, each in m/s

    Phase 1 is first-stage burn; 0 is separation coast; 2 is upper-stage burn;
    3 is post-cutoff coast. Integrated loss terms are independent of plot spacing.
    """

    def __init__(self, c=None, root=None):
        c = copy.deepcopy(load_config() if c is None else c)
        self.c = c
        self.root = Path(root) if root is not None else REPOSITORY_ROOT
        self.atm = Atmosphere(c["atmosphere"])
        if c["solver"].get("method", "RK45") != "RK45":
            raise ValueError("This simulator requires method=RK45.")
        if c["solver"]["max_time_s"] <= 0:
            raise ValueError("max_time_s must be positive.")
        for st in c["vehicle"]["stages"]:
            if not 0 <= st["residual_kg"] < st["propellant_kg"]:
                raise ValueError("Propellant must exceed residual inventory.")
            if not 0 < st["min_throttle"] <= 1:
                raise ValueError("Minimum throttle must be in (0,1].")
        self.vehicle = Vehicle(c["vehicle"])
        self.table = AeroTable(self.root / c["aero"]["file"], c["aero"])
        self.events = []
        self.rows = []
        self.phase = 1
        self.attached = True
        self.fairing = True
        self.cutoff = False
        self.ignition_time = 0.0

    def initial(self):
        """Build the release state, including Earth rotation and startup propellant."""
        lat, lon = np.deg2rad(
            [self.c["launch"]["latitude_deg"], self.c["launch"]["longitude_deg"]]
        )
        props = [s["propellant_kg"] for s in self.c["vehicle"]["stages"]]
        props[0] -= self.c["vehicle"]["stages"][0]["startup_propellant_kg"]
        initial_cg = self.vehicle.properties(props, True, True)[1]
        r = (RE + self.c["launch"]["altitude_m"] + initial_cg) * np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        e, n, u = local_frame(r)
        q = Rotation.from_matrix(np.column_stack([u, e, n])).as_quat()
        # Start at release with launch vehicle co-rotating with the Earth.
        return np.r_[
            r,
            np.cross(OMEGA, r),
            q,
            OMEGA @ matrix(q),
            props,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def guidance(self, t, r, vrel, v, h, qbar):
        """Return the desired body-axis direction in ECI.

        The first stage follows the pitch schedule. The upper stage targets
        altitude while building orbital speed. An air-relative AoA limit relaxes
        as dynamic pressure falls. Inputs use seconds, SI vectors and pascals.
        """
        guidance = self.c["guidance"]
        east, north, up = local_frame(r)
        azimuth = np.deg2rad(self.c["launch"]["azimuth_deg"])
        horizontal = np.sin(azimuth) * east + np.cos(azimuth) * north
        pitch = np.deg2rad(
            np.interp(t, guidance["pitch_time_s"], guidance["pitch_deg"])
        )
        if self.phase == 2:
            # Altitude hold with radial-velocity damping while building orbital speed.
            radial_speed = np.dot(v, up)
            horizontal_speed = np.linalg.norm(v - radial_speed * up)
            radial_acceleration = np.clip(
                guidance["altitude_gain_s2"] * (guidance["target_altitude_m"] - h)
                - guidance["vertical_damping_s"] * radial_speed,
                -12.0,
                12.0,
            )
            mass = self.vehicle.properties(self._props, self.attached, self.fairing)[0]
            thrust = self.c["vehicle"]["stages"][1]["thrust_N"]
            pitch_sine = np.clip(
                (
                    MU / np.linalg.norm(r) ** 2
                    - horizontal_speed * horizontal_speed / np.linalg.norm(r)
                    + radial_acceleration
                )
                / (thrust / mass),
                -0.7,
                0.95,
            )
            pitch = np.arcsin(pitch_sine)
        target = np.cos(pitch) * horizontal + np.sin(pitch) * up
        if (
            t > guidance.get("aoa_limit_start_s", 25.0)
            and qbar > 1.0
            and np.linalg.norm(vrel) > 100
            and np.dot(vrel, horizontal) > 0
        ):
            air_velocity_direction = unit(vrel)
            angle = np.arccos(np.clip(np.dot(target, air_velocity_direction), -1, 1))
            # Continuous relaxation as atmosphere thins; avoids a command step at max-Q exit.
            limit = np.deg2rad(
                min(
                    60.0,
                    guidance["aoa_command_limit_deg"]
                    * np.sqrt(max(1.0, guidance["aoa_limit_min_q_Pa"] / qbar)),
                )
            )
            if angle > limit:
                tangent = unit(
                    target
                    - air_velocity_direction * np.dot(target, air_velocity_direction)
                )
                target = (
                    np.cos(limit) * air_velocity_direction + np.sin(limit) * tangent
                )
        return unit(target)

    def evaluate(self, t, y, record=False):
        """Evaluate one state: return its RK45 derivative or a named output row.

        Steps follow the physical chain: air state, aerodynamic loads, thrust
        command, attitude control, actuator response, rigid-body dynamics, and
        payload loads. ``record=True`` exposes those intermediate quantities.
        """
        # 1. Recover the mass properties and air-relative state.
        config = self.c
        position = y[:3]
        velocity = y[3:6]
        body_to_eci = matrix(y[6:10])
        body_rate = y[10:13]
        propellant = np.maximum(y[13:15], 0)
        self._props = propellant
        mass, cg, inertia = self.vehicle.properties(
            propellant, self.attached, self.fairing
        )
        altitude = np.linalg.norm(position) - RE
        east, north, up = local_frame(position)
        temperature, pressure, rho, sound_speed, air_viscosity = self.atm.sample(
            altitude
        )
        wind = self.atm.wind(altitude, t, position)
        air_velocity = velocity - np.cross(OMEGA, position) - wind
        air_velocity_body = body_to_eci.T @ air_velocity
        speed = np.linalg.norm(air_velocity)
        mach = speed / sound_speed
        alpha = (
            np.arctan2(np.linalg.norm(air_velocity_body[1:]), air_velocity_body[0])
            if speed > 1e-5
            else 0.0
        )
        qbar = 0.5 * rho * speed**2
        powered = self.phase in [1, 2] and not self.cutoff
        ca, cn, cp_from_nose, outside_table = self.table.sample(mach, alpha, powered)
        # 2. Resolve aerodynamic force and its moment about the current CM.
        reference_area = config["aero"]["reference_area_m2"]
        normal_force = (
            -qbar * reference_area * cn * unit(np.r_[0.0, air_velocity_body[1:]])
            if np.linalg.norm(air_velocity_body[1:]) > 1e-10
            else np.zeros(3)
        )
        aero_force = np.r_[-qbar * reference_area * ca, 0.0, 0.0] + normal_force
        # The raw table describes the complete vehicle. Upper-stage reuse is explicit.
        nose = sum(s["length_m"] for s in config["vehicle"]["stages"])
        if not self.fairing:
            nose -= config["vehicle"]["fairing_length_m"]
        cp = nose - cp_from_nose + config["aero"].get("cp_station_offset_m", 0.0)
        aero_moment_body = np.cross(np.array([cp - cg, 0.0, 0.0]), aero_force)
        # 3. Apply the throttle schedule and CM axial-acceleration command cap.
        stage_index = 0 if self.phase == 1 else 1
        stage = config["vehicle"]["stages"][stage_index]
        engine = self.vehicle.offset[stage_index] + stage["engine_pivot_m"]
        arm = cg - engine
        full_thrust = stage["thrust_N"]
        # Optional known nozzle-area correction. Default zero because source lacks area.
        full_thrust = max(
            0.0,
            full_thrust
            + (stage["reference_pressure_Pa"] - pressure) * stage["exit_area_m2"],
        )
        throttle = 0.0
        if powered:
            schedule = np.interp(
                t, config["throttle"]["time_s"], config["throttle"]["fraction"]
            )
            axial_cap = (
                mass * G0 * config["limits"]["command_axial_g"] - aero_force[0]
            ) / max(full_thrust, 1.0)
            throttle = np.clip(min(schedule, axial_cap), stage["min_throttle"], 1.0)
        thrust = full_thrust * throttle
        # 4. Convert attitude error into a limited angular-acceleration command.
        target = self.guidance(t, position, air_velocity, velocity, altitude, qbar)
        target_body = body_to_eci.T @ target
        attitude_error = np.cross(np.array([1.0, 0.0, 0.0]), target_body)
        control = config["control"]
        natural_frequency = control["natural_frequency_rad_s"]
        earth_rate_body = body_to_eci.T @ OMEGA
        angular_command = (
            natural_frequency * natural_frequency * attitude_error
            - 2
            * control["damping_ratio"]
            * natural_frequency
            * (body_rate - earth_rate_body)
        )
        command_magnitude = np.linalg.norm(angular_command)
        angular_acceleration_limit = np.deg2rad(
            control.get("angular_acceleration_limit_deg_s2", 0.3)
        )
        if self.phase == 2:
            angular_acceleration_limit = np.deg2rad(
                control.get("upper_angular_acceleration_limit_deg_s2", 1.5)
            )
        if command_magnitude > angular_acceleration_limit:
            angular_command *= angular_acceleration_limit / command_magnitude
        # Axial roll is held by a separately stated RCS allocation, not engine gimbal.
        mdot = thrust / (stage["isp_s"] * G0) if powered else 0.0
        propellant_rate = np.zeros(2)
        propellant_rate[stage_index] = -mdot
        # Propellant flow changes CM and inertia; use one-kilogram differences.
        inertia_rate = np.zeros(3)
        cgdot = 0.0
        cgddot = 0.0
        if mdot:
            altered = propellant.copy()
            altered[stage_index] = max(0.0, altered[stage_index] - 1.0)
            minus = self.vehicle.properties(altered, self.attached, self.fairing)
            inertia_rate = (minus[2] - inertia) * mdot
            added_propellant = propellant.copy()
            added_propellant[stage_index] += 1.0
            plus = self.vehicle.properties(
                added_propellant, self.attached, self.fairing
            )
            cg_per_kg = (plus[1] - minus[1]) / 2
            cgdot = -cg_per_kg * mdot
            # Slow mass-flow variation: exact schedule slope or late-burn g-limit slope.
            knot = min(
                len(config["throttle"]["time_s"]) - 2,
                max(0, np.searchsorted(config["throttle"]["time_s"], t) - 1),
            )
            slope = (
                config["throttle"]["fraction"][knot + 1]
                - config["throttle"]["fraction"][knot]
            ) / (
                config["throttle"]["time_s"][knot + 1]
                - config["throttle"]["time_s"][knot]
            )
            mass_flow_derivative = (
                full_thrust * slope / (stage["isp_s"] * G0)
                if schedule < axial_cap and stage["min_throttle"] < throttle < 1
                else (
                    -mdot * config["limits"]["command_axial_g"] / stage["isp_s"]
                    if axial_cap < schedule and throttle > stage["min_throttle"]
                    else 0.0
                )
            )
            cgddot = (
                plus[1] - 2 * cg + minus[1]
            ) * mdot**2 - cg_per_kg * mass_flow_derivative
        exhaust_momentum_flux = mdot * arm * arm * np.r_[0.0, body_rate[1:]]
        required_moment = (
            inertia * angular_command
            + np.cross(body_rate, inertia * body_rate)
            + inertia_rate * body_rate
            + exhaust_momentum_flux
            - aero_moment_body
        )
        # 5. Allocate pitch/yaw moment to gimbal motion and roll to the stated RCS.
        gimbal_limit = np.deg2rad(stage["gimbal_limit_deg"])
        gimbal_rate_limit = np.deg2rad(stage["gimbal_rate_deg_s"])
        # State y[15:17] represents force-direction tilt in body y,z (radians).
        desired = (
            np.array([-required_moment[2], required_moment[1]]) / max(thrust * arm, 1.0)
            if thrust > 0 and arm > 0
            else np.zeros(2)
        )
        desired_magnitude = np.linalg.norm(desired)
        if desired_magnitude > np.sin(gimbal_limit):
            desired *= np.sin(gimbal_limit) / desired_magnitude
        desired = np.arcsin(np.clip(desired, -1.0, 1.0))
        gimbal_rate = (desired - y[15:17]) / control["actuator_time_constant_s"]
        if np.linalg.norm(gimbal_rate) > gimbal_rate_limit:
            gimbal_rate *= gimbal_rate_limit / np.linalg.norm(gimbal_rate)
        gimbal_angles = y[15:17].copy()
        gimbal_magnitude = np.linalg.norm(gimbal_angles)
        if gimbal_magnitude > gimbal_limit:
            gimbal_angles *= gimbal_limit / gimbal_magnitude
        thrust_force = (
            thrust
            * np.r_[
                np.sqrt(max(0.0, 1.0 - np.sum(np.sin(gimbal_angles) ** 2))),
                np.sin(gimbal_angles),
            ]
        )
        thrust_moment = np.cross(np.array([-arm, 0.0, 0.0]), thrust_force)
        rcs_capacity = (
            control["rcs_moment_Nm"] if not powered else control["roll_rcs_moment_Nm"]
        )
        rcs_moment = np.zeros(3)
        if not powered:
            required_magnitude = np.linalg.norm(required_moment)
            rcs_moment = required_moment * min(
                1.0, rcs_capacity / max(required_magnitude, 1e-12)
            )
        else:
            rcs_moment[0] = np.clip(required_moment[0], -rcs_capacity, rcs_capacity)
        # 6. Integrate translation and rotation, including changing inertia.
        angular_acceleration = (
            aero_moment_body
            + thrust_moment
            + rcs_moment
            - np.cross(body_rate, inertia * body_rate)
            - inertia_rate * body_rate
            - exhaust_momentum_flux
        ) / inertia
        gravity = -MU * position / np.linalg.norm(position) ** 3
        acceleration = body_to_eci @ (aero_force + thrust_force) / mass + gravity
        # 7. Compute payload-point proper acceleration and accumulated velocity losses.
        point = np.array([config["vehicle"]["payload_station_m"] - cg, 0.0, 0.0])
        proper = (
            (aero_force + thrust_force) / mass
            + np.cross(angular_acceleration, point)
            + np.cross(body_rate, np.cross(body_rate, point))
        )
        proper += 2 * np.cross(body_rate, np.array([-cgdot, 0.0, 0.0])) + np.array(
            [-cgddot, 0.0, 0.0]
        )
        velocity_direction = unit(velocity)
        steering = (
            thrust
            / mass
            * (
                1.0
                - np.clip(
                    np.dot(body_to_eci @ unit(thrust_force), velocity_direction), -1, 1
                )
            )
            if thrust
            else 0.0
        )
        gimbal_loss = (
            thrust / mass * (1.0 - thrust_force[0] / thrust) if thrust else 0.0
        )
        drag_loss = -np.dot(body_to_eci @ aero_force, velocity_direction) / mass
        grav_loss = -np.dot(gravity, velocity_direction)
        derivative = np.r_[
            velocity,
            acceleration,
            quaternion_derivative(y[6:10], body_rate),
            angular_acceleration,
            propellant_rate,
            gimbal_rate,
            thrust / mass,
            steering,
            gimbal_loss,
            drag_loss,
            grav_loss,
        ]
        if not record:
            return derivative
        available = thrust * max(arm, 0.0) * np.sin(gimbal_limit) + (
            control["rcs_moment_Nm"] if not powered else 0.0
        )
        aero_moment = np.linalg.norm(aero_moment_body[1:])
        demand = np.linalg.norm(required_moment[1:])
        relative = velocity - np.cross(OMEGA, position)

        def elevation(vector):
            return np.rad2deg(
                np.arctan2(
                    np.dot(vector, up), np.linalg.norm(vector - np.dot(vector, up) * up)
                )
            )

        inertial_speed = np.linalg.norm(velocity)
        radial = np.dot(velocity, up)
        angular_momentum = np.cross(position, velocity)
        eccentricity = np.linalg.norm(
            np.cross(velocity, angular_momentum) / MU - unit(position)
        )
        energy = 0.5 * inertial_speed**2 - MU / np.linalg.norm(position)
        semimajor_axis = -MU / (2 * energy) if energy < 0 else np.inf
        sample = dict(
            t_s=t,
            phase=self.phase,
            altitude_m=altitude,
            mass_kg=mass,
            cg_from_base_m=cg,
            Ixx_kg_m2=inertia[0],
            Iyy_kg_m2=inertia[1],
            Izz_kg_m2=inertia[2],
            prop1_kg=propellant[0],
            prop2_kg=propellant[1],
            cg_rate_m_s=cgdot,
            cg_acceleration_m_s2=cgddot,
            airspeed_m_s=speed,
            inertial_speed_m_s=inertial_speed,
            vertical_speed_m_s=radial,
            mach=mach,
            temperature_K=temperature,
            pressure_Pa=pressure,
            density_kg_m3=rho,
            wind_m_s=np.linalg.norm(wind),
            reynolds_length=rho
            * speed
            * (
                sum(s["length_m"] for s in config["vehicle"]["stages"])
                - (0.0 if self.attached else config["vehicle"]["stages"][0]["length_m"])
                - (0.0 if self.fairing else config["vehicle"]["fairing_length_m"])
            )
            / air_viscosity,
            q_Pa=qbar,
            alpha_deg=np.rad2deg(alpha),
            qalpha_Pa_deg=qbar * np.rad2deg(alpha),
            pitch_deg=elevation(body_to_eci[:, 0]),
            heading_deg=np.rad2deg(
                np.arctan2(
                    np.dot(body_to_eci[:, 0], east), np.dot(body_to_eci[:, 0], north)
                )
            )
            % 360,
            flight_path_deg=elevation(relative),
            inertial_flight_path_deg=elevation(velocity),
            command_pitch_deg=elevation(target),
            throttle=throttle,
            thrust_N=thrust,
            gimbal_y_deg=np.rad2deg(gimbal_angles[0]),
            gimbal_z_deg=np.rad2deg(gimbal_angles[1]),
            gimbal_total_deg=np.rad2deg(np.linalg.norm(gimbal_angles)),
            gimbal_rate_deg_s=np.rad2deg(np.linalg.norm(gimbal_rate)),
            axial_g=proper[0] / G0,
            lateral_g=np.linalg.norm(proper[1:]) / G0,
            cm_proper_g=np.linalg.norm(aero_force + thrust_force) / mass / G0,
            aero_moment_Nm=aero_moment,
            available_moment_Nm=available,
            required_moment_Nm=demand,
            actual_control_moment_Nm=np.linalg.norm((thrust_moment + rcs_moment)[1:]),
            moment_margin_Nm=available - aero_moment,
            ratio2_margin_Nm=available
            - config["limits"]["control_ratio"] * aero_moment,
            normal_force_N=np.linalg.norm(normal_force),
            axial_aero_force_N=-aero_force[0],
            cp_from_base_m=cp,
            aero_out_of_domain=int(outside_table),
            aero_configuration_proxy=int(not self.attached or not self.fairing),
            ideal_dv_m_s=y[17],
            steering_loss_m_s=y[18],
            gimbal_loss_m_s=y[19],
            drag_loss_m_s=y[20],
            gravity_loss_m_s=y[21],
            osculating_perigee_m=semimajor_axis * (1 - eccentricity) - RE,
            osculating_apogee_m=semimajor_axis * (1 + eccentricity) - RE,
            quaternion_norm=np.linalg.norm(y[6:10]),
        )
        for i, k in enumerate(["x", "y", "z"]):
            sample[f"eci_{k}_m"] = position[i]
            sample[f"eci_v{k}_m_s"] = velocity[i]
            sample[f"body_rate_{k}_deg_s"] = np.rad2deg(body_rate[i])
        for i, k in enumerate(["x", "y", "z", "w"]):
            sample[f"quaternion_{k}"] = y[6 + i]
        return sample

    def integrate(self, y, t0, t1, events=None):
        """Advance one continuous phase and sample its dense RK45 solution."""
        solver = self.c["solver"]
        if solver.get("verbose", False):
            print(f"Integrating phase {self.phase}: t={t0:.2f} s", flush=True)
        atol = np.r_[
            np.full(3, solver["position_atol_m"]),
            np.full(3, solver["velocity_atol_m_s"]),
            np.full(4, 1e-9),
            np.full(3, 1e-9),
            np.full(2, 1e-4),
            np.full(2, 1e-9),
            np.full(5, 1e-5),
        ]
        solution = solve_ivp(
            self.evaluate,
            (t0, t1),
            y,
            method="RK45",
            rtol=solver["rtol"],
            atol=atol,
            max_step=solver["max_step_s"],
            dense_output=True,
            events=events,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        if solver.get("verbose", False):
            print(
                f"  reached {solution.t[-1]:.2f} s, {solution.nfev} RHS evaluations",
                flush=True,
            )
        sample_times = np.arange(t0, solution.t[-1], solver["output_step_s"])
        sample_times = np.unique(np.r_[sample_times, solution.t[-1]])
        self.rows.extend(
            self.evaluate(float(t), solution.sol(t), True) for t in sample_times
        )
        return solution.y[:, -1], float(solution.t[-1]), solution

    def run(self):
        """Run ascent phases, applying CM and velocity changes at mass shedding."""
        y = self.initial()
        t = 0.0
        config = self.c

        def burnout1(t, y):
            return y[13] - config["vehicle"]["stages"][0]["residual_kg"]

        burnout1.terminal = True
        burnout1.direction = -1

        def ground(t, y):
            return np.linalg.norm(y[:3]) - RE + 1.0

        ground.terminal = True
        ground.direction = -1
        self.events.append({"event": "liftoff", "t_s": 0.0})
        y, t, solution = self.integrate(
            y, t, config["solver"]["max_time_s"], [burnout1, ground]
        )
        if not len(solution.t_events[0]):
            self.events.append({"event": "terminated_before_stage1_burnout", "t_s": t})
            return self.rows, self.events
        self.phase = 0
        self.events.append({"event": "MECO", "t_s": t})
        y, t, coast_solution = self.integrate(
            y,
            t,
            min(t + config["separation"]["delay_s"], config["solver"]["max_time_s"]),
            [ground],
        )
        if len(coast_solution.t_events[0]) or t >= config["solver"]["max_time_s"]:
            self.events.append(
                {
                    "event": (
                        "ground_contact"
                        if len(coast_solution.t_events[0])
                        else "time_limit"
                    ),
                    "t_s": t,
                }
            )
            return self.rows, self.events
        mass_before, cg_before, _ = self.vehicle.properties(y[13:15], True, True)
        self.attached = False
        mass_after, cg_after, _ = self.vehicle.properties(y[13:15], False, True)
        shift = matrix(y[6:10]) @ np.array([cg_after - cg_before, 0.0, 0.0])
        y[:3] += shift
        y[3:6] += np.cross(matrix(y[6:10]) @ y[10:13], shift)
        separation_delta_v = config["separation"]["upper_delta_v_m_s"]
        y[3:6] += matrix(y[6:10])[:, 0] * separation_delta_v
        self.events.append(
            {
                "event": "stage_separation",
                "t_s": t,
                "shed_mass_kg": mass_before - mass_after,
                "cg_shift_m": cg_after - cg_before,
                "upper_delta_v_m_s": separation_delta_v,
                "lower_delta_v_m_s": -mass_after
                / (mass_before - mass_after)
                * separation_delta_v,
            }
        )
        self.phase = 2
        self.ignition_time = t
        cg_before = self.vehicle.properties(y[13:15], False, self.fairing)[1]
        y[14] -= config["vehicle"]["stages"][1]["startup_propellant_kg"]
        cg_after = self.vehicle.properties(y[13:15], False, self.fairing)[1]
        shift = matrix(y[6:10]) @ np.array([cg_after - cg_before, 0.0, 0.0])
        y[:3] += shift
        y[3:6] += np.cross(matrix(y[6:10]) @ y[10:13], shift)
        self.events.append({"event": "stage2_ignition", "t_s": t})

        def burnout2(t, y):
            return y[14] - config["vehicle"]["stages"][1]["residual_kg"]

        burnout2.terminal = True
        burnout2.direction = -1

        def insertion(t, y):
            r = y[:3]
            v = y[3:6]
            radial = np.dot(v, unit(r))
            vt = np.linalg.norm(v - radial * unit(r))
            return vt - np.sqrt(MU / np.linalg.norm(r))

        insertion.terminal = True
        insertion.direction = 1

        def fairing(t, y):
            return (
                np.linalg.norm(y[:3]) - RE - config["separation"]["fairing_altitude_m"]
            )

        fairing.terminal = True
        fairing.direction = 1
        event = "time_limit"
        while t < config["solver"]["max_time_s"]:
            if (
                self.fairing
                and np.linalg.norm(y[:3]) - RE
                >= config["separation"]["fairing_altitude_m"]
            ):
                before = self.vehicle.properties(y[13:15], False, True)
                self.fairing = False
                after = self.vehicle.properties(y[13:15], False, False)
                shift = matrix(y[6:10]) @ np.array([after[1] - before[1], 0.0, 0.0])
                y[:3] += shift
                y[3:6] += np.cross(matrix(y[6:10]) @ y[10:13], shift)
                self.events.append(
                    {
                        "event": "fairing_jettison",
                        "t_s": t,
                        "shed_mass_kg": before[0] - after[0],
                    }
                )
            phase_events = [burnout2, insertion, ground] + (
                [fairing] if self.fairing else []
            )
            y, t, solution = self.integrate(
                y, t, config["solver"]["max_time_s"], phase_events
            )
            if len(solution.t_events[0]):
                event = "stage2_propellant_limit"
                break
            if len(solution.t_events[1]):
                event = "orbital_speed_cutoff"
                break
            if len(solution.t_events[2]):
                event = "ground_contact"
                break
            if self.fairing and len(solution.t_events[3]):
                # Move across root by a negligible distance to execute mass jump next loop.
                y[:3] += unit(y[:3]) * 1e-6
                continue
            event = "time_limit"
            break
        self.events.append({"event": event, "t_s": t})
        if event == "orbital_speed_cutoff":
            self.phase = 3
            self.cutoff = True
            y, t, _ = self.integrate(
                y, t, t + config["guidance"]["post_cutoff_coast_s"], [ground]
            )
            self.events.append({"event": "end_coast", "t_s": t})
        return self.rows, self.events


def summarize(df, events, c):
    """Return the canonical engineering summary and sampled governing load cases."""
    peaks = {}
    for key in [
        "q_Pa",
        "qalpha_Pa_deg",
        "axial_g",
        "lateral_g",
        "normal_force_N",
        "axial_aero_force_N",
        "aero_moment_Nm",
        "required_moment_Nm",
        "thrust_N",
        "gimbal_total_deg",
    ]:
        row = df.loc[df[key].abs().idxmax()]
        peaks[key] = {
            k: float(row[k])
            for k in [
                "t_s",
                key,
                "altitude_m",
                "mach",
                "alpha_deg",
                "mass_kg",
                "aero_out_of_domain",
                "aero_configuration_proxy",
            ]
        }
    powered = df[df.thrust_N > 0]
    loaded = df[df.q_Pa > 100]
    active = df[df.aero_moment_Nm > 1.0]
    coast = df[df.thrust_N == 0]
    ratios = active.available_moment_Nm / active.aero_moment_Nm
    powered_ratios = (
        powered[powered.aero_moment_Nm > 1.0].available_moment_Nm
        / powered[powered.aero_moment_Nm > 1.0].aero_moment_Nm
    )
    cutoff = next(
        (x["t_s"] for x in events if x["event"] == "orbital_speed_cutoff"), None
    )
    end = df.iloc[-1] if cutoff is None else df.iloc[np.argmin(np.abs(df.t_s - cutoff))]
    orbit_ok = bool(
        cutoff is not None
        and abs(end.altitude_m - c["guidance"]["target_altitude_m"]) < 5000
        and abs(end.vertical_speed_m_s) < 30
        and end.osculating_perigee_m > 190000
    )
    stage2 = c["vehicle"]["stages"][1]
    burnout_mass = (
        stage2["dry_mass_kg"] + c["vehicle"]["payload_kg"] + stage2["residual_kg"]
    )
    reserve = (
        stage2["isp_s"]
        * G0
        * np.log(
            (stage2["dry_mass_kg"] + c["vehicle"]["payload_kg"] + end.prop2_kg)
            / burnout_mass
        )
    )
    checks = dict(
        axial_acceleration_pass=bool(df.axial_g.abs().max() <= c["limits"]["axial_g"]),
        lateral_acceleration_pass=bool(df.lateral_g.max() <= c["limits"]["lateral_g"]),
        nominal_moment_inequality_all_samples=bool(
            (df.moment_margin_Nm >= -1e-8).all()
        ),
        nominal_moment_ratio_requirement_all_samples=bool(
            (df.ratio2_margin_Nm >= -1e-8).all()
        ),
        moment_demand_within_capacity_powered=bool(
            (powered.required_moment_Nm <= powered.available_moment_Nm + 1e-6).all()
        ),
        circular_200km_target_screen_pass=orbit_ok,
        validated_control_authority=False,
    )
    # Rigid-body load cases: interface proper acceleration times supported payload.
    cases = []
    for name, key in [
        ("Max-Q", "q_Pa"),
        ("Max-Qalpha", "qalpha_Pa_deg"),
        ("Max axial acceleration", "axial_g"),
        ("Max lateral acceleration", "lateral_g"),
        ("Max aerodynamic moment", "aero_moment_Nm"),
        ("Max TVC demand", "required_moment_Nm"),
        ("Max thrust", "thrust_N"),
        ("Max aerodynamic normal force", "normal_force_N"),
        ("Max aerodynamic axial force", "axial_aero_force_N"),
    ]:
        row = df.loc[df[key].abs().idxmax()].to_dict()
        row["case"] = name
        row["payload_axial_force_N"] = row["axial_g"] * G0 * c["vehicle"]["payload_kg"]
        row["payload_lateral_force_N"] = (
            row["lateral_g"] * G0 * c["vehicle"]["payload_kg"]
        )
        cases.append(row)
    checks["significant_q_table_domain_pass"] = bool(
        (loaded.aero_out_of_domain == 0).all()
    )
    checks["powered_moment_ratio_requirement_pass"] = bool(
        (powered.ratio2_margin_Nm >= -1e-8).all()
    )
    summary = dict(
        peaks=peaks,
        checks=checks,
        events=events,
        duration_s=float(df.t_s.max()),
        initial_mass_kg=float(df.mass_kg.iloc[0]),
        ideal_remaining_delta_v_m_s=float(reserve),
        reference_TMI_delta_v_m_s=c.get("mission", {}).get("TMI_delta_v_m_s", 3570.5),
        min_moment_ratio_above_1Nm=float(ratios.min()) if len(ratios) else None,
        min_powered_moment_ratio_above_1Nm=(
            float(powered_ratios.min()) if len(powered_ratios) else None
        ),
        worst_ratio2_margin_Nm=float(df.ratio2_margin_Nm.min()),
        coast_peak_aero_moment_Nm=(
            float(coast.aero_moment_Nm.max()) if len(coast) else 0.0
        ),
        additional_coast_moment_for_ratio2_Nm=(
            float(
                np.maximum(
                    0,
                    coast.aero_moment_Nm * c["limits"]["control_ratio"]
                    - coast.available_moment_Nm,
                ).max()
            )
            if len(coast)
            else 0.0
        ),
        aero_outside_table_samples=int(df.aero_out_of_domain.sum()),
        aero_outside_table_above_100Pa_samples=int(loaded.aero_out_of_domain.sum()),
        post_separation_aero_proxy_samples=int(df.aero_configuration_proxy.sum()),
        separation_average_acceleration_g=c["separation"]["upper_delta_v_m_s"]
        / c["separation"]["impulse_duration_s"]
        / G0,
        peak_aero_outside_table_q_Pa=(
            float(df.loc[df.aero_out_of_domain == 1, "q_Pa"].max())
            if df.aero_out_of_domain.any()
            else 0.0
        ),
        cutoff_state={
            k: float(end[k])
            for k in [
                "t_s",
                "altitude_m",
                "inertial_speed_m_s",
                "vertical_speed_m_s",
                "osculating_perigee_m",
                "osculating_apogee_m",
                "prop2_kg",
                "ideal_dv_m_s",
                "steering_loss_m_s",
                "gimbal_loss_m_s",
                "drag_loss_m_s",
                "gravity_loss_m_s",
            ]
        },
        assumptions=c["assumptions"],
    )
    return summary, pd.DataFrame(cases)


def run_trajectory(config=None):
    """Run the shared model and return ``(trajectory, summary, simulation)``.

    ``config`` may be an editable dictionary or a JSON path. Omitting it loads
    the selected case from ``inputs/vehicle.toml`` and ``inputs/flight.toml``. Output samples
    include both sides of instantaneous phase/mass changes, so event times may
    repeat intentionally; do not silently discard those rows for load checks.
    Units appear in column names; ``mach`` and quaternions are dimensionless.
    ``simulation.load_cases`` contains sampled maximum-load rows.

    Numerical verification is distinct from engineering verification: the
    returned summary retains failed coast-control and aero-coverage checks.
    """
    if config is None or isinstance(config, (str, Path)):
        config = load_config(config)
    simulation = Simulation(config)
    rows, events = simulation.run()
    trajectory = pd.DataFrame(rows)
    summary, cases = summarize(trajectory, events, simulation.c)
    simulation.load_cases = cases
    return trajectory, summary, simulation
