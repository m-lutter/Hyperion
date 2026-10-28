"""Annotated plots shared by Hyperion notebooks."""

from pathlib import Path
import numpy as np
from .trajectory import RE


def plot_results(df, s, c, out=None):
    """Return five annotated figure sheets; optionally save PNGs to ``out``.

    The caller owns the returned figures and may display or close them. This
    helper neither selects a backend nor modifies the input trajectory frame.
    ``df`` is the trajectory history, ``s`` its summary, and ``c`` the run config.
    """
    import matplotlib.pyplot as plt

    figures = []
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "#f7f9fc",
            "axes.facecolor": "white",
            "lines.linewidth": 1.7,
        }
    )
    times = [
        s["peaks"][k]["t_s"] for k in ["q_Pa", "qalpha_Pa_deg", "axial_g", "lateral_g"]
    ]
    colors = ["#b76404", "#b8226b", "#bd2424", "#7a48aa"]
    labels = ["Max-Q", "Max-Qα", "Max axial g", "Max lateral g"]

    def marks(axes, allmarks=True):
        for x, color, label in zip(times, colors, labels):
            if allmarks:
                axes.axvline(x, color=color, ls=":", lw=1, alpha=0.8, label=label)
        for ev in s["events"]:
            if ev["event"] in [
                "stage_separation",
                "fairing_jettison",
                "orbital_speed_cutoff",
            ]:
                axes.axvline(ev["t_s"], color="#546780", ls="--", lw=0.8)
        axes.set_xlabel("Time since release (s)")

    def save(fig, name, title):
        fig.suptitle(title, fontsize=16, fontweight="bold", x=0.05, ha="left")
        fig.text(
            0.05,
            0.015,
            "Hyperion • Current configuration • Preliminary rigid-body model; actuator capability assumed",
            fontsize=9,
            color="#586579",
        )
        if name != "04_orientation_atmosphere.png":
            from matplotlib.lines import Line2D

            fig.legend(
                [Line2D([0], [0], color=color, ls=":") for color in colors],
                labels,
                ncol=4,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.952),
                frameon=False,
            )
        fig.tight_layout(rect=[0, 0.035, 1, 0.915])
        if out is not None:
            fig.savefig(out / name, dpi=150)
        figures.append(fig)

    # 1. Position, attitude, angle of attack and commanded thrust.
    time = df.t_s
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes[0, 0].plot(time, df.altitude_m / 1000)
    axes[0, 0].axhline(c["guidance"]["target_altitude_m"] / 1000, color="gray", ls="--")
    axes[0, 0].set_ylabel("Altitude (km)")
    # ECEF ground-track arc from initial position, independent of Earth's rotation.
    earth_rotation = 7.292115e-5 * time.to_numpy()
    x = df.eci_x_m.to_numpy()
    y = df.eci_y_m.to_numpy()
    z = df.eci_z_m.to_numpy()
    earth_fixed_direction = np.column_stack(
        [
            np.cos(earth_rotation) * x + np.sin(earth_rotation) * y,
            -np.sin(earth_rotation) * x + np.cos(earth_rotation) * y,
            z,
        ]
    )
    earth_fixed_direction /= np.linalg.norm(earth_fixed_direction, axis=1)[:, None]
    ground_arc_km = (
        RE
        * np.arccos(np.clip(earth_fixed_direction @ earth_fixed_direction[0], -1, 1))
        / 1000
    )
    axes[0, 1].plot(ground_arc_km, df.altitude_m / 1000)
    axes[0, 1].set(xlabel="Ground arc from launch (km)", ylabel="Altitude (km)")
    for peak_time, color, label in zip(times, colors, labels):
        index = np.argmin(abs(time - peak_time))
        axes[0, 1].scatter(
            ground_arc_km[index],
            df.altitude_m.iloc[index] / 1000,
            color=color,
            s=30,
            label=label,
        )
    axes[1, 0].plot(time, df.pitch_deg, label="Body elevation")
    axes[1, 0].plot(time, df.command_pitch_deg, ls="--", label="Command")
    axes[1, 0].set_ylabel("Orientation / elevation (deg)")
    axes[1, 0].legend()
    axes[1, 1].plot(time, df.flight_path_deg, label="Earth-relative")
    axes[1, 1].plot(time, df.inertial_flight_path_deg, label="Inertial", alpha=0.6)
    axes[1, 1].set_ylabel("Flight path angle (deg)")
    axes[1, 1].legend()
    axes[2, 0].plot(time, df.alpha_deg)
    axes[2, 0].axhline(4, color="red", ls="--", label="Table AoA boundary")
    axes[2, 0].set_ylabel("Total angle of attack (deg)")
    axes[2, 0].legend()
    axes[2, 1].plot(time, df.thrust_N / 1e6, label="Thrust")
    axes[2, 1].set_ylabel("Commanded thrust (MN)")
    for axis in axes.flat:
        if axis is not axes[0, 1]:
            marks(axis)
    axes[0, 1].legend(fontsize=8)
    save(fig, "01_trajectory.png", "Trajectory, attitude, and thrust")
    # 2. Aerodynamic and payload loads, with their governing samples labelled.
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes[0, 0].plot(time, df.q_Pa / 1000)
    axes[0, 0].set_ylabel("Dynamic pressure (kPa)")
    axes[0, 1].plot(time, df.qalpha_Pa_deg / 1000)
    axes[0, 1].set_ylabel("q × |α| (kPa·deg)")
    if s["peaks"]["qalpha_Pa_deg"]["aero_out_of_domain"]:
        axes[0, 1].text(
            0.43,
            0.80,
            "Max-Qα is outside the 4° AoA table.\nForce/moment prediction is conditional.",
            transform=axes[0, 1].transAxes,
            color="#b8226b",
            fontsize=9,
        )
    axes[1, 0].plot(time, df.axial_g, label="Payload axial")
    axes[1, 0].axhline(
        c["limits"]["axial_g"],
        color="red",
        ls="--",
        label=f"{c['limits']['axial_g']:g} g limit",
    )
    axes[1, 0].set_ylabel("Axial proper acceleration (g)")
    axes[1, 0].legend()
    axes[1, 1].plot(time, df.lateral_g)
    axes[1, 1].axhline(
        c["limits"]["lateral_g"],
        color="red",
        ls="--",
        label=f"{c['limits']['lateral_g']:g} g limit",
    )
    axes[1, 1].set_ylabel("Payload lateral acceleration (g)")
    axes[1, 1].legend()
    axes[2, 0].plot(time, df.normal_force_N / 1000, label="Normal")
    axes[2, 0].plot(time, df.axial_aero_force_N / 1000, label="Axial")
    axes[2, 0].set_ylabel("Aerodynamic force (kN)")
    axes[2, 0].legend()
    for force_key, offset in [
        ("normal_force_N", (15, 15)),
        ("axial_aero_force_N", (15, -20)),
    ]:
        peak = s["peaks"][force_key]
        axes[2, 0].scatter(peak["t_s"], peak[force_key] / 1000, s=25, color="#b8226b")
        axes[2, 0].annotate(
            f"{peak[force_key]/1000:.2f} kN @ {peak['t_s']:.1f} s",
            (peak["t_s"], peak[force_key] / 1000),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    axes[2, 1].plot(time, df.aero_moment_Nm / 1e6)
    axes[2, 1].set_ylabel("Aerodynamic moment about CM (MN·m)")
    for axis, key, scale, units in [
        (axes[0, 0], "q_Pa", 1000, "kPa"),
        (axes[0, 1], "qalpha_Pa_deg", 1000, "kPa·deg"),
        (axes[1, 0], "axial_g", 1, "g"),
        (axes[1, 1], "lateral_g", 1, "g"),
        (axes[2, 1], "aero_moment_Nm", 1e6, "MN·m"),
    ]:
        peak = s["peaks"][key]
        axis.scatter(peak["t_s"], peak[key] / scale, s=35, color="#b8226b", zorder=5)
        axis.annotate(
            f"{peak[key]/scale:.3f} {units} @ {peak['t_s']:.1f} s",
            (peak["t_s"], peak[key] / scale),
            xytext=(12, 18 if key == "lateral_g" else -22),
            textcoords="offset points",
            fontsize=9,
        )
    for axis in axes.flat:
        marks(axis)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    save(fig, "02_load_cases.png", "Maximum aerodynamic and payload loads")
    # 3. Control demand/capacity, actuator motion, mass and velocity losses.
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes[0, 0].plot(time, df.gimbal_y_deg, label="Body y tilt")
    axes[0, 0].plot(time, df.gimbal_z_deg, label="Body z tilt")
    axes[0, 0].plot(time, df.gimbal_total_deg, label="Total", color="black", alpha=0.5)
    axes[0, 0].set_ylabel("Gimbal angle (deg)")
    axes[0, 0].legend()
    axes[0, 1].plot(time, df.throttle * 100)
    axes[0, 1].set_ylabel("Throttle command (%)")
    for name, label in [
        ("aero_moment_Nm", "Aero disturbance"),
        ("available_moment_Nm", "Available control"),
        ("required_moment_Nm", "Tracking demand"),
    ]:
        axes[1, 0].plot(time, df[name] / 1e6, label=label)
    axes[1, 0].set_ylabel("Moment magnitude (MN·m)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(time, df.ratio2_margin_Nm / 1e6)
    axes[1, 1].axhline(0, color="red", ls="--")
    axes[1, 1].fill_between(
        time,
        df.ratio2_margin_Nm / 1e6,
        0,
        where=df.ratio2_margin_Nm < 0,
        color="red",
        alpha=0.3,
    )
    axes[1, 1].set_ylabel("Available − 2 × aero (MN·m)")
    meco = next((ev["t_s"] for ev in s["events"] if ev["event"] == "MECO"), None)
    if meco is not None:
        coast_inset = axes[1, 1].inset_axes([0.48, 0.30, 0.48, 0.42])
        coast_inset.plot(time, df.ratio2_margin_Nm / 1000, color="#b8226b")
        coast_inset.axhline(0, color="red", ls="--", lw=0.8)
        coast_inset.set_xlim(meco - 0.2, meco + c["separation"]["delay_s"] + 0.2)
        deficit = s["additional_coast_moment_for_ratio2_Nm"] / 1000
        inset_scale = max(1.0, deficit)
        coast_inset.set_ylim(-inset_scale * 1.25, inset_scale * 0.35)
        coast_inset.tick_params(labelsize=7)
        coast_inset.set_title("Engine-off gap (kN·m)", fontsize=8)
        axes[1, 1].text(
            0.48,
            0.86,
            f"Coast requirement shortfall: {deficit:.1f} kN·m",
            transform=axes[1, 1].transAxes,
            color="#b8226b",
            fontsize=9,
        )
    axes[2, 0].plot(time, df.mass_kg / 1000, label="Mass")
    axes[2, 0].set_ylabel("Mass (tonnes)")
    other = axes[2, 0].twinx()
    other.plot(time, df.cg_from_base_m, color="#b76404")
    other.set_ylabel("CM from original base (m)", color="#b76404")
    for key, label in [
        ("steering_loss_m_s", "Thrust direction"),
        ("gimbal_loss_m_s", "Gimbal diagnostic (non-additive)"),
        ("drag_loss_m_s", "Aero work"),
        ("gravity_loss_m_s", "Gravity"),
    ]:
        axes[2, 1].plot(time, df[key], label=label)
    axes[2, 1].set_ylabel("Integrated velocity losses (m/s)")
    axes[2, 1].legend(fontsize=8)
    for axis in axes.flat:
        marks(axis)
    axes[0, 1].legend(loc="lower left", fontsize=8)
    save(
        fig,
        "03_control_and_mass.png",
        "Control authority, gimbal motion, and mass depletion",
    )
    # 4. Quaternion state, angular rates, atmosphere and inertia.
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    for k in ["x", "y", "z", "w"]:
        axes[0, 0].plot(time, df["quaternion_" + k], label=k)
    axes[0, 0].set_ylabel("Body-to-ECI quaternion")
    axes[0, 0].legend(ncol=4)
    for k in ["x", "y", "z"]:
        axes[0, 1].plot(time, df["body_rate_" + k + "_deg_s"], label=k)
    axes[0, 1].set_ylabel("Body angular rates (deg/s)")
    axes[0, 1].legend()
    axes[1, 0].plot(time, df.temperature_K)
    axes[1, 0].set_ylabel("Temperature (K)")
    axes[1, 1].semilogy(time, np.maximum(df.pressure_Pa, 1e-12), label="Pressure (Pa)")
    axes[1, 1].semilogy(
        time, np.maximum(df.density_kg_m3, 1e-12), label="Density (kg/m³)"
    )
    axes[1, 1].legend()
    axes[2, 0].plot(time, df.wind_m_s)
    axes[2, 0].set_ylabel("Wind speed (m/s)")
    axes[2, 1].plot(time, df.Iyy_kg_m2 / 1e6, label="Pitch/yaw")
    axes[2, 1].plot(time, df.Ixx_kg_m2 / 1e6, label="Roll")
    axes[2, 1].set_ylabel("Principal inertia (10⁶ kg·m²)")
    axes[2, 1].legend()
    for axis in axes.flat:
        marks(axis, False)
    save(
        fig,
        "04_orientation_atmosphere.png",
        "Full attitude state and atmospheric environment",
    )
    # 5. Distinguish air-relative Mach from inertial orbital speed.
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(time, df.mach, color="#205e96", label="Air-relative Mach number")
    axes[0].axhline(1, color="#8b98a7", ls="--", lw=0.9, label="Mach 1")
    axes[0].set_ylabel("Mach number (air-relative speed / local sound speed)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].plot(
        time, df.airspeed_m_s / 1000, color="#205e96", label="Air-relative speed"
    )
    axes[1].plot(
        time,
        df.inertial_speed_m_s / 1000,
        color="#148579",
        ls="--",
        label="Earth-centered inertial speed",
    )
    axes[1].set_ylabel("Speed (km/s)")
    axes[1].legend(loc="upper left", fontsize=9)
    for axis in axes:
        marks(axis)
    for event in s["events"]:
        if event["event"] in [
            "stage_separation",
            "fairing_jettison",
            "orbital_speed_cutoff",
        ]:
            event_label = {
                "stage_separation": "Stage separation",
                "fairing_jettison": "Fairing jettison",
                "orbital_speed_cutoff": "Engine cutoff",
            }[event["event"]]
            axes[0].text(
                event["t_s"] + 3,
                0.97,
                event_label,
                transform=axes[0].get_xaxis_transform(),
                rotation=90,
                ha="left",
                va="top",
                fontsize=8,
                color="#546780",
            )
    save(fig, "05_velocity_mach.png", "Velocity and Mach history")
    return figures
