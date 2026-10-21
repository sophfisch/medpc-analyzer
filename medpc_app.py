import io
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


APP_VERSION = "2026-10-21.3"


st.set_page_config(page_title="Behavioral Training Progress", page_icon="🐀", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: #f7f9fc; color: #172033;}
    [data-testid="stHeader"] {background: rgba(247,249,252,.92);}
    div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #d9e2ef;
      padding: 14px; border-radius: 14px; box-shadow: 0 2px 8px rgba(24,45,76,.06);}
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] {
      color: #172033;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {background: #eaf1f8; border-radius: 9px; color: #23344f;}
    .stTabs [aria-selected="true"] {background: #d6e8f7;}
    h1, h2, h3 {letter-spacing: .01em;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_saved_progress():
    data_dir = Path(__file__).parent / "data"
    drl_path = data_dir / "drl_progress.xlsx"
    pit_path = data_dir / "pit_progress.xlsx"
    return drl_path, pit_path


def wide_metric(path, sheet):
    frame = pd.read_excel(path, sheet_name=sheet)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    return frame


def metric_long(path, sheet, value_name):
    frame = wide_metric(path, sheet)
    return frame.melt(id_vars="Rat", var_name="Session", value_name=value_name)


def drl_target(session_label):
    """Return the programmed DRL criterion encoded in a session label."""
    match = re.search(r"DRL-(\d+)", str(session_label))
    return float(match.group(1)) if match else np.nan


def plot_drl_series(frame, ylabel, title, target_line=False, percent=False):
    """Plot every saved session explicitly so later DRL stages stay visible."""
    sessions = frame.index.astype(str).tolist()
    x = np.arange(len(sessions))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        ax.plot(x, values, marker="o", linewidth=2.2, markersize=5, label=column)
    if target_line:
        targets = np.array([drl_target(label) for label in sessions], dtype=float)
        ax.step(x, targets, where="mid", color="#e34a4a", linestyle="--",
                linewidth=2.2, label="Programmed criterion")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=.22)
    ax.margins(x=.02)
    if percent:
        ax.set_ylim(0, 100)
    if len(frame.columns) > 1 or target_line:
        ax.legend(frameon=False, ncol=min(3, len(frame.columns) + int(target_line)))
    fig.tight_layout()
    return fig


def shade_drl_stages(ax, sessions):
    """Add subtle alternating bands and stage names to DRL figures."""
    stages = [re.search(r"DRL-(\d+)", str(s)).group(1) for s in sessions]
    start = 0
    colors = ["#eaf3fb", "#f4f7fa"]
    block = 0
    while start < len(stages):
        end = start
        while end + 1 < len(stages) and stages[end + 1] == stages[start]:
            end += 1
        ax.axvspan(start - .5, end + .5, color=colors[block % 2], zorder=0)
        ax.text((start + end) / 2, 1.02, f"DRL-{stages[start]}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#35526f")
        start = end + 1
        block += 1


def individual_drl_figure(rat_id, metrics):
    sessions = metrics.index.astype(str).tolist()
    x = np.arange(len(sessions))
    targets = np.array([drl_target(label) for label in sessions], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(f"{rat_id} — DRL training overview", x=.08, ha="left",
                 fontsize=17, fontweight="bold")
    for ax in axes:
        shade_drl_stages(ax, sessions)
        ax.grid(axis="y", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot(x, metrics["Active presses"], "o-", color="#285f8f", lw=2.4,
                 label="Active presses")
    axes[0].plot(x, metrics["Successful presses"], "o-", color="#2ca25f", lw=2.4,
                 label="Rewards")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(x, metrics["Efficiency (%)"], "o-", color="#7a5195", lw=2.4)
    axes[1].set_ylabel("Efficiency (%)")
    axes[1].set_ylim(0, 105)
    axes[2].plot(x, metrics["Median IRT (s)"], "o-", color="#ef8a32", lw=2.4,
                 label="Median IRT")
    axes[2].step(x, targets, where="mid", color="#d73027", ls="--", lw=2.2,
                 label="Target")
    axes[2].set_ylabel("Median IRT (s)")
    axes[2].legend(frameon=False, ncol=2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sessions, rotation=45, ha="right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, .96])
    return fig


def individual_pit_figure(rat_id, frame):
    sessions = frame.index.astype(str).tolist()
    x = np.arange(len(sessions))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(f"{rat_id} — PIT instrumental training overview", x=.08, ha="left",
                 fontsize=17, fontweight="bold")
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot(x, frame["Active presses"], "o-", color="#285f8f", lw=2.4,
                 label="Active presses")
    axes[0].plot(x, frame["Inactive presses"], "o-", color="#9aa6b2", lw=2.2,
                 label="Inactive presses")
    axes[0].set_ylabel("Presses")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(x, frame["Pellets"], "o-", color="#2ca25f", lw=2.4)
    axes[1].set_ylabel("Pellets")
    if "Head entries" in frame:
        axes[2].plot(x, frame["Head entries"], "o-", color="#7a5195", lw=2.4,
                     label="Head entries")
    if "Post-press checks" in frame:
        axes[2].plot(x, frame["Post-press checks"], "o-", color="#ef8a32", lw=2.4,
                     label="Post-press checks")
    axes[2].set_ylabel("Checks")
    axes[2].legend(frameon=False, ncol=2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sessions, rotation=45, ha="right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, .96])
    return fig


def render_progress_dashboard():
    drl_path, pit_path = load_saved_progress()
    st.title("Behavioral Training Progress")
    st.caption("DRL and PIT cohorts · updated October 21, 2026")
    if not drl_path.exists() or not pit_path.exists():
        st.error("Saved dashboard data are missing. Add both files to the app's data folder.")
        st.stop()

    drl_tab, pit_tab = st.tabs(["DRL cohort", "PIT cohort"])
    with drl_tab:
        active = wide_metric(drl_path, "Active Presses")
        successful = wide_metric(drl_path, "Successful Presses")
        efficiency = wide_metric(drl_path, "Efficiency")
        median_irt = wide_metric(drl_path, "Median IRT")
        latest = active.columns[-1]
        latest_eff = pd.to_numeric(efficiency[latest], errors="coerce")
        latest_irt = pd.to_numeric(median_irt[latest], errors="coerce")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Animals", len(active))
        c2.metric("Current session", latest)
        c3.metric("Mean efficiency", f"{latest_eff.mean():.1%}")
        c4.metric("Median IRT", f"{latest_irt.median():.1f} s")
        st.info("DRL head-entry measures from box 4 are excluded because that detector is unreliable.")

        overview_tab, individual_tab = st.tabs(["Cohort overview", "Individual rats"])
        with overview_tab:
            measure = st.selectbox(
                "DRL measure",
                ["Active presses", "Successful presses", "Efficiency", "Median IRT"],
                key="drl_measure",
            )
            source = {"Active presses": active, "Successful presses": successful,
                      "Efficiency": efficiency, "Median IRT": median_irt}[measure]
            group = source.set_index("Rat").apply(pd.to_numeric, errors="coerce").mean(axis=0)
            plot_values = group.mul(100) if measure == "Efficiency" else group
            plot_frame = plot_values.rename(f"Mean {measure}").to_frame()
            fig = plot_drl_series(
                plot_frame,
                "Efficiency (%)" if measure == "Efficiency" else measure,
                f"Cohort mean — {measure}",
                target_line=measure == "Median IRT",
                percent=measure == "Efficiency",
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("All saved stages are shown: DRL-5, DRL-10, DRL-15, and DRL-20.")

        with individual_tab:
            st.subheader("Individual rat progress")
            rat_id = st.selectbox("Choose a rat", active["Rat"].tolist(), key="drl_rat")
            rat_metrics = pd.DataFrame({
                "Active presses": active.set_index("Rat").loc[rat_id],
                "Successful presses": successful.set_index("Rat").loc[rat_id],
                "Efficiency (%)": efficiency.set_index("Rat").loc[rat_id].astype(float) * 100,
                "Median IRT (s)": median_irt.set_index("Rat").loc[rat_id],
            }).apply(pd.to_numeric, errors="coerce")
            last = rat_metrics.iloc[-1]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Latest active presses", f"{last['Active presses']:.0f}")
            r2.metric("Latest rewards", f"{last['Successful presses']:.0f}")
            r3.metric("Latest efficiency", f"{last['Efficiency (%)']:.1f}%")
            r4.metric("Latest median IRT", f"{last['Median IRT (s)']:.1f} s")

            fig = individual_drl_figure(rat_id, rat_metrics)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            if rat_metrics["Active presses"].eq(0).any():
                zero_days = ", ".join(rat_metrics.index[rat_metrics["Active presses"].eq(0)].astype(str))
                st.warning(f"Zero active presses recorded: {zero_days}")
        st.download_button("Download current DRL workbook", drl_path.read_bytes(),
                           file_name=drl_path.name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with pit_tab:
        long = pd.read_excel(pit_path, sheet_name="RI5 Long Data")
        long["Date"] = pd.to_datetime(long["Date"], errors="coerce")
        latest_day = int(pd.to_numeric(long["RI5 day"], errors="coerce").max())
        current = long[pd.to_numeric(long["RI5 day"], errors="coerce").eq(latest_day)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Animals", long["Rat"].nunique())
        c2.metric("Current session", "RI-60 Day 4")
        c3.metric("Mean active presses", f"{current['Active presses'].mean():.1f}")
        c4.metric("Mean active preference", f"{current['Active preference'].mean():.1%}")

        pit_overview_tab, pit_individual_tab = st.tabs(["Cohort overview", "Individual rats"])
        with pit_overview_tab:
            grouped = long.groupby("RI5 day", sort=True).agg(
                active_presses=("Active presses", "mean"),
                pellets=("Pellets", "mean"),
                active_preference=("Active preference", "mean"),
            )
            st.line_chart(grouped[["active_presses", "pellets"]], height=330)

        with pit_individual_tab:
            st.subheader("Individual rat progress")
            pit_rat = st.selectbox("Choose a rat", sorted(long["Rat"].dropna().unique()), key="pit_rat")
            one = long[long["Rat"].eq(pit_rat)].sort_values("RI5 day").copy()
            one["Session"] = ["RI-5 D1", "RI-5 D2", "RI-5 D3", "RI-15 D1", "RI-30 D1",
                              "RI-30 D2", "RI-60 D1", "RI-60 D2", "RI-60 D3", "RI-60 D4"][:len(one)]
            one = one.set_index("Session")
            last = one.iloc[-1]
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Latest active presses", f"{last['Active presses']:.0f}")
            p2.metric("Latest pellets", f"{last['Pellets']:.0f}")
            p3.metric("Active preference", f"{last['Active preference']:.1%}")
            checks = pd.to_numeric(pd.Series([last.get("Post-press checks", np.nan)]), errors="coerce").iloc[0]
            p4.metric("Post-press checks", "—" if pd.isna(checks) else f"{checks:.0f}")
            fig = individual_pit_figure(pit_rat, one)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.download_button("Download current PIT workbook", pit_path.read_bytes(),
                           file_name=pit_path.name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


render_progress_dashboard()
st.stop()

# Keep the app working even if the image has not yet been uploaded to GitHub.
logo_path = Path(__file__).parent / "assets" / "lab_logo.png"
if logo_path.exists():
    st.image(str(logo_path), width="stretch")

st.title("MED-PC Behavioral Data Converter")
st.caption(f"App version: {APP_VERSION}")
st.write(
    "Upload one or more daily MED-PC text files. The app detects DRL training FR1, DRL-20, "
    "PIT instrumental training, Pavlovian conditioning, and PIT transfer tests."
)


def field(block, name, default=""):
    match = re.search(rf"^{re.escape(name)}:\s*(.*?)\s*$", block, re.MULTILINE)
    return match.group(1) if match else default


def number(block, name, default=0.0):
    try:
        return float(field(block, name, default))
    except (TypeError, ValueError):
        return float(default)


def array(block, name):
    match = re.search(
        rf"^{re.escape(name)}:\s*$\n((?:^\s*\d+:.*(?:\n|$))*)",
        block,
        re.MULTILINE,
    )
    if not match:
        return []
    values = []
    for line in match.group(1).splitlines():
        index_text, value_text = line.split(":", 1)
        start = int(index_text.strip())
        while len(values) < start:
            values.append(np.nan)
        values.extend(float(x) for x in value_text.split())
    return values


def at(values, index, default=np.nan):
    return values[index] if index < len(values) else default


def first_n(values, count, start=0):
    return values[start : start + max(int(count), 0)]


def decode_upload(upload):
    raw = upload.getvalue()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def base_row(block, source_file, session_id):
    return {
        "session_id": session_id,
        "source_file": source_file,
        "date": field(block, "Start Date"),
        "medpc_subject": field(block, "Subject"),
        "subject": field(block, "Subject"),
        "box": int(number(block, "Box")),
        "procedure": field(block, "MSN"),
        "start_time": field(block, "Start Time"),
        "end_time": field(block, "End Time"),
    }


def parse_drl(block, base, tables):
    active_n = int(number(block, "J"))
    success_n = int(number(block, "Q"))
    fail_n = int(number(block, "P"))
    inactive_n = int(number(block, "N"))
    reinforced_head_n = int(number(block, "O"))
    nonreinforced_head_n = int(number(block, "R"))
    duration_s = number(block, "C") / 10

    schedule_match = re.search(r"DRL_(\d+)", base["procedure"])
    schedule_seconds = int(schedule_match.group(1)) if schedule_match else 20
    row = {
        **base,
        "family": f"DRL-{schedule_seconds}",
        "schedule_seconds": schedule_seconds,
        "duration_s": duration_s,
        "duration_min": duration_s / 60,
        "active_presses": active_n,
        "successful_presses": success_n,
        "failed_presses": fail_n,
        "inactive_presses": inactive_n,
        "reinforcers": int(number(block, "F")),
        "efficiency": success_n / active_n if active_n else np.nan,
        "reinforced_head_entries": int(number(block, "V")),
        "nonreinforced_head_entries": int(number(block, "W")),
    }
    tables["sessions"].append(row)

    times = first_n(array(block, "A"), active_n, start=1)
    irts = first_n(array(block, "B"), active_n, start=1)
    outcomes = first_n(array(block, "E"), active_n, start=1)
    for i, (time, irt, outcome) in enumerate(zip(times, irts, outcomes), 1):
        tables["drl_active"].append(
            {
                **base,
                "press_number": i,
                "time_s": time / 10,
                "irt_s": irt / 10,
                "outcome": "Successful" if outcome == 1 else "Failed",
            }
        )

    for i, time in enumerate(first_n(array(block, "T"), inactive_n, start=1), 1):
        tables["drl_inactive"].append({**base, "press_number": i, "time_s": time / 10})

    head_sets = [
        ("X", reinforced_head_n, "After successful press"),
        ("Y", nonreinforced_head_n, "After failed press"),
    ]
    for variable, count, outcome in head_sets:
        for i, time in enumerate(first_n(array(block, variable), count, start=1), 1):
            tables["drl_head"].append(
                {**base, "entry_number": i, "time_s": time / 10, "entry_type": outcome}
            )


def parse_drl_training_fr1(block, base, tables):
    """Parse session totals produced by the DRL_training_FR1 program."""
    active_n = int(number(block, "F"))
    inactive_n = int(number(block, "I"))
    elapsed_min = number(block, "M")

    tables["sessions"].append(
        {
            **base,
            "family": "DRL training FR1",
            "schedule": "FR1",
            "duration_s": elapsed_min * 60,
            "duration_min": elapsed_min,
            "active_presses": active_n,
            "inactive_presses": inactive_n,
            "earned_reinforcers": active_n,
            "seconds_since_last_press_or_free_pellet": number(block, "S"),
            "completed_minutes_since_last_active_press": number(block, "U"),
        }
    )


def parse_instrumental(block, base, tables):
    a = array(block, "A")
    b = array(block, "B")
    schedule = at(a, 0, 0)
    duration_s = number(block, "T") / 100
    schedule_name = "FR1" if schedule == 0 else f"RI{int(schedule)}"
    row = {
        **base,
        "family": "PIT instrumental",
        "schedule": schedule_name,
        "schedule_seconds": schedule,
        "active_lever": "Right" if at(a, 1, 1) == 1 else "Left",
        "planned_duration_min": at(a, 2),
        "duration_s": duration_s,
        "duration_min": duration_s / 60,
        "active_presses": int(at(b, 0, 0)),
        "inactive_presses": int(at(b, 1, 0)),
        "pellets": int(at(b, 2, 0)),
        "head_entries": int(at(b, 3, 0)),
        "beam_breaks": int(at(b, 4, 0)),
        "left_position_transitions": int(at(b, 5, 0)),
        "right_position_transitions": int(at(b, 6, 0)),
        "licks": int(at(b, 7, 0)),
    }
    tables["sessions"].append(row)

    event_specs = [
        ("C", "J", "Active press"),
        ("D", "K", "Inactive press"),
        ("E", "L", "Head entry"),
        ("F", "M", "Beam break"),
        ("G", "N", "Pellet delivery"),
    ]
    for variable, counter, event_type in event_specs:
        count = int(number(block, counter))
        for i, time in enumerate(first_n(array(block, variable), count), 1):
            tables["instrumental_events"].append(
                {**base, "event_number": i, "event_type": event_type, "time_s": time}
            )


def parse_pit_ri(block, base, tables):
    """Parse the newer PIT RI programs (PIT_2b/2c/2d/2e_RI...)."""
    b = array(block, "B")
    schedule_match = re.search(r"RI(\d+)$", base["procedure"])
    schedule_seconds = int(schedule_match.group(1)) if schedule_match else int(at(b, 12, 0))
    active_n, inactive_n = int(at(b, 4, 0)), int(at(b, 5, 0))
    duration_s = number(block, "T") / 100
    row = {
        **base,
        "family": "PIT instrumental",
        "schedule": f"RI{schedule_seconds}",
        "schedule_seconds": schedule_seconds,
        "active_lever": "Right" if int(at(b, 13, 1)) == 1 else "Left",
        "duration_s": duration_s,
        "duration_min": duration_s / 60,
        "active_presses": active_n,
        "inactive_presses": inactive_n,
        "pellets": int(at(b, 2, 0)),
        "head_entries": int(at(b, 0, 0)),
        "head_in_time_s": at(b, 1, 0) / 100,
        "beam_breaks": int(at(b, 4, 0)),
    }
    tables["sessions"].append(row)
    event_specs = [("J", active_n, "Active press"), ("I", int(at(b, 2, 0)), "Pellet delivery")]
    for variable, count, event_type in event_specs:
        for i, time in enumerate(first_n(array(block, variable), count), 1):
            if time <= 0:
                break
            tables["instrumental_events"].append(
                {**base, "event_number": i, "event_type": event_type, "time_s": time}
            )
    heads = array(block, "C")
    head_number = 0
    for i in range(0, len(heads) - 1, 2):
        start, duration = heads[i], heads[i + 1]
        if start == 0 and duration == 0:
            break
        if start > 0 and duration > 0:
            head_number += 1
            tables["instrumental_events"].append(
                {**base, "event_number": head_number, "event_type": "Head entry", "time_s": start}
            )


def cue_label(trial_type):
    return "CS+" if trial_type == 1 else "CS-" if trial_type == 2 else "Unknown"


def modality_label(code):
    return "Tone" if code == 1 else "Clicker" if code == 2 else "Unknown"


def parse_pavlovian(block, base, tables):
    a = array(block, "A")
    b = array(block, "B")
    trial_n = int(at(a, 6, 8))
    cs_plus_modality = int(at(b, 2, at(a, 8, 0)))
    duration_s = number(block, "T") / 100
    row = {
        **base,
        "family": "PIT Pavlovian",
        "cs_plus_modality": modality_label(cs_plus_modality),
        "cue_duration_s": at(a, 5),
        "planned_trials": trial_n,
        "duration_s": duration_s,
        "duration_min": duration_s / 60,
        "head_entries": int(at(b, 0, 0)),
        "pellets": int(at(b, 1, 0)),
        "right_presses_check": int(at(b, 3, 0)),
        "left_presses_check": int(at(b, 4, 0)),
        "beam_breaks": int(at(b, 5, 0)),
        "iti_head_entries": int(at(b, 6, 0)),
        "left_position_transitions": int(at(b, 7, 0)),
        "right_position_transitions": int(at(b, 8, 0)),
        "licks": int(at(b, 9, 0)),
        "trial_order_index": int(at(b, 10, 0)),
    }
    tables["sessions"].append(row)

    event_times = first_n(array(block, "E"), int(number(block, "P")))
    event_codes = first_n(array(block, "F"), int(number(block, "P")))
    code_names = {1: "CS+ on", 2: "CS+ off", 3: "CS- on", 4: "CS- off", 5: "Pre-CS opens"}
    for i, (time, code) in enumerate(zip(event_times, event_codes), 1):
        tables["cue_events"].append(
            {**base, "event_number": i, "time_s": time, "event_code": int(code), "event": code_names.get(int(code), "Unknown")}
        )

    cue_on_codes = [int(code) for code in event_codes if code in (1, 3)]
    pre_counts = array(block, "G")
    cue_counts = array(block, "H")
    pellet_counts = array(block, "I")
    for trial in range(trial_n):
        trial_type = 1 if trial < len(cue_on_codes) and cue_on_codes[trial] == 1 else 2
        cue = cue_label(trial_type)
        modality = modality_label(cs_plus_modality if cue == "CS+" else 3 - cs_plus_modality)
        tables["pav_trials"].append(
            {
                **base,
                "trial": trial + 1,
                "cue_type": cue,
                "cue_modality": modality,
                "pre_cs_head_entries": int(at(pre_counts, trial, 0)),
                "cue_head_entries": int(at(cue_counts, trial, 0)),
                "pellets": int(at(pellet_counts, trial, 0)),
            }
        )

    timestamp_specs = [("C", "N", "Head entry"), ("D", "O", "Pellet delivery")]
    for variable, counter, event_type in timestamp_specs:
        for i, time in enumerate(first_n(array(block, variable), int(number(block, counter))), 1):
            tables["pav_events"].append(
                {**base, "event_number": i, "event_type": event_type, "time_s": time}
            )


def parse_pit_test(block, base, tables):
    a = array(block, "A")
    b = array(block, "B")
    trial_n = int(at(a, 6, 8))
    cs_plus_modality = int(at(b, 9, at(a, 8, 0)))
    duration_s = number(block, "T") / 100
    row = {
        **base,
        "family": "PIT transfer test",
        "cs_plus_modality": modality_label(cs_plus_modality),
        "cue_order": int(at(b, 10, at(a, 7, 0))),
        "extinction_block_min": at(a, 9),
        "cue_duration_s": at(a, 5),
        "duration_s": duration_s,
        "duration_min": duration_s / 60,
        "active_presses": int(at(b, 0, 0)),
        "inactive_presses": int(at(b, 1, 0)),
        "head_entries": int(at(b, 2, 0)),
        "beam_breaks": int(at(b, 5, 0)),
        "active_presses_outside_windows": int(at(b, 6, 0)),
        "head_entries_outside_windows": int(at(b, 7, 0)),
        "left_position_transitions": int(at(b, 3, 0)),
        "right_position_transitions": int(at(b, 4, 0)),
        "licks": int(at(b, 8, 0)),
    }
    tables["sessions"].append(row)

    trial_types = array(block, "M")
    active_windows = array(block, "I")
    head_windows = array(block, "K")
    for trial in range(trial_n):
        trial_type = int(at(trial_types, trial, 0))
        cue = cue_label(trial_type)
        modality = modality_label(cs_plus_modality if cue == "CS+" else 3 - cs_plus_modality)
        pre_active = int(at(active_windows, trial, 0))
        cue_active = int(at(active_windows, 8 + trial, 0))
        pre_head = int(at(head_windows, trial, 0))
        cue_head = int(at(head_windows, 8 + trial, 0))
        tables["pit_trials"].append(
            {
                **base,
                "trial": trial + 1,
                "cue_type": cue,
                "cue_modality": modality,
                "pre_cs_active_presses": pre_active,
                "cue_active_presses": cue_active,
                "active_press_change": cue_active - pre_active,
                "pre_cs_head_entries": pre_head,
                "cue_head_entries": cue_head,
                "head_entry_change": cue_head - pre_head,
            }
        )

    cue_times = first_n(array(block, "G"), int(number(block, "R")))
    cue_codes = first_n(array(block, "H"), int(number(block, "R")))
    code_names = {1: "CS+ on", 2: "CS+ off", 3: "CS- on", 4: "CS- off", 5: "Pre-CS opens"}
    for i, (time, code) in enumerate(zip(cue_times, cue_codes), 1):
        tables["cue_events"].append(
            {**base, "event_number": i, "time_s": time, "event_code": int(code), "event": code_names.get(int(code), "Unknown")}
        )

    timestamp_specs = [("C", "N", "Active press"), ("E", "P", "Head entry"), ("F", "J", "Beam break")]
    for variable, counter, event_type in timestamp_specs:
        for i, time in enumerate(first_n(array(block, variable), int(number(block, counter))), 1):
            tables["pit_events"].append(
                {**base, "event_number": i, "event_type": event_type, "time_s": time}
            )


def parse_output_test(block, base, tables):
    row = {**base, "family": "Hardware output test"}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        row[f"raw_{letter}"] = number(block, letter)
    tables["hardware_tests"].append(row)


def empty_tables():
    names = [
        "sessions", "drl_active", "drl_inactive", "drl_head",
        "instrumental_events", "pav_trials", "pav_events",
        "pit_trials", "pit_events", "cue_events", "hardware_tests",
    ]
    return {name: [] for name in names}


def parse_uploads(uploads):
    tables = empty_tables()
    unknown = []
    session_id = 0
    for upload in uploads:
        text = decode_upload(upload).replace("\r", "")
        blocks = re.split(r"(?=^Start Date:)", text, flags=re.MULTILINE)
        for block in blocks:
            procedure = field(block, "MSN")
            if not procedure:
                continue
            session_id += 1
            base = base_row(block, upload.name, session_id)
            if procedure in {
                "DRL_5_HFD", "DRL_10_HFD", "DRL_10_HFD_new",
                "DRL_15_HFD_new", "DRL_20_HFD", "DRL_20_HFD_new",
            }:
                parse_drl(block, base, tables)
            elif procedure == "DRL_training_FR1":
                parse_drl_training_fr1(block, base, tables)
            elif re.match(r"^PIT_\d+[A-Za-z]?_RI\d+$", procedure):
                parse_pit_ri(block, base, tables)
            elif procedure.startswith("PIT_Instrumental_"):
                parse_instrumental(block, base, tables)
            elif procedure.startswith("PIT_Pav_"):
                parse_pavlovian(block, base, tables)
            elif procedure.startswith("PIT_Test_"):
                parse_pit_test(block, base, tables)
            elif procedure == "PIT_OutputTest":
                parse_output_test(block, base, tables)
            elif procedure in {
                "PR TEST HFD",
            }:
                # These can coexist in the same daily file as DRL-20.
                # They are intentionally outside this app's requested analyses.
                continue
            else:
                unknown.append({**base, "reason": "No parser is defined for this MSN"})
    frames = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    frames["unknown"] = pd.DataFrame(unknown)
    return frames


def apply_subjects(frames, subject_map):
    for frame in frames.values():
        if not frame.empty and "session_id" in frame.columns:
            mapped = frame["session_id"].map(subject_map)
            if "subject" in frame.columns:
                frame["subject"] = mapped.fillna(frame["subject"])


def excel_bytes(frames):
    sheet_names = {
        "sessions": "Session Summary",
        "drl_active": "DRL Active Presses",
        "drl_inactive": "DRL Inactive Presses",
        "drl_head": "DRL Head Entries",
        "instrumental_events": "Instrumental Events",
        "pav_trials": "Pavlovian Trials",
        "pav_events": "Pavlovian Events",
        "pit_trials": "PIT Test Trials",
        "pit_events": "PIT Test Events",
        "cue_events": "Cue Events",
        "hardware_tests": "Hardware Tests",
        "unknown": "Unrecognized Sessions",
    }
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for key, sheet in sheet_names.items():
            frame = frames[key]
            if not frame.empty:
                frame.to_excel(writer, sheet_name=sheet, index=False)
    output.seek(0)
    return output.getvalue()


def frames_for_sessions(frames, session_ids):
    """Keep only rows belonging to the selected MED-PC sessions."""
    selected = {}
    session_ids = set(session_ids)
    for key, frame in frames.items():
        if frame.empty:
            selected[key] = frame.copy()
        elif "session_id" in frame.columns:
            selected[key] = frame[frame["session_id"].isin(session_ids)].copy()
        else:
            selected[key] = frame.copy()

        # Remove rows and columns that are completely empty for this
        # particular procedure. Legitimate numeric zeros are preserved.
        cleaned = selected[key].dropna(axis=0, how="all")
        empty_columns = []
        for column in cleaned.columns:
            values = cleaned[column]
            if values.isna().all():
                empty_columns.append(column)
            elif values.dtype == "object" and values.fillna("").astype(str).str.strip().eq("").all():
                empty_columns.append(column)
        cleaned = cleaned.drop(columns=empty_columns)

        # The master session table contains the union of every procedure's
        # fields. Procedure-specific workbooks should show only the concise
        # summary measures relevant to that procedure.
        if key == "sessions" and not cleaned.empty and "procedure" in cleaned.columns:
            procedure = cleaned["procedure"].iloc[0]
            common = [
                "source_file", "date", "subject", "box", "procedure",
                "start_time", "end_time", "duration_s", "duration_min",
            ]
            if procedure.startswith("DRL_"):
                relevant = common + [
                    "active_presses", "successful_presses", "failed_presses",
                    "inactive_presses", "reinforcers", "efficiency",
                    "reinforced_head_entries", "nonreinforced_head_entries",
                ]
            elif procedure == "DRL_training_FR1":
                relevant = common + [
                    "schedule", "active_presses", "inactive_presses",
                    "earned_reinforcers",
                    "seconds_since_last_press_or_free_pellet",
                    "completed_minutes_since_last_active_press",
                ]
            elif procedure.startswith("PIT_Instrumental_") or re.match(r"^PIT_\d+[A-Za-z]?_RI\d+$", procedure):
                relevant = common + [
                    "schedule", "active_lever", "active_presses",
                    "inactive_presses", "pellets", "head_entries", "beam_breaks",
                ]
            elif procedure.startswith("PIT_Pav_"):
                relevant = common + [
                    "cs_plus_modality", "cue_duration_s", "planned_trials",
                    "head_entries", "pellets", "iti_head_entries", "beam_breaks",
                ]
            elif procedure.startswith("PIT_Test_"):
                relevant = common + [
                    "cs_plus_modality", "cue_order", "extinction_block_min",
                    "cue_duration_s", "active_presses", "inactive_presses",
                    "head_entries", "beam_breaks",
                    "active_presses_outside_windows",
                    "head_entries_outside_windows",
                ]
            else:
                relevant = list(cleaned.columns)
            cleaned = cleaned[[column for column in relevant if column in cleaned.columns]]

        selected[key] = cleaned
    return selected


def safe_filename(text):
    """Convert a procedure name into a safe filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_.")
    return cleaned or "MED-PC"


uploads = st.file_uploader(
    "Upload daily MED-PC text files",
    type=["txt"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Upload one or more MED-PC text files to begin.")
    st.stop()

frames = parse_uploads(uploads)

all_sessions = []
for key in ("sessions", "hardware_tests", "unknown"):
    if not frames[key].empty:
        cols = ["session_id", "source_file", "date", "box", "procedure", "medpc_subject"]
        all_sessions.append(frames[key][[c for c in cols if c in frames[key].columns]])

if not all_sessions:
    st.error("No MED-PC sessions were found in the uploaded files.")
    st.stop()

detected = pd.concat(all_sessions, ignore_index=True).sort_values("session_id")
st.success(f"Detected {len(detected)} sessions across {len(uploads)} file(s).")
st.write("Detected procedures:", ", ".join(sorted(detected["procedure"].unique())))

st.subheader("Assign animal IDs")
editor = detected.copy()
editor["subject"] = editor.apply(
    lambda row: row["medpc_subject"] if str(row.get("medpc_subject", "")).strip() else f"Box{int(row['box'])}",
    axis=1,
)
editor = st.data_editor(
    editor[["session_id", "date", "box", "procedure", "subject"]],
    hide_index=True,
    disabled=["session_id", "date", "box", "procedure"],
    width="stretch",
)
subject_map = dict(zip(editor["session_id"], editor["subject"]))
apply_subjects(frames, subject_map)

if not frames["unknown"].empty:
    st.error("Some sessions use an unrecognized procedure. They are listed in the Excel workbook.")

st.subheader("Session summary")
if not frames["sessions"].empty:
    for procedure in sorted(frames["sessions"]["procedure"].unique()):
        ids = frames["sessions"].loc[
            frames["sessions"]["procedure"].eq(procedure), "session_id"
        ].tolist()
        concise_summary = frames_for_sessions(frames, ids)["sessions"]
        st.markdown(f"**{procedure}**")
        st.dataframe(concise_summary, hide_index=True, width="stretch")
else:
    st.info("No behavioral sessions were detected; only hardware-test records were found.")

if not frames["pit_trials"].empty:
    st.subheader("PIT transfer trial preview")
    st.dataframe(frames["pit_trials"], hide_index=True, width="stretch")

if not frames["drl_active"].empty:
    st.subheader("DRL-20 IRT distribution")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=frames["drl_active"], x="irt_s", hue="outcome", binwidth=2, multiple="stack", ax=ax)
    ax.axvline(20, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Inter-response time (seconds)")
    st.pyplot(fig)
    plt.close(fig)

with st.expander("View all output tables"):
    for key, frame in frames.items():
        if not frame.empty:
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            st.dataframe(frame, hide_index=True, width="stretch")

st.subheader("Download separate Excel files")
st.write("Each detected MED-PC program is exported to its own Excel workbook.")

procedure_workbooks = {}
for procedure in sorted(detected["procedure"].dropna().unique()):
    ids = detected.loc[detected["procedure"].eq(procedure), "session_id"].tolist()
    procedure_frames = frames_for_sessions(frames, ids)
    workbook_name = f"{safe_filename(procedure)}_analysis.xlsx"
    procedure_workbooks[workbook_name] = excel_bytes(procedure_frames)

for workbook_name, workbook_data in procedure_workbooks.items():
    label = Path(workbook_name).stem.replace("_analysis", "").replace("_", " ")
    st.download_button(
        label=f"Download {label}",
        data=workbook_data,
        file_name=workbook_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{workbook_name}",
    )

if len(procedure_workbooks) > 1:
    zip_output = io.BytesIO()
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for workbook_name, workbook_data in procedure_workbooks.items():
            archive.writestr(workbook_name, workbook_data)
    zip_output.seek(0)

    st.download_button(
        label="Download all program files as ZIP",
        data=zip_output.getvalue(),
        file_name="MED-PC_separate_program_files.zip",
        mime="application/zip",
        type="primary",
        key="download_all_programs_zip",
    )
