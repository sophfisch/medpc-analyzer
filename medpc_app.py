import io
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


APP_VERSION = "2026-09-24.3"


st.set_page_config(page_title="MED-PC Data Converter", page_icon="🐀", layout="wide")
st.title("MED-PC Behavioral Data Converter")
st.caption(f"App version: {APP_VERSION}")
st.write(
    "Upload one or more daily MED-PC text files. The app detects DRL-20, "
    "PIT instrumental training, Pavlovian conditioning, and PIT transfer tests."
)
with st.sidebar:
    st.image(
        "assets/lab_logo.png",
        width="stretch",
    )

    st.markdown("## How to use")

    st.markdown(
        """
        1. Upload MED-PC text files.
        2. Enter the animal IDs.
        3. Review the detected sessions.
        4. Download the Excel workbooks.
        """
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

    row = {
        **base,
        "family": "DRL-20",
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
            if procedure == "DRL_20_HFD":
                parse_drl(block, base, tables)
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
                "DRL_training_FR1",
                "DRL_5_HFD",
                "DRL_10_HFD",
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
            if procedure == "DRL_20_HFD":
                relevant = common + [
                    "active_presses", "successful_presses", "failed_presses",
                    "inactive_presses", "reinforcers", "efficiency",
                    "reinforced_head_entries", "nonreinforced_head_entries",
                ]
            elif procedure.startswith("PIT_Instrumental_"):
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
