import time
from pathlib import Path
from datetime import datetime

import cv2
import streamlit as st

from config import (
    APP_ICON,
    APP_TITLE,
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_PERSON_CONFIDENCE,
    DEFAULT_ROI_EXPANSION,
    DEFAULT_SAMPLE_EVERY_N,
    MAX_UPLOAD_MB,
    PERSON_MODEL_PATH,
    WEAPON_MODEL_PATH,
)
from pipeline import (
    analyze_frame,
    analyze_image,
    analyze_weapon_only_frame,
    get_video_metadata,
    process_video,
    _draw_weapon_only_result,
)
from inference import YOLODetector
from tracking import PersonTracker
from temporal_analysis import TemporalAnalyzer
from events import EventManager
from cctv import (
    CCTVIncidentManager,
    CCTVProcessor,
    normalize_cctv_source,
    validate_cctv_source,
)
from ui import (
    render_header,
    render_sidebar,
    render_image_results,
    render_video_results,
    render_detection_summary,
)


CCTV_LIBRARY_DIR = Path(__file__).resolve().parent / "incidents"
CCTV_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)


def _safe_cctv_label(label: str) -> str:
    return "".join(
        ch if ch.isalnum() else "_"
        for ch in str(label).lower()
    ).strip("_") or "weapon"


def _next_incident_number() -> int:
    existing = list(CCTV_LIBRARY_DIR.rglob("incident_*.jpg"))
    return len(existing) + 1


def _save_cctv_incident(
    incident,
    incident_number: int,
    session_id: str,
) -> Path | None:
    """Persist the peak-confidence evidence frame for one incident."""
    if not incident.snapshot:
        return None

    day_dir = CCTV_LIBRARY_DIR / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    safe_label = _safe_cctv_label(incident.label)
    path = day_dir / (
        f"incident_{session_id}_{incident_number:04d}_"
        f"{safe_label}_{incident.peak_confidence:.0%}.jpg"
    )

    try:
        path.write_bytes(incident.snapshot)
    except OSError:
        return None

    return path


def _incident_key(incident, session_id: str) -> tuple:
    return (
        session_id,
        round(float(incident.start_time), 3),
        round(float(incident.peak_time), 3),
        int(incident.frame_index),
    )


def _persist_new_cctv_incidents(manager):
    """Save closed incidents from the current CCTV session."""
    session_id = st.session_state["cctv_session_id"]
    persisted = st.session_state.setdefault("cctv_persisted_ids", set())
    items = st.session_state.setdefault("cctv_library_items", [])

    for incident in manager.incidents:
        key = _incident_key(incident, session_id)

        if key in persisted:
            continue

        number = _next_incident_number()
        path = _save_cctv_incident(
            incident,
            number,
            session_id,
        )

        if path is not None:
            item = {
                "id": key,
                "path": str(path),
                "label": incident.label,
                "confidence": float(incident.peak_confidence),
                "start": float(incident.start_time),
                "end": float(incident.end_time),
                "peak_time": float(incident.peak_time),
                "frames": int(incident.detection_count),
                "source": str(
                    st.session_state.get(
                        "cctv_source",
                        "",
                    )
                ),
            }
            items.append(item)

        persisted.add(key)


def _load_saved_cctv_library():
    """Load evidence images created by previous monitoring sessions."""
    items = []

    for path in sorted(
        CCTV_LIBRARY_DIR.rglob("incident_*.jpg"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        items.append(
            {
                "id": str(path),
                "path": str(path),
                "label": path.stem,
                "confidence": None,
                "start": None,
                "end": None,
                "peak_time": None,
                "frames": None,
                "source": None,
            }
        )

    return items



def _clear_cctv_current_session():
    """Clear only the current/last CCTV session from the UI."""
    processor = st.session_state.get("cctv_processor")
    if processor is not None:
        processor.release()

    for key in (
        "cctv_processor",
        "cctv_weapon_detector",
        "cctv_person_detector",
        "cctv_incidents",
        "cctv_frame_index",
        "cctv_processed",
        "cctv_weapon_hits",
        "cctv_last_weapon",
        "cctv_last_frame",
        "cctv_selected_incident",
        "cctv_start_time",
        "cctv_tracker",
        "cctv_temporal",
        "cctv_events",
        "cctv_persisted_ids",
        "cctv_session_id",
        "cctv_last_session",
        "cctv_library_items",
    ):
        st.session_state.pop(key, None)

    st.session_state["cctv_running"] = False


def _clear_cctv_saved_evidence():
    """Delete saved CCTV evidence images and clear the visible library."""
    processor = st.session_state.get("cctv_processor")
    if processor is not None:
        processor.release()

    if CCTV_LIBRARY_DIR.exists():
        for path in CCTV_LIBRARY_DIR.rglob("incident_*.jpg"):
            try:
                path.unlink()
            except OSError:
                pass

    _clear_cctv_current_session()
    st.session_state["cctv_library_items"] = []


def _render_incident_cards(items, title="Incident Library"):
    st.markdown(f"#### {title}")

    if not items:
        st.caption(
            "Captured weapon incidents will appear here with their "
            "peak-confidence evidence frame."
        )
        return

    st.caption(f"{len(items)} incident(s) captured.")

    columns = st.columns(4)

    for index, item in enumerate(items):
        with columns[index % 4]:
            path = Path(item["path"])

            if path.exists():
                st.image(
                    str(path),
                    use_container_width=True,
                )

            label = item.get("label") or "Saved incident"
            confidence = item.get("confidence")
            start = item.get("start")
            end = item.get("end")
            frames = item.get("frames")

            if confidence is not None:
                st.markdown(
                    f"**{label}**  \n"
                    f"{confidence:.1%} peak  \n"
                    f"{start:.2f}s – {end:.2f}s"
                )
                if frames is not None:
                    st.caption(f"{frames} detection frame(s)")
            else:
                st.markdown(f"**Saved incident**  \n{path.stem}")

            with st.expander("Evidence details"):
                st.code(str(path), language=None)


def _archive_cctv_session():
    """Keep the finished live session visible after the camera is stopped."""
    manager = st.session_state.get("cctv_incidents")

    if manager is not None:
        manager.flush_active()
        _persist_new_cctv_incidents(manager)
        incidents = manager.all_incidents(include_active=False)
    else:
        incidents = []

    st.session_state["cctv_last_session"] = {
        "source": st.session_state.get("cctv_source", ""),
        "duration": max(
            0.0,
            time.monotonic()
            - st.session_state.get(
                "cctv_start_time",
                time.monotonic(),
            ),
        ),
        "processed": int(
            st.session_state.get(
                "cctv_processed",
                0,
            )
        ),
        "weapon_hits": int(
            st.session_state.get(
                "cctv_weapon_hits",
                0,
            )
        ),
        "last_frame": st.session_state.get("cctv_last_frame"),
        "incidents": list(incidents),
    }


def _render_last_cctv_session():
    session = st.session_state.get("cctv_last_session")

    if not session:
        return

    st.markdown("#### Last monitoring session")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames processed", session["processed"])
    c2.metric("Detection frames", session["weapon_hits"])
    c3.metric("Incidents", len(session["incidents"]))
    c4.metric("Duration", f"{session['duration']:.1f}s")

    if session["last_frame"] is not None:
        st.image(
            session["last_frame"],
            caption="Last analyzed frame",
            use_container_width=True,
        )


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header()

settings = render_sidebar(
    confidence=DEFAULT_CONFIDENCE,
    person_confidence=DEFAULT_PERSON_CONFIDENCE,
    roi_expansion=DEFAULT_ROI_EXPANSION,
    max_upload_mb=MAX_UPLOAD_MB,
    max_video_frames=DEFAULT_MAX_VIDEO_FRAMES,
    sample_every_n=max(DEFAULT_SAMPLE_EVERY_N, 10),
)

input_mode = st.radio(
    "Input source",
    ["Image", "Video", "Live CCTV"],
    horizontal=True,
)

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

if input_mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image for weapon analysis.",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        st.stop()

    if (
        settings["max_upload_bytes"] is not None
        and uploaded_file.size > settings["max_upload_bytes"]
    ):
        st.error(
            f"File exceeds the {settings['max_upload_mb']} MB upload limit."
        )
        st.stop()

    if st.button("Analyze image", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing image..."):
                result = analyze_image(
                    image_bytes=uploaded_file.getvalue(),
                    confidence_threshold=settings["confidence"],
                    person_confidence=settings["person_confidence"],
                    top_k=2,
                    roi_expansion=settings["roi_expansion"],
                    person_model_path=PERSON_MODEL_PATH,
                    weapon_model_path=WEAPON_MODEL_PATH,
                    advanced_person_analysis=settings["advanced_person_analysis"],
                )

            render_image_results(result)

        except Exception as e:
            st.error(f"Image analysis failed: {e}")

# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

elif input_mode == "Video":
    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload video for sampled weapon analysis.",
    )

    if uploaded_file is None:
        st.info("Upload a video to begin.")
        st.stop()

    if (
        settings["max_upload_bytes"] is not None
        and uploaded_file.size > settings["max_upload_bytes"]
    ):
        st.error(
            f"File exceeds the {settings['max_upload_mb']} MB upload limit."
        )
        st.stop()

    file_bytes = uploaded_file.getvalue()

    try:
        metadata = get_video_metadata(file_bytes)
    except Exception as e:
        st.error(f"Could not read video metadata: {e}")
        st.stop()

    st.markdown("### Video overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resolution", metadata["resolution"])
    c2.metric("FPS", f"{metadata['fps']:.1f}")
    c3.metric("Frames", metadata["frame_count"])
    c4.metric("Duration", f"{metadata['duration']:.1f}s")

    mode_text = (
        "Advanced person-aware analysis"
        if settings["advanced_person_analysis"]
        else "Weapon-only analysis"
    )

    st.caption(
        f"{mode_text} • every {settings['sample_every_n']} frame(s) • "
        f"maximum {settings['max_video_frames']} analyzed frames"
    )

    video_key = (
        getattr(uploaded_file, "name", ""),
        int(uploaded_file.size),
    )

    if st.session_state.get("wd_video_key") != video_key:
        st.session_state["wd_video_key"] = video_key
        st.session_state.pop("wd_video_result", None)
        st.session_state.pop("wd_video_bytes", None)
        st.session_state["wd_selected_group"] = 0

    if st.button("Analyze video", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing video..."):
                result = process_video(
                    video_bytes=file_bytes,
                    confidence_threshold=settings["confidence"],
                    person_confidence=settings["person_confidence"],
                    top_k=2,
                    roi_expansion=settings["roi_expansion"],
                    person_model_path=PERSON_MODEL_PATH,
                    weapon_model_path=WEAPON_MODEL_PATH,
                    sample_every_n=settings["sample_every_n"],
                    max_frames=settings["max_video_frames"],
                    advanced_person_analysis=settings["advanced_person_analysis"],
                )

            st.session_state["wd_video_result"] = result
            st.session_state["wd_video_bytes"] = file_bytes
            st.session_state["wd_selected_group"] = 0

        except Exception as e:
            st.error(f"Video analysis failed: {e}")

    saved_result = st.session_state.get("wd_video_result")
    saved_bytes = st.session_state.get("wd_video_bytes")

    if saved_result is not None and saved_bytes:
        render_video_results(
            saved_result,
            video_bytes=saved_bytes,
        )

# ---------------------------------------------------------------------------
# Live CCTV
# ---------------------------------------------------------------------------

else:
    st.markdown("### Live CCTV")

    if "cctv_library_items" not in st.session_state:
        st.session_state["cctv_library_items"] = []

    if "cctv_running" not in st.session_state:
        st.session_state["cctv_running"] = False

    st.caption(
        "Weapon detection is the default. Person-aware analysis is optional "
        "for difficult scenes."
    )

    source_text = st.text_input(
        "Camera / stream source",
        value=str(
            st.session_state.get(
                "cctv_source",
                0,
            )
        ),
        help=(
            "Use a webcam index such as 0, or an RTSP/HTTP/RTMP stream URL."
        ),
    )

    normalized_source = normalize_cctv_source(source_text)
    valid, message = validate_cctv_source(normalized_source)

    if not valid:
        st.warning(message)
        st.stop()

    running = bool(st.session_state.get("cctv_running", False))

    # ---------------------------------------------------------------
    # Idle state
    # ---------------------------------------------------------------
    if not running:
        clear_left, clear_right = st.columns(2)

        with clear_left:
            if st.button(
                "Clear current session",
                use_container_width=True,
                help="Remove the previous CCTV session from the interface. Saved evidence stays on disk.",
            ):
                _clear_cctv_current_session()
                st.rerun()

        with clear_right:
            if st.button(
                "Clear saved evidence",
                use_container_width=True,
                help="Permanently remove incident images saved in the incidents folder.",
            ):
                _clear_cctv_saved_evidence()
                st.rerun()

        if st.button(
            "Start monitoring",
            type="primary",
            use_container_width=True,
        ):
            cctv = CCTVProcessor(normalized_source)

            if not cctv.connect():
                st.error("Unable to connect to the CCTV source.")
                cctv.release()
                st.stop()

            session_id = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            st.session_state["cctv_running"] = True
            st.session_state["cctv_source"] = normalized_source
            st.session_state["cctv_processor"] = cctv
            st.session_state["cctv_weapon_detector"] = YOLODetector(
                WEAPON_MODEL_PATH
            )
            st.session_state["cctv_person_detector"] = (
                YOLODetector(PERSON_MODEL_PATH)
                if settings["advanced_person_analysis"]
                else None
            )
            st.session_state["cctv_incidents"] = CCTVIncidentManager()
            st.session_state["cctv_library_items"] = []
            st.session_state.pop("cctv_last_session", None)
            st.session_state["cctv_session_id"] = session_id
            st.session_state["cctv_persisted_ids"] = set()
            st.session_state["cctv_frame_index"] = 0
            st.session_state["cctv_processed"] = 0
            st.session_state["cctv_weapon_hits"] = 0
            st.session_state["cctv_last_weapon"] = {}
            st.session_state["cctv_last_frame"] = None
            st.session_state["cctv_selected_incident"] = None
            st.session_state["cctv_start_time"] = time.monotonic()

            if settings["advanced_person_analysis"]:
                st.session_state["cctv_tracker"] = PersonTracker()
                st.session_state["cctv_temporal"] = TemporalAnalyzer()
                st.session_state["cctv_events"] = EventManager()

            st.rerun()

        _render_last_cctv_session()

        _render_incident_cards(
            st.session_state["cctv_library_items"],
            title="Incident Library",
        )

        st.info(
            "Enter a camera index or stream URL, then start monitoring."
        )
        st.stop()

    # ---------------------------------------------------------------
    # Stop
    # ---------------------------------------------------------------
    if st.button(
        "Stop monitoring",
        type="secondary",
        use_container_width=True,
    ):
        processor = st.session_state.get("cctv_processor")

        _archive_cctv_session()

        if processor is not None:
            processor.release()

        # Release live resources, but intentionally keep the archived
        # session and persistent incident library in session state.
        for key in (
            "cctv_processor",
            "cctv_weapon_detector",
            "cctv_person_detector",
            "cctv_incidents",
            "cctv_frame_index",
            "cctv_processed",
            "cctv_weapon_hits",
            "cctv_last_weapon",
            "cctv_last_frame",
            "cctv_selected_incident",
            "cctv_start_time",
            "cctv_tracker",
            "cctv_temporal",
            "cctv_events",
            "cctv_persisted_ids",
            "cctv_session_id",
        ):
            st.session_state.pop(key, None)

        st.session_state["cctv_running"] = False
        st.rerun()

    processor = st.session_state.get("cctv_processor")
    weapon_detector = st.session_state.get("cctv_weapon_detector")
    advanced = bool(settings["advanced_person_analysis"])

    if (
        processor is None
        or weapon_detector is None
        or not processor.is_connected()
    ):
        st.session_state["cctv_running"] = False
        st.error("The CCTV connection is no longer available.")
        st.stop()

    source_started = st.session_state.get(
        "cctv_start_time",
        None,
    )

    if source_started is None:
        source_started = time.monotonic()
        st.session_state["cctv_start_time"] = source_started

    # Latest-frame capture keeps the monitor close to the live camera.
    frame_ok, frame = processor.read_latest(drain_reads=6)

    if not frame_ok or frame is None:
        st.warning("No frame received. Attempting to reconnect...")

        if processor.reconnect(attempts=1):
            st.rerun()

        st.error("The CCTV stream could not be read.")
        _archive_cctv_session()
        st.session_state["cctv_running"] = False
        processor.release()
        st.stop()

    frame_index = int(
        st.session_state.get(
            "cctv_frame_index",
            0,
        )
    )
    elapsed = time.monotonic() - source_started

    # ---------------------------------------------------------------
    # Default: weapon-only detection
    # ---------------------------------------------------------------
    if not advanced:
        weapon = analyze_weapon_only_frame(
            frame,
            weapon_detector,
            confidence_threshold=settings["confidence"],
            top_k=2,
        )
        annotated = _draw_weapon_only_result(
            frame,
            weapon,
        )

        if weapon.get("present"):
            manager = st.session_state["cctv_incidents"]

            manager.update(
                timestamp=elapsed,
                frame_index=frame_index,
                weapon=weapon,
                frame=annotated,
            )

            st.session_state["cctv_weapon_hits"] += 1

        # A newly completed incident is persisted as soon as a gap appears.
        _persist_new_cctv_incidents(
            st.session_state["cctv_incidents"]
        )

        st.session_state["cctv_last_weapon"] = weapon
        st.session_state["cctv_last_frame"] = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB,
        )

    # ---------------------------------------------------------------
    # Advanced: person-aware analysis
    # ---------------------------------------------------------------
    else:
        if "cctv_tracker" not in st.session_state:
            st.session_state["cctv_tracker"] = PersonTracker()
            st.session_state["cctv_temporal"] = TemporalAnalyzer()
            st.session_state["cctv_events"] = EventManager()

        person_detector = st.session_state["cctv_person_detector"]

        people, annotated = analyze_frame(
            frame,
            person_detector,
            weapon_detector,
            settings["confidence"],
            settings["person_confidence"],
            2,
            settings["roi_expansion"],
        )

        tracker = st.session_state["cctv_tracker"]
        temporal = st.session_state["cctv_temporal"]
        events = st.session_state["cctv_events"]

        people = tracker.update(people)

        for track_id in tracker.consume_expired_tracks():
            temporal.remove_track(track_id)
            events.close_track(track_id)

        current_best = {}

        for person in people:
            temporal_result = temporal.update(
                person["track_id"],
                person.get("weapon", {}),
            )
            person["temporal"] = temporal_result

            event = events.create_event(
                temporal_result,
                timestamp=elapsed,
                frame_index=frame_index,
            )

            if event is not None:
                person["event_id"] = event.event_id

            weapon = person.get("weapon", {})

            if weapon.get("present") and (
                not current_best
                or weapon.get("confidence", 0.0)
                > current_best.get("confidence", 0.0)
            ):
                current_best = weapon

        from pipeline import _draw_results
        annotated = _draw_results(
            frame,
            people,
        )

        st.session_state["cctv_last_weapon"] = current_best
        st.session_state["cctv_last_frame"] = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB,
        )

        if current_best.get("present"):
            st.session_state["cctv_weapon_hits"] += 1

    st.session_state["cctv_processed"] += 1
    st.session_state["cctv_frame_index"] = frame_index + 1

    last_weapon = st.session_state.get(
        "cctv_last_weapon",
        {},
    )

    # ---------------------------------------------------------------
    # Live monitor
    # ---------------------------------------------------------------
    left, right = st.columns(
        [1.55, 1],
        gap="large",
    )

    with left:
        st.markdown("#### Live feed")

        st.image(
            st.session_state["cctv_last_frame"],
            caption=(
                f"Live analysis • {elapsed:.1f}s • "
                f"Frame {frame_index}"
            ),
            use_container_width=True,
        )

    with right:
        st.markdown("#### Detection status")

        if last_weapon.get("present"):
            confidence = float(
                last_weapon.get(
                    "confidence",
                    0.0,
                )
            )
            top = last_weapon.get(
                "top_classes",
                [],
            )

            label = (
                str(
                    top[0].get(
                        "class",
                        "Weapon",
                    )
                )
                .replace("_", " ")
                .title()
                if top
                else "Weapon"
            )

            st.markdown(
                f"""
                <div class="wd-status wd-alert">
                    ⚠ WEAPON DETECTED
                </div>
                <div class="wd-card">
                    <div style="font-size:2.1rem;font-weight:900">
                        {confidence:.1%}
                    </div>
                    <div class="wd-note">Detector confidence</div>
                    <strong>{label}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="wd-status wd-safe">✓ NO WEAPON DETECTED</div>',
                unsafe_allow_html=True,
            )

        st.metric(
            "Frames processed",
            st.session_state["cctv_processed"],
        )

        st.metric(
            "Detection frames",
            st.session_state["cctv_weapon_hits"],
        )

    # ---------------------------------------------------------------
    # Incident review
    # ---------------------------------------------------------------
    manager = st.session_state.get("cctv_incidents")

    if manager is not None:
        incidents = list(
            reversed(
                manager.all_incidents(
                    include_active=True
                )
            )
        )

        st.markdown("#### Current session incidents")

        if not incidents:
            st.caption("No weapon incidents recorded yet.")
        else:
            columns = st.columns(3)

            for index, incident in enumerate(incidents):
                with columns[index % 3]:
                    title = (
                        f"{incident.time_range}  •  "
                        f"{incident.label}"
                    )
                    st.markdown(
                        f"**{title}**  \n"
                        f"Peak {incident.peak_confidence:.1%}  •  "
                        f"{incident.detection_count} frame(s)"
                    )

                    if incident.snapshot:
                        st.image(
                            incident.snapshot,
                            use_container_width=True,
                        )

    _render_incident_cards(
        list(
            reversed(
                st.session_state.get(
                    "cctv_library_items",
                    [],
                )
            )
        ),
        title="Incident Library • Current Session",
    )

    st.caption(
        f"Source: {st.session_state.get('cctv_source')} • "
        f"{'Advanced person-aware analysis' if advanced else 'Weapon-only analysis'}"
    )

    # One rerun = one fresh live inference cycle.
    st.rerun()
