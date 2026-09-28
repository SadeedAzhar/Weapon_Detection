import base64
import html

import streamlit as st

from config import PERSON_MODEL_PATH, WEAPON_MODEL_PATH


def inject_css():
    st.markdown(
        """
        <style>
        :root{
            --bg:#eef2f7;
            --surface:#ffffff;
            --soft:#f7f9fc;
            --border:#d9e1eb;
            --text:#142033;
            --muted:#657286;
            --navy:#0d1728;
            --navy2:#17253b;
            --accent:#e0313f;
            --accent-soft:#fff0f2;
            --green:#087443;
            --green-soft:#eafaf2;
            --shadow:0 12px 32px rgba(15,23,42,.07)
        }

        .stApp{background:var(--bg)}
        .block-container{max-width:1240px;padding-top:2rem;padding-bottom:4rem}
        [data-testid="stHeader"]{background:rgba(238,242,247,.94)}

        /* Sidebar */
        [data-testid="stSidebar"]{background:var(--navy)!important;border-right:1px solid #233047!important}
        [data-testid="stSidebar"]>div:first-child{background:var(--navy)!important}
        [data-testid="stSidebar"] *{color:#f4f7fb}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] small{color:#dfe6ef!important}
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3{color:#ffffff!important}

        /* Advanced analysis is intentionally dark, with high-contrast text. */
        [data-testid="stSidebar"] [data-testid="stExpander"]{
            background:#ffffff!important;
            border:1px solid #d7dee8!important;
            border-radius:13px!important;
            overflow:hidden
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary *,
        [data-testid="stSidebar"] [data-testid="stExpander"] button,
        [data-testid="stSidebar"] [data-testid="stExpander"] button *,
        [data-testid="stSidebar"] [data-testid="stExpander"] label,
        [data-testid="stSidebar"] [data-testid="stExpander"] p,
        [data-testid="stSidebar"] [data-testid="stExpander"] span,
        [data-testid="stSidebar"] [data-testid="stExpander"] div{
            color:#142033!important
        }
        [data-testid="stSidebar"] [data-testid="stExpander"]>div{background:#ffffff!important}
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="switch"]{background:#d5dbe4!important}
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="switch"] > div{background:#ffffff!important}

        /* Header */
        .wd-header{display:flex;align-items:center;gap:16px;margin-bottom:28px}
        .wd-mark{
            width:54px;height:54px;border-radius:17px;background:linear-gradient(145deg,#0b1424,#1b2c46);
            display:flex;align-items:center;justify-content:center;color:#fff;
            font-weight:950;letter-spacing:.05em;box-shadow:0 9px 23px rgba(13,23,40,.17)
        }
        .wd-kicker{color:var(--accent);font-size:.68rem;font-weight:900;letter-spacing:.16em;text-transform:uppercase}
        .wd-title{color:var(--text);font-size:2.45rem;font-weight:900;letter-spacing:-.045em;line-height:1.02}
        .wd-subtitle{color:var(--muted);font-size:.94rem;margin-top:7px}

        /* Common cards */
        .wd-card{
            background:var(--surface);border:1px solid var(--border);border-radius:18px;
            padding:20px;box-shadow:var(--shadow);margin-bottom:18px
        }
        .wd-section{color:var(--text);font-size:1.08rem;font-weight:900;margin-bottom:4px}
        .wd-note,.wd-event-help{color:var(--muted);font-size:.81rem;margin-bottom:12px}

        /* Main result */
        .wd-hero{
            background:linear-gradient(135deg,#0c1728,#182b45);border-radius:20px;
            padding:25px 27px;color:#fff;box-shadow:0 16px 36px rgba(13,23,40,.15);margin-bottom:18px
        }
        .wd-hero-kicker{color:#a9b7c9;font-size:.67rem;font-weight:900;letter-spacing:.14em;text-transform:uppercase}
        .wd-hero-status{font-size:1.02rem;font-weight:900;margin-top:11px}
        .wd-hero-value{font-size:3.35rem;line-height:1;font-weight:950;margin-top:8px}
        .wd-hero-label{color:#aab8c9;font-size:.67rem;letter-spacing:.14em;margin-top:6px}
        .wd-class-pill{
            display:inline-block;background:#263b58;color:#f6f8fb;border:1px solid #3e5573;
            border-radius:999px;padding:7px 11px;font-size:.76rem;font-weight:800;margin-top:13px
        }
        .wd-status{border:1px solid;border-radius:14px;padding:12px 15px;font-weight:900;margin-bottom:14px}
        .wd-safe{background:var(--green-soft);border-color:#b6e6ca;color:var(--green)}
        .wd-alert{background:var(--accent-soft);border-color:#fecdd3;color:#be123c}

        /* Top-2 */
        .wd-pred{
            background:var(--soft);border:1px solid var(--border);border-radius:13px;
            padding:12px 14px;margin:8px 0
        }
        .wd-pred-name{color:var(--text);font-weight:850}
        .wd-pred-conf{float:right;color:#b42318;font-weight:900}
        .wd-score-track{height:6px;border-radius:999px;background:#e4eaf1;overflow:hidden;margin-top:8px}
        .wd-score-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#e0313f,#f26770)}

        /* Video */
        .wd-video{
            background:#0a1321;border-radius:17px;padding:9px;box-shadow:var(--shadow);margin-bottom:16px
        }
        .wd-video video{
            width:100%;display:block;border-radius:10px;background:#04080e;max-height:560px
        }
        .wd-video-hint{
            color:#9eafc2;font-size:11px;margin:10px 3px 6px
        }
        .wd-range-title{
            color:#dfe7f1;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;margin:7px 3px 6px
        }
        .wd-shortcut{
            display:flex;gap:7px;flex-wrap:wrap;max-height:92px;overflow-y:auto;padding:1px
        }
        .wd-shortcut button{
            border:1px solid #334660;background:#15243a;color:#eff4fa;border-radius:10px;
            padding:8px 10px;font-size:11px;cursor:pointer
        }
        .wd-shortcut button:hover{background:#203551;border-color:#637895}
        .wd-exact{
            display:flex;gap:6px;flex-wrap:wrap;max-height:64px;overflow-y:auto;padding-top:3px
        }
        .wd-exact button{
            border:1px solid #2d3d55;background:#0f1d30;color:#cdd8e5;border-radius:999px;
            padding:5px 8px;font-size:10px;cursor:pointer
        }
        .wd-exact button:hover{background:#1c304a}

        /* Compact detection cards */
        .wd-detection{
            border:1px solid var(--border);background:var(--surface);border-radius:12px;
            padding:9px 11px;margin:6px 0;box-shadow:0 3px 10px rgba(15,23,42,.035)
        }
        .wd-detection-title{color:var(--text);font-weight:850;font-size:.83rem}
        .wd-detection-meta{color:var(--muted);font-size:.72rem;margin-top:2px}

        div[data-testid="stMetric"]{
            background:var(--surface);border:1px solid var(--border);border-radius:14px;
            padding:13px 15px;box-shadow:0 4px 15px rgba(16,24,40,.04)
        }
        div[data-testid="stMetricLabel"]{color:var(--muted)}
        div[data-testid="stMetricValue"]{color:var(--text)}
        .stButton>button{border-radius:11px!important;font-weight:850!important;min-height:42px}
        .stButton>button[kind="primary"]{background:var(--navy)!important;border-color:var(--navy)!important}

        div[data-testid="stRadio"] label{
            color:var(--text)!important;font-weight:700!important
        }

        div[data-testid="stExpander"]{
            border-color:var(--border);border-radius:14px;background:var(--surface)
        }

        @media(max-width:850px){
            .wd-title{font-size:1.9rem}
            .wd-hero-value{font-size:2.7rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    inject_css()
    st.markdown(
        """
        <div class="wd-header">
            <div class="wd-mark">WD</div>
            <div>
                <div class="wd-kicker">Intelligent Vision</div>
                <div class="wd-title">Weapon Detection</div>
                <div class="wd-subtitle">
                    Weapon-first analysis for images, video and live CCTV.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    confidence,
    person_confidence,
    roi_expansion,
    max_upload_mb,
    max_video_frames,
    sample_every_n=10,
):
    st.sidebar.markdown("## Analysis")
    st.sidebar.caption("Primary weapon-detection controls")

    confidence = st.sidebar.slider(
        "Weapon confidence",
        0.05,
        0.95,
        float(confidence),
        0.05,
        help="Minimum confidence required for a weapon detection.",
    )

    sample_every_n = st.sidebar.slider(
        "Video sampling",
        5,
        20,
        max(5, int(sample_every_n)),
        1,
        help="Analyze every Nth frame. Higher values reduce processing time.",
    )

    with st.sidebar.expander("Advanced analysis"):
        advanced = st.toggle(
            "Person-aware analysis",
            value=False,
            help=(
                "Off: V3 analyzes the full frame directly. "
                "On: detect people first and analyze their regions."
            ),
        )

        if advanced:
            person_confidence = st.slider(
                "Person confidence",
                0.05,
                0.95,
                float(person_confidence),
                0.05,
            )

            roi_expansion = st.slider(
                "ROI expansion",
                0.00,
                0.60,
                float(roi_expansion),
                0.05,
            )

            st.caption(
                "Use this only for difficult scenes where the subject is very small."
            )

    return {
        "confidence": confidence,
        "advanced_person_analysis": advanced,
        "person_confidence": person_confidence,
        "roi_expansion": float(roi_expansion),
        "top_k": 2,
        "sample_every_n": int(sample_every_n),
        "person_model_path": str(PERSON_MODEL_PATH),
        "weapon_model_path": str(WEAPON_MODEL_PATH),
        "max_upload_mb": max_upload_mb,
        "max_upload_bytes": (
            max_upload_mb * 1024 * 1024
            if max_upload_mb is not None
            else None
        ),
        "max_video_frames": int(max_video_frames),
    }


def _best_weapon(result):
    best = None
    for weapon in getattr(result, "weapon_results", []) or []:
        if weapon and weapon.get("present"):
            if best is None or float(weapon.get("confidence", 0.0)) > float(
                best.get("confidence", 0.0)
            ):
                best = weapon
    return best


def _weapon_label(weapon):
    top = (weapon or {}).get("top_classes", []) or []
    if top:
        return str(top[0].get("class", "Weapon")).replace("_", " ").title()
    return "Weapon"


def _render_top_two(top_classes):
    top_classes = (top_classes or [])[:2]

    if not top_classes:
        st.caption("No class evidence available.")
        return

    st.markdown("**Top 2 class scores**")
    st.caption(
        "Detector scores are shown as model scores, not calibrated probabilities."
    )

    for index, prediction in enumerate(top_classes, 1):
        name = (
            str(prediction.get("class", "unknown"))
            .replace("_", " ")
            .title()
        )
        confidence = float(prediction.get("confidence", 0.0))
        width = max(0.0, min(confidence, 1.0)) * 100

        st.markdown(
            f"""
            <div class="wd-pred">
                <span class="wd-pred-name">{index}. {html.escape(name)}</span>
                <span class="wd-pred-conf">{confidence:.1%}</span>
                <div class="wd-score-track">
                    <div class="wd-score-fill" style="width:{width:.1f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_weapon_assessment(weapon):
    if not weapon or not weapon.get("present"):
        st.markdown(
            '<div class="wd-status wd-safe">✓ NO WEAPON DETECTED</div>',
            unsafe_allow_html=True,
        )
        return

    confidence = float(weapon.get("confidence", 0.0))
    best_class = _weapon_label(weapon)

    st.markdown(
        f"""
        <div class="wd-hero">
            <div class="wd-hero-kicker">Weapon detection</div>
            <div class="wd-hero-status">⚠ WEAPON DETECTED</div>
            <div class="wd-hero-value">{confidence:.1%}</div>
            <div class="wd-hero-label">DETECTOR CONFIDENCE</div>
            <div class="wd-class-pill">
                Primary class: {html.escape(best_class)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_top_two(weapon.get("top_classes", []))


def render_detection_summary(summary, event_summary=None):
    status = summary.get("status", "Ready")
    weapons = summary.get("potential_weapons", 0)
    confirmed = summary.get("confirmed_alerts", 0)

    if event_summary is not None:
        confirmed = event_summary.get("confirmed_alerts", confirmed)

    st.markdown(
        '<div class="wd-section">Detection overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="wd-note">Weapon detection is the primary signal. '
        'Person-aware analysis is optional.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", status)
    c2.metric("Weapon hits", weapons)
    c3.metric("Confirmed events", confirmed)

    resolution = summary.get("resolution")
    fps = summary.get("fps")

    if resolution:
        extra = f" • {float(fps):.1f} FPS" if fps is not None else ""
        st.caption(f"{resolution}{extra}")


def render_image_results(result):
    best = _best_weapon(result)

    _render_weapon_assessment(best)

    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.markdown(
            '<div class="wd-card">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="wd-section">Scene evidence</div>',
            unsafe_allow_html=True,
        )
        st.caption("The analyzed image shows weapon evidence only.")
        st.image(
            result.annotated_image,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="wd-card">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="wd-section">Detection summary</div>',
            unsafe_allow_html=True,
        )

        if best:
            st.metric(
                "Detected confidence",
                f"{float(best.get('confidence', 0.0)):.1%}",
            )
            st.write(f"**Primary class:** {_weapon_label(best)}")
            _render_top_two(best.get("top_classes", []))
        else:
            st.info("No weapon evidence above the selected threshold.")

        if result.persons:
            with st.expander("Advanced person analysis"):
                for index, person in enumerate(result.persons, 1):
                    track_id = (
                        person.track_id
                        if person.track_id is not None
                        else index
                    )
                    st.write(
                        f"Person #{track_id} • confidence {person.confidence:.1%}"
                    )

        st.markdown("</div>", unsafe_allow_html=True)


def _detected_frames(result):
    return [
        (index, frame_result)
        for index, frame_result in enumerate(result.frame_results)
        if frame_result.get("weapon", {}).get("present")
    ]


def _group_detected_frames(result, detected_frames):
    if not detected_frames:
        return []

    timestamps = [float(item[1].get("timestamp", 0.0)) for item in detected_frames]
    if len(timestamps) == 1:
        expected_interval = 0.5
    else:
        intervals = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
        expected_interval = sorted(intervals)[len(intervals) // 2] if intervals else 0.5

    # A small gap is allowed inside one incident; a larger gap starts a new one.
    max_gap = max(expected_interval * 2.5, expected_interval + 0.35, 0.75)
    max_window_duration = 4.0
    groups = []
    current = [detected_frames[0]]

    for item in detected_frames[1:]:
        previous_time = float(current[-1][1].get("timestamp", 0.0))
        current_time = float(item[1].get("timestamp", 0.0))
        window_start = float(current[0][1].get("timestamp", 0.0))
        if (
            current_time - previous_time <= max_gap
            and current_time - window_start <= max_window_duration
        ):
            current.append(item)
        else:
            groups.append(current)
            current = [item]

    groups.append(current)
    return groups


def _group_label(group):
    start_time = float(group[0][1].get("timestamp", 0.0))
    end_time = float(group[-1][1].get("timestamp", start_time))
    best_item = max(
        group,
        key=lambda item: float(item[1].get("weapon", {}).get("confidence", 0.0)),
    )
    best_weapon = best_item[1].get("weapon", {})
    label = _weapon_label(best_weapon)
    confidence = float(best_weapon.get("confidence", 0.0))
    return start_time, end_time, label, confidence, best_item


def _render_video_player(video_bytes, detection_groups, detected_frames, selected_group=0):
    if not video_bytes:
        return

    encoded = base64.b64encode(video_bytes).decode("ascii")
    range_buttons = []
    for group_index, group in enumerate(detection_groups[:30]):
        start_time, end_time, label, confidence, _ = _group_label(group)
        if end_time > start_time:
            range_text = f"{start_time:.2f}s → {end_time:.2f}s · {label} · peak {confidence:.1%}"
        else:
            range_text = f"{start_time:.2f}s · {label} · {confidence:.1%}"
        range_buttons.append(
            f'<button onclick="jumpTo({start_time:.4f})">{html.escape(range_text)}</button>'
        )

    exact_buttons = []
    for _, frame_result in detected_frames[:80]:
        timestamp = float(frame_result.get("timestamp", 0.0))
        weapon = frame_result.get("weapon", {})
        label = _weapon_label(weapon)
        exact_buttons.append(
            f'<button onclick="jumpTo({timestamp:.4f})">{timestamp:.2f}s · {html.escape(label)}</button>'
        )

    range_html = "".join(range_buttons) or '<span style="color:#9eafc2;font-size:11px">No detection ranges.</span>'
    exact_html = "".join(exact_buttons) or '<span style="color:#9eafc2;font-size:11px">No detected frames.</span>'

    # The Streamlit range cards below the player live outside this HTML
    # component. When one is clicked, Streamlit reruns the page and passes
    # the selected range back here so the player can seek to its start.
    selected_group = min(
        max(int(selected_group), 0),
        max(len(detection_groups) - 1, 0),
    )
    selected_seek = 0.0
    if detection_groups:
        selected_seek = float(
            detection_groups[selected_group][0][1].get(
                "timestamp",
                0.0,
            )
        )

    component = f"""
    <style>
        body{{margin:0;background:transparent;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
        .player{{background:#0a1321;border-radius:17px;padding:9px;box-sizing:border-box}}
        video{{width:100%;display:block;border-radius:10px;background:#04080e;max-height:560px}}
        .hint{{color:#9eafc2;font-size:11px;margin:10px 3px 6px}}
        .title{{color:#dfe7f1;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;margin:7px 3px 6px}}
        .ranges,.exact{{display:flex;gap:7px;flex-wrap:wrap;overflow-y:auto;padding:1px}}
        .ranges{{max-height:92px}}
        .exact{{max-height:62px}}
        button{{border:1px solid #334660;background:#15243a;color:#eff4fa;border-radius:10px;padding:8px 10px;font-size:11px;cursor:pointer}}
        button:hover{{background:#203551;border-color:#637895}}
        .exact button{{border-color:#2d3d55;background:#0f1d30;color:#cdd8e5;border-radius:999px;padding:5px 8px;font-size:10px}}
        .exact button:hover{{background:#1c304a}}
    </style>
    <div class="player">
        <video id="wdVideo" controls preload="metadata">
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
        <div class="hint">Click a detection window or exact timestamp to seek the player. It stays paused after seeking.</div>
        <div class="title">Detection windows</div>
        <div class="ranges">{range_html}</div>
        <div class="title">Detected timestamps</div>
        <div class="exact">{exact_html}</div>
    </div>
    <script>
        function seekPaused(seconds){{
            const video=document.getElementById("wdVideo");
            const seek = () => {{
                video.pause();
                try {{
                    video.currentTime = Math.max(0, Number(seconds) || 0);
                }} catch (e) {{
                    /* Ignore a transient seek error while metadata is loading. */
                }}
                video.pause();
            }};
            if (video.readyState >= 1) seek();
            else video.addEventListener("loadedmetadata", seek, {{once:true}});
        }}

        function jumpTo(seconds){{
            seekPaused(seconds);
        }}

        // A range selected from the Streamlit controls below the player
        // arrives here as the initial seek position for this rerender.
        const initialSeek = {selected_seek:.4f};
        if (initialSeek > 0) {{
            if (document.readyState === "loading")
                document.addEventListener("DOMContentLoaded", () => seekPaused(initialSeek), {{once:true}});
            else
                seekPaused(initialSeek);
        }}
    </script>
    """

    import streamlit.components.v1 as components
    components.html(component,height=640,scrolling=False)


def _render_detection_ranges(detection_groups):
    if not detection_groups:
        return 0

    selected = int(st.session_state.get("wd_selected_group", 0))
    selected = min(max(selected, 0), len(detection_groups) - 1)

    st.markdown(
        '<div class="wd-section">Detection windows</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="wd-event-help">Select a range to move the main video to its start. Playback stays paused.</div>',
        unsafe_allow_html=True,
    )

    # These are real Streamlit controls, not decorative cards. Clicking one
    # updates the selected range and reruns the page; the video player then
    # seeks to the same range on its next render.
    columns_per_row = 4
    for start in range(0, len(detection_groups), columns_per_row):
        row = detection_groups[start:start + columns_per_row]
        cols = st.columns(len(row))

        for offset, group in enumerate(row):
            group_index = start + offset
            start_time, end_time, label, confidence, _ = _group_label(group)
            if end_time > start_time:
                range_text = f"{start_time:.2f}s → {end_time:.2f}s"
            else:
                range_text = f"{start_time:.2f}s"

            with cols[offset]:
                if st.button(
                    range_text,
                    key=f"wd_detection_range_{group_index}",
                    use_container_width=True,
                    type=(
                        "primary"
                        if group_index == selected
                        else "secondary"
                    ),
                    help=(
                        f"{label} · peak {confidence:.1%} · "
                        f"{len(group)} detected frame(s)"
                    ),
                ):
                    st.session_state["wd_selected_group"] = group_index
                    st.rerun()

                duration = max(0.0, end_time - start_time)
                st.markdown(
                    f"""
                    <div class="wd-detection">
                        <div class="wd-detection-title">
                            {html.escape(label)} · peak {confidence:.1%}
                        </div>
                        <div class="wd-detection-meta">
                            {len(group)} frame(s) · span {duration:.2f}s
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    return selected


def _render_detection_gallery(result, detection_groups, selected_group=0):
    if not detection_groups:
        return

    group = detection_groups[min(max(selected_group, 0), len(detection_groups) - 1)]
    preview_items = sorted(
        group,
        key=lambda item: float(item[1].get("weapon", {}).get("confidence", 0.0)),
        reverse=True,
    )[:8]

    st.markdown(
        '<div class="wd-section">Detection frames</div>',
        unsafe_allow_html=True,
    )
    st.caption("Compact evidence from the selected detection window.")

    cols = st.columns(min(4, len(preview_items)))
    for col, (frame_index, frame_result) in zip(cols, preview_items):
        timestamp = float(frame_result.get("timestamp", 0.0))
        weapon = frame_result.get("weapon", {})
        label = _weapon_label(weapon)
        confidence = float(weapon.get("confidence", 0.0))
        with col:
            st.image(
                result.annotated_frames[frame_index],
                caption=f"{timestamp:.2f}s · {label} · {confidence:.1%}",
                width=170,
            )


def _render_all_detection_frames(result, detected_frames):
    if not detected_frames:
        return

    with st.expander(f"All detected frames ({len(detected_frames)})"):
        cols_per_row = 5
        for start in range(0, len(detected_frames), cols_per_row):
            row = detected_frames[start:start + cols_per_row]
            cols = st.columns(len(row))
            for col, (frame_index, frame_result) in zip(cols, row):
                timestamp = float(frame_result.get("timestamp", 0.0))
                weapon = frame_result.get("weapon", {})
                label = _weapon_label(weapon)
                confidence = float(weapon.get("confidence", 0.0))
                with col:
                    st.image(
                        result.annotated_frames[frame_index],
                        caption=f"{timestamp:.2f}s · {label} · {confidence:.1%}",
                        width=150,
                    )


def render_video_results(result, video_bytes=None):
    metadata = result.metadata
    advanced = bool(metadata.get("advanced_person_analysis", False))
    event_summary = metadata.get("event_summary", {})

    if not result.frames:
        st.warning("No frames were processed.")
        return

    detected_frames = _detected_frames(result)
    detection_groups = _group_detected_frames(result, detected_frames)

    render_detection_summary(
        {
            "status": "Weapon detected" if detected_frames else "No weapon detected",
            "potential_weapons": len(detected_frames),
            "confirmed_alerts": event_summary.get("confirmed_alerts", 0),
            "resolution": metadata.get("resolution", "-"),
            "fps": metadata.get("fps", 0.0),
        },
        event_summary=event_summary if advanced else None,
    )

    st.markdown(
        '<div class="wd-section">Video review</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Use the player as the main review surface. Detection controls seek the video without autoplaying it."
    )

    selected_group = int(st.session_state.get("wd_selected_group", 0))
    selected_group = min(
        max(selected_group, 0),
        max(len(detection_groups) - 1, 0),
    )

    _render_video_player(
        video_bytes,
        detection_groups,
        detected_frames,
        selected_group=selected_group,
    )
    _render_detection_ranges(detection_groups)

    if detection_groups:
        st.markdown(
            '<div class="wd-section">Detection frames</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Compact evidence frames from the detected moments are shown below the player."
        )

        visible_frames = detected_frames[:12]
        if visible_frames:
            cols_per_row = min(4, len(visible_frames))
            for start_index in range(0, len(visible_frames), cols_per_row):
                row = visible_frames[start_index:start_index + cols_per_row]
                cols = st.columns(len(row))
                for col, (frame_index, frame_result) in zip(cols, row):
                    timestamp = float(frame_result.get("timestamp", 0.0))
                    weapon = frame_result.get("weapon", {})
                    label = _weapon_label(weapon)
                    confidence = float(weapon.get("confidence", 0.0))
                    with col:
                        st.image(
                            result.annotated_frames[frame_index],
                            caption=f"{timestamp:.2f}s · {label} · {confidence:.1%}",
                            use_container_width=True,
                        )

        if len(detected_frames) > len(visible_frames):
            with st.expander(f"View all detected frames ({len(detected_frames)})"):
                remaining = detected_frames[len(visible_frames):]
                for start_index in range(0, len(remaining), 4):
                    row = remaining[start_index:start_index + 4]
                    cols = st.columns(len(row))
                    for col, (frame_index, frame_result) in zip(cols, row):
                        timestamp = float(frame_result.get("timestamp", 0.0))
                        weapon = frame_result.get("weapon", {})
                        label = _weapon_label(weapon)
                        confidence = float(weapon.get("confidence", 0.0))
                        with col:
                            st.image(
                                result.annotated_frames[frame_index],
                                caption=f"{timestamp:.2f}s · {label} · {confidence:.1%}",
                                use_container_width=True,
                            )
    else:
        st.info("Nothing to review because no weapon evidence was detected.")

    if advanced:
        _render_events(metadata.get("events", []))

def _render_events(events):
    st.markdown(
        '<div class="wd-card">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="wd-section">Confirmed events</div>',
        unsafe_allow_html=True,
    )

    if not events:
        st.info("No confirmed weapon events in the analyzed footage.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    for event in events:
        timestamp = float(event.get("timestamp", 0.0))
        track_id = event.get("track_id", "?")
        weapon_class = (
            str(event.get("weapon_class") or "Unknown")
            .replace("_", " ")
            .title()
        )
        confidence = float(event.get("confidence", 0.0))

        st.markdown(
            f"""
            <div class="wd-pred">
                <span class="wd-pred-name">
                    {timestamp:.2f}s • {html.escape(weapon_class)}
                </span>
                <span class="wd-pred-conf">{confidence:.1%}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(f"Advanced track: Person #{track_id}")

    st.markdown('</div>', unsafe_allow_html=True)


def _count_current_persons(result):
    if not result.frame_results:
        return 0
    return len(result.frame_results[-1].get("persons", []))


def _count_potential_weapons(result):
    if not result.frame_results:
        return 0
    return sum(
        1
        for person in result.frame_results[-1].get("persons", [])
        if person.get("weapon", {}).get("present")
    )
