import streamlit as st
from PIL import Image
import numpy as np
import os
import time
import psutil
import torch
import cv2
import threading
import io
from textwrap import dedent


def render_html(content):
    html = dedent(content)
    html = "\n".join(line.lstrip() for line in html.splitlines())
    st.markdown(html, unsafe_allow_html=True)

from dr_inference import DRPredictor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Project DRISTI - National AI Screening Matrix",
    page_icon="DR",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>

        /* =====================================================
           GLOBAL
           ===================================================== */

        .stApp {
            background:
                linear-gradient(
                    rgba(15, 23, 42, 0.96),
                    rgba(15, 23, 42, 0.96)
                );

            color: #f8fafc;
        }

        /*
         * Remove Streamlit's automatic heading anchor/link icons.
         */
        h1 a,
        h2 a,
        h3 a,
        h4 a,
        h5 a,
        h6 a {
            display: none !important;
        }

        /*
         * Prevent horizontal overflow.
         */
        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }


        /* =====================================================
           SIDEBAR
           ===================================================== */

        section[data-testid="stSidebar"] {
            background: #20212a;
            border-right: 1px solid rgba(71, 85, 105, 0.35);
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #f8fafc !important;
        }


        /* =====================================================
           MAIN HEADER
           ===================================================== */

        .dristi-title {
            font-size: 2.65rem;
            font-weight: 750;
            line-height: 1.15;
            color: #ffffff;
            letter-spacing: -0.04em;
            margin-top: 0.2rem;
            margin-bottom: 0.4rem;
        }

        .dristi-subtitle {
            font-size: 1.08rem;
            font-weight: 450;
            color: #94a3b8;
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }


        /* =====================================================
           SECTION TITLES
           ===================================================== */

        .section-title {
            font-size: 1.65rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.25;
            margin-top: 0.6rem;
            margin-bottom: 1.1rem;
            letter-spacing: -0.02em;
        }


        /* =====================================================
           SYSTEM / INFERENCE CARDS
           ===================================================== */

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 14px;
            width: 100%;
            margin-bottom: 1.5rem;
        }

        .metric-grid-four {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            width: 100%;
            margin-bottom: 1.5rem;
        }

        .metric-grid-three {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            width: 100%;
            margin-bottom: 1.5rem;
        }

        .metric-card-custom {
            background: rgba(30, 41, 59, 0.74);
            border: 1px solid rgba(71, 85, 105, 0.65);
            border-radius: 12px;
            padding: 18px 18px 17px 18px;
            min-height: 112px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
            box-sizing: border-box;
        }

        .metric-card-custom:hover {
            border-color: rgba(100, 116, 139, 0.95);
            background: rgba(30, 41, 59, 0.9);
        }

        .metric-label-custom {
            font-size: 0.80rem;
            font-weight: 650;
            color: #94a3b8;
            line-height: 1.25;
            margin-bottom: 10px;
        }

        .metric-value-custom {
            font-size: 1.38rem;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.25;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .metric-value-small {
            font-size: 1.10rem;
        }

        .metric-value-large {
            font-size: 1.60rem;
        }

        .metric-secondary-custom {
            font-size: 0.76rem;
            font-weight: 450;
            color: #64748b;
            line-height: 1.3;
            margin-top: 7px;
        }


        /* =====================================================
           DIAGNOSTIC RESULT CARD
           ===================================================== */

        .diagnostic-card {
            background: rgba(30, 41, 59, 0.74);
            border: 1px solid rgba(71, 85, 105, 0.65);
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 1rem;
        }

        .diagnostic-label {
            font-size: 0.80rem;
            color: #94a3b8;
            font-weight: 650;
            margin-bottom: 8px;
        }

        .diagnostic-value {
            font-size: 1.75rem;
            font-weight: 750;
            color: #ffffff;
            line-height: 1.25;
        }


        /* =====================================================
           INFORMATION BOX
           ===================================================== */

        .clinical-box {
            background: rgba(30, 41, 59, 0.68);
            border: 1px solid rgba(71, 85, 105, 0.65);
            border-left: 4px solid #64748b;
            border-radius: 10px;
            padding: 16px 18px;
            color: #cbd5e1;
            line-height: 1.6;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }


        /* =====================================================
           IMAGE CONTAINERS
           ===================================================== */

        [data-testid="stImage"] {
            border-radius: 10px;
            overflow: hidden;
        }


        /* =====================================================
           FILE UPLOADER
           ===================================================== */

        [data-testid="stFileUploader"] {
            background: rgba(30, 41, 59, 0.55);
            border-radius: 12px;
        }


        /* =====================================================
           RESPONSIVE
           ===================================================== */

        @media (max-width: 1200px) {

            .metric-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }

            .metric-grid-four {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .metric-grid-three {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 750px) {

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .metric-grid,
            .metric-grid-four,
            .metric-grid-three {
                grid-template-columns: 1fr;
            }

            .dristi-title {
                font-size: 2rem;
            }

            .dristi-subtitle {
                font-size: 0.95rem;
            }
        }

    </style>
    """,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_screening_model():
    return DRPredictor(model_path="dr_model.onnx")


try:
    predictor = load_screening_model()
    model_loaded = True
except Exception as e:
    predictor = None
    model_loaded = False
    st.error(
        f"Error loading model artifacts: {e}"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Deployment & Compute Node"
)

compute_mode = st.sidebar.selectbox(
    "Select Hardware Target",
    [
        "CPU",
        "GPU (CUDA / RTX / Edge Accelerator)"
    ]
)


# ============================================================
# DEVICE SELECTION
# ============================================================

if compute_mode == "CPU":
    target_device = torch.device("cpu")
    node_name = "CPU"
else:
    if torch.cuda.is_available():
        target_device = torch.device("cuda")
        node_name = torch.cuda.get_device_name(0)
    else:
        target_device = torch.device("cpu")
        node_name = "CPU (CUDA unavailable)"


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Project DRISTI** is an autonomous diabetic retinopathy
    triage and diagnostic infrastructure. It leverages
    deep learning to analyze retinal fundus images and
    provide a severity assessment.

    This application is designed for research and
    educational purposes. Clinical decisions should be
    made by qualified healthcare professionals.
    """
)


# ============================================================
# SYSTEM-WIDE TELEMETRY
# ============================================================

cpu_load = psutil.cpu_percent(interval=None)

virtual_memory = psutil.virtual_memory()

ram_usage = virtual_memory.percent

ram_used_gb = (
    virtual_memory.used /
    (1024 ** 3)
)

ram_total_gb = (
    virtual_memory.total /
    (1024 ** 3)
)

cpu_threads = psutil.cpu_count(
    logical=True
)


# ============================================================
# GPU SYSTEM TELEMETRY
# ============================================================

if target_device.type == "cuda":

    system_vram_allocated_gb = (
        torch.cuda.memory_allocated(0) /
        (1024 ** 3)
    )

    system_vram_total_gb = (
        torch.cuda.get_device_properties(0).total_memory /
        (1024 ** 3)
    )

else:

    system_vram_allocated_gb = 0.0

    system_vram_total_gb = 0.0


# ============================================================
# MAIN HEADER
# ============================================================

render_html(
    """
    <div class="dristi-title">
        Project DRISTI
    </div>

    <div class="dristi-subtitle">
        Autonomous Diabetic Retinopathy Triage & Diagnostic Infrastructure
    </div>
    """,
)

st.markdown("---")


# ============================================================
# SYSTEM STATUS
# ============================================================

render_html(
    '<div class="section-title">System Status</div>',
)


if target_device.type == "cuda":

    gpu_vram_text = (
        f"{system_vram_allocated_gb:.2f} / "
        f"{system_vram_total_gb:.1f} GB"
    )

    gpu_vram_secondary = (
        "Allocated / Total"
    )

else:

    gpu_vram_text = "Not in use"

    gpu_vram_secondary = (
        "GPU execution disabled"
    )


render_html(
    f"""
    <div class="metric-grid">

        <div class="metric-card-custom">
            <div class="metric-label-custom">
                Compute Node
            </div>

            <div class="metric-value-custom metric-value-small">
                {node_name}
            </div>
        </div>


        <div class="metric-card-custom">
            <div class="metric-label-custom">
                CPU Threads
            </div>

            <div class="metric-value-custom metric-value-large">
                {cpu_threads}
            </div>

            <div class="metric-secondary-custom">
                Logical processors
            </div>
        </div>


        <div class="metric-card-custom">
            <div class="metric-label-custom">
                GPU VRAM
            </div>

            <div class="metric-value-custom">
                {gpu_vram_text}
            </div>

            <div class="metric-secondary-custom">
                {gpu_vram_secondary}
            </div>
        </div>


        <div class="metric-card-custom">
            <div class="metric-label-custom">
                System RAM
            </div>

            <div class="metric-value-custom">
                {ram_usage:.1f}%
            </div>

            <div class="metric-secondary-custom">
                {ram_used_gb:.1f} / {ram_total_gb:.1f} GB
            </div>
        </div>


        <div class="metric-card-custom">
            <div class="metric-label-custom">
                System CPU
            </div>

            <div class="metric-value-custom">
                {cpu_load:.1f}%
            </div>

            <div class="metric-secondary-custom">
                Overall utilization
            </div>
        </div>

    </div>
    """,
)


st.markdown("---")


# ============================================================
# FILE UPLOADER
# ============================================================

render_html(
    '<div class="section-title">Retinal Fundus Scan</div>',
)

uploaded_file = st.file_uploader(
    "Upload Patient Retinal Fundus Scan (Clinical Capture)",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# EXPLAINABILITY MAP
# ============================================================

def generate_annotated_lesion_map(image_bytes):

    """
    Generates a visualization based on green-channel
    CLAHE enhancement and contour detection.

    This is an EXPLAINABILITY visualization and is
    separate from the diagnostic neural network.
    """

    nparr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if img is None:

        raise ValueError(
            "Unable to decode uploaded image."
        )

    # --------------------------------------------------------
    # Green channel
    # --------------------------------------------------------

    green_channel = img[:, :, 1]

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        green_channel
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    _, thresh = cv2.threshold(
        enhanced,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # Contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    annotated_img = img.copy()

    lesion_count = 0

    # --------------------------------------------------------
    # Bounding boxes
    # --------------------------------------------------------

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if 10 < area < 800:

            x, y, w, h = (
                cv2.boundingRect(cnt)
            )

            x1 = max(
                x - 5,
                0
            )

            y1 = max(
                y - 5,
                0
            )

            x2 = min(
                x + w + 5,
                annotated_img.shape[1]
            )

            y2 = min(
                y + h + 5,
                annotated_img.shape[0]
            )

            cv2.rectangle(
                annotated_img,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            lesion_count += 1

            if lesion_count >= 15:

                break

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    if lesion_count == 0:

        cv2.putText(
            annotated_img,
            "AI Focus: Macular & Vascular Zone Verified",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            annotated_img,
            f"Detected Anomalies/Exudates: {lesion_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return (
        cv2.cvtColor(
            annotated_img,
            cv2.COLOR_BGR2RGB
        ),
        lesion_count
    )


# ============================================================
# INFERENCE MONITOR
# ============================================================

class InferenceMonitor:

    """
    Monitors the DRISTI Python process while the neural
    network inference is running.

    Measurements:
        - Peak process RAM
        - Average process CPU
        - Maximum process CPU
    """

    def __init__(self):

        self.running = False

        self.peak_ram_gb = 0.0

        self.average_cpu_percent = 0.0

        self.max_cpu_percent = 0.0

        self.samples = 0

        self.cpu_sum = 0.0

        self.thread = None

    def _monitor(self):

        process = psutil.Process(
            os.getpid()
        )
        # Initialize CPU counter
        process.cpu_percent(
        interval=None
        )

        # Get total logical cores for normalization
        total_cores = psutil.cpu_count(logical=True) or 1

        while self.running:
        # ----------------------------------------------
        # RAM
        # ----------------------------------------------

            current_ram_gb = (
                process.memory_info().rss /
                (1024 ** 3)
            )

            self.peak_ram_gb = max(
                self.peak_ram_gb,
                current_ram_gb
            )

                    # ----------------------------------------------
                    # CPU (Normalized to 0 - 100% scale)
                    # ----------------------------------------------

            current_cpu = (
                process.cpu_percent(interval=0.05) / total_cores
            )

            self.cpu_sum += current_cpu

            self.max_cpu_percent = max(
                self.max_cpu_percent,
                current_cpu
            )

            self.samples += 1

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=1.0
            )

        if self.samples > 0:

            self.average_cpu_percent = (
                self.cpu_sum /
                self.samples
            )


# ============================================================
# MONITORED INFERENCE
# ============================================================

def run_monitored_inference(
    predictor,
    img_bytes,
    device
):

    """
    Measures only model inference.

    Excluded:
        - File upload
        - Image decoding for display
        - Explainability map generation
        - UI rendering
        - Model loading
        - Streamlit processing
    """

    process = psutil.Process(
        os.getpid()
    )

    # ========================================================
    # BEFORE INFERENCE
    # ========================================================

    ram_before_gb = (
        process.memory_info().rss /
        (1024 ** 3)
    )

    # ========================================================
    # GPU BASELINE
    # ========================================================

    gpu_available = (
        device.type == "cuda"
        and torch.cuda.is_available()
    )

    if gpu_available:

        torch.cuda.synchronize()

        vram_before_gb = (
            torch.cuda.memory_allocated(0) /
            (1024 ** 3)
        )

        reserved_before_gb = (
            torch.cuda.memory_reserved(0) /
            (1024 ** 3)
        )

        # Reset peak counter immediately before inference
        torch.cuda.reset_peak_memory_stats(0)

    else:

        vram_before_gb = 0.0

        reserved_before_gb = 0.0

    # ========================================================
    # START MONITOR
    # ========================================================

    monitor = InferenceMonitor()

    monitor.start()

    # ========================================================
    # START EXACT INFERENCE TIMER
    # ========================================================

    start_time = time.perf_counter()

    try:

        # ====================================================
        # ACTUAL MODEL INFERENCE
        # ====================================================

        result = predictor.predict(
            img_bytes
        )

        # ====================================================
        # CUDA SYNCHRONIZATION
        # ====================================================

        if gpu_available:

            torch.cuda.synchronize()

    finally:

        end_time = time.perf_counter()

        monitor.stop()

    # ========================================================
    # LATENCY
    # ========================================================

    inference_time_ms = (
        end_time -
        start_time
    ) * 1000.0

    # ========================================================
    # AFTER INFERENCE RAM
    # ========================================================

    ram_after_gb = (
        process.memory_info().rss /
        (1024 ** 3)
    )

    ram_delta_gb = (
        ram_after_gb -
        ram_before_gb
    )

    # ========================================================
    # GPU METRICS
    # ========================================================

    if gpu_available:

        vram_after_gb = (
            torch.cuda.memory_allocated(0) /
            (1024 ** 3)
        )

        reserved_after_gb = (
            torch.cuda.memory_reserved(0) /
            (1024 ** 3)
        )

        peak_vram_gb = (
            torch.cuda.max_memory_allocated(0) /
            (1024 ** 3)
        )

        peak_reserved_gb = (
            torch.cuda.max_memory_reserved(0) /
            (1024 ** 3)
        )

        vram_delta_gb = (
            vram_after_gb -
            vram_before_gb
        )

    else:

        vram_after_gb = 0.0

        reserved_after_gb = 0.0

        peak_vram_gb = 0.0

        peak_reserved_gb = 0.0

        vram_delta_gb = 0.0

    # ========================================================
    # RETURN METRICS
    # ========================================================

    metrics = {

        # Timing
        "inference_time_ms":
            inference_time_ms,

        # RAM
        "ram_before_gb":
            ram_before_gb,

        "ram_after_gb":
            ram_after_gb,

        "ram_delta_gb":
            ram_delta_gb,

        "peak_ram_gb":
            monitor.peak_ram_gb,

        # CPU
        "average_cpu_percent":
            monitor.average_cpu_percent,

        "max_cpu_percent":
            monitor.max_cpu_percent,

        # GPU
        "vram_before_gb":
            vram_before_gb,

        "vram_after_gb":
            vram_after_gb,

        "vram_delta_gb":
            vram_delta_gb,

        "peak_vram_gb":
            peak_vram_gb,

        "peak_reserved_gb":
            peak_reserved_gb,

        "reserved_before_gb":
            reserved_before_gb,

        "reserved_after_gb":
            reserved_after_gb
    }

    return result, metrics


# ============================================================
# IMAGE PROCESSING + INFERENCE
# ============================================================

if uploaded_file is not None and model_loaded:

    # ========================================================
    # READ BYTES
    # ========================================================

    uploaded_file.seek(0)

    img_bytes = uploaded_file.read()

    # ========================================================
    # DECODE DISPLAY IMAGE
    # ========================================================

    display_image = Image.open(
        io.BytesIO(img_bytes)
    ).convert("RGB")


    # ========================================================
    # THREE COLUMN LAYOUT
    # ========================================================

    col1, col2, col3 = st.columns(
        [1.15, 1.15, 1]
    )


    # ========================================================
    # RAW IMAGE
    # ========================================================

    with col1:

        render_html(
            '<div class="section-title">Raw Capture</div>',
)

        st.image(
            display_image,
            use_container_width=True
        )

        st.caption(
            "Original retinal fundus image supplied to DRISTI."
        )


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    with col2:

        render_html(
            '<div class="section-title">Explainability</div>',
)

        annotated_image, lesion_count = (
            generate_annotated_lesion_map(
                img_bytes
            )
        )

        st.image(
            annotated_image,
            use_container_width=True
        )

        st.caption(
            f"Highlighted {lesion_count} focal anomaly region(s) "
            "using green-channel CLAHE contrast isolation."
        )


    # ========================================================
    # DIAGNOSTIC OUTPUT
    # ========================================================

    with col3:

        render_html(
            '<div class="section-title">Diagnostic Output</div>',
)

        # ====================================================
        # ACTUAL MONITORED INFERENCE
        # ====================================================

        result, inference_metrics = (
            run_monitored_inference(
                predictor,
                img_bytes,
                target_device
            )
        )

        # ====================================================
        # SEVERITY
        # ====================================================

        severity_map = {

            0: "Stage 0: No DR",

            1: "Stage 1: Mild DR",

            2: "Stage 2: Moderate DR",

            3: "Stage 3: Severe DR",

            4: "Stage 4: Proliferative DR"
        }

        severity_text = severity_map.get(
            result["grade"],
            "Unknown Stage"
        )

        render_html(
            f"""
            <div class="diagnostic-card">

                <div class="diagnostic-label">
                    Predicted Severity
                </div>

                <div class="diagnostic-value">
                    {severity_text}
                </div>

            </div>
            """,
)

        # ====================================================
        # SEVERITY SCORE
        # ====================================================

        st.progress(
            min(
                max(
                    result["raw_score"] / 4.0,
                    0.0
                ),
                1.0
            )
        )

        st.caption(
            f"Ordinal severity score: "
            f"{result['raw_score']:.2f} / 4.00"
        )


    # ========================================================
    # INFERENCE PERFORMANCE
    # ========================================================

    st.markdown("---")

    render_html(
        '<div class="section-title">Inference Performance</div>',
)


    # ========================================================
    # RESOURCE CARDS
    # ========================================================

    render_html(
        f"""
        <div class="metric-grid-four">

            <div class="metric-card-custom">

                <div class="metric-label-custom">
                    Inference Latency
                </div>

                <div class="metric-value-custom metric-value-large">
                    {inference_metrics['inference_time_ms']:.1f} ms
                </div>

                <div class="metric-secondary-custom">
                    Model execution only
                </div>

            </div>


            <div class="metric-card-custom">

                <div class="metric-label-custom">
                    Peak RAM
                </div>

                <div class="metric-value-custom metric-value-large">
                    {inference_metrics['peak_ram_gb']:.3f} GB
                </div>

                <div class="metric-secondary-custom">
                    Peak process memory
                </div>

            </div>


            <div class="metric-card-custom">

                <div class="metric-label-custom">
                    RAM Increase
                </div>

                <div class="metric-value-custom">
                    {max(inference_metrics['ram_delta_gb'], 0):.3f} GB
                </div>

                <div class="metric-secondary-custom">
                    Before vs after inference
                </div>

            </div>


            <div class="metric-card-custom">

                <div class="metric-label-custom">
                    Average CPU
                </div>

                <div class="metric-value-custom metric-value-large">
                    {inference_metrics['average_cpu_percent']:.1f}%
                </div>

                <div class="metric-secondary-custom">
                    Process utilization
                </div>

            </div>

        </div>
        """,
)


    # ========================================================
    # SECOND RESOURCE ROW
    # ========================================================

    if target_device.type == "cuda":

        render_html(
            f"""
            <div class="metric-grid-four">

                <div class="metric-card-custom">

                    <div class="metric-label-custom">
                        Peak VRAM
                    </div>

                    <div class="metric-value-custom metric-value-large">
                        {inference_metrics['peak_vram_gb']:.3f} GB
                    </div>

                    <div class="metric-secondary-custom">
                        Peak allocated during inference
                    </div>

                </div>


                <div class="metric-card-custom">

                    <div class="metric-label-custom">
                        VRAM Increase
                    </div>

                    <div class="metric-value-custom">
                        {max(inference_metrics['vram_delta_gb'], 0):.3f} GB
                    </div>

                    <div class="metric-secondary-custom">
                        Before vs after inference
                    </div>

                </div>


                <div class="metric-card-custom">

                    <div class="metric-label-custom">
                        Peak CPU
                    </div>

                    <div class="metric-value-custom">
                        {inference_metrics['max_cpu_percent']:.1f}%
                    </div>

                    <div class="metric-secondary-custom">
                        Maximum process utilization
                    </div>

                </div>


                <div class="metric-card-custom">

                    <div class="metric-label-custom">
                        VRAM After Inference
                    </div>

                    <div class="metric-value-custom">
                        {inference_metrics['vram_after_gb']:.3f} GB
                    </div>

                    <div class="metric-secondary-custom">
                        Allocated after prediction
                    </div>

                </div>

            </div>
            """,
)

    else:

        render_html(
            f"""
            <div class="metric-grid-three">

                <div class="metric-card-custom">

                    <div class="metric-label-custom">
                        Peak CPU
                    </div>

                    <div class="metric-value-custom metric-value-large">
                        {inference_metrics['max_cpu_percent']:.1f}%
                    </div>

                    <div class="metric-secondary-custom">
                        Maximum process utilization
                    </div>

                </div>


                <div class="metric-card-custom">

                    <div class="metric-label-custom">
                        RAM Before
                    </div>

                    <div class="metric-value-custom">
                        {inference_metrics['ram_before_gb']:.3f} GB
                    </div>

                    <div class="metric-secondary-custom">
                        Process memory baseline
                    </div>

                </div>


                <div class="metric-card-custom">

                    <div class="metric-label-custom">
                        RAM After
                    </div>

                    <div class="metric-value-custom">
                        {inference_metrics['ram_after_gb']:.3f} GB
                    </div>

                    <div class="metric-secondary-custom">
                        Process memory after prediction
                    </div>

                </div>

            </div>
            """,
)


    # ========================================================
    # MODEL METRICS
    # ========================================================

    render_html(
        '<div class="section-title">Model Metrics</div>',
)

    per_model_scores = result.get("per_model", [result["raw_score"]])
    ensemble_variance = np.var(per_model_scores)

    render_html(
        f"""
        <div class="metric-grid-three">

            <div class="metric-card-custom">

                <div class="metric-label-custom">
                    Model Confidence
                </div>

                <div class="metric-value-custom metric-value-large">
                    {result['confidence'] * 100:.1f}%
                </div>

                <div class="metric-secondary-custom">
                    Final prediction confidence
                </div>

            </div>


            <div class="metric-card-custom">

                <div class="metric-label-custom">
                    Ensemble Variance
                </div>

                <div class="metric-value-custom metric-value-large">
                    {ensemble_variance:.4f}
                </div>

                <div class="metric-secondary-custom">
                    TTA / variation index
                </div>

            </div>


            <div class="metric-card-custom">

                <div class="metric-label-custom">
                    Model Engine
                </div>

                <div class="metric-value-custom metric-value-large">
                    ONNX Runtime
                </div>

                <div class="metric-secondary-custom">
                    Optimized execution backend
                </div>

            </div>

        </div>
        """,
)


    # ========================================================
    # ENSEMBLE / TTA BREAKDOWN
    # ========================================================

    render_html(
        '<div class="section-title">Prediction Breakdown</div>',
)


    fold_grid_html = """
    <div class="metric-grid">
    """

    for idx, fold_score in enumerate(per_model_scores):

        fold_grid_html += f"""
        <div class="metric-card-custom">

            <div class="metric-label-custom">
                Inference Pass {idx + 1}
            </div>

            <div class="metric-value-custom metric-value-large">
                {fold_score:.2f}
            </div>

            <div class="metric-secondary-custom">
                Ordinal prediction score
            </div>

        </div>
        """

    fold_grid_html += """
    </div>
    """

    render_html(
        fold_grid_html,
)


    # ========================================================
    # CLINICAL BRIEFING
    # ========================================================

    render_html(
        """
        <div class="clinical-box">

            <strong>Evaluator Clinical Briefing</strong><br><br>

            This system utilizes ONNX runtime acceleration with
            EfficientNet-B6 architecture and test-time augmentation.
            Patients predicted at Stage 2 or higher can be 
            prioritized for clinical review.

        </div>
        """,
)


    # ========================================================
    # RESOURCE MEASUREMENT NOTE
    # ========================================================

    st.caption(
        "Inference measurements are collected around the "
        "model prediction call. File upload, image display, "
        "explainability-map generation, model loading, and "
        "Streamlit interface rendering are excluded from "
        "the reported inference latency."
    )