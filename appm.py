# 🌍 EcoEyes - Smart Waste Monitoring Dashboard
# UI Refined Version based on requested structure
import os
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import cv2
import math
import time
from pathlib import Path
from datetime import datetime
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import sqlite3
from textwrap import dedent
import streamlit.components.v1 as components
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False
# ═══════════════════════════════════════════════════════════════════════════
# 🎨 COLORS
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#224C43",
    "primary_dark": "#183A32",
    "secondary": "#8EB69B",
    "bg": "#DCE9DA",
    "white": "#FFFFFF",
    "gray_light": "#F4F8F3",
    "gray_medium": "#D7E2D4",
    "gray_dark": "#5E6B64",
    "black": "#1A1F1C",
    "success": "#2E9B55",
    "warning": "#D9A31A",
    "danger": "#C94C4C",
    "info": "#2B8CA3",
}
GREEN_COLORS = [
  "#1B7854",
  "#2BA876",
  "#4CAF7A",
  "#7ED6A5",
  "#A8E6C1"
]

RISK_STYLES = {
    "High": {
        "bg": "#FDE7E7",
        "border": "#D9534F",
        "badge": "#D9534F",
        "text": "#7A1B1B",
    },
    "Medium": {
        "bg": "#FFF4D6",
        "border": "#E0A800",
        "badge": "#E0A800",
        "text": "#6F5300",
    },
    "Low": {
        "bg": "#E5F5E8",
        "border": "#28A745",
        "badge": "#28A745",
        "text": "#155724",
    },
    "Multiple Danger": {
        "bg": "#F3E8FF",
        "border": "#7E57C2",
        "badge": "#7E57C2",
        "text": "#4A2B8A",
    }
}

def normalize_risk_level(level):
    if level is None:
        return "Low"

    level = str(level).strip().lower()

    if level == "high":
        return "High"
    elif level == "medium":
        return "Medium"
    elif level == "low":
        return "Low"
    elif level in ["multiple danger", "multiple_danger"]:
        return "Multiple Danger"
    else:
        return "Low"
    
RISK_MAP = {
    "Paper": "Low",
    "Tissues": "Low",
    "Paper bags": "Low",
    "Organic": "Low",
    "Plastic": "Medium",
    "Metal": "High",
    "Metal cans": "High",
    "Glass": "High",
    "Chemi.": "High",
}

CLASS_NAMES = {
    0: "Glass",
    1: "Plastic",
    2: "metal",
    3: "Paper",
    4: "Organic",
}

CLASS_NORMALIZE = {
    "metal": "Metal",
    "Metal": "Metal",
}


# ═══════════════════════════════════════════════════════════════════════════
# ⚙️ PATHS
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(".")
DB_PATH = BASE_DIR / "database" / "ecoeyes.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "weights" / "best.pt"

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DET_DIR = BASE_DIR / "data" / "detected"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DET_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 🧠 STREAMLIT CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EcoEyes - Waste Monitoring",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════════════════════════════════════════════════════════════════
# 🎨 CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
<style>
    .stApp {{
        background: {COLORS['bg']};
    }}
    
    section[data-testid="stSidebar"],
    div[data-testid="collapsedControl"] {{
        display: none !important;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
        max-width: 100%;
    }}

/* الكروت */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: white;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 16px;
    border: 1px solid #E3EAE3;
}}


div[data-testid="stPlotlyChart"] {{
    padding-top: 6px;
}}


.card-header-title {{
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 4px;
}}

.card-header-sub {{
    font-size: 13px;
    color: #6B7280;
    margin-bottom: 10px;
}}

    h1, h2, h3 {{
        color: {COLORS['black']};
        font-weight: 800;
        letter-spacing: -0.3px;
    }}

    .topbar {{
        background: {COLORS['primary']};
        border-radius: 22px;
        padding: 22px 28px;
        margin-top: 40px;
        margin-bottom: 14px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.08);
    }}

    .brand-title {{
        color: white;
        font-size: 32px;
        font-weight: 900;
        margin: 0;
        line-height: 1.15;
    }}

    .brand-subtitle {{
        color: #DDEBE0;
        font-size: 14px;
        margin: 6px 0 0 0;
        line-height: 1.4;
    }}

    .navbar-title {{
        text-align: right;
        font-size: 14px;
        color: {COLORS['gray_dark']};
        margin-bottom: 18px;
        font-weight: 600;
    }}

    .page-shell {{
        background: transparent;
        border-radius: 20px;
    }}

    .hero-card {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        border-radius: 22px;
        padding: 28px;
        margin-bottom: 18px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.10);
    }}

    .hero-title {{
        color: white;
        font-size: 31px;
        font-weight: 900;
        margin: 0;
    }}

    .hero-text {{
        color: #E7F2E8;
        font-size: 14px;
        margin-top: 10px;
        line-height: 1.85;
    }}

    .section-title {{
        font-size: 22px;
        font-weight: 900;
        margin-bottom: 14px;
        color: {COLORS['black']};
    }}

    .section-card {{
        background: {COLORS['white']};
        border: 1px solid {COLORS['gray_medium']};
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }}

    /* =========================
       MAIN NAVIGATION CARDS
       ========================= */
    .nav-card {{
        background: {COLORS['white']};
        border: 1px solid {COLORS['gray_medium']};
        border-radius: 20px;
        padding: 16px 18px;
        min-height: 92px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        transition: all 0.22s ease;
        margin-bottom: 14px;
    }}

    .nav-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(0,0,0,0.08);
    }}

    .nav-card-title {{
        color: {COLORS['primary']};
        font-size: 19px;
        font-weight: 900;
        margin-bottom: 8px;
    }}

    .nav-card-text {{
        color: {COLORS['gray_dark']};
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 0;
    }}

    /* =========================
       ST.METRIC STYLE
       ========================= */
    div[data-testid="metric-container"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    div[data-testid="metric-container"] > div {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }}

    div[data-testid="metric-container"] label {{
        color: {COLORS['gray_dark']} !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {COLORS['primary']} !important;
        font-size: 40px !important;
        font-weight: 900 !important;
        line-height: 1.1 !important;
        letter-spacing: -0.5px;
    }}

    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
        display: none !important;
    }}

    .stCaption {{
        color: {COLORS['gray_dark']} !important;
        font-size: 12px !important;
        margin-top: -2px !important;
    }}

  
    .metric-card {{
        background: {COLORS['white']};
        border: 1px solid {COLORS['gray_medium']};
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.04);
        min-height: 120px;
    }}

    .metric-label {{
        color: {COLORS['gray_dark']};
        font-size: 13px;
        margin-bottom: 8px;
        font-weight: 600;
    }}

    .metric-value {{
        color: {COLORS['primary']};
        font-size: 30px;
        font-weight: 900;
        line-height: 1.1;
    }}
  
    .login-logo {{
       text-align: center;
       font-size: 58px;
       margin-bottom: 8px;
    }}

    .login-title {{
      text-align: center;
      font-size: 36px;
      font-weight: 900;
      color: #092E24;
      margin-bottom: 4px;
    }}

    .login-subtitle {{
      text-align: center;
      font-size: 17px;
      color: #5E6B64;
      margin-bottom: 20px;
    }}

    .login-divider {{
      text-align: center;
      color: #6FAF78;
      margin: 10px 0 22px 0;
      font-size: 20px;
    }}

    div[data-testid="stForm"] {{
      background: white;
      border-radius: 28px;
      padding: 42px 48px 34px 48px;
      max-width: 620px;
      margin: 55px auto 0 auto;
      box-shadow: 0 18px 40px rgba(0,0,0,0.12);
      border: 1px solid rgba(255,255,255,0.8);
    }}

    div[data-testid="stForm"] label {{
      color: #1F6B43 !important;
      font-weight: 800 !important;
      font-size: 16px !important;
    }}

    div[data-testid="stTextInput"] input {{
      border-radius: 14px !important;
      height: 54px !important;
      border: 1.5px solid #D7DDD7 !important;
      font-size: 16px !important;
      padding-left: 16px !important;
    }}

    div[data-testid="stTextInput"] input:focus {{
      border-color: #2E7D4F !important;
      box-shadow: 0 0 0 1px #2E7D4F !important;
    }}

    div[data-testid="stFormSubmitButton"] button {{
      width: 100%;
      height: 54px;
      border-radius: 14px;
      background: linear-gradient(135deg, #1F6B43, #224C43) !important;
      color: white !important;
      font-size: 17px;
      font-weight: 900;
      margin-top: 12px;
    }}

    .login-footer {{
      text-align: center;
      color: #7A857D;
      font-size: 13px;
      margin-top: 24px;
    }}

    .metric-sub {{
        color: {COLORS['gray_dark']};
        font-size: 12px;
        margin-top: 8px;
    }}

    .alert-box {{
        border-radius: 18px;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 5px 14px rgba(0,0,0,0.04);
    }}

    .subtle-note {{
        color: {COLORS['gray_dark']};
        font-size: 13px;
        line-height: 1.75;
    }}

    .mini-note {{
        font-size: 12px;
        color: {COLORS['gray_dark']};
        margin-top: 8px;
    }}

    .report-card-header {{
    background: {COLORS['white']};
    border: 2px solid {COLORS['gray_medium']};
    border-radius: 22px;
    padding: 18px;
    margin-bottom: 10px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}}


    .card-header-title {{
        color: {COLORS['primary']};
        font-size: 18px;
        font-weight: 900;
        margin-bottom: 6px;
    }}

    .card-header-sub {{
        color: {COLORS['gray_dark']};
        font-size: 13px;
        margin-bottom: 14px;
    }}

    .status-banner {{
        background: #EDF7EE;
        border: 1px solid #B9DEC0;
        color: #1E5F34;
        padding: 12px 14px;
        border-radius: 14px;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 16px;
    }}

    .success-banner {{
        background: #E7F6EA;
        border: 1px solid #BCE0C4;
        color: #1B6C37;
        padding: 14px 16px;
        border-radius: 16px;
        font-size: 14px;
        font-weight: 700;
        margin: 14px 0;
    }}

    .link-banner {{
        background: #EEF5FF;
        border: 1px solid #C9DBF7;
        color: #174A8B;
        padding: 12px 14px;
        border-radius: 14px;
        font-size: 13px;
        margin-top: 10px;
    }}

    .stButton > button {{
        background: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: 14px;
        padding: 10px 18px;
        font-size: 14px;
        font-weight: 800;
        box-shadow: 0 6px 14px rgba(0,0,0,0.10);
        transition: all 0.18s ease;
    }}

    .stButton > button:hover {{
        background: {COLORS['primary_dark']};
        transform: translateY(-1px);
    }}

    div[data-testid="stHorizontalBlock"] > div {{
        gap: 1.2rem;
    }}

    .footer-box {{
        text-align: center;
        padding: 14px;
        color: {COLORS['gray_dark']};
    }}
</style>
""",
    unsafe_allow_html=True,
)



# ═══════════════════════════════════════════════════════════════════════════
# 🗄️ DB HELPERS يشغل قاعدة البيانات بالكامل
# ═══════════════════════════════════════════════════════════════════════════

def get_conn():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def init_db():
    with get_conn() as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source_type TEXT NOT NULL,
            conf_threshold REAL NOT NULL,
            iou REAL,
            model_path TEXT NOT NULL
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            image_name TEXT NOT NULL,
            image_path TEXT,
            detected_path TEXT,
            total_detections INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            det_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            waste_type TEXT NOT NULL,
            count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS detection_boxes (
            box_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            class_id INTEGER NOT NULL,
            waste_type TEXT NOT NULL,
            conf REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            level TEXT NOT NULL,
            waste_type TEXT NOT NULL,
            conf REAL NOT NULL,
            message TEXT NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        """)

            
        conn.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY,
            location_name TEXT,
            stream_url TEXT,
            map_x REAL,
            map_y REAL
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            password_hash TEXT,
            role TEXT
        );
        """)  
        columns = [col[1] for col in conn.execute("PRAGMA table_info(alerts)")]

        if "camera_id" not in columns:
            conn.execute("ALTER TABLE alerts ADD COLUMN camera_id INTEGER")
        
        cam_exists = conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        if cam_exists == 0:
            conn.execute("""
            INSERT INTO cameras (camera_id, location_name, stream_url, map_x, map_y)
            VALUES (1, 'Default Location', '0', 100, 200)
            """)
        
        user_exists = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

        if user_exists == 0:
           conn.execute("""
        INSERT INTO users (username, password_hash, role)
        VALUES ('user1', '1234', 'user')
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_run ON images(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_det_image ON detections(image_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_box_image ON detection_boxes(image_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);")

        conn.commit()


def db_create_run(source_type: str, conf: float, iou: float | None, model_path: str) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO runs(created_at, source_type, conf_threshold, iou, model_path) VALUES (?, ?, ?, ?, ?)",
            (now, source_type, float(conf), iou, model_path)
        )
        conn.commit()
        return int(cur.lastrowid)


def db_add_image(run_id: int, image_name: str, image_path: str | None, detected_path: str | None, total_detections: int) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO images(run_id, image_name, image_path, detected_path, total_detections, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (int(run_id), str(image_name), image_path, detected_path, int(total_detections), now)
        )
        conn.commit()
        return int(cur.lastrowid)


def db_add_detection_summary(image_id: int, summary_rows: list[tuple[str, int]]):
    if not summary_rows:
        return
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO detections(image_id, waste_type, count, created_at) VALUES (?, ?, ?, ?)",
            [(int(image_id), str(wt), int(cnt), now) for (wt, cnt) in summary_rows]
        )
        conn.commit()


def db_add_boxes(image_id: int, boxes_rows: list[tuple[int, str, float, float, float, float, float]]):
    if not boxes_rows:
        return
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO detection_boxes(image_id, class_id, waste_type, conf, x1, y1, x2, y2, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(int(image_id), int(cid), str(wt), float(cf), float(x1), float(y1), float(x2), float(y2), now)
             for (cid, wt, cf, x1, y1, x2, y2) in boxes_rows]
        )
        conn.commit()


def db_add_alert(image_id: int, level: str, waste_type: str, conf: float, message: str) -> int:
    now = datetime.now().isoformat(timespec="seconds")

    CAMERA_ID = "CAM_01"   

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO alerts(
                image_id,
                created_at,
                level,
                waste_type,
                conf,
                message,
                camera_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (int(image_id), now, str(level), str(waste_type), float(conf), str(message), CAMERA_ID)
        )
        conn.commit()
        return int(cur.lastrowid)


def db_get_total_stats():
    with get_conn() as conn:
        total_images = pd.read_sql_query("SELECT COUNT(*) AS c FROM images", conn).iloc[0]["c"]
        total_boxes = pd.read_sql_query("SELECT COUNT(*) AS c FROM detection_boxes", conn).iloc[0]["c"]
        total_alerts = pd.read_sql_query("SELECT COUNT(*) AS c FROM alerts", conn).iloc[0]["c"]
        total_runs = pd.read_sql_query("SELECT COUNT(*) AS c FROM runs", conn).iloc[0]["c"]
    return int(total_images), int(total_boxes), int(total_alerts), int(total_runs)


def db_get_recent_alerts(limit: int = 10) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                a.alert_id, a.created_at, a.level, a.waste_type, a.conf, a.message,
                i.image_name, i.image_id, i.image_path, i.detected_path
            FROM alerts a
            JOIN images i ON i.image_id = a.image_id
            ORDER BY a.alert_id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),)
        )
    return df


def db_get_alert_by_id(alert_id: int) -> dict | None:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                a.alert_id, a.created_at, a.level, a.waste_type, a.conf, a.message,
                a.camera_id,
                i.image_name, i.image_id, i.image_path, i.detected_path, i.total_detections
            FROM alerts a
            JOIN images i ON i.image_id = a.image_id
            WHERE a.alert_id = ?
            """,
            conn,
            params=(int(alert_id),)
        )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def db_get_waste_distribution() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT waste_type, COUNT(*) AS count
            FROM detection_boxes
            GROUP BY waste_type
            ORDER BY count DESC
            """,
            conn
        )
    return df


def db_get_detection_timeline_last10() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT DATE(created_at) as date, COUNT(*) as detection_count
            FROM detection_boxes
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT 10
            """,
            conn
        )
    return df


def db_get_risk_distribution() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT level, COUNT(*) as count
            FROM alerts
            GROUP BY level
            ORDER BY count DESC
            """,
            conn
        )
    return df


def db_get_alerts_table(limit: int = 100) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                a.alert_id AS "Alert ID",
                i.image_name AS "Image Name",
                a.waste_type AS "Waste Type",
                a.level AS "Risk Level",
                ROUND(a.conf, 2) AS "Confidence",
                a.created_at AS "Detected At"
            FROM alerts a
            JOIN images i ON i.image_id = a.image_id
            ORDER BY a.alert_id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),)
        )
    return df


def db_get_alerts_filtered(level: str | None = None, waste_type: str | None = None, limit: int = 50) -> pd.DataFrame:
    query = """
       SELECT
           a.alert_id, a.created_at, a.level, a.waste_type, a.conf, a.message,
           a.camera_id,   
           i.image_name, i.image_id, i.image_path, i.detected_path
       FROM alerts a
       JOIN images i ON i.image_id = a.image_id
       WHERE 1=1
    """
    params = []

    if level and level != "All":
        query += " AND a.level = ?"
        params.append(level)

    if waste_type and waste_type != "All":
        query += " AND a.waste_type = ?"
        params.append(waste_type)

    query += " ORDER BY a.alert_id DESC LIMIT ?"
    params.append(int(limit))

    with get_conn() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def db_get_waste_types_for_filter() -> list[str]:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT waste_type
            FROM alerts
            ORDER BY waste_type ASC
            """,
            conn
        )
    return df["waste_type"].tolist() if not df.empty else []


def db_get_recent_camera_activity() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT r.run_id, r.created_at, r.source_type, COUNT(i.image_id) AS images_count
            FROM runs r
            LEFT JOIN images i ON i.run_id = r.run_id
            WHERE r.source_type = 'camera'
            GROUP BY r.run_id, r.created_at, r.source_type
            ORDER BY r.run_id DESC
            LIMIT 5
            """,
            conn
        )
    return df


def db_get_boxes_for_image(image_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT waste_type, conf, x1, y1, x2, y2, class_id, created_at
            FROM detection_boxes
            WHERE image_id = ?
            ORDER BY box_id DESC
            """,
            conn,
            params=(int(image_id),)
        )
    return df
# **********
#  MODEL
# **********
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

# 🧩 HELPERS 

def normalize_waste_name(name: str) -> str:
    if name is None:
        return ""
    name = str(name).strip()
    return CLASS_NORMALIZE.get(name, name)


def pick_alert_level(waste_types: list[str]) -> tuple[str, str]:
    if not waste_types:
        return ("Low", "Unknown")

    normalized = [normalize_waste_name(w) for w in waste_types]
    unique_levels = {RISK_MAP.get(w, "Low") for w in normalized}

    if len(set(normalized)) > 1 and len(unique_levels) > 1:
        return ("Multiple Danger", "Multiple Waste Types")

    rank = {"Low": 1, "Medium": 2, "High": 3}
    best_level = "Low"
    best_type = normalized[0]

    for wt in normalized:
        lvl = RISK_MAP.get(wt, "Low")
        if rank[lvl] > rank[best_level]:
            best_level = lvl
            best_type = wt

    return best_level, best_type


def render_metric_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nav_card(title: str, text: str):
    st.markdown(
        f"""
        <div class="nav-card">
            <div class="nav-card-title">{title}</div>
            <div class="nav-card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_card(row: dict, btn_key: str):
    level = normalize_risk_level(row.get("level", "Low"))
    style = RISK_STYLES.get(level, RISK_STYLES["Low"])

    card_html = dedent(f"""
    <div class="alert-box" style="
        background:{style['bg']};
        border:2px solid {style['border']};
    ">
        <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start;">
            <div style="flex:1;">
                <div style="font-size:16px; font-weight:900; color:{style['text']};">
                    📷 {row.get('image_name','')}
                </div>
                <div style="font-size:13px; color:{style['text']}; opacity:.85; margin-top:4px;">
                    🕒 {row.get('created_at','')}
                </div>
                <div style="font-size:14px; color:{style['text']}; margin-top:10px;">
                    <b>Camera:</b> {row.get('camera_id','-')}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Waste:</b> {row.get('waste_type','')}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Conf:</b> {float(row.get('conf',0)):.2f}
                </div>
                <div style="font-size:13px; color:{style['text']}; margin-top:8px; opacity:.95;">
                    {row.get('message','')}
                </div>
            </div>
            <div style="
                background:{style['badge']};
                color:white;
                padding:8px 14px;
                border-radius:999px;
                font-weight:900;
                font-size:12px;
                white-space:nowrap;
            ">
                {level}
            </div>
        </div>
    </div>
    """)
    st.markdown(card_html, unsafe_allow_html=True)

    if st.button("View Details", key=btn_key):
      st.session_state["selected_alert_id"] = int(row["alert_id"])
      st.session_state["page"] = "Alerts"
      st.session_state["scroll_to_alert_details"] = True
      st.rerun()



def set_page(page_name: str):
    st.session_state["page"] = page_name
    st.rerun()


def alert_message_for_types(waste_types: list[str], level: str) -> str:
    unique_types = sorted(set(waste_types))
    if not unique_types:
        return "No pollution detected."

    if level == "Multiple Danger":
        return f"Multiple pollution types detected: {', '.join(unique_types)}"

    if len(unique_types) == 1:
        return f"Pollution detected in the image ({unique_types[0]})"

    return f"Pollution detected in the image ({', '.join(unique_types)})"


def safe_file_signature(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    return f"{uploaded_file.name}_{len(raw)}"

# 🚀 INIT

init_db()

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "selected_alert_id" not in st.session_state:
    st.session_state["selected_alert_id"] = None

if "last_processed_files" not in st.session_state:
    st.session_state["last_processed_files"] = []

if "last_created_alert_id" not in st.session_state:
    st.session_state["last_created_alert_id"] = None
# ================= LOGIN PAGE =================
if not st.session_state["logged_in"]:

    with st.form("login_form"):
        st.markdown(
            """
            <div class="login-logo">🌍</div>
            <div class="login-title">EcoEyes Login</div>
            <div class="login-subtitle">Sign in to access the dashboard</div>
            <div class="login-divider">──────── 🍃 ────────</div>
            """,
            unsafe_allow_html=True
        )

        username = st.text_input("Username", placeholder="👤  Enter your username")
        password = st.text_input("Password", placeholder="🔒  Enter your password", type="password")

        remember = st.checkbox("Remember me")

        submitted = st.form_submit_button("↪  Login")

        if submitted:
            if username == "admin" and password == "1234":
                st.session_state["logged_in"] = True
                st.session_state["page"] = "Home"
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown(
            """
            <div class="login-footer">
                🍃 © 2026 EcoEyes. All rights reserved. 🍃
            </div>
            """,
            unsafe_allow_html=True
        )

    st.stop()
# ═══════════════════════════════════════════════════════════════════════════
# 🌍  Header + TOP BAR
# ═══════════════════════════════════════════════════════════════════════════
# ================= HEADER =================

st.markdown(
    """
    <div class="topbar">
        <p class="brand-title">🌍 EcoEyes</p>
        <p class="brand-subtitle">Smart Waste Monitoring Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================= NAV BAR =================

st.markdown(
    """
    <div class="navbar-title">
        Environmental Monitoring • Detection • Alerts • Reports
    </div>
    """,
    unsafe_allow_html=True,
)

nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(6)

with nav1:
    if st.button("Home", key="nav_home"):
        set_page("Home")

with nav2:
    if st.button("Image Analysis", key="nav_analysis"):
        set_page("Image Analysis")

with nav3:
    if st.button("Alerts", key="nav_alerts"):
        set_page("Alerts")

with nav4:
    if st.button("Reports", key="nav_reports"):
        set_page("Reports")

with nav5:
    if st.button("Map", key="nav_map"):
        set_page("Map")

with nav6:
    if st.button("Settings", key="nav_settings"):
        set_page("Settings")


#*********
# 🏠 HOME
#*********
if st.session_state["page"] == "Home":
    total_img, total_det, total_alerts, total_runs = db_get_total_stats()

    # ================= MAIN MEASURES =================
    st.markdown('<div class="section-title"></div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Total Images Processed", total_img)
        st.caption("All stored uploaded and camera images")

    with m2:
        st.metric("Total Detections", total_det)
        st.caption("All detected objects from YOLO")

    with m3:
        st.metric("Total Alerts", total_alerts)
        st.caption("Generated alert records")

    with m4:
        st.metric("System Runs", total_runs)
        st.caption("Upload and camera processing runs")

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= MAIN NAVIGATION =================
    st.markdown('<div class="section-title"></div>', unsafe_allow_html=True)

    left, center, right = st.columns([1, 2.5, 1])

        

    with center:

        def card(title, desc, key, page):
            c1, c2 = st.columns([5, 1])

            with c1:
                st.markdown(f"""
                <div class="nav-card">
                    <div class="nav-card-title">{title}</div>
                    <div class="nav-card-text">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown("<div style='height:34px;'></div>", unsafe_allow_html=True)
                if st.button("Open", key=key):
                    set_page(page)

        card("📤 Image Analysis",
             "Upload image files, run waste detection, save results, and create alerts directly.",
             "home_analysis", "Image Analysis")

        card("🚨 Alerts",
             "Review all pollution alerts and inspect full alert details with original and detected images.",
             "home_alerts", "Alerts")

        card("📊 Reports",
             "See analytics, distributions, trend charts, and summary tables of stored detections.",
             "home_reports", "Reports")

        card("🗺️ Map",
             "View camera locations and monitoring points on the interactive map.",
             "home_map", "Map")

        card("⚙️ Settings",
             "Refresh data, clear saved records safely, and export the detection records as CSV.",
             "home_settings", "Settings")

    st.markdown("<br>", unsafe_allow_html=True)

#*********
# 📤 IMAGE ANALYSIS
#*********

elif st.session_state["page"] == "Image Analysis":
    st.header("📤 Image Analysis")
    st.caption("Upload images, analyze them, save the detections, and create alerts.")

    top1, top2 = st.columns(2)

    ctrl1, ctrl2, ctrl3 = st.columns([1.3, 1, 1])

    with ctrl1:
            conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.25, 0.05)

    with ctrl2:
            show_images = st.checkbox("Show Original/Detected", value=True)

    with ctrl3:
            img_width = st.selectbox("Image Width", [350, 420, 500, 600], index=2)

    uploaded_files = st.file_uploader(
            "Upload Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

    

    if uploaded_files:
     st.markdown(
        f"""
        <div class="status-banner">
            {len(uploaded_files)} image(s) selected and ready for analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_signatures = [safe_file_signature(f) for f in uploaded_files]

    if current_signatures != st.session_state.get("last_processed_files", []):
        run_id = None
        saved_count = 0
        latest_alert_id = None

        for uploaded in uploaded_files:
            image = Image.open(uploaded).convert("RGB")
            img_np = np.array(image)

            results = model.predict(source=img_np, conf=conf_threshold, verbose=False)
            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                st.warning(f"No detection found in: {uploaded.name} (not saved in DB)")
                continue

            if run_id is None:
                run_id = db_create_run(
                    source_type="upload_multi",
                    conf=conf_threshold,
                    iou=None,
                    model_path=str(MODEL_PATH)
                )

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_name = f"{stamp}_{Path(uploaded.name).stem}.jpg"

            orig_path = UPLOAD_DIR / safe_name
            det_path = DET_DIR / safe_name

            image.save(orig_path)

            det_img = Image.fromarray(r.plot()[:, :, ::-1])
            det_img.save(det_path)

            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy().astype(float)
            xyxy = r.boxes.xyxy.cpu().numpy().astype(float)

            waste_types = []
            boxes_rows = []

            for cid, cf, (x1, y1, x2, y2) in zip(cls_ids, confs, xyxy):
                wt_raw = CLASS_NAMES.get(int(cid), f"class_{int(cid)}")
                wt = normalize_waste_name(wt_raw)
                waste_types.append(wt)
                boxes_rows.append((int(cid), wt, float(cf), float(x1), float(y1), float(x2), float(y2)))

            total_det = int(len(waste_types))
            summary = pd.Series(waste_types).value_counts()
            summary_rows = [(str(wt), int(cnt)) for wt, cnt in summary.items()]

            image_id = db_add_image(
                run_id=run_id,
                image_name=uploaded.name,
                image_path=str(orig_path),
                detected_path=str(det_path),
                total_detections=total_det
            )

            db_add_detection_summary(image_id, summary_rows)
            db_add_boxes(image_id, boxes_rows)

            level, top_waste = pick_alert_level(waste_types)
            max_conf = float(np.max(confs)) if len(confs) else 0.0
            message = alert_message_for_types(waste_types, level)

            alert_id = db_add_alert(
                image_id=image_id,
                level=level,
                waste_type=top_waste,
                conf=max_conf,
                message=message
            )

            latest_alert_id = int(alert_id)
            saved_count += 1

            st.markdown(
                f"""
                <div class="success-banner">
                    ✅ Detection completed successfully for <b>{uploaded.name}</b><br>
                    Alert created successfully. <b>Alert ID = {alert_id}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
            result_info = pd.DataFrame({
                "Waste Type": waste_types,
                "Confidence": [round(x, 2) for x in confs],
                "Risk Level": [
                    ("Multiple Danger" if level == "Multiple Danger"
                     else RISK_MAP.get(normalize_waste_name(w), "Low"))
                    for w in waste_types
                ]
            })

            if show_images:
                left, right = st.columns(2)
                with left:
                    st.image(image, caption=f"Original - {uploaded.name}", width=img_width)
                with right:
                    st.image(det_img, caption=f"Detected - {uploaded.name}", width=img_width)

            st.dataframe(result_info, use_container_width=True)
            st.markdown("---")

        if saved_count > 0:
            st.session_state["last_processed_files"] = current_signatures
            st.session_state["latest_alert_for_navigation"] = latest_alert_id

    else:
        st.info("This uploaded image was already analyzed. Upload a new image to run analysis again.")
    # =====================
    # Camera settings
    # =====================
    USERNAME = "admin"
    PASSWORD = "Eco162534"
    IP_ADDRESS = "192.168.0.110"
    RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:554/h264Preview_01_main"

    # =====================
    # Load model
    # =====================
    @st.cache_resource
    def load_camera_model():
        return YOLO("weights/best.pt")

    camera_model = load_camera_model()

    # =====================
    # Smart detection settings
    # =====================
    ANALYZE_EVERY = 2
    CONF_TH = 0.20

    CONFIRM_HITS = 6
    CANDIDATE_TTL = 20
    ALERT_MEMORY_TTL = 1800

    DIST_THRESHOLD = 40
    AREA_DIFF_THRESHOLD = 0.20

    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    if "candidates" not in st.session_state:
        st.session_state.candidates = []

    if "alerted_objects" not in st.session_state:
        st.session_state.alerted_objects = []

    if "last_analysis_time" not in st.session_state:
        st.session_state.last_analysis_time = 0

    if "last_display_frame" not in st.session_state:
        st.session_state.last_display_frame = None


    def cleanup_memory():
        now = time.time()

        st.session_state.candidates = [
            c for c in st.session_state.candidates
            if now - c["last_seen"] <= CANDIDATE_TTL
        ]

        st.session_state.alerted_objects = [
            a for a in st.session_state.alerted_objects
            if now - a["alerted_at"] <= ALERT_MEMORY_TTL
        ]


    def same_object(obj1, obj2):
        if obj1["cls"] != obj2["cls"]:
            return False

        dist = math.sqrt((obj1["cx"] - obj2["cx"]) ** 2 + (obj1["cy"] - obj2["cy"]) ** 2)

        area1 = max(obj1["area"], 1)
        area2 = max(obj2["area"], 1)
        area_diff = abs(area1 - area2) / max(area1, area2)

        return dist <= DIST_THRESHOLD and area_diff <= AREA_DIFF_THRESHOLD


    def find_match(obj, memory_list):
        for item in memory_list:
            if same_object(obj, item):
                return item
        return None


    def update_candidate(candidate, obj):
        candidate["cx"] = int((candidate["cx"] + obj["cx"]) / 2)
        candidate["cy"] = int((candidate["cy"] + obj["cy"]) / 2)
        candidate["area"] = int((candidate["area"] + obj["area"]) / 2)
        candidate["last_seen"] = time.time()
        candidate["hits"] += 1


    # =====================
    # UI
    # =====================
    st.markdown("### 📹 Live Camera Detection")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Camera"):
            st.session_state.camera_on = True

    with col2:
        if st.button("Stop Camera"):
            st.session_state.camera_on = False

    frame_placeholder = st.empty()
    alert_placeholder = st.empty()

    # =====================
    # Camera loop
    # =====================
    if st.session_state.camera_on:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            st.error("Failed to open camera stream.")
        else:
            while st.session_state.camera_on:
                ret, frame = cap.read()

                if not ret:
                    st.warning("Failed to read frame from camera.")
                    time.sleep(1)
                    continue

                current_time = time.time()

                if st.session_state.last_display_frame is None:
                    st.session_state.last_display_frame = frame.copy()

                if current_time - st.session_state.last_analysis_time >= ANALYZE_EVERY:
                    st.session_state.last_analysis_time = current_time
                    cleanup_memory()

                    try:
                        results = camera_model(frame, conf=CONF_TH, verbose=False)
                        r = results[0]
                    except Exception as e:
                        st.warning(f"Inference error: {e}")
                        st.session_state.last_display_frame = frame.copy()
                        continue

                    st.session_state.last_display_frame = r.plot()

                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                        boxes_cls = r.boxes.cls.cpu().numpy()
                        boxes_conf = r.boxes.conf.cpu().numpy()

                        for box, cls_id, conf in zip(boxes_xyxy, boxes_cls, boxes_conf):
                            x1, y1, x2, y2 = map(int, box)
                            cls_id = int(cls_id)

                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            area = max((x2 - x1) * (y2 - y1), 1)

                            obj = {
                                "cls": cls_id,
                                "cx": cx,
                                "cy": cy,
                                "area": area,
                            }

                            matched_alert = find_match(obj, st.session_state.alerted_objects)
                            if matched_alert is not None:
                                continue

                            matched_candidate = find_match(obj, st.session_state.candidates)

                            if matched_candidate is not None:
                                update_candidate(matched_candidate, obj)

                                if matched_candidate["hits"] >= CONFIRM_HITS:
                                    alert_placeholder.success(
                                        f"ALERT: Confirmed waste detected | class={cls_id} | center=({matched_candidate['cx']}, {matched_candidate['cy']})"
                                    )
                                    waste_name = camera_model.names[cls_id]
                                    confidence = float(conf)

                                    if waste_name.lower() == "glass":
                                        level = "HIGH"
                                    elif waste_name.lower() in ["plastic", "metal"]:
                                        level = "MEDIUM"
                                    else:
                                        level = "LOW"

                                    message = f"{waste_name} detected with confidence {confidence:.2f}"

                                    # Save waste snapshot
                                    os.makedirs("data/alert_snapshots", exist_ok=True)

                                    x1, y1, x2, y2 = map(int, box)
                                    waste_crop = frame[y1:y2, x1:x2]

                                    snapshot_name = f"{waste_name}_{int(time.time())}.jpg"
                                    snapshot_path = os.path.join("data/alert_snapshots", snapshot_name)

                                    cv2.imwrite(snapshot_path, waste_crop)

                                    # Save to DB
                                    image_id = db_add_image(
                                        run_id=1,
                                        image_name=snapshot_name,
                                        image_path=snapshot_path,
                                        detected_path=snapshot_path,
                                        total_detections=1
                                    )
                                    db_add_alert(
                                        image_id=image_id,
                                        level=level,
                                        waste_type=waste_name,
                                        conf=confidence,
                                        message=message
                                    )
                                    st.session_state.alerted_objects.append({
                                        "cls": matched_candidate["cls"],
                                        "cx": matched_candidate["cx"],
                                        "cy": matched_candidate["cy"],
                                        "area": matched_candidate["area"],
                                        "alerted_at": time.time()
                                    })

                                    if matched_candidate in st.session_state.candidates:
                                        st.session_state.candidates.remove(matched_candidate)

                            else:
                                st.session_state.candidates.append({
                                    "cls": cls_id,
                                    "cx": cx,
                                    "cy": cy,
                                    "area": area,
                                    "hits": 1,
                                    "last_seen": time.time()
                                })

                    else:
                        st.session_state.last_display_frame = frame.copy()

                display_frame = st.session_state.last_display_frame
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                frame_placeholder.image(
                    display_frame,
                    caption="EcoEyes Live Detection",
                    width="stretch"
                )

                time.sleep(0.05)

            cap.release()
            current_signatures = [safe_file_signature(f) for f in uploaded_files]

            # تحليل تلقائي فقط إذا الملفات جديدة
            if current_signatures != st.session_state.get("last_processed_files", []):
                run_id = None
                saved_count = 0
                latest_alert_id = None

                for uploaded in uploaded_files:
                    image = Image.open(uploaded).convert("RGB")
                    img_np = np.array(image)

                    results = model.predict(source=img_np, conf=conf_threshold, verbose=False)
                    r = results[0]

                    if r.boxes is None or len(r.boxes) == 0:
                        st.warning(f"No detection found in: {uploaded.name} (not saved in DB)")
                        continue

                    if run_id is None:
                        run_id = db_create_run(
                            source_type="upload_multi",
                            conf=conf_threshold,
                            iou=None,
                            model_path=str(MODEL_PATH)
                        )

                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    safe_name = f"{stamp}_{Path(uploaded.name).stem}.jpg"

                    orig_path = UPLOAD_DIR / safe_name
                    det_path = DET_DIR / safe_name

                    image.save(orig_path)

                    det_img = Image.fromarray(r.plot()[:, :, ::-1])
                    det_img.save(det_path)

                    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy().astype(float)
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(float)

                    waste_types = []
                    boxes_rows = []

                    for cid, cf, (x1, y1, x2, y2) in zip(cls_ids, confs, xyxy):
                        wt_raw = CLASS_NAMES.get(int(cid), f"class_{int(cid)}")
                        wt = normalize_waste_name(wt_raw)
                        waste_types.append(wt)
                        boxes_rows.append((int(cid), wt, float(cf), float(x1), float(y1), float(x2), float(y2)))

                    total_det = int(len(waste_types))
                    summary = pd.Series(waste_types).value_counts()
                    summary_rows = [(str(wt), int(cnt)) for wt, cnt in summary.items()]

                    image_id = db_add_image(
                        run_id=run_id,
                        image_name=uploaded.name,
                        image_path=str(orig_path),
                        detected_path=str(det_path),
                        total_detections=total_det
                    )

                    db_add_detection_summary(image_id, summary_rows)
                    db_add_boxes(image_id, boxes_rows)

                    level, top_waste = pick_alert_level(waste_types)
                    max_conf = float(np.max(confs)) if len(confs) else 0.0
                    message = alert_message_for_types(waste_types, level)

                    alert_id = db_add_alert(
                        image_id=image_id,
                        level=level,
                        waste_type=top_waste,
                        conf=max_conf,
                        message=message
                    )

                    latest_alert_id = int(alert_id)
                    st.session_state["last_created_alert_id"] = latest_alert_id
                    saved_count += 1

                    st.markdown(
                        f"""
                        <div class="success-banner">
                            ✅ Detection completed successfully for <b>{uploaded.name}</b><br>
                            Alert created successfully. <b>Alert ID = {alert_id}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    result_info = pd.DataFrame({
                        "Waste Type": waste_types,
                        "Confidence": [round(x, 2) for x in confs],
                        "Risk Level": [
                            ("Multiple Danger" if level == "Multiple Danger"
                            else RISK_MAP.get(normalize_waste_name(w), "Low"))
                            for w in waste_types
                        ]
                    })

                    if show_images:
                        left, right = st.columns(2)
                        with left:
                            st.image(image, caption=f"Original - {uploaded.name}", width=img_width)
                        with right:
                            st.image(det_img, caption=f"Detected - {uploaded.name}", width=img_width)

                    st.dataframe(result_info, use_container_width=True)
                    st.markdown("---")

                if saved_count > 0:
                    st.session_state["last_processed_files"] = current_signatures
                    st.session_state["latest_alert_for_navigation"] = latest_alert_id

        
        latest_alert_for_navigation = st.session_state.get("latest_alert_for_navigation")

        if latest_alert_for_navigation is not None:
            nav_col1, nav_col2 = st.columns([1, 3])

            with nav_col1:
                if st.button("Go to Alert Details", key="go_to_latest_alert_details"):
                  st.session_state["selected_alert_id"] = int(latest_alert_for_navigation)
                  st.session_state["page"] = "Alerts"
                  st.rerun()


            with nav_col2:
                st.markdown(
                    f"""
                    <div class="link-banner">
                        You can view the full details of this detection in the Alerts page using Alert ID <b>{latest_alert_for_navigation}</b>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════
# 🚨 ALERTS
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state["page"] == "Alerts":
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    st.header("🚨 Alert Details")
    st.caption("Review pollution alerts, filter them, and inspect all details for each record.")

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

    with filter_col1:
        level_filter = st.selectbox("Risk Level", ["All", "High", "Medium", "Low", "Multiple Danger"])

    alerts_df = db_get_alerts_filtered(level=level_filter, limit=50)

    st.subheader("Recent Alert Cards")
    if alerts_df.empty:
        st.info("No alerts matched the selected filters.")
    else:
        for i, row in alerts_df.iterrows():
            render_alert_card(row.to_dict(), btn_key=f"alerts_page_{row['alert_id']}_{i}")

    st.markdown("---")

    full_alerts_df = db_get_recent_alerts(limit=100)
    selected = st.session_state.get("selected_alert_id")

    if full_alerts_df.empty:
        st.info("No alert details available.")
    else:
        options = full_alerts_df["alert_id"].tolist()
        default_idx = options.index(selected) if selected in options else 0

        chosen_alert_id = st.selectbox("Select Alert ID", options, index=default_idx)
        st.session_state["selected_alert_id"] = int(chosen_alert_id)

        row = db_get_alert_by_id(int(chosen_alert_id))

        st.markdown('<div id="alert-details-section"></div>', unsafe_allow_html=True)
        st.subheader("Selected Alert Details")

        if st.session_state.get("scroll_to_alert_details", False):
            components.html(
                """
                <script>
                    const target = window.parent.document.getElementById("alert-details-section");
                    if (target) {
                        target.scrollIntoView({behavior: "smooth", block: "start"});
                    }
                </script>
                """,
                height=0,
            )
            st.session_state["scroll_to_alert_details"] = False

        if not row:
            st.error("Alert not found.")
        else:
            level = normalize_risk_level(row.get("level", "Low"))
            style = RISK_STYLES.get(level, RISK_STYLES["Low"])
            image_id = int(row["image_id"])
            boxes_df = db_get_boxes_for_image(image_id)

            st.markdown(
                f"""
                <div class="alert-box" style="
                    background:{style['bg']};
                    border:2px solid {style['border']};
                ">
                    <div style="display:flex; justify-content:space-between; gap:14px; align-items:flex-start;">
                        <div style="flex:1;">
                            <div style="font-size:18px; font-weight:900; color:{style['text']};">
                                📷 {row.get('image_name','')}
                            </div>
                            <div style="margin-top:6px; color:{style['text']}; opacity:.9;">
                                🕒 {row.get('created_at','')}
                            </div>
                            <div style="margin-top:10px; color:{style['text']}; line-height:1.9;">
                                <b>Camera ID:</b> {row.get('camera_id')}<br>
                                <b>Waste Type:</b> {row.get('waste_type','')}<br>
                                <b>Risk Level:</b> {level}<br>
                                <b>Confidence:</b> {float(row.get('conf',0)):.2f}<br>
                                <b>Total Detections:</b> {int(row.get('total_detections', 0))}
                            </div>
                            <div style="margin-top:10px; color:{style['text']}; opacity:.96;">
                                {row.get('message','')}
                            </div>
                        </div>
                        <div style="
                            background:{style['badge']};
                            color:white;
                            padding:8px 14px;
                            border-radius:999px;
                            font-weight:900;
                            font-size:12px;
                            white-space:nowrap;
                        ">
                            {level}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            colL, colR = st.columns(2)

            with colL:
                st.markdown(
                    """
                    <div class="section-card">
                        <div class="card-header-title">Original Image</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                p = row.get("image_path")
                if p and Path(p).exists():
                    st.image(Image.open(p), use_container_width=True)
                else:
                    st.info("Original image path not available.")

            with colR:
                st.markdown(
                    """
                    <div class="section-card">
                        <div class="card-header-title">Detected Image</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                p2 = row.get("detected_path")
                if p2 and Path(p2).exists():
                    st.image(Image.open(p2), use_container_width=True)
                else:
                    st.info("Detected image path not available.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Detection Breakdown")

            if boxes_df.empty:
                st.info("")
            else:
                detail_df = boxes_df.copy()
                detail_df["conf"] = detail_df["conf"].round(2)
                detail_df = detail_df.rename(columns={
                    "waste_type": "Waste Type",
                    "conf": "Confidence",
                    "x1": "x1",
                    "y1": "y1",
                    "x2": "x2",
                    "y2": "y2",
                    "class_id": "Class ID",
                    "created_at": "Created At",
                })
                st.dataframe(detail_df, use_container_width=True)
    st.markdown(
    """
    <style>
    .scroll-top-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #224C43;
        color: white;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 22px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        z-index: 9999;
    }
    </style>

    <a href="#top" class="scroll-top-btn">↑</a>
    """,
    unsafe_allow_html=True
    )
# ═══════════════════════════════════════════════════════════════════════════
# 📊 REPORTS
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "Reports":
    st.header("📊 Reports")
    st.caption("Charts and analytics of detections, alerts, and pollution activity.")

    dist = db_get_waste_distribution()
    timeline_df = db_get_detection_timeline_last10()
    risk_df = db_get_risk_distribution()
    alerts_table = db_get_alerts_table(limit=100)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= ROW 1 =================
    row1_col1, row1_col2 = st.columns(2, gap="large")

    with row1_col1:
        with st.container(border=True):
            st.markdown("### Waste Distribution")
            st.caption("Bar chart of detected waste categories.")

            if dist.empty:
                st.info("No waste distribution data available.")
            else:
                if PLOTLY_OK:
                    fig_bar = px.bar(
                        dist,
                        x="waste_type",
                        y="count",
                        color="waste_type",
                        color_discrete_sequence=GREEN_COLORS,
                        title=""
                    )
                    fig_bar.update_layout(
                        template="plotly_white",
                        height=360,
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor="white",
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.dataframe(dist, use_container_width=True)

    with row1_col2:
        with st.container(border=True):
            st.markdown("### Waste Ratio")
            st.caption("Pie chart of pollution categories.")

            if dist.empty:
                st.info("No waste ratio data available.")
            else:
                if PLOTLY_OK:
                    fig_pie = px.pie(
                        dist,
                        values="count",
                        names="waste_type",
                        color_discrete_sequence=GREEN_COLORS,
                        title=""
                    )
                    fig_pie.update_layout(
                        template="plotly_white",
                        height=360,
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor="white",
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.dataframe(dist, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= ROW 2 =================
    row2_col1, row2_col2 = st.columns(2, gap="large")

    with row2_col1:
        with st.container(border=True):
            st.markdown("### Detection Timeline")
            st.caption("Line chart for the latest detection activity.")

            if timeline_df.empty:
                st.info("No timeline data available.")
            else:
                timeline_sorted = timeline_df.sort_values("date")
                if PLOTLY_OK:
                    fig_line = go.Figure()
                    fig_line.add_trace(
                        go.Scatter(
                            x=timeline_sorted["date"],
                            y=timeline_sorted["detection_count"],
                            mode="lines+markers",
                            line=dict(color="#2BA876", width=3),
                            marker=dict(size=8),
                            name="Detections"
                        )
                    )
                    fig_line.update_layout(
                        template="plotly_white",
                        title="",
                        height=360,
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor="white",
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.line_chart(timeline_sorted.set_index("date"))

    with row2_col2:
        with st.container(border=True):
            st.markdown("### Alert Risk Level")
            st.caption("Donut chart showing alert level distribution.")

            if risk_df.empty:
                st.info("No risk-level data available.")
            else:
                if PLOTLY_OK:
                    fig_donut = px.pie(
                        risk_df,
                        values="count",
                        names="level",
                        hole=0.55,
                        color_discrete_sequence=GREEN_COLORS,
                        title=""
                    )
                    fig_donut.update_layout(
                        template="plotly_white",
                        height=360,
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor="white",
                        plot_bgcolor="white"
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                else:
                    st.dataframe(risk_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= TABLE =================
    with st.container(border=True):
        st.markdown("### Alerts Summary Table")
        st.caption("All recent alert records in table form.")

        if alerts_table.empty:
            st.info("No alert records available.")
        else:
            st.dataframe(alerts_table, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 🗺️ Monitoring Map
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "Map":
    st.header("🗺️ Monitoring Map")
    st.caption("View camera locations and monitoring points.")

    m = folium.Map(location=[18.2465, 42.5117], zoom_start=13)

    conn = get_conn()

    cameras = conn.execute("""
    SELECT camera_id, location_name, map_x, map_y
    FROM cameras
    """).fetchall()

    for cam in cameras:
        cam_id, name, x, y = cam

        folium.Marker(
             [18.2465, 42.5117],  
             popup=f"Camera {cam_id} - {name}",
             tooltip=f"Camera {cam_id}",
             icon=folium.Icon(color="green", icon="camera", prefix="fa")
        ).add_to(m)

    st_folium(m, width=None, height=500)
# ═══════════════════════════════════════════════════════════════════════════
# ⚙️ SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state["page"] == "Settings":
    st.header("⚙️ Settings")
    st.caption("Manage data refresh, database maintenance, and record export.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="section-card">
                <div class="card-header-title">Refresh Data</div>
                <div class="card-header-sub">Reload the dashboard and refresh all current values.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔄 Refresh", key="settings_refresh"):
            st.rerun()

    with c2:
        st.markdown(
            """
            <div class="section-card">
                <div class="card-header-title">Clear Database</div>
                <div class="card-header-sub">Delete all stored runs, images, detections, and alerts.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        confirm = st.checkbox("I confirm deleting all data", key="settings_confirm_delete")
        if st.button("🗑️ Delete All Data", key="settings_delete") and confirm:
            with get_conn() as conn:
                conn.execute("DELETE FROM alerts")
                conn.execute("DELETE FROM detection_boxes")
                conn.execute("DELETE FROM detections")
                conn.execute("DELETE FROM images")
                conn.execute("DELETE FROM runs")
                conn.commit()
            st.success("All database records were deleted successfully.")
            st.session_state["selected_alert_id"] = None
            st.session_state["last_created_alert_id"] = None
            st.session_state["last_processed_files"] = []

    with c3:
        st.markdown(
            """
            <div class="section-card">
                <div class="card-header-title">Export CSV</div>
                <div class="card-header-sub">Export detection box records for reporting or documentation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📥 Prepare CSV Export", key="settings_export"):
            with get_conn() as conn:
                df = pd.read_sql_query("SELECT * FROM detection_boxes ORDER BY created_at DESC", conn)
            if df.empty:
                st.info("No data available to export.")
            else:
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False),
                    file_name=f"ecoeyes_boxes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="settings_download_csv"
                )


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f"""
    <div class="footer-box">
        <p style="margin-bottom:6px;"><b style="color:{COLORS['primary']};">🌍 EcoEyes</b> - Smart Waste Monitoring System</p>
        <p style="font-size:12px; margin:0;">© EcoEyes Team</p>
    </div>
    """,
    unsafe_allow_html=True,
)
