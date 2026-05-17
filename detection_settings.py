"""
detection_settings.py  –  RoadSight Detection Control Panel
============================================================
A PySide6 dialog that lets the user:
  • Enable / disable individual CV models before running detection
  • Adjust mask opacity (alpha) per model with a live slider
  • Adjust bounding-box / contour opacity per model

Settings are written to  <BASE_DIR>/detection_settings.json
detect.py reads that file at startup and overrides MODEL_CONFIGS accordingly.

Usage (standalone test):
    python detection_settings.py

Usage (integrated call from app.py):
    from detection_settings import show_detection_settings
    show_detection_settings(parent=window)
"""

import json
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "detection_settings.json")

# ---------------------------------------------------------------------------
# Model metadata
# Must stay in sync with MODEL_CONFIGS in detect.py
# ---------------------------------------------------------------------------

MODEL_META = [
    {
        "name": "footpath",
        "label": "Footpath",
        "description": "Sidewalk / footpath segmentation",
        "color": "#00ff00",   # green  (BGR 0,255,0  → #00ff00)
        "default_alpha": 0.40,
        "has_mask": True,
    },
    {
        "name": "road_evidence",
        "label": "Road Evidence",
        "description": "Crosswalk markings · Lane markings · Stop lines",
        "color": "#ff0000",   # red (dominant lane_marking colour)
        "default_alpha": 0.50,
        "has_mask": True,
    },
    {
        "name": "trunk",
        "label": "Tree Trunks",
        "description": "Tree trunk detection",
        "color": "#ffa500",   # orange
        "default_alpha": 0.45,
        "has_mask": True,
    },
    {
        "name": "pole",
        "label": "Poles",
        "description": "Street / utility pole detection",
        "color": "#ff00ff",   # purple
        "default_alpha": 0.45,
        "has_mask": True,
    },
    {
        "name": "traffic_light",
        "label": "Traffic Lights",
        "description": "Traffic light bounding-box detection",
        "color": "#ffff00",   # cyan in BGR = yellow in RGB display
        "default_alpha": 0.45,
        "has_mask": False,
    },
    {
        "name": "road_barrier",
        "label": "Road Barriers",
        "description": "Road barrier / fence detection",
        "color": "#0080ff",   # light blue
        "default_alpha": 0.45,
        "has_mask": True,
    },
    {
        "name": "bike_lane",
        "label": "Bike Lanes",
        "description": "Dedicated bicycle lane segmentation",
        "color": "#8000ff",   # violet
        "default_alpha": 0.45,
        "has_mask": True,
    },
    {
        "name": "traffic_sign",
        "label": "Traffic Signs",
        "description": "Speed limits · Stop signs · Warnings · Parking etc.",
        "color": "#ff8000",   # orange-like
        "default_alpha": 0.45,
        "has_mask": False,
    },
]


# ---------------------------------------------------------------------------
# Settings I/O
# ---------------------------------------------------------------------------

def load_settings() -> dict:
    """
    Return persisted settings dict.
    Falls back to defaults if file is missing / corrupt.
    """
    defaults = {
        m["name"]: {"enabled": True, "alpha": m["default_alpha"]}
        for m in MODEL_META
    }

    if not os.path.exists(SETTINGS_FILE):
        return defaults

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)

        # Merge: keep defaults for any model not yet in the file
        for name, vals in defaults.items():
            if name not in saved:
                saved[name] = vals
            else:
                # Ensure both keys exist
                saved[name].setdefault("enabled", True)
                saved[name].setdefault("alpha", vals["alpha"])

        return saved

    except Exception:
        return defaults


def save_settings(settings: dict) -> None:
    """Write settings dict to JSON file."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


# ---------------------------------------------------------------------------
# Colour swatch widget
# ---------------------------------------------------------------------------

class ColorSwatch(QWidget):
    """Small filled rounded rectangle showing the model's overlay colour."""

    def __init__(self, hex_color: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 22)
        self._color = QColor(hex_color)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor("#555555"), 1))
        p.setBrush(QBrush(self._color))
        p.drawRoundedRect(2, 2, 18, 18, 4, 4)


# ---------------------------------------------------------------------------
# Per-model row widget
# ---------------------------------------------------------------------------

SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 5px;
    background: #cccccc;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: white;
    border: 1px solid #888;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #4a90d9;
    border-radius: 3px;
}
QSlider::groove:horizontal:disabled {
    background: #e0e0e0;
}
QSlider::handle:horizontal:disabled {
    background: #cccccc;
    border-color: #bbb;
}
"""

CHECKBOX_STYLE = """
QCheckBox {
    font-size: 13px;
    font-weight: bold;
    color: #1a1a1a;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #aaa;
    background: white;
}
QCheckBox::indicator:checked {
    background: #4a90d9;
    border-color: #357abd;
    image: none;
}
"""


class ModelRow(QFrame):
    """One row in the control panel for a single model."""

    def __init__(self, meta: dict, settings: dict, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.name = meta["name"]

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: white; border-radius: 10px; border: 1px solid #dde3ed; }"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(6)

        # ── Row 1: swatch + checkbox + description ────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)

        self.swatch = ColorSwatch(meta["color"])
        top.addWidget(self.swatch)

        self.checkbox = QCheckBox(meta["label"])
        self.checkbox.setStyleSheet(CHECKBOX_STYLE)
        self.checkbox.setChecked(settings.get(self.name, {}).get("enabled", True))
        top.addWidget(self.checkbox)

        top.addStretch()

        desc = QLabel(meta["description"])
        desc.setStyleSheet("color: #666; font-size: 11px; border: none;")
        desc.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(desc)

        outer.addLayout(top)

        # ── Row 2: opacity slider ─────────────────────────────────────────
        slider_row = QHBoxLayout()
        slider_row.setSpacing(10)

        lbl_mask = QLabel("Opacity:")
        lbl_mask.setStyleSheet("color: #333; font-size: 11px; border: none;")
        lbl_mask.setFixedWidth(56)
        slider_row.addWidget(lbl_mask)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        saved_alpha = settings.get(self.name, {}).get("alpha", meta["default_alpha"])
        self.alpha_slider.setValue(int(round(saved_alpha * 100)))
        self.alpha_slider.setStyleSheet(SLIDER_STYLE)
        self.alpha_slider.setFixedHeight(22)
        slider_row.addWidget(self.alpha_slider, stretch=1)

        self.alpha_label = QLabel(f"{int(round(saved_alpha * 100))}%")
        self.alpha_label.setFixedWidth(36)
        self.alpha_label.setStyleSheet("color: #333; font-size: 11px; border: none;")
        self.alpha_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider_row.addWidget(self.alpha_label)

        outer.addLayout(slider_row)

        # ── Connections ───────────────────────────────────────────────────
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        self.checkbox.stateChanged.connect(self._on_toggle)

        # Apply initial enabled/disabled state
        self._set_enabled_ui(self.checkbox.isChecked())

    # ------------------------------------------------------------------

    def _on_alpha_changed(self, value: int):
        self.alpha_label.setText(f"{value}%")

    def _on_toggle(self, state):
        self._set_enabled_ui(bool(state))

    def _set_enabled_ui(self, enabled: bool):
        self.alpha_slider.setEnabled(enabled)
        self.alpha_label.setEnabled(enabled)
        opacity = 1.0 if enabled else 0.45
        self.setStyleSheet(
            f"QFrame {{ background: {'white' if enabled else '#f5f5f5'}; "
            f"border-radius: 10px; border: 1px solid #dde3ed; }}"
        )

    # ------------------------------------------------------------------

    def get_values(self) -> dict:
        return {
            "enabled": self.checkbox.isChecked(),
            "alpha": self.alpha_slider.value() / 100.0,
        }


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

DIALOG_STYLE = """
QDialog {
    background: #f0f4f9;
}
QPushButton {
    background: white;
    color: #1a1a1a;
    border: 1px solid #ccc;
    border-radius: 8px;
    font-size: 13px;
    padding: 6px 18px;
}
QPushButton:hover {
    background: #e8eef7;
    border-color: #4a90d9;
}
QPushButton#apply_btn {
    background: #4a90d9;
    color: white;
    border-color: #357abd;
    font-weight: bold;
}
QPushButton#apply_btn:hover {
    background: #357abd;
}
"""


class DetectionSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Settings")
        self.setMinimumWidth(560)
        self.setStyleSheet(DIALOG_STYLE)

        self._settings = load_settings()
        self._rows: list[ModelRow] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ── Header ────────────────────────────────────────────────────────
        header = QLabel("Detection Control Panel")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setStyleSheet("color: #1a1a1a;")
        root.addWidget(header)

        sub = QLabel(
            "Choose which models to run and adjust their overlay opacity.\n"
            "Settings are saved and applied on the next detection run."
        )
        sub.setStyleSheet("color: #555; font-size: 12px;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ccc;")
        root.addWidget(sep)

        # ── Quick toggles ─────────────────────────────────────────────────
        quick = QHBoxLayout()
        quick.setSpacing(8)

        enable_all_btn = QPushButton("Enable All")
        enable_all_btn.setFixedHeight(32)
        enable_all_btn.clicked.connect(lambda: self._set_all(True))
        quick.addWidget(enable_all_btn)

        disable_all_btn = QPushButton("Disable All")
        disable_all_btn.setFixedHeight(32)
        disable_all_btn.clicked.connect(lambda: self._set_all(False))
        quick.addWidget(disable_all_btn)

        reset_btn = QPushButton("Reset Defaults")
        reset_btn.setFixedHeight(32)
        reset_btn.clicked.connect(self._reset_defaults)
        quick.addWidget(reset_btn)

        quick.addStretch()
        root.addLayout(quick)

        # ── Scrollable model list ─────────────────────────────────────────
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background: transparent; }")

        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 6, 0)
        scroll_layout.setSpacing(8)

        for meta in MODEL_META:
            row = ModelRow(meta, self._settings)
            self._rows.append(row)
            scroll_layout.addWidget(row)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        root.addWidget(scroll_area, stretch=1)

        # ── Footer buttons ────────────────────────────────────────────────
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #ccc;")
        root.addWidget(sep2)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        footer.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedSize(100, 36)
        cancel_btn.clicked.connect(self.reject)
        footer.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply & Save")
        apply_btn.setObjectName("apply_btn")
        apply_btn.setFixedSize(130, 36)
        apply_btn.clicked.connect(self._apply)
        footer.addWidget(apply_btn)

        root.addLayout(footer)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_all(self, state: bool):
        for row in self._rows:
            row.checkbox.setChecked(state)

    def _reset_defaults(self):
        for row in self._rows:
            default_alpha = row.meta["default_alpha"]
            row.checkbox.setChecked(True)
            row.alpha_slider.setValue(int(round(default_alpha * 100)))

    def _apply(self):
        new_settings = {row.name: row.get_values() for row in self._rows}
        save_settings(new_settings)
        self.accept()


# ---------------------------------------------------------------------------
# Public helper for app.py
# ---------------------------------------------------------------------------

def show_detection_settings(parent=None) -> bool:
    """
    Open the dialog.
    Returns True if the user clicked 'Apply & Save', False if cancelled.
    """
    dlg = DetectionSettingsDialog(parent)
    return dlg.exec() == QDialog.Accepted


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = DetectionSettingsDialog()
    dlg.resize(580, 680)
    result = dlg.exec()
    print("Accepted:", result == QDialog.Accepted)
    if result == QDialog.Accepted:
        print("Saved settings:", json.dumps(load_settings(), indent=2))
    sys.exit(0)
