#!/usr/bin/env python3
"""
PPG Patient Monitor — MAX30100 + PyQt5

Real-time pulse oximetry / heart-rate monitor styled like a bedside clinical
patient monitor. Reads raw IR/RED samples over serial from an Arduino running
the matching sketch, does bandpass filtering + beat detection on the PC, and
draws a non-scrolling sweep waveform at 60 FPS.

Usage:
    python clinical_monitor.py --port COM5
    python clinical_monitor.py --simulate
"""

import argparse
import math
import random
import sys
import time
from collections import deque

import numpy as np
import serial
from serial.tools import list_ports
from scipy.signal import butter, sosfilt, sosfilt_zi

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


# --------------------------------------------------------------------------- #
#  Configuration defaults
# --------------------------------------------------------------------------- #
DEFAULT_FS = 100.0            # sensor sample rate (Hz)
DEFAULT_IR_THRESHOLD = 5000   # raw IR below this → "no finger"

WAVEFORM_SECONDS = 8.0        # sweep window length
SWEEP_FPS = 60
BLANK_BAR_FRACTION = 0.04     # width of the erasing bar, fraction of window
TRAIL_FADE = 0.55             # exponential fade factor behind the sweep

# Neon-cyan palette
COLOR_BG = "#000000"
COLOR_GRID = "#0a2a2e"
COLOR_WAVE = "#00ffff"
COLOR_TEXT = "#00ffff"
COLOR_ALARM = "#ff2040"
COLOR_OK = "#00ff88"


# --------------------------------------------------------------------------- #
#  Serial reader thread
# --------------------------------------------------------------------------- #
class SerialReader(QtCore.QThread):
    """Reads ASCII lines from the serial port and pushes samples into a deque.

    Emits status signals for connection changes; never raises to the GUI.
    """

    sample_ready = QtCore.pyqtSignal(int, int, int, int)   # ir, red, bpm, spo2
    status_changed = QtCore.pyqtSignal(str)
    connection_changed = QtCore.pyqtSignal(bool)

    def __init__(self, port, baud=115200, parent=None):
        super().__init__(parent)
        self._port = port
        self._baud = baud
        self._running = True
        self._serial = None

    def stop(self):
        self._running = False
        self.wait(1500)

    def _open(self):
        try:
            self._serial = serial.Serial(self._port, self._baud, timeout=1.0)
            self._serial.reset_input_buffer()
            self.status_changed.emit(f"CONNECTED {self._port}")
            self.connection_changed.emit(True)
            return True
        except Exception as exc:
            self.status_changed.emit(f"WAITING FOR {self._port}: {exc}")
            self.connection_changed.emit(False)
            self._serial = None
            return False

    def run(self):
        while self._running:
            if self._serial is None:
                if not self._open():
                    for _ in range(10):
                        if not self._running:
                            return
                        self.msleep(100)
                    continue

            try:
                line = self._serial.readline()
            except Exception:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
                self.connection_changed.emit(False)
                self.status_changed.emit("DISCONNECTED — RECONNECTING")
                continue

            if not line:
                continue

            try:
                text = line.decode("ascii", errors="ignore").strip()
            except Exception:
                continue

            if not text or text.startswith("#"):
                continue

            parts = text.split(",")
            if len(parts) != 4:
                continue

            try:
                ir = int(float(parts[0]))
                red = int(float(parts[1]))
                bpm = int(float(parts[2]))
                spo2 = int(float(parts[3]))
            except ValueError:
                continue

            if self._running:
                self.sample_ready.emit(ir, red, bpm, spo2)


# --------------------------------------------------------------------------- #
#  Simulated source (no hardware)
# --------------------------------------------------------------------------- #
class SimulatedSource(QtCore.QObject):
    """Generates a synthetic PPG waveform at ~100 Hz and emits sample_ready."""

    sample_ready = QtCore.pyqtSignal(int, int, int, int)

    def __init__(self, fs=DEFAULT_FS, parent=None):
        super().__init__(parent)
        self._fs = fs
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000 / fs))
        self._timer.timeout.connect(self._tick)
        self._t = 0.0
        self._hr = 72.0
        self._finger = True
        self._finger_toggle = 0.0
        self._finger_off_until = 0.0

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _tick(self):
        dt = 1.0 / self._fs
        self._t += dt

        self._hr += random.uniform(-0.05, 0.05)
        self._hr = max(55.0, min(110.0, self._hr))
        f = self._hr / 60.0

        phase = 2 * math.pi * f * self._t
        pulse = (
            0.60 * math.sin(phase)
            + 0.25 * math.sin(2 * phase - 0.4)
            + 0.10 * math.sin(3 * phase - 0.9)
            + 0.05 * math.sin(0.25 * 2 * math.pi * self._t)
        )

        self._finger_toggle += dt
        if self._finger and self._finger_toggle > 25.0:
            self._finger = False
            self._finger_off_until = self._t + 2.5
            self._finger_toggle = 0.0
        elif not self._finger and self._t > self._finger_off_until:
            self._finger = True
            self._finger_toggle = 0.0

        if self._finger:
            dc = 60000.0
            amp = 8000.0
            ir = dc + amp * pulse + random.gauss(0, 60)
            red = 45000.0 + 0.6 * amp * pulse + random.gauss(0, 60)
            bpm = int(round(self._hr))
            spo2 = int(round(97 + random.uniform(-1, 1)))
        else:
            ir = random.gauss(500, 80)
            red = random.gauss(500, 80)
            bpm = 0
            spo2 = 0

        self.sample_ready.emit(int(ir), int(red), int(bpm), int(spo2))


# --------------------------------------------------------------------------- #
#  Waveform widget — non-scrolling sweep with fading trail
# --------------------------------------------------------------------------- #
class SweepWaveform(pg.PlotWidget):
    """A bedside-monitor-style sweeping waveform."""

    def __init__(self, fs, seconds, parent=None):
        super().__init__(parent)
        self._fs = fs
        self._n = int(fs * seconds)
        self._seconds = seconds

        self._intensity = np.zeros(self._n, dtype=np.float32)
        self._values = np.zeros(self._n, dtype=np.float32)
        self._head = 0

        self.setBackground(COLOR_BG)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setAntialiasing(False)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.getPlotItem().getAxis("left").setPen(pg.mkPen(COLOR_GRID))
        self.getPlotItem().getAxis("bottom").setPen(pg.mkPen(COLOR_GRID))
        self.getPlotItem().getAxis("left").setTextPen(pg.mkPen(COLOR_GRID))
        self.getPlotItem().getAxis("bottom").setTextPen(pg.mkPen(COLOR_GRID))
        self.setXRange(0, self._n, padding=0)
        self.setYRange(-1.2, 1.2, padding=0)
        self.getPlotItem().setContentsMargins(0, 0, 0, 0)

        self._curve = pg.PlotCurveItem(pen=pg.mkPen(COLOR_WAVE, width=2))
        self.addItem(self._curve)

        self._blank = QtWidgets.QGraphicsRectItem()
        self._blank.setBrush(QtGui.QBrush(QtGui.QColor(COLOR_BG)))
        self._blank.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.addItem(self._blank)

        self._x = np.arange(self._n, dtype=np.float32)

    def push(self, value):
        head = self._head
        self._values[head] = value
        self._intensity[head] = 1.0

        bar = max(1, int(self._n * BLANK_BAR_FRACTION))
        for k in range(1, bar + 1):
            idx = (head + k) % self._n
            self._intensity[idx] = 0.0
            self._values[idx] = 0.0

        self._head = (head + 1) % self._n

    def advance_fade(self):
        self._intensity *= TRAIL_FADE
        if self._head > 0:
            self._intensity[self._head - 1] = 1.0

    def redraw(self):
        y = self._values.astype(np.float32).copy()
        y[self._intensity < 0.02] = np.nan
        self._curve.setData(self._x, y)

        bar = max(1, int(self._n * BLANK_BAR_FRACTION))
        x0 = self._head
        x1 = (self._head + bar) % self._n
        if x1 > x0:
            self._blank.setRect(QtCore.QRectF(x0, -1.2, bar, 2.4))
        else:
            self._blank.setRect(QtCore.QRectF(x0, -1.2, self._n - x0, 2.4))


# --------------------------------------------------------------------------- #
#  Readout panel
# --------------------------------------------------------------------------- #
class Readout(QtWidgets.QWidget):
    """Big numeric readouts for BPM and SpO2, plus status line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COLOR_BG};")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        self.hr_label = self._make_label("HR", 22, COLOR_TEXT)
        self.hr_value = self._make_label("--", 96, COLOR_TEXT, bold=True)
        self.hr_unit = self._make_label("BPM", 20, COLOR_TEXT)

        self.spo2_label = self._make_label("SpO₂", 22, COLOR_TEXT)
        self.spo2_value = self._make_label("--", 96, COLOR_TEXT, bold=True)
        self.spo2_unit = self._make_label("%", 20, COLOR_TEXT)

        self.status = self._make_label("INITIALISING", 16, COLOR_OK)

        layout.addWidget(self.hr_label)
        layout.addWidget(self.hr_value)
        layout.addWidget(self.hr_unit)
        layout.addSpacing(24)
        layout.addWidget(self.spo2_label)
        layout.addWidget(self.spo2_value)
        layout.addWidget(self.spo2_unit)
        layout.addStretch(1)
        layout.addWidget(self.status)

    @staticmethod
    def _make_label(text, size, color, bold=False):
        lbl = QtWidgets.QLabel(text)
        weight = "bold" if bold else "normal"
        lbl.setStyleSheet(
            f"color:{color}; font-size:{size}px; font-weight:{weight};"
            f"font-family:'DejaVu Sans Mono','Consolas',monospace;"
        )
        lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        return lbl

    def set_hr(self, bpm):
        self.hr_value.setText(f"{bpm:3d}" if bpm > 0 else "---")

    def set_spo2(self, spo2):
        self.spo2_value.setText(f"{spo2:2d}" if spo2 > 0 else "--")

    def set_status(self, text, ok=True):
        self.status.setText(text)
        self.status.setStyleSheet(
            f"color:{COLOR_OK if ok else COLOR_ALARM};"
            f"font-size:16px; font-family:'DejaVu Sans Mono',monospace;"
        )

    def set_alarm(self, on):
        color = COLOR_ALARM if on else COLOR_TEXT
        self.hr_value.setStyleSheet(
            f"color:{color}; font-size:96px; font-weight:bold;"
            f"font-family:'DejaVu Sans Mono','Consolas',monospace;"
        )
        self.spo2_value.setStyleSheet(
            f"color:{color}; font-size:96px; font-weight:bold;"
            f"font-family:'DejaVu Sans Mono','Consolas',monospace;"
        )


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #
class MonitorWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("PPG Patient Monitor — MAX30100")
        self.setStyleSheet(f"background:{COLOR_BG};")

        self.fs = float(args.fs)
        self.ir_threshold = int(args.ir_threshold)
        self.invert = bool(args.invert)

        nyq = 0.5 * self.fs
        low = 0.5 / nyq
        high = min(5.0 / nyq, 0.99)
        self._sos = butter(2, [low, high], btype="bandpass", output="sos")
        self._zi = sosfilt_zi(self._sos) * 0.0
        self._filter_initialised = False

        self._dc = None
        self._bpm = 0
        self._spo2 = 0

        self._finger_present = False
        self._alarm_phase = False
        self._last_finger_off = 0.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.wave = SweepWaveform(self.fs, WAVEFORM_SECONDS)
        self.readout = Readout()
        root.addWidget(self.wave, 7)
        root.addWidget(self.readout, 3)

        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setInterval(int(1000 / SWEEP_FPS))
        self._render_timer.timeout.connect(self._on_render)
        self._render_timer.start()

        self._alarm_timer = QtCore.QTimer(self)
        self._alarm_timer.setInterval(400)
        self._alarm_timer.timeout.connect(self._blink_alarm)
        self._alarm_timer.start()

        self.reader = None
        self.simulator = None
        self._start_source()

        QtWidgets.QShortcut(QtGui.QKeySequence("F11"), self, self._toggle_fs)
        QtWidgets.QShortcut(QtGui.QKeySequence("F"), self, self._toggle_fs)
        QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, self._exit_fs)
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, self.close)

        if not args.windowed:
            self.showFullScreen()
        else:
            self.showMaximized()

    def _start_source(self):
        if self.args.simulate:
            self.simulator = SimulatedSource(self.fs)
            self.simulator.sample_ready.connect(self._on_sample)
            self.simulator.start()
            self.readout.set_status("SIMULATION MODE", ok=True)
            return

        port = self.args.port or self._autodetect_port()
        if not port:
            self.readout.set_status("NO SERIAL PORT FOUND", ok=False)
            return

        self.reader = SerialReader(port)
        self.reader.sample_ready.connect(self._on_sample)
        self.reader.status_changed.connect(self._on_status)
        self.reader.connection_changed.connect(self._on_connection)
        self.reader.start()

    @staticmethod
    def _autodetect_port():
        ports = list(list_ports.comports())
        if not ports:
            return None
        preferred = []
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in ("arduino", "ch340", "cp210", "ft232",
                                       "usb serial", "usb-serial")):
                preferred.append(p.device)
        return preferred[0] if preferred else ports[0].device

    def _on_status(self, text):
        self.readout.set_status(text, ok="CONNECTED" in text)

    def _on_connection(self, connected):
        if not connected:
            self.readout.set_status("SENSOR DISCONNECTED", ok=False)

    def _on_sample(self, ir, red, bpm, spo2):
        finger = ir >= self.ir_threshold

        if finger != self._finger_present:
            if not finger:
                self._last_finger_off = time.time()
            self._finger_present = finger

        if not finger:
            self.wave.push(0.0)
            self._bpm = 0
            self._spo2 = 0
            self.readout.set_hr(0)
            self.readout.set_spo2(0)
            self.readout.set_alarm(True)
            self.readout.set_status("SENSOR DISCONNECTED", ok=False)
            return

        self.readout.set_alarm(False)

        if self._dc is None:
            self._dc = float(ir)
        else:
            self._dc = 0.995 * self._dc + 0.005 * float(ir)

        ac = float(ir) - self._dc
        if not self._filter_initialised:
            self._zi = sosfilt_zi(self._sos) * ac
            self._filter_initialised = True
        filtered, self._zi = sosfilt(self._sos, [ac], zi=self._zi)
        y = float(filtered[0])

        env = max(500.0, abs(y) * 4.0)
        norm = max(-1.0, min(1.0, y / env))
        if self.invert:
            norm = -norm

        self.wave.push(norm)

        self._bpm = bpm
        self._spo2 = spo2
        self.readout.set_hr(bpm)
        self.readout.set_spo2(spo2)
        if bpm > 0:
            self.readout.set_status("MONITORING", ok=True)

    def _on_render(self):
        self.wave.advance_fade()
        self.wave.redraw()

    def _blink_alarm(self):
        if self._finger_present:
            return
        self._alarm_phase = not self._alarm_phase
        if self._alarm_phase:
            self.readout.set_status("⚠ SENSOR DISCONNECTED ⚠", ok=False)
        else:
            self.readout.set_status("", ok=False)

    def _toggle_fs(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def _exit_fs(self):
        if self.isFullScreen():
            self.showMaximized()

    def closeEvent(self, event):
        self._render_timer.stop()
        self._alarm_timer.stop()
        if self.reader:
            self.reader.stop()
        if self.simulator:
            self.simulator.stop()
        event.accept()


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="PPG Patient Monitor")
    p.add_argument("--port", default=None,
                   help="Serial port (default: auto-detect)")
    p.add_argument("--fs", type=float, default=DEFAULT_FS,
                   help="Sensor sample rate in Hz (default: 100)")
    p.add_argument("--ir-threshold", type=int, default=DEFAULT_IR_THRESHOLD,
                   help="Raw IR below this = no finger (default: 5000)")
    p.add_argument("--invert", action="store_true",
                   help="Flip pleth polarity")
    p.add_argument("--simulate", action="store_true",
                   help="Synthetic waveform, no hardware")
    p.add_argument("--windowed", action="store_true",
                   help="Start maximized instead of fullscreen")
    return p.parse_args()


def main():
    args = parse_args()
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MonitorWindow(args)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
