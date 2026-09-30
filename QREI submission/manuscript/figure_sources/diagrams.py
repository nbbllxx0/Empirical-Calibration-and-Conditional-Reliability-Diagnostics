"""Vector schematics for the QREI manuscript (Figures 1-3).

Run from the repository root:
    python "QREI submission/manuscript/figure_sources/diagrams.py"
Writes SVG sources next to this file and exports PDFs to ../figures with Inkscape (INKSCAPE environment variable or PATH); text fit uses Pillow.
All counts and fold roles are taken from the fixed protocol in
QREI submission/protocol_endpoint_v3.json.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
PROTOCOL = HERE.parents[1] / "protocol_endpoint_v3.json"
INKSCAPE = (os.environ.get("INKSCAPE") or shutil.which("inkscape")
            or r"D:\Program Files\Inkscape\bin\inkscape.exe")

FONT = "Arial"
INK = "#1f2a36"
MUTED = "#55616c"
LINE = "#44505b"
PALETTE = {
    "blue": ("#2f5d8a", "#eaf1f7"),
    "orange": ("#b65a24", "#fbefe6"),
    "teal": ("#237a73", "#e6f3f1"),
    "violet": ("#5a4a8c", "#efecf6"),
    "navy": ("#1f3b5a", "#e8edf3"),
    "gray": ("#6b7780", "#f2f4f5"),
}
CHAR_WIDTH = 0.53  # mean Arial advance / font size, used for fit checks
FS = 1.12  # global text scale for legibility at print size
PROBLEMS: list[str] = []
MIN_ARROW = 24  # canvas units; arrowhead is about 8


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Rendered width in canvas units, measured with the Arial font files."""
    from PIL import ImageFont
    font = ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", 200)
    return font.getlength(text) * size / 200


class Canvas:
    def __init__(self, width: float, height: float, name: str):
        self.width, self.height, self.name = width, height, name
        self.items: list[str] = []

    def add(self, item: str) -> None:
        self.items.append(item)

    def text(self, x, y, s, size=10.5, weight="normal", color=INK, anchor="middle", style=""):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size*FS:.2f}" '
                 f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" {style}>{escape(s)}</text>')

    def rect(self, x, y, w, h, fill, stroke, rx=5, sw=1.1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, points, color=LINE, sw=1.1, arrow=True, dash=None):
        if arrow:
            (x0, y0), (x1, y1) = points[-2], points[-1]
            if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 < MIN_ARROW:
                PROBLEMS.append(f"{self.name}: arrow ending at ({x1:.0f}, {y1:.0f}) has a shaft shorter than {MIN_ARROW}")
        path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
        m = ' marker-end="url(#arrow)"' if arrow else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}"{m}{d} '
                 f'stroke-linejoin="round"/>')

    def card(self, x, y, w, h, title, lines, color="blue", badge=None, size=10.2, title_size=10.8,
             header=24, align="middle"):
        stroke, fill = PALETTE[color]
        self.rect(x, y, w, h, fill, stroke)
        # header band (square bottom corners)
        self.add(f'<path d="M{x:.1f},{y+header:.1f} L{x:.1f},{y+5:.1f} Q{x:.1f},{y:.1f} {x+5:.1f},{y:.1f} '
                 f'L{x+w-5:.1f},{y:.1f} Q{x+w:.1f},{y:.1f} {x+w:.1f},{y+5:.1f} L{x+w:.1f},{y+header:.1f} Z" '
                 f'fill="{stroke}"/>')
        tx = x + w / 2
        if badge is not None:
            self.add(f'<circle cx="{x+13:.1f}" cy="{y+header/2:.1f}" r="8" fill="#ffffff"/>')
            self.text(x + 13, y + header / 2 + 3.6, str(badge), size=9.5, weight="bold", color=stroke)
            tx = x + (w + 22) / 2
        self.text(tx, y + header / 2 + 3.8, title, size=title_size, weight="bold", color="#ffffff")
        self.check(title, title_size, w - (30 if badge is not None else 8), bold=True)
        top = y + header + 6
        free = h - header - 6
        step = min(14.5, free / max(len(lines), 1))
        first = top + (free - step * len(lines)) / 2 + step * 0.72
        for i, s in enumerate(lines):
            if align == "start":
                self.text(x + 9, first + i * step, s, size=size, anchor="start")
            else:
                self.text(x + w / 2, first + i * step, s, size=size)
            self.check(s, size, w - 12)

    def check(self, s, size, room, bold=False):
        size = size * FS
        if text_width(s, size, bold) > room:
            PROBLEMS.append(f"{self.name}: '{s}' needs {text_width(s, size, bold):.0f} > {room:.0f}")

    def svg(self) -> str:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width/4:.2f}mm" '
                f'height="{self.height/4:.2f}mm" viewBox="0 0 {self.width} {self.height}">'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                f'markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{LINE}"/>'
                '</marker></defs><rect width="100%" height="100%" fill="#ffffff"/>')
        return head + "\n".join(self.items) + "</svg>"

    def panel(self, x, y, letter, title):
        """Nature-style panel label: bold lowercase letter followed by a plain title."""
        self.text(x, y, letter, size=13.5, weight="bold", anchor="start")
        self.text(x + 16, y, title, size=11.2, anchor="start")

    def embed(self, other: "Canvas", dy: float) -> None:
        self.add(f'<g transform="translate(0,{dy})">' + "\n".join(other.items) + "</g>")


def figure_design(protocol: dict) -> Canvas:
    c = Canvas(700, 418, "design")
    c.panel(0, 14, "a", "Evaluation pipeline")
    cards = [
        ("Test records", ["10 archived tests", "14,297 acquisitions", "Load and speed", "vary within tests"], "blue"),
        ("Endpoint rules", ["6 vibration stops", "2 temperature stops", "B01, B05 kept as", "diagnostic records"], "orange"),
        ("Signal chain", ["Channels A, C in g", "Common 64 kHz rate", "Envelope and orders", "Causal temperature"], "teal"),
        ("Forecasts", ["8 learned predictors", "6 reference controls", "Output in hours", "No test-life scaling"], "violet"),
        ("Evaluation", ["8 held-out bearings", "Bearing bootstrap", "Error, direction,", "intervals, lead time"], "navy"),
    ]
    w, gap, y, h = 116, 30, 26, 118
    for i, (title, lines, color) in enumerate(cards):
        x = i * (w + gap)
        c.card(x, y, w, h, title, lines, color=color, badge=i + 1, size=9.6, title_size=10.2)
        if i < len(cards) - 1:
            c.line([(x + w + 3, y + h / 2), (x + w + gap - 3, y + h / 2)])

    events = protocol["event_bearings"]
    diagnostic = sorted(protocol["diagnostic_only"])
    y0 = 190
    c.panel(0, y0, "b", "Bearing roles in the eight leave-one-bearing-out folds")
    cell_w, cell_h, gap_x, gap_y = 38, 19, 3, 3
    x0, top = 104, y0 + 34
    roles = {"T": ("Test", "#1f3b5a", "#ffffff"), "V": ("Validation", "#6f9cc4", "#ffffff"),
             "C": ("Calibration", "#dd9a45", "#ffffff"), "F": ("Fitting", "#dfe7ef", INK),
             "D": ("Not used", "#f4f5f6", "#9aa3aa")}
    columns = events + diagnostic
    xs = []
    for j, b in enumerate(columns):
        x = x0 + j * (cell_w + gap_x) + (14 if j >= len(events) else 0)
        xs.append(x)
        c.text(x + cell_w / 2, top - 7, b, size=9.8, weight="bold",
               color=MUTED if b in diagnostic else INK)
    c.text(xs[len(events)] + (cell_w * 2 + gap_x) / 2, top - 21, "Diagnostic", size=9.2, color=MUTED)
    c.text(xs[0] + (xs[len(events) - 1] + cell_w - xs[0]) / 2, top - 21, "Documented completed tests",
           size=9.2, color=MUTED)
    n = len(events)
    for i, test in enumerate(events):
        y = top + i * (cell_h + gap_y)
        c.text(x0 - 10, y + cell_h / 2 + 3.5, f"Fold {i+1}", size=9.8, anchor="end")
        for j, b in enumerate(columns):
            if b in diagnostic:
                role = "D"
            elif b == test:
                role = "T"
            elif b == events[(i + 1) % n]:
                role = "V"
            elif b == events[(i + 2) % n]:
                role = "C"
            else:
                role = "F"
            label, fill, fg = roles[role]
            c.rect(xs[j], y, cell_w, cell_h, fill, "#ffffff" if role != "D" else "#d4d9dd", rx=2.5, sw=0.8)
            c.text(xs[j] + cell_w / 2, y + cell_h / 2 + 3.6, role if role != "D" else "–", size=9.6,
                   weight="bold", color=fg)
    lx = xs[-1] + cell_w + 34
    c.text(lx, top - 7, "Role in a fold", size=9.8, weight="bold", anchor="start")
    notes = {"T": "one bearing, scored", "V": "selects training epoch", "C": "sets interval width",
             "F": "five bearings", "D": "never fitted or scored"}
    for k, key in enumerate("TVCFD"):
        label, fill, fg = roles[key]
        y = top + k * 30
        c.rect(lx, y, 22, 17, fill, "#d4d9dd" if key == "D" else fill, rx=2.5, sw=0.8)
        c.text(lx + 11, y + 12.3, key if key != "D" else "–", size=9.4, weight="bold", color=fg)
        c.text(lx + 30, y + 8, label, size=9.8, weight="bold", anchor="start")
        c.text(lx + 30, y + 19.5, notes[key], size=9, color=MUTED, anchor="start")
    c.text(x0 + (xs[n - 1] + cell_w - x0) / 2, top + n * (cell_h + gap_y) + 14,
           "Validation and calibration bearings follow the test bearing in circular order.",
           size=9.2, color=MUTED)
    return c


def column_heads(c: Canvas, heads, y=12):
    for x, w, label in heads:
        c.text(x + w / 2, y, label, size=9.4, weight="bold", color=MUTED)


def figure_signal_chain() -> Canvas:
    """Four columns (records, conditioning, features, model inputs); straight arrows, no crossings."""
    c = Canvas(700, 318, "signal")
    C1, C2, C3, C4 = (0, 128), (158, 136), (324, 170), (524, 176)
    column_heads(c, [(*C1, "Raw records"), (*C2, "Conditioning"), (*C3, "Features"), (*C4, "Model inputs")])
    top = 22
    # row y positions
    W_Y, W_H = top, 54          # waveform branch (features column)
    V_Y, V_H = top + 68, 96     # vibration record, conditioning, record descriptors
    H_Y, H_H = V_Y + V_H + 32, 112   # causal histories
    O_Y, O_H = H_Y + H_H - 40, 40    # not used (operating context shares the histories row bottom)
    c.card(C3[0], W_Y, C3[1], W_H, "Waveform branch", ["Anti-alias to 1,280 Hz"], color="teal", header=22,
           size=9.8)
    c.card(C1[0], V_Y, C1[1], V_H, "Vibration", ["Channels A and C", "1.6 s every ~12 s", "128 or 64 kHz, volts"],
           color="blue", size=9.6)
    c.card(C2[0], V_Y, C2[1], V_H, "Conditioning", ["Convert at 10 g/V", "Remove channel mean", "Resample to 64 kHz"],
           color="teal", size=9.6)
    c.card(C3[0], V_Y, C3[1], V_H, "Record descriptors", ["Time-domain statistics", "Power in five bands",
                                                          "6–10 kHz envelope", "Power at defect orders"],
           color="teal", size=9.6)
    c.card(C1[0], H_Y + 22, C1[1], 72, "Temperature", ["Positions T1 and T2", "Ambient"], color="orange", size=9.6)
    c.card(C3[0], H_Y, C3[1], H_H, "Causal histories", ["Averages, 5 min half-life", "Slopes, 15–60 min",
                                                        "Margins to 110 °C, 6–10 g", "Threshold crossings",
                                                        "Elapsed time"], color="orange", size=9.6)
    op_y = H_Y + H_H + 18
    c.card(C1[0], op_y, C1[1], 56, "Operating record", ["Load and speed"], color="gray", size=9.6, header=22)
    # model inputs column
    x4, w4 = C4
    c.card(x4, W_Y, w4, W_H, "Waveforms", ["2 × 2,048 samples; neural only"], color="navy", header=22, size=9.2)
    eng_h = H_Y + H_H - V_Y
    c.card(x4, V_Y, w4, eng_h, "Engineered vector (104)", [], color="navy")
    rows = [("42", "current descriptors"), ("40", "smoothed descriptors"), ("2", "vibration indicators"),
            ("19", "endpoint histories"), ("1", "elapsed time")]
    for k, (n, label) in enumerate(rows):
        y = V_Y + 54 + (eng_h - 96) / 4 * k
        c.text(x4 + 36, y, n, size=10, weight="bold", anchor="end")
        c.text(x4 + 44, y, label, size=9.6, anchor="start")
        c.check(label, 9.6, w4 - 50)
    c.add(f'<line x1="{x4+12}" y1="{V_Y+eng_h-26}" x2="{x4+w4-12}" y2="{V_Y+eng_h-26}" stroke="#b9c5d1" '
          f'stroke-width="0.8"/>')
    c.text(x4 + w4 / 2, V_Y + eng_h - 10, "all learned predictors", size=9, color=MUTED)
    c.card(x4, op_y, w4, 56, "Operating context (3)", ["Static and dynamic load, speed"], color="navy",
           header=22, size=9.2)
    # arrows: straight, no crossings
    c.line([(C1[0] + C1[1], V_Y + V_H / 2 + 6), (C2[0] - 3, V_Y + V_H / 2 + 6)])
    c.line([(C2[0] + C2[1], V_Y + V_H / 2 + 6), (C3[0] - 3, V_Y + V_H / 2 + 6)])
    c.line([(C2[0] + C2[1] / 2, V_Y), (C2[0] + C2[1] / 2, W_Y + W_H / 2 + 5), (C3[0] - 3, W_Y + W_H / 2 + 5)])
    c.line([(C3[0] + C3[1] / 2, V_Y + V_H), (C3[0] + C3[1] / 2, H_Y - 3)])
    c.line([(C1[0] + C1[1], H_Y + 58), (C3[0] - 3, H_Y + 58)])
    c.line([(C3[0] + C3[1], W_Y + W_H / 2 + 5), (x4 - 3, W_Y + W_H / 2 + 5)])
    c.line([(C3[0] + C3[1], V_Y + V_H / 2 + 6), (x4 - 3, V_Y + V_H / 2 + 6)])
    c.line([(C3[0] + C3[1], H_Y + 58), (x4 - 3, H_Y + 58)])
    c.line([(C1[0] + C1[1], op_y + 35), (x4 - 3, op_y + 35)])
    c.height = op_y + 58
    return c


def figure_architecture() -> Canvas:
    """Encoders -> concatenation bar -> fusion -> two heads -> forecast; orthogonal arrows only."""
    c = Canvas(700, 232, "architecture")
    E, B, F, Hd, O = (0, 156), (186, 28), (244, 100), (392, 100), (540, 160)
    cards = [(0, "Waveform encoder", ["Input 2 × 2,048", "Conv 32→48→96 channels", "Kernels 9, 7, 5; pooling"]),
             (80, "Feature encoder", ["Input 104 values", "Linear 104→96"]),
             (160, "Context encoder", ["Input 3 values", "Linear 3→48→96"])]
    for y, title, lines in cards:
        c.card(E[0], y, E[1], 72, title, lines, color="blue", size=9.4)
        c.line([(E[0] + E[1], y + 43), (B[0] - 3, y + 43)])
    stroke, fill = PALETTE["teal"]
    c.rect(B[0], 0, B[1], 232, fill, stroke)
    cx, cy = B[0] + B[1] / 2, 116
    c.add(f'<text x="{cx}" y="{cy}" font-family="{FONT}" font-size="{10*FS:.2f}" font-weight="bold" fill="{stroke}" '
          f'text-anchor="middle" dominant-baseline="central" transform="rotate(-90 {cx} {cy})">'
          f'Concatenate, 3 × 96 = 288</text>')
    c.card(F[0], 78, F[1], 76, "Fusion", ["288→96→96", "Dropout 0.1"], color="teal", size=9.8)
    c.line([(B[0] + B[1], 116), (F[0] - 3, 116)])
    c.card(Hd[0], 30, Hd[1], 62, "State head", ["Sigmoid → d"], color="violet", size=9.8)
    c.card(Hd[0], 140, Hd[1], 62, "Rate head", ["Softplus → r"], color="violet", size=9.8)
    xm = (F[0] + F[1] + Hd[0]) / 2 - 8
    c.line([(F[0] + F[1], 116), (xm, 116), (xm, 67), (Hd[0] - 3, 67)])
    c.line([(xm, 116), (xm, 177), (Hd[0] - 3, 177)])
    c.card(O[0], 60, O[1], 112, "Forecast in hours", ["ŷ = min{s(1 − d)/(r + 10⁻⁴), C}", "",
                                                      "s: median fitting life", "C: twice the longest life"],
           color="navy", size=9.0)
    xo = (Hd[0] + Hd[1] + O[0]) / 2 - 8
    c.line([(Hd[0] + Hd[1], 67), (xo, 67), (xo, 104), (O[0] - 3, 104)])
    c.line([(Hd[0] + Hd[1], 177), (xo, 177), (xo, 140), (O[0] - 3, 140)])
    return c


def export(canvas: Canvas, name: str) -> None:
    svg = HERE / f"{name}.svg"
    svg.write_text(canvas.svg(), encoding="utf-8")
    pdf = FIGURES / f"{name}.pdf"
    subprocess.run([INKSCAPE, str(svg), "--export-type=pdf", f"--export-filename={pdf}",
                    "--export-text-to-path"], check=True, capture_output=True)
    subprocess.run([INKSCAPE, str(svg), "--export-type=png", f"--export-filename={HERE / (name + '.png')}",
                    "--export-dpi=200"], check=True, capture_output=True)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    FIGURES.mkdir(exist_ok=True)
    export(figure_design(protocol), "study_design")
    export(figure_signal_chain(), "signal_chain")
    export(figure_architecture(), "architecture")
    if PROBLEMS:
        print("TEXT FIT WARNINGS:")
        print("\n".join(PROBLEMS))
    else:
        print("All text fits its box under the width estimate.")


if __name__ == "__main__":
    main()
