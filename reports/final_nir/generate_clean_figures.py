from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1800, 1100
BLACK = (0, 0, 0)
GRID = (218, 218, 218)
GRAY_DARK = (85, 85, 85)
GRAY_MID = (145, 145, 145)
GRAY_LIGHT = (210, 210, 210)
WHITE = (255, 255, 255)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
TITLE = ImageFont.truetype(BOLD, 44)
AXIS = ImageFont.truetype(FONT, 31)
LABEL = ImageFont.truetype(FONT, 27)
SMALL = ImageFont.truetype(FONT, 23)

MODELS = [
    ("baseline PC", 0.9689, 0.9741, 0.9783, 8919.5, 4.451),
    ("edge SBC", 0.9702, 0.9745, 0.9544, 6057.6, 4.267),
    ("micro PyTorch", 0.9767, 0.9804, 0.9783, 101.3, 0.479),
    ("micro INT8", 0.9754, 0.9795, 0.9848, 31.0, 0.441),
]


def base(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    d.text((W / 2, 38), title, font=TITLE, fill=BLACK, anchor="mt")
    return im, d


def save(im: Image.Image, stem: str) -> None:
    im.save(OUT / f"{stem}.png", dpi=(300, 300))
    im.save(OUT / f"{stem}.pdf", "PDF", resolution=300)


def wrap_label(label: str) -> str:
    return label.replace(" ", "\n", 1)


def centered_lines(d: ImageDraw.ImageDraw, x: float, y: float, text: str, font: ImageFont.FreeTypeFont) -> None:
    for i, line in enumerate(text.split("\n")):
        d.text((x, y + i * 31), line, font=font, fill=BLACK, anchor="mt")


def quality() -> None:
    im, d = base("Сравнение качества моделей на deduplicated split")
    x0, y0, x1, y1 = 170, 145, W - 100, H - 260
    ymin, ymax = 0.93, 1.00
    for tick in [0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]:
        y = y1 - (tick - ymin) / (ymax - ymin) * (y1 - y0)
        d.line((x0, y, x1, y), fill=GRID, width=2)
        d.text((x0 - 12, y), f"{tick:.2f}", font=SMALL, fill=BLACK, anchor="rm")
    d.line((x0, y1, x1, y1), fill=BLACK, width=3)
    d.line((x0, y0, x0, y1), fill=BLACK, width=3)
    group = (x1 - x0) / len(MODELS)
    bw = 80
    for i, (name, _, f1, defect_recall, _, _) in enumerate(MODELS):
        cx = x0 + group * (i + 0.5)
        for j, (value, fill) in enumerate([(f1, GRAY_LIGHT), (defect_recall, WHITE)]):
            bx0 = cx - 90 + j * 96
            bx1 = bx0 + bw
            by = y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)
            d.rectangle((bx0, by, bx1, y1), fill=fill, outline=BLACK, width=3)
            d.text(((bx0 + bx1) / 2, by - 12), f"{value:.3f}", font=SMALL, fill=BLACK, anchor="mb")
        centered_lines(d, cx, y1 + 28, wrap_label(name), LABEL)
    d.text(((x0 + x1) / 2, H - 86), "Профиль модели", font=AXIS, fill=BLACK, anchor="mm")
    d.text((56, (y0 + y1) / 2), "Значение метрики", font=AXIS, fill=BLACK, anchor="mm")
    lx, ly = W / 2 - 230, H - 165
    d.rectangle((lx, ly, lx + 45, ly + 28), fill=GRAY_LIGHT, outline=BLACK, width=2)
    d.text((lx + 60, ly + 14), "F1", font=LABEL, fill=BLACK, anchor="lm")
    d.rectangle((lx + 170, ly, lx + 215, ly + 28), fill=WHITE, outline=BLACK, width=2)
    d.text((lx + 230, ly + 14), "defective recall", font=LABEL, fill=BLACK, anchor="lm")
    save(im, "quality_comparison_clean")
    save(im, "quality_comparison")
    save(im, "model_quality_comparison")


def size() -> None:
    im, d = base("Сравнение размера моделей")
    x0, y0, x1, y1 = 170, 145, W - 100, H - 230
    ticks = [10, 30, 100, 300, 1000, 3000, 10000]
    for tick in ticks:
        y = y1 - (math.log10(tick) - 1) / 3 * (y1 - y0)
        d.line((x0, y, x1, y), fill=GRID, width=2)
        d.text((x0 - 12, y), str(tick), font=SMALL, fill=BLACK, anchor="rm")
    d.line((x0, y1, x1, y1), fill=BLACK, width=3)
    d.line((x0, y0, x0, y1), fill=BLACK, width=3)
    fills = [GRAY_DARK, GRAY_MID, GRAY_LIGHT, WHITE]
    group = (x1 - x0) / len(MODELS)
    for i, (name, _, _, _, kb, _) in enumerate(MODELS):
        cx = x0 + group * (i + 0.5)
        y = y1 - (math.log10(kb) - 1) / 3 * (y1 - y0)
        d.rectangle((cx - 70, y, cx + 70, y1), fill=fills[i], outline=BLACK, width=3)
        d.text((cx, y - 12), f"{kb:.0f} KB", font=SMALL, fill=BLACK, anchor="mb")
        centered_lines(d, cx, y1 + 28, wrap_label(name), LABEL)
    d.text(((x0 + x1) / 2, H - 82), "Профиль модели", font=AXIS, fill=BLACK, anchor="mm")
    d.text((56, (y0 + y1) / 2), "Размер модели, KB (логарифмическая шкала)", font=AXIS, fill=BLACK, anchor="mm")
    save(im, "size_comparison_clean")
    save(im, "size_comparison_log")
    save(im, "model_size_comparison")


def pareto() -> None:
    im, d = base("Pareto-компромисс качества и размера модели")
    x0, y0, x1, y1 = 190, 145, W - 120, H - 220
    ymin, ymax = 0.970, 0.983
    for tick in [10, 30, 100, 300, 1000, 3000, 10000]:
        x = x0 + (math.log10(tick) - 1) / 3 * (x1 - x0)
        d.line((x, y0, x, y1), fill=GRID, width=1)
        d.text((x, y1 + 18), str(tick), font=SMALL, fill=BLACK, anchor="mt")
    for tick in [0.970, 0.975, 0.980]:
        y = y1 - (tick - ymin) / (ymax - ymin) * (y1 - y0)
        d.line((x0, y, x1, y), fill=GRID, width=2)
        d.text((x0 - 12, y), f"{tick:.3f}", font=SMALL, fill=BLACK, anchor="rm")
    d.rectangle((x0, y0, x1, y1), outline=BLACK, width=3)
    offsets = [(20, -48), (20, 28), (22, -12), (-240, -48)]
    for i, (name, _, f1, _, kb, _) in enumerate(MODELS):
        x = x0 + (math.log10(kb) - 1) / 3 * (x1 - x0)
        y = y1 - (f1 - ymin) / (ymax - ymin) * (y1 - y0)
        if name == "micro INT8":
            d.text((x, y), "★", font=ImageFont.truetype(BOLD, 56), fill=BLACK, anchor="mm")
        elif name == "edge SBC":
            d.rectangle((x - 18, y - 18, x + 18, y + 18), fill=WHITE, outline=BLACK, width=3)
        elif name == "micro PyTorch":
            d.polygon([(x, y - 24), (x - 22, y + 18), (x + 22, y + 18)], fill=WHITE, outline=BLACK)
        else:
            d.ellipse((x - 18, y - 18, x + 18, y + 18), fill=WHITE, outline=BLACK, width=3)
        ox, oy = offsets[i]
        d.text((x + ox, y + oy), name, font=LABEL, fill=BLACK, anchor="lm")
    d.text(((x0 + x1) / 2, H - 78), "Размер модели, KB (логарифмическая шкала)", font=AXIS, fill=BLACK, anchor="mm")
    d.text((58, (y0 + y1) / 2), "F1", font=AXIS, fill=BLACK, anchor="mm")
    save(im, "pareto_f1_size_clean")
    save(im, "pareto_f1_size")


def latency() -> None:
    im, d = base("Proxy-оценка latency")
    x0, y0, x1, y1 = 170, 145, W - 100, H - 255
    maxv = 5.0
    for tick in [0, 1, 2, 3, 4, 5]:
        y = y1 - tick / maxv * (y1 - y0)
        d.line((x0, y, x1, y), fill=GRID, width=2)
        d.text((x0 - 12, y), str(tick), font=SMALL, fill=BLACK, anchor="rm")
    d.line((x0, y1, x1, y1), fill=BLACK, width=3)
    d.line((x0, y0, x0, y1), fill=BLACK, width=3)
    fills = [GRAY_DARK, GRAY_MID, GRAY_LIGHT, WHITE]
    group = (x1 - x0) / len(MODELS)
    for i, (name, _, _, _, _, lat) in enumerate(MODELS):
        cx = x0 + group * (i + 0.5)
        y = y1 - lat / maxv * (y1 - y0)
        d.rectangle((cx - 70, y, cx + 70, y1), fill=fills[i], outline=BLACK, width=3)
        d.text((cx, y - 12), f"{lat:.3f} ms", font=SMALL, fill=BLACK, anchor="mb")
        centered_lines(d, cx, y1 + 28, wrap_label(name), LABEL)
    d.text((56, (y0 + y1) / 2), "Latency, ms", font=AXIS, fill=BLACK, anchor="mm")
    d.text((W / 2, H - 82), "Измерено на текущей машине; не является аппаратным замером на Raspberry Pi/OpenMV", font=LABEL, fill=BLACK, anchor="mm")
    save(im, "latency_comparison_clean")
    save(im, "latency_comparison")


def triage() -> None:
    im, d = base("Режим ACCEPT/REVIEW для снижения риска автоматического решения")
    items = [("coverage", 0.8277), ("review rate", 0.1723), ("accepted acc.", 0.9984), ("accepted def. recall", 0.9977)]
    x0, y0, x1, y1 = 170, 145, W - 100, H - 260
    for tick in [0, 0.25, 0.50, 0.75, 1.00]:
        y = y1 - tick * (y1 - y0)
        d.line((x0, y, x1, y), fill=GRID, width=2)
        d.text((x0 - 12, y), f"{tick:.2f}", font=SMALL, fill=BLACK, anchor="rm")
    d.line((x0, y1, x1, y1), fill=BLACK, width=3)
    d.line((x0, y0, x0, y1), fill=BLACK, width=3)
    fills = [GRAY_MID, GRAY_LIGHT, WHITE, (235, 235, 235)]
    group = (x1 - x0) / len(items)
    for i, (name, value) in enumerate(items):
        cx = x0 + group * (i + 0.5)
        y = y1 - value * (y1 - y0)
        d.rectangle((cx - 75, y, cx + 75, y1), fill=fills[i], outline=BLACK, width=3)
        d.text((cx, y - 12), f"{value:.4f}", font=SMALL, fill=BLACK, anchor="mb")
        centered_lines(d, cx, y1 + 28, name.replace(" ", "\n"), LABEL)
    d.text((W / 2, H - 78), "Threshold = 0.95; unsafe errors = 1", font=AXIS, fill=BLACK, anchor="mm")
    d.text((56, (y0 + y1) / 2), "Доля / метрика", font=AXIS, fill=BLACK, anchor="mm")
    save(im, "triage_summary_clean")
    save(im, "triage_tradeoff")


def confusion() -> None:
    cm = [[299, 12], [7, 454]]
    im, d = base("Confusion matrix: micro-edge INT8 exact")
    x0, y0, cell = 560, 230, 250
    labels = ["ok", "defective"]
    shades = [[225, 248], [250, 175]]
    for i in range(2):
        for j in range(2):
            x = x0 + j * cell
            y = y0 + i * cell
            shade = shades[i][j]
            d.rectangle((x, y, x + cell, y + cell), fill=(shade, shade, shade), outline=BLACK, width=4)
            d.text((x + cell / 2, y + cell / 2), str(cm[i][j]), font=TITLE, fill=BLACK, anchor="mm")
    for j, label in enumerate(labels):
        d.text((x0 + j * cell + cell / 2, y0 - 35), f"Pred: {label}", font=LABEL, fill=BLACK, anchor="mb")
    for i, label in enumerate(labels):
        d.text((x0 - 38, y0 + i * cell + cell / 2), f"True: {label}", font=LABEL, fill=BLACK, anchor="rm")
    d.text((W / 2, H - 165), "false OK = 7; false defective = 12", font=AXIS, fill=BLACK, anchor="mm")
    save(im, "confusion_matrix_micro_int8_clean")
    save(im, "confusion_matrix_micro_int8")


if __name__ == "__main__":
    quality()
    size()
    pareto()
    latency()
    triage()
    confusion()
