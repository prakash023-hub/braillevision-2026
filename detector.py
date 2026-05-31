"""
BrailleVision — Core Detection Engine
Camera image → Braille dots → Cell segmentation → English text

Pipeline:
  1. Preprocess (grayscale, CLAHE, denoise)
  2. Detect Braille dots via blob detection or adaptive threshold + contours
  3. Cluster dots into rows & columns → Braille cells
  4. Map each cell's 2×3 dot pattern → character
  5. Return decoded text + annotated image
"""

import cv2
import numpy as np
from typing import Optional
from braille_map import decode_cells


# ─── Constants ───────────────────────────────────────────────────────────────

DOT_MIN_AREA = 60       # px²  – smaller blobs are noise (real Braille dots cast larger shadows)
DOT_MAX_AREA = 2500     # px²  – larger blobs are noise / artefacts
DOT_MIN_CIRC = 0.45     # circularity (1.0 = perfect circle)
DOT_ASPECT_MAX = 2.2    # bounding-box width/height ratio guard

# Braille cell geometry (relative units – computed from detected dots)
ROW_GAP_FACTOR = 0.9    # gap > factor × median_dot_h  → new row (within-cell row gap ~1.1x dot_h)
COL_GAP_FACTOR = 1.4    # gap > factor × median_dot_w  → new cell column within row
CELL_COL_FACTOR = 1.75  # gap > factor × median_dot_w  → inter-cell gap (new character)
                         # Must satisfy: intra-cell x-span < threshold < inter-cell x-gap


# ─── Image Preprocessing ─────────────────────────────────────────────────────

def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        gray  – grayscale float image for display
        proc  – processed binary image for detection
    """
    if image is None:
        raise ValueError("Input image is None")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # CLAHE – equalise contrast across uneven lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    # Mild blur to reduce paper texture noise
    blurred = cv2.GaussianBlur(eq, (5, 5), 0)

    # Adaptive threshold – works under variable lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=8
    )

    # Small morphological open to remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    return gray, clean


# ─── Dot Detection ───────────────────────────────────────────────────────────

def detect_dots(binary: np.ndarray) -> list[dict]:
    """
    Find Braille dots as circular blobs in binary image.
    Returns list of dicts: {cx, cy, w, h, area}
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < DOT_MIN_AREA or area > DOT_MAX_AREA:
            continue

        # Circularity filter
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < DOT_MIN_CIRC:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        if max(w, h) / min(w, h) > DOT_ASPECT_MAX:
            continue

        cx = x + w // 2
        cy = y + h // 2
        dots.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area})

    return dots


# ─── Cell Segmentation ───────────────────────────────────────────────────────

def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n else 1


def cluster_into_cells(dots: list[dict]) -> list[list[list[dict]]]:
    """
    Cluster dots → rows → cells → dot positions within cell.

    Returns: list of text-lines, each a list of cells,
             each cell is a list of dot dicts.
    Structure: lines[ cells[ dots[] ] ]
    """
    if not dots:
        return []

    med_h = _median([d['h'] for d in dots])
    med_w = _median([d['w'] for d in dots])

    # ── Step 1: cluster dots into horizontal rows ──
    dots_sorted_y = sorted(dots, key=lambda d: d['cy'])
    rows = []
    current_row = [dots_sorted_y[0]]

    for dot in dots_sorted_y[1:]:
        if dot['cy'] - current_row[-1]['cy'] > ROW_GAP_FACTOR * med_h:
            rows.append(current_row)
            current_row = [dot]
        else:
            current_row.append(dot)
    rows.append(current_row)

    # ── Step 2: Braille lines = groups of 3 dot-rows ──
    # Each Braille character occupies 3 vertical dot rows.
    # Group consecutive rows into Braille line-groups of 3.
    braille_lines = []
    i = 0
    while i < len(rows):
        line_rows = rows[i:i + 3]
        braille_lines.append(line_rows)
        i += 3

    # ── Step 3: Within each line-group, cluster into cells by x-position ──
    result_lines = []

    for line_rows in braille_lines:
        # Flatten all dots in line, sort by x
        all_dots = []
        for r_idx, row in enumerate(line_rows):
            for dot in row:
                all_dots.append({**dot, '_row_in_cell': r_idx})  # 0,1,2

        if not all_dots:
            continue

        all_dots.sort(key=lambda d: d['cx'])

        # Split by x gaps into individual cells
        cells = []
        current_cell = [all_dots[0]]

        for dot in all_dots[1:]:
            gap = dot['cx'] - current_cell[-1]['cx']
            if gap > CELL_COL_FACTOR * med_w:
                cells.append(current_cell)
                current_cell = [dot]
            else:
                current_cell.append(dot)
        cells.append(current_cell)

        result_lines.append(cells)

    return result_lines


# ─── Dot Pattern → Braille Cell ──────────────────────────────────────────────

def cell_to_pattern(cell_dots: list[dict], med_w: float) -> frozenset:
    """
    Convert dots in a cell to a Braille dot pattern (frozenset of 1–6).

    Braille dot numbering:
      col0  col1
       1     4    row0
       2     5    row1
       3     6    row2
    """
    if not cell_dots:
        return frozenset()

    xs = [d['cx'] for d in cell_dots]
    min_x, max_x = min(xs), max(xs)
    x_span = max_x - min_x

    # If all dots are tightly clustered (single-column cell), put them all in col0
    # Use half a dot-width as minimum meaningful column separation
    if x_span < med_w * 0.6:
        # All dots are in one column — treat as col0
        active_dots = set()
        for dot in cell_dots:
            row = dot.get('_row_in_cell', 0)
            dot_num = row + 1          # col0: 1, 2, 3
            active_dots.add(dot_num)
        return frozenset(active_dots)

    # Two-column cell: split at midpoint
    x_mid = (min_x + max_x) / 2.0

    active_dots = set()
    for dot in cell_dots:
        row = dot.get('_row_in_cell', 0)
        col = 0 if dot['cx'] <= x_mid else 1
        dot_num = row + 1 + col * 3   # col0: 1,2,3  col1: 4,5,6
        active_dots.add(dot_num)

    return frozenset(active_dots)


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def process_image(image: np.ndarray) -> dict:
    """
    Main entry point.

    Args:
        image: BGR numpy array from camera or file

    Returns:
        {
          'text': decoded English string,
          'annotated': BGR annotated image,
          'dot_count': int,
          'cell_count': int,
          'confidence': str,
          'debug_binary': grayscale binary image
        }
    """
    gray, binary = preprocess(image)

    dots = detect_dots(binary)

    if not dots:
        annotated = image.copy()
        cv2.putText(annotated, "No Braille dots detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return {
            'text': '',
            'annotated': annotated,
            'dot_count': 0,
            'cell_count': 0,
            'confidence': 'low',
            'debug_binary': binary
        }

    med_w = _median([d['w'] for d in dots])

    lines = cluster_into_cells(dots)

    # ── Decode each line ──
    full_text_parts = []
    cell_count = 0
    all_cell_patterns = []

    for line in lines:
        line_patterns = []
        for cell_dots in line:
            pat = cell_to_pattern(cell_dots, med_w)
            line_patterns.append(pat)
            all_cell_patterns.append(pat)
            cell_count += 1
        decoded_line = decode_cells(line_patterns)
        full_text_parts.append(decoded_line)

    text = '\n'.join(full_text_parts).strip()

    # ── Annotate image ──
    annotated = image.copy()

    # Draw detected dots
    for dot in dots:
        cv2.circle(annotated, (dot['cx'], dot['cy']),
                   max(dot['w'], dot['h']) // 2 + 2, (0, 220, 100), 2)

    # Confidence heuristic
    readable = sum(1 for p in all_cell_patterns if p in _KNOWN_PATTERNS)
    total = max(len(all_cell_patterns), 1)
    conf_ratio = readable / total

    if conf_ratio > 0.7:
        confidence = 'high'
        conf_color = (0, 200, 0)
    elif conf_ratio > 0.4:
        confidence = 'medium'
        conf_color = (0, 165, 255)
    else:
        confidence = 'low'
        conf_color = (0, 0, 200)

    # Text overlay
    label = f"Dots: {len(dots)} | Cells: {cell_count} | Conf: {confidence}"
    cv2.putText(annotated, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)

    y_offset = 65
    for line in full_text_parts[:5]:  # show up to 5 lines
        cv2.putText(annotated, line[:60], (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 50), 2)
        y_offset += 35

    return {
        'text': text,
        'annotated': annotated,
        'dot_count': len(dots),
        'cell_count': cell_count,
        'confidence': confidence,
        'debug_binary': binary
    }


# Build known-patterns set for confidence scoring
from braille_map import BRAILLE_TO_CHAR, NUMBER_MAP
_KNOWN_PATTERNS = set(BRAILLE_TO_CHAR.keys()) | set(NUMBER_MAP.keys())


# ─── Synthetic Braille Generator (for testing without physical paper) ─────────

def generate_synthetic_braille(text: str, img_size=(800, 160)) -> np.ndarray:
    """
    Generate a synthetic grayscale image with Braille dots for a given text.
    Useful for demo/testing when physical Braille paper is not available.
    img_size: (width, height) — note: width first for intuitive use.
    """
    from braille_map import BRAILLE_TO_CHAR

    # Invert lookup (char → pattern)
    CHAR_TO_PATTERN = {v: k for k, v in BRAILLE_TO_CHAR.items() if v != '?'}

    img_w, img_h = img_size                        # width × height
    img = np.ones((img_h, img_w), dtype=np.uint8) * 220  # light gray background

    DOT_R = 8       # dot radius px
    CELL_W = 60     # horizontal spacing between cell origins (generous gap for segmentation)
    ROW_H = 20      # vertical distance between dot rows in one cell
    COL_W = 18      # horizontal distance between dot cols in one cell
    START_X = 35
    START_Y = 50

    x = START_X
    for ch in text.lower():
        pat = CHAR_TO_PATTERN.get(ch, frozenset())
        for dot_num in pat:
            col = (dot_num - 1) // 3       # 0 or 1
            row = (dot_num - 1) % 3        # 0, 1, or 2
            px = x + col * COL_W
            py = START_Y + row * ROW_H
            if 0 <= px < img_w and 0 <= py < img_h:
                cv2.circle(img, (px, py), DOT_R, 30, -1)   # dark filled dot
        x += CELL_W
        if x > img_w - CELL_W:
            break

    # Add gaussian noise to simulate paper texture
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img
