"""Generate docs/assets/algorithm-comparison.svg from real matcher output.

Runs both algorithms over a 10-record company-name fixture (typos, legal-suffix
variations, one rename, one acronym expansion) and renders a grouped bar chart.
The SVG is theme-aware: opaque light surface by default, opaque dark surface
under prefers-color-scheme: dark, so text stays readable on any README theme.

Usage:
    uv run python scripts/generate_comparison_chart.py [--dark-preview OUT.svg]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzz_match.core import matrix_cosine, rapid_fuzz_wratio

FIXTURE = [
    ("Apple Inc.", "Appel Incorporated"),
    ("Microsoft Corporation", "Microsft Corp"),
    ("Google LLC", "Googel L.L.C."),
    ("Amazon.com, Inc.", "Amazonn Inc"),
    ("Tesla Motors", "Telsa Motors Inc"),
    ("Facebook", "Meta Platforms"),
    ("Netflix, Inc.", "NETFLIX INTL"),
    ("Twitter Inc.", "Twiiter, Inc"),
    ("IBM", "International Business Machines"),
    ("Oracle Corporation", "Oracle Corp."),
]

LIGHT = {
    "surface": "#fcfcfb", "ring": "rgba(11,11,11,0.10)",
    "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "baseline": "#c3c2b7",
    "s1": "#2a78d6", "s2": "#1baf7a",
}
DARK = {
    "surface": "#1a1a19", "ring": "rgba(255,255,255,0.10)",
    "primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "baseline": "#383835",
    "s1": "#3987e5", "s2": "#199e70",
}

# Plot geometry (viewBox 880 x 640)
X0, X100 = 172, 848
PLOT_TOP, ROW_H, BAR_H, PAIR_GAP = 104, 50, 13, 2


def bar_path(x0, y, w, h, r=4):
    """Left-anchored bar with rounded data end."""
    w = max(w, r)
    return (
        f"M {x0} {y} h {w - r:.1f} a {r} {r} 0 0 1 {r} {r} "
        f"v {h - 2 * r} a {r} {r} 0 0 1 -{r} {r} h -{w - r:.1f} Z"
    )


def css_block(theme):
    return (
        f".surface {{ fill:{theme['surface']}; stroke:{theme['ring']}; }}\n"
        f".title {{ font-size:19px; font-weight:700; fill:{theme['primary']}; }}\n"
        f".sub {{ font-size:12.5px; fill:{theme['secondary']}; }}\n"
        f".cat {{ font-size:11.5px; fill:{theme['secondary']}; }}\n"
        f".val {{ font-size:10.5px; fill:{theme['secondary']}; font-variant-numeric:tabular-nums; }}\n"
        f".tick {{ font-size:11px; fill:{theme['muted']}; font-variant-numeric:tabular-nums; }}\n"
        f".leg {{ font-size:12px; fill:{theme['secondary']}; }}\n"
        f".grid {{ stroke:{theme['grid']}; stroke-dasharray:2,3; }}\n"
        f".baseline {{ stroke:{theme['baseline']}; }}\n"
        f".s1 {{ fill:{theme['s1']}; }}\n"
        f".s2 {{ fill:{theme['s2']}; }}\n"
    )


def render(rows, theme, dark_block=True):
    px = (X100 - X0) / 100.0
    plot_bot = PLOT_TOP + ROW_H * len(rows)
    dark_css = (
        "@media (prefers-color-scheme: dark) {\n" + css_block(DARK) + "}\n"
        if dark_block
        else ""
    )
    parts = [
        '<svg viewBox="0 0 880 640" xmlns="http://www.w3.org/2000/svg" '
        "font-family=\"system-ui, -apple-system, 'Segoe UI', sans-serif\">",
        "<style>",
        css_block(theme),
        dark_css,
        "</style>",
        '<rect class="surface" x="1" y="1" width="878" height="638" rx="14"/>',
        '<text class="title" x="32" y="44">Algorithm comparison — best-match scores</text>',
        '<text class="sub" x="32" y="66">10-record company-name fixture: typos, '
        "legal-suffix variants, one rename, one acronym</text>",
        '<rect class="s1" x="618" y="35" width="12" height="12" rx="3"/>',
        '<text class="leg" x="638" y="45">Levenshtein WRatio</text>',
        '<rect class="s2" x="618" y="57" width="12" height="12" rx="3"/>',
        '<text class="leg" x="638" y="67">TF-IDF Cosine</text>',
    ]
    for v in range(0, 101, 20):
        x = X0 + v * px
        cls = "baseline" if v == 0 else "grid"
        parts.append(
            f'<line class="{cls}" x1="{x:.1f}" y1="{PLOT_TOP - 8}" x2="{x:.1f}" '
            f'y2="{plot_bot + 4}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{plot_bot + 22}" '
            f'text-anchor="middle">{v}</text>'
        )
    for i, row in enumerate(rows):
        y_pair = PLOT_TOP + i * ROW_H + (ROW_H - 2 * BAR_H - PAIR_GAP) / 2
        parts.append(
            f'<text class="cat" x="{X0 - 12}" y="{y_pair + BAR_H + 4:.1f}" '
            f'text-anchor="end">{row["name"]}</text>'
        )
        for j, series in enumerate(["wratio", "cosine"]):
            y = y_pair + j * (BAR_H + PAIR_GAP)
            w = row[series] * px
            parts.append(
                f'<path class="s{j + 1}" d="{bar_path(X0, y, w, BAR_H)}"/>'
            )
            parts.append(
                f'<text class="val" x="{X0 + w + 8:.1f}" y="{y + BAR_H - 3.5:.1f}">'
                f"{row[series]:.1f}</text>"
            )
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    left = [c for c, _ in FIXTURE]
    right = [v for _, v in FIXTURE]
    wr = rapid_fuzz_wratio(left, right)
    mc = matrix_cosine(left, right, topn=1)
    rows = [
        {
            "name": left[i].replace("&", "&amp;"),
            "wratio": float(wr.iloc[i]["score"]),
            "cosine": float(mc.iloc[i]["score"]),
        }
        for i in range(len(left))
    ]
    rows.sort(key=lambda r: r["wratio"], reverse=True)

    out = Path(__file__).resolve().parents[1] / "docs" / "assets" / "algorithm-comparison.svg"
    out.write_text(render(rows, LIGHT))
    print(f"Wrote {out}")

    for r in rows:
        print(f"{r['name']:26s} WRatio {r['wratio']:5.1f}   Cosine {r['cosine']:5.1f}")
    print(f"{'AVERAGE':26s} WRatio {sum(r['wratio'] for r in rows)/10:5.1f}   "
          f"Cosine {sum(r['cosine'] for r in rows)/10:5.1f}")

    if "--dark-preview" in sys.argv:
        dark_out = Path(sys.argv[sys.argv.index("--dark-preview") + 1])
        dark_out.write_text(render(rows, DARK, dark_block=False))
        print(f"Wrote {dark_out}")


if __name__ == "__main__":
    main()
