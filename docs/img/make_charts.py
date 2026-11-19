#!/usr/bin/env python3
"""Draw the README's two charts from the bench tables in docs/measurements.md (standard library only).
usage: python3 docs/img/make_charts.py [REPO_ROOT]   -> docs/img/{decode-by-context,first-token}-{light,dark}.svg
Every number comes from the tables (the closing bench and the prefix-sharing measure); nothing is typed here."""
import math, pathlib, re, sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2])
TEXT = (ROOT / "docs/measurements.md").read_text()
LENGTHS = ["1K", "8K", "32K", "64K", "128K", "256K"]
KINDS = ["Code", "Legal"]
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#1a1a19", ink2="#5c5b56", grid="#e4e3df", s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34332f", s1="#3987e5", s2="#d95926"),
}

def tables():
    """{heading: [rows]} for every '### ' table of measurements.md."""
    out, cur = {}, None
    for line in TEXT.splitlines():
        m = re.match(r"### (.+)$", line)
        if m:
            cur = m.group(1).strip(); out.setdefault(cur, [])
        elif cur and line.startswith("| ") and not line.startswith("| Prompt") and not line.startswith("|---"):
            out[cur].append([c.strip() for c in line.strip().strip("|").split("|")])
    return out

def mean(cell):
    return float(cell.split(" [")[0].split(" (")[0])

def cell(tabs, heading, length, streams, col):
    rows = [r for r in tabs[heading] if r[0] == length and r[1] == streams]
    if len(rows) != 1:
        raise SystemExit(f"INCONCLUSIVE: {heading} {length} streams {streams}: {len(rows)} rows")
    return mean(rows[0][col])

def data():
    t = tables()
    d = {}
    for k in KINDS:
        d[k] = dict(
            one=[cell(t, k, L, "1", 4) for L in LENGTHS],                 # decode tok/s, one stream
            three=[cell(t, k, L, "3", 6) for L in LENGTHS],               # sum of decode rates, three streams
            cold=[cell(t, k, L, "1", 2) for L in LENGTHS],                # first token, cold read
            cached=[cell(t, k + ", with prefix sharing", L, "1", 2) for L in LENGTHS],  # first token on a hit
        )
    return d

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")

def text(x, y, s, size, fill, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

W, H = 880, 392
PANEL_X = [64, 468]; PANEL_W = 316; TOP = 112; BOTTOM = 330

def frame(th, title, subtitle, legend):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">',
         f'<rect width="{W}" height="{H}" fill="{th["surface"]}"/>',
         text(24, 30, title, 16, th["ink"], weight="600"), text(24, 50, subtitle, 11.5, th["ink2"])]
    x = 24
    for name, color in legend:
        s.append(f'<circle cx="{x+5}" cy="{70}" r="5" fill="{color}"/>')
        s.append(text(x + 14, 74, name, 11.5, th["ink2"]))
        x += 6.6 * len(name) + 34
    return s

def decode_chart(d, mode):
    th = THEMES[mode]
    s = frame(th, "Decode speed by context length (tok/s)",
              "One request alone, and three requests together (each stream's rate while all three stream, summed). MEASURED 2026-11-19, 175 W, cold reads.",
              [("one request", th["s1"]), ("three together, summed", th["s2"])])
    ymax = 450.0
    for pi, k in enumerate(KINDS):
        x0 = PANEL_X[pi]; plot_h = BOTTOM - TOP
        s.append(text(x0, TOP - 12, k, 13, th["ink"], weight="600"))
        for g in range(0, 451, 150):
            y = BOTTOM - g / ymax * plot_h
            s.append(f'<line x1="{x0}" x2="{x0+PANEL_W}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["grid"]}" stroke-width="1"/>')
            s.append(text(x0 - 8, y + 4, f"{g}", 10.5, th["ink2"], anchor="end"))
        xs = [x0 + 18 + i * (PANEL_W - 36) / (len(LENGTHS) - 1) for i in range(len(LENGTHS))]
        for i, L in enumerate(LENGTHS):
            s.append(text(xs[i], BOTTOM + 18, L, 10.5, th["ink2"], anchor="middle"))
        for key, color, label in (("three", th["s2"], "three together"), ("one", th["s1"], "one request")):
            pts = [(xs[i], BOTTOM - v / ymax * plot_h) for i, v in enumerate(d[k][key])]
            s.append('<polyline fill="none" stroke="%s" stroke-width="2" stroke-linejoin="round" points="%s"/>'
                     % (color, " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)))
            for x, y in pts:
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{th["surface"]}" stroke-width="2"/>')
            # selective direct labels: the 128K point of each line
            i = LENGTHS.index("128K"); x, y = pts[i]
            s.append(text(x, y - 10, f"{d[k][key][i]:.0f}", 11, th["ink"], anchor="middle", weight="600"))
            s.append(text(pts[-1][0] + 8, pts[-1][1] + 4, label, 10.5, th["ink2"]))
    s.append(text(24, H - 14, "Prompts of 1K to 256K tokens; 1,024 tokens out; mean of two runs. Three together: the throughput once every prompt is in; long prompts are read one after another.", 10.5, th["ink2"]))
    s.append("</svg>")
    return "\n".join(s)

def first_token_chart(d, mode):
    th = THEMES[mode]
    s = frame(th, "First token: cold read against a document already in the cache (seconds, log scale)",
              "One request. MEASURED 2026-11-19, 175 W. A cached document gives the first token in under a second at every length.",
              [("cold read", th["s1"]), ("document already cached", th["s2"])])
    lo, hi = math.log10(0.1), math.log10(100)
    def y_of(v):
        return BOTTOM - (math.log10(v) - lo) / (hi - lo) * (BOTTOM - TOP)
    for pi, k in enumerate(KINDS):
        x0 = PANEL_X[pi]
        s.append(text(x0, TOP - 12, k, 13, th["ink"], weight="600"))
        for g, lab in ((0.1, "0.1 s"), (1, "1 s"), (10, "10 s"), (100, "100 s")):
            y = y_of(g)
            s.append(f'<line x1="{x0}" x2="{x0+PANEL_W}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["grid"]}" stroke-width="1"/>')
            s.append(text(x0 - 8, y + 4, lab, 10.5, th["ink2"], anchor="end"))
        slot = PANEL_W / len(LENGTHS); bw = 20
        for i, L in enumerate(LENGTHS):
            cx = x0 + slot * (i + 0.5)
            s.append(text(cx, BOTTOM + 18, L, 10.5, th["ink2"], anchor="middle"))
            for j, (key, color) in enumerate((("cold", th["s1"]), ("cached", th["s2"]))):
                v = d[k][key][i]; x = cx - bw - 1 + j * (bw + 2); y = y_of(v); h = BOTTOM - y
                s.append(f'<path d="M{x:.1f},{BOTTOM} v{-(h-4):.1f} a4,4 0 0 1 4,-4 h{bw-8} a4,4 0 0 1 4,4 v{h-4:.1f} z" fill="{color}"/>')
                if L in ("128K", "256K"):
                    s.append(text(x + bw / 2, y - 5, f"{v:.1f}", 10.5, th["ink"], anchor="middle", weight="600"))
    s.append(text(24, H - 14, "Cached: the server has read the document before (another slot or an earlier turn); the first time a second or third stream shares it the server reads it once more.", 10.5, th["ink2"]))
    s.append("</svg>")
    return "\n".join(s)

if __name__ == "__main__":
    d = data()
    out = ROOT / "docs/img"
    for mode in THEMES:
        (out / f"decode-by-context-{mode}.svg").write_text(decode_chart(d, mode))
        (out / f"first-token-{mode}.svg").write_text(first_token_chart(d, mode))
    for k in KINDS:
        print(k, "one", d[k]["one"], "three", d[k]["three"], "cold", d[k]["cold"], "cached", d[k]["cached"])
    print("wrote 4 charts to", out)
