"""Build the self-contained M1 sweep report (single HTML, data-URI images,
static SVG chart, no scripts). Reads output/sweep/results.json.

    .venv/Scripts/python.exe m1_differentiable_bc1/build_report.py
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
SWEEP = HERE / "output" / "sweep"
IMGDIR = SWEEP / "report"
OUT = SWEEP / "report.html"

RES_ORDER = [512, 384, 256, 192, 128]
FULL_VIEWS = [("ref", "参考"), ("bc1_384", "BC1 4×384²（甜点）"), ("bc1_128", "BC1 4×128²（极限）")]


def data_uri(name: str) -> str:
    b = (IMGDIR / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def build_chart(rows: list[dict]) -> str:
    W, H, L, R, T, B = 720, 380, 56, 128, 18, 46
    x0, x1, y0, y1 = -0.15, 4.25, 24.0, 66.0

    def X(bpp: float) -> float:
        return L + (math.log2(bpp) - x0) / (x1 - x0) * (W - L - R)

    def Y(p: float) -> float:
        return T + (y1 - p) / (y1 - y0) * (H - T - B)

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block" role="img" '
         f'aria-label="PSNR 随 bpp 变化的率失真曲线，float 与 BC1 两条序列" '
         f'font-family="system-ui, sans-serif">']
    for p in (30, 40, 50, 60):
        s.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(p):.1f}" y2="{Y(p):.1f}" stroke="var(--grid)"/>')
        s.append(f'<text x="{L-8}" y="{Y(p)+4:.1f}" font-size="11.5" fill="var(--muted)" text-anchor="end">{p}</text>')
    for b in (1, 2, 4, 8, 16):
        s.append(f'<line x1="{X(b):.1f}" x2="{X(b):.1f}" y1="{H-B}" y2="{H-B+4}" stroke="var(--axis)"/>')
        s.append(f'<text x="{X(b):.1f}" y="{H-B+18}" font-size="11.5" fill="var(--muted)" text-anchor="middle">{b}</text>')
    s.append(f'<line x1="{L}" x2="{W-R}" y1="{H-B}" y2="{H-B}" stroke="var(--axis)"/>')
    s.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-8}" font-size="11.5" fill="var(--muted)" text-anchor="middle">bits per texel（log₂）</text>')
    s.append(f'<text x="14" y="{T+6}" font-size="11.5" fill="var(--muted)">PSNR dB</text>')

    for key, color, label in (("psnr_float", "var(--float)", "float"), ("psnr_bc1", "var(--bc1)", "BC1")):
        pts = " ".join(f"{X(r['bpp']):.1f},{Y(r[key]):.1f}" for r in rows)
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        first = rows[0]
        s.append(f'<text x="{X(first["bpp"])+12:.1f}" y="{Y(first[key])+4:.1f}" font-size="12" '
                 f'font-weight="600" fill="var(--ink-2)">{label} {first[key]:.1f} dB</text>')
        for r in rows:
            tip = (f"4×{r['latent_res']}² · {r['bpp']:.2f} bpp\n"
                   f"float {r['psnr_float']:.2f} dB · BC1 {r['psnr_bc1']:.2f} dB\n"
                   f"量化代价 {r['quantization_cost_db']:.2f} dB")
            s.append(f'<circle cx="{X(r["bpp"]):.1f}" cy="{Y(r[key]):.1f}" r="4" fill="{color}" '
                     f'stroke="var(--surface)" stroke-width="2"><title>{tip}</title></circle>')
    s.append("</svg>")
    return "".join(s)


def build_memchart(rows: list[dict], baselines: dict) -> str:
    entries = [
        ("BC7+BC5+BC7（常规质量）", baselines["bc7_bc5_set"]["kib"], 200.0, False),
        ("直接 BC1 ×3（基准 100%）", baselines["direct_bc1"]["kib"], 100.0, False),
    ] + [(f"Neural 4×{r['latent_res']}²", r["kib"], r["pct_vs_direct_bc1"], True) for r in rows]
    W, ROW, BARH, TOP, LX, X0 = 760, 32, 20, 8, 160, 170
    vmax = max(e[1] for e in entries)
    span = 470.0
    H = TOP + len(entries) * ROW + 30

    def X(v: float) -> float:
        return X0 + v / vmax * span

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block" role="img" '
         f'aria-label="各方案 512² 材质集显存占用条形图，以直接 BC1 为 100% 基准" '
         f'font-family="system-ui, sans-serif">']
    ref_x = X(baselines["direct_bc1"]["kib"])
    s.append(f'<line x1="{ref_x:.1f}" x2="{ref_x:.1f}" y1="{TOP}" y2="{H-24}" '
             f'stroke="var(--muted)" stroke-dasharray="4 4"/>')
    s.append(f'<text x="{ref_x:.1f}" y="{H-8}" font-size="11" fill="var(--muted)" '
             f'text-anchor="middle">直接 BC1 = 100%</text>')
    for i, (label, v, pct, ours) in enumerate(entries):
        y = TOP + i * ROW
        fill = "var(--bc1)" if ours else "var(--axis)"
        s.append(f'<text x="{LX}" y="{y+BARH/2+4:.1f}" font-size="12" fill="var(--ink-2)" '
                 f'text-anchor="end">{label}</text>')
        s.append(f'<rect x="{X0}" y="{y}" width="{X(v)-X0:.1f}" height="{BARH}" rx="3" fill="{fill}">'
                 f'<title>{label}：{v:.0f} KiB（{pct:.1f}% 于直接 BC1）</title></rect>')
        s.append(f'<text x="{X(v)+8:.1f}" y="{y+BARH/2+4:.1f}" font-size="11.5" '
                 f'fill="var(--ink-2)">{v:.0f} KiB · {pct:.0f}%</text>')
    s.append("</svg>")
    return "".join(s)


def cmp_rows(rows: list[dict]) -> str:
    out = ['<div class="rowlab"></div>',
           '<div class="head">BaseColor 放大</div><div class="head">BaseColor 误差 ×4</div>',
           '<div class="head">Normal 放大</div><div class="head">Normal 误差 ×4</div>',
           '<div class="rowlab"><b>参考</b><span class="m">源材质（未压缩）</span></div>',
           f'<img src="{data_uri("ref_basecolor_zoom.png")}" alt="参考 BaseColor 放大裁剪">',
           '<div class="ph">—</div>',
           f'<img src="{data_uri("ref_normal_zoom.png")}" alt="参考 Normal 放大裁剪">',
           '<div class="ph">—</div>']
    by_res = {r["latent_res"]: r for r in rows}
    for res in RES_ORDER:
        r = by_res[res]
        out.append(f'<div class="rowlab"><b>4 × {res}²</b><span class="m">{r["bpp"]:.1f} bpp · '
                   f'{r["ratio_vs_bc7"]:.1f}× · {r["psnr_bc1"]:.1f} dB</span></div>')
        for view in ("basecolor", "normal"):
            out.append(f'<img src="{data_uri(f"bc1_{res}_{view}_zoom.png")}" alt="BC1 {res} {view} 重建">')
            out.append(f'<img src="{data_uri(f"bc1_{res}_{view}_diff.png")}" alt="BC1 {res} {view} 误差图">')
    return "".join(out)


def table_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        cls = ' class="pick"' if r["latent_res"] == 384 else ""
        best = ' class="best"' if r["latent_res"] == 512 else ""
        out.append(
            f'<tr{cls}><td>4 × {r["latent_res"]}²</td><td>{r["bpp"]:.2f}</td>'
            f'<td>{r["kib"]:.0f}</td><td>{r["pct_vs_direct_bc1"]:.1f}%</td>'
            f'<td>{r["ratio_vs_uncompressed"]:.1f}×</td><td>{r["ratio_vs_bc7"]:.1f}×</td>'
            f'<td>{r["psnr_float"]:.2f}</td><td{best}>{r["psnr_bc1"]:.2f}</td>'
            f'<td>{r["quantization_cost_db"]:.2f}</td>'
            f'<td>{r["bc1_psnr_basecolor"]:.2f}</td><td>{r["bc1_psnr_normal"]:.2f}</td>'
            f'<td>{r["bc1_psnr_ao"]:.2f}</td><td>{r["bc1_psnr_roughness"]:.2f}</td>'
            f'<td>{r["bc1_psnr_metallic"]:.2f}</td></tr>')
    return "".join(out)


def fulls() -> str:
    out = []
    for view in ("basecolor", "normal"):
        for prefix, caption in FULL_VIEWS:
            name = f"{prefix}_{view}_full.png"
            cap = f"{caption} · {'BaseColor' if view == 'basecolor' else 'Normal'}"
            out.append(f'<figure><img src="{data_uri(name)}" alt="{cap} 整图">'
                       f'<figcaption>{cap}</figcaption></figure>')
    return "".join(out)


def main() -> None:
    results = json.loads((SWEEP / "results.json").read_text())
    rows = results["summary"]  # ordered 512 -> 128
    tpl = (HERE / "report_template.html").read_text(encoding="utf-8")
    html = (tpl.replace("<!--CHART-->", build_chart(rows))
               .replace("<!--MEMCHART-->", build_memchart(rows, results["baselines"]))
               .replace("<!--TABLE_ROWS-->", table_rows(rows))
               .replace("<!--CMP_ROWS-->", cmp_rows(rows))
               .replace("<!--FULLS-->", fulls()))
    OUT.write_text(html, encoding="utf-8")
    print(f"report ({OUT.stat().st_size / 1e6:.1f} MB) -> {OUT}")


if __name__ == "__main__":
    main()
