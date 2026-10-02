"""Styling and small HTML/Plotly helpers for the Streamlit UI."""
import plotly.graph_objects as go

PALETTE = {
    "ink": "#15191C", "green": "#0F7B63", "plate": "#F6C90E", "paper": "#F3F5F4",
    "line": "#DCE2DF", "muted": "#5A6670", "red": "#C2410C", "amber": "#B7791F",
}
FUEL_COLORS = {"Petrol": "#0F7B63", "Diesel": "#15191C", "CNG": "#E0A800"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Figtree:wght@400;500;600&family=Oswald:wght@600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Figtree', sans-serif; }
h1, h2, h3 { font-family: 'Bricolage Grotesque', sans-serif !important; letter-spacing: -0.02em; color: #15191C; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

.hero h1 { font-size: clamp(2rem, 4vw, 3rem); margin: 0; line-height: 1.05; font-weight: 800; }
.hero p { color: #5A6670; font-size: 1.05rem; margin: .5rem 0 0; max-width: 46rem; }
.hero .stats { display:flex; gap:1.6rem; flex-wrap:wrap; margin-top:1rem; color:#15191C; font-weight:600; font-size:.95rem; }
.hero .stats span small { display:block; color:#5A6670; font-weight:400; font-size:.8rem; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid #DCE2DF; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: .6rem 1rem; }
.stTabs [aria-selected="true"] { color: #0F7B63; }

/* panels */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; border-color: #DCE2DF; background: #fff; }

/* the one memorable thing: the number plate */
.plate-wrap { display:flex; flex-direction:column; align-items:flex-start; gap:.5rem; margin: .2rem 0 .4rem; }
.plate { display:inline-flex; align-items:stretch; background:#F6C90E; border:4px solid #15191C; border-radius:12px;
  box-shadow: 0 0 0 2px #F6C90E, 0 0 0 4px #15191C; overflow:hidden; max-width:100%; }
.plate .ind { background:#1D3FA8; color:#fff; font-family:'Oswald',sans-serif; font-size:.75rem; letter-spacing:.08em;
  display:flex; align-items:flex-end; justify-content:center; padding:.5rem .55rem; writing-mode:horizontal-tb; }
.plate .amt { font-family:'Oswald',sans-serif; font-weight:700; color:#15191C; font-size:clamp(2.4rem, 6vw, 4rem);
  line-height:1; padding:.55rem 1.1rem .5rem; letter-spacing:.03em; white-space:nowrap; }
.plate .amt small { font-size:.4em; letter-spacing:.12em; margin-left:.4rem; }
.plate-caption { color:#5A6670; font-size:.95rem; }

.band { position:relative; height:10px; border-radius:999px; background:#E7ECEA; margin:1.6rem 0 .3rem; }
.band .fill { position:absolute; top:0; bottom:0; border-radius:999px; background:#0F7B63; opacity:.28; }
.band .dot { position:absolute; top:-5px; width:20px; height:20px; border-radius:50%; background:#0F7B63; border:3px solid #fff;
  box-shadow:0 1px 4px rgba(0,0,0,.35); transform:translateX(-50%); }
.band-labels { display:flex; justify-content:space-between; font-size:.88rem; color:#5A6670; }
.band-labels b { color:#15191C; }

.kpi { background:#fff; border:1px solid #DCE2DF; border-radius:12px; padding:.8rem 1rem; height:100%; }
.kpi .l { color:#5A6670; font-size:.82rem; }
.kpi .v { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.45rem; color:#15191C; }
.kpi .d { font-size:.85rem; font-weight:600; }
.neg { color:#C2410C; } .pos { color:#0F7B63; } .neu { color:#5A6670; }

.verdict { border-radius:12px; padding:.9rem 1.1rem; margin:.6rem 0 1rem; border-left:6px solid; }
.verdict b { font-family:'Bricolage Grotesque',sans-serif; font-size:1.15rem; }
.verdict.good { background:#E6F4EF; border-color:#0F7B63; }
.verdict.warn { background:#FFF6DC; border-color:#E0A800; }
.verdict.bad  { background:#FDECE3; border-color:#C2410C; }

.empty { text-align:center; color:#5A6670; padding:2.5rem 1rem; border:1.5px dashed #DCE2DF; border-radius:14px; background:#fff; }
.small { color:#5A6670; font-size:.85rem; }
</style>
"""


def fmt_price(lakh: float) -> str:
    return f"₹{lakh / 100:.2f} Cr" if lakh >= 100 else f"₹{lakh:.2f} L"


def plate_html(lakh: float, caption: str) -> str:
    if lakh >= 100:
        amt, unit = f"{lakh / 100:.2f}", "CRORE"
    else:
        amt, unit = f"{lakh:.2f}", "LAKH"
    return (f'<div class="plate-wrap"><div class="plate"><div class="ind">IND</div>'
            f'<div class="amt">₹ {amt}<small>{unit}</small></div></div>'
            f'<div class="plate-caption">{caption}</div></div>')


def band_html(low: float, price: float, high: float) -> str:
    span = max(high * 1.08 - low * 0.92, 1e-6)
    lo0 = low * 0.92
    pos = lambda v: (v - lo0) / span * 100
    return (f'<div class="band"><div class="fill" style="left:{pos(low):.1f}%;width:{pos(high) - pos(low):.1f}%"></div>'
            f'<div class="dot" style="left:{pos(price):.1f}%"></div></div>'
            f'<div class="band-labels"><span>Likely low <b>{fmt_price(low)}</b></span>'
            f'<span>Likely high <b>{fmt_price(high)}</b></span></div>')


def kpi_html(label: str, value: str, delta: str = "", tone: str = "neu") -> str:
    d = f'<div class="d {tone}">{delta}</div>' if delta else ""
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>{d}</div>'


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=30, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, sans-serif", color=PALETTE["ink"], size=13),
        legend=dict(orientation="h", y=1.12, x=0), hoverlabel=dict(font_family="Figtree"),
    )
    fig.update_xaxes(gridcolor="#E7ECEA", zeroline=False, linecolor=PALETTE["line"])
    fig.update_yaxes(gridcolor="#E7ECEA", zeroline=False, linecolor=PALETTE["line"])
    return fig
