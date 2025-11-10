import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------- helpers ----------
def box(ax, key, text, x, y, w, h, fs=9.5, store=None):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           ec="gray", fc="skyblue", alpha=0.6, linewidth=1.4)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, wrap=True)
    if store is not None:
        store[key] = dict(x=x, y=y, w=w, h=h)
    return patch

def anchor(b, side, pad=0.04):
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    if side == "left":   return (x - pad, y + h/2)
    if side == "right":  return (x + w + pad, y + h/2)
    if side == "top":    return (x + w/2, y + h + pad)
    if side == "bottom": return (x + w/2, y - pad)
    raise ValueError(side)

def edge_anchor(b, side, t=0.5, pad=0.04):
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    if side == "bottom": return (x + t*w, y - pad)
    if side == "top":    return (x + t*w, y + h + pad)
    if side == "left":   return (x - pad,     y + t*h)
    if side == "right":  return (x + w + pad, y + t*h)
    raise ValueError(side)

def arrow(ax, p1, p2, lw=1.5):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=10,
                                 color="gray", lw=lw))

# ---------- canvas ----------
fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(-1, 15)
ax.set_ylim(1.0, 12.0)
ax.axis("off")

plt.title("Fig-1: Updated Methodology — Dual Engines & Explainability Boundaries",
          fontsize=12, weight="bold")

B = {}

# uniform vertical rows
top    = 10.5
row1   = 8.8
row2   = 7.0
row3   = 5.2
row4   = 3.5
bottom = 1.5

# =================== TOP ===================
box(ax, "ui_top",
    "AI-Powered Decision UI\n(Streamlit Web Interface)\n• Upload resume & JD\n• Mode select (Classical / LLM)",
    0.6, top, 3.2, 1.1, store=B)

box(ax, "proc",
    "Resume Processing Layer\n• Text extraction (pdfplumber, python-docx)\n• Clean text (NLTK)\n• Named entity recognition (spaCy)",
    4.5, top, 4.0, 1.1, store=B)

box(ax, "sbert",
    "Feature Extraction Layer\n• SBERT embeddings (all-MiniLM-L6-v2)",
    9.2, top, 3.2, 1.1, store=B)

arrow(ax, anchor(B["ui_top"], "right"), anchor(B["proc"],  "left"))
arrow(ax, anchor(B["proc"],   "right"), anchor(B["sbert"], "left"))

# =================== BRANCH HEADERS (with in-box subtitles) ===================
box(ax, "head_cls",
    "Branch A — Classical Engine\n(SBERT → XGBoost)\n— SHAP explainability",
    5.0, row1, 3.2, 1.2, fs=9.2, store=B)

box(ax, "head_llm",
    "Branch B — LLM Baseline\n(GPT-4o / GPT-4o-mini)\n— TF-IDF transparency (no SHAP)",
    9.4, row1, 3.0, 1.2, fs=9.2, store=B)

# diagonals from SBERT to branch headers (edge-to-edge)
arrow(ax, edge_anchor(B["sbert"], "bottom", t=0.35), anchor(B["head_cls"], "top"))
# choose t so the diagonal aims at the vertical center of the LLM column
t_llm = (B["head_llm"]["x"] + B["head_llm"]["w"]/2 - B["sbert"]["x"]) / B["sbert"]["w"]
t_llm = max(0.05, min(0.95, t_llm))
arrow(ax, edge_anchor(B["sbert"], "bottom", t=t_llm), anchor(B["head_llm"], "top"))

# =================== ROW 2 ===================
box(ax, "train", "Training & Evaluation\n• 80/20 stratified split\n• XGBoost classifier",
    5.0, row2, 3.2, 1.0, store=B)
arrow(ax, anchor(B["head_cls"], "bottom"), anchor(B["train"], "top"))

box(ax, "llm", "LLM Screening\n• Resume + JD prompt\n• JSON output parsing",
    9.4, row2, 3.0, 1.0, store=B)
arrow(ax, anchor(B["head_llm"], "bottom"), anchor(B["llm"], "top"))

# =================== ROW 3 ===================
box(ax, "xai", "Explainability (XAI)\n• SHAP (global + local) over embeddings\n• TF-IDF terms (human-readable)",
    5.0, row3, 3.2, 1.1, store=B)
arrow(ax, anchor(B["train"], "bottom"), anchor(B["xai"], "top"))

box(ax, "transp", "Transparency (Heuristic)\n• TF-IDF key terms only\n( no SHAP on LLM )",
    9.4, row3, 3.0, 1.1, store=B)
arrow(ax, anchor(B["llm"], "bottom"), anchor(B["transp"], "top"))

# =================== ROW 4 OUTPUTS ===================
box(ax, "out_cls",
    "Outputs (Classical)\n• Predicted category + probability\n• Classification report & normalized confusion matrix",
    5.0, row4, 3.2, 1.1, store=B)
arrow(ax, anchor(B["xai"], "bottom"), anchor(B["out_cls"], "top"))

# LLM output same size/positioning under Transparency; vertical arrow
out_llm_w = B["transp"]["w"]
out_llm_x = B["transp"]["x"]
out_llm_h = 1.1
out_llm_y = row4

box(ax, "out_llm",
    "Outputs (LLM Baseline)\n• Match score (0–1) + predicted category\n• Optional: sample-based classification report\n  & normalized confusion matrix",
    out_llm_x, out_llm_y, out_llm_w, out_llm_h, store=B)

arrow(ax, anchor(B["transp"], "bottom"), anchor(B["out_llm"], "top"))

# =================== FINAL UI + ETHICS ===================
box(ax, "ui_disp",
    "UI Display Elements\n• Resume match score\n• Predicted category / prob.\n• SHAP (Classical)\n• TF-IDF key terms (both)\n• Reports & confusion matrices",
    7.0, bottom, 4.5, 1.3, store=B)

arrow(ax, anchor(B["out_cls"], "bottom"), anchor(B["ui_disp"], "top"))
arrow(ax, anchor(B["out_llm"], "bottom"), anchor(B["ui_disp"], "top"))

box(ax, "ethics",
    "Ethics & Governance\n• Human-in-the-loop\n• Bias & transparency awareness\n• Reproducibility vs API dependency",
    2.0, bottom, 4.5, 1.3, store=B)

arrow(ax, anchor(B["ethics"], "right"), anchor(B["ui_disp"], "left"))

plt.tight_layout()
plt.show()
