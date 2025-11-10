import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, text, xy, width, height):
    x, y = xy
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02",
                         ec="gray", fc="skyblue", alpha=0.6)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha="center", va="center", fontsize=11, wrap=True)

def draw_arrow(ax, xy_from, xy_to):
    ax.annotate("",
                xy=xy_to, xycoords='data',
                xytext=xy_from, textcoords='data',
                arrowprops=dict(arrowstyle="->", color='gray', lw=1.5))

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(-1, 11)
ax.set_ylim(0, 10)
ax.axis("off")

# Draw boxes
draw_box(ax, "AI-Powered Decision UI\n(Streamlit Web Interface)", (0.5, 7.5), 2.5, 1.3)
draw_box(ax, "Resume Processing Layer\n1) Text Extraction (pdfplumber, python-docx)\n2) Clean Text (NLTK)\n3) Named Entity Recognition (spaCy)", (3.5, 7.5), 3.5, 1.3)
draw_box(ax, "Feature Extraction Layer\n1) SBERT Embeddings ", (7.5, 7.5), 2.5, 1.3)

draw_box(ax, "ML Model Training\n1) Split the dataset into 80/20\n2) XGBoost Classifier", (7.5, 5.3), 2.5, 1.3)
draw_box(ax, "Resume Job Matching\n1) Semantic Scoring (SBERT + Cosine)\n2) Resume Ranking Algorithm", (3.5, 5.3), 3.5, 1.3)
draw_box(ax, "Explainability Layer\n1) SHAP\n2) Influential Words", (0.5, 5.3), 2.5, 1.3)
draw_box(ax, "AI-Powered Decision UI\n1) Resume Match Score\n2) Display Extracted Details\n3) Influential Words", (0.5, 2.6), 2.5, 1.5)

# Draw arrows
draw_arrow(ax, (3.0, 8.0), (3.5, 8.0))  # UI → Processing
draw_arrow(ax, (7.0, 8.0), (7.5, 8.0))  # Processing → Feature
draw_arrow(ax, (8.75, 7.5), (8.75, 6.6))  # Feature → ML
draw_arrow(ax, (7.5, 6.0), (7.0, 6.0))  # ML → Matching
draw_arrow(ax, (3.5, 6.0), (3.0, 6.0))  # Matching → Explainability
draw_arrow(ax, (1.75, 5.3), (1.75, 4.1))  # Explainability → UI

plt.title("Fig-1: Methodology Flow Diagram", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()
