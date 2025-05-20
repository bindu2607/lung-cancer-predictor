# shap_explain.py

import os
import pickle
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

from tensorflow.keras.models import load_model

# ─── Configuration ──────────────────────────────────────────────
MODEL_PATH        = 'models/lung_cancer_model_final.keras'
SCALER_PATH       = 'models/scaler_final.pkl'
DATA_PATH         = 'lung_cancer_clean_50k.csv'

# All outputs go under Flask’s static/ folder
EXPLAIN_PLOTS_DIR = 'static/shap_plots'
PDF_OUTPUT_PATH   = 'static/shap_report.pdf'


# Top‑15 features
TOP_15_FEATURES = [
    'Obesity','Coughing of Blood','Alcohol use','Dust Allergy','Balanced Diet',
    'Genetic Risk','Passive Smoker','OccuPational Hazards','Chest Pain',
    'Air Pollution','Fatigue','chronic Lung Disease','Smoking',
    'Shortness of Breath','Frequent Cold'
]

# Groups for the grouped-bar chart
FEATURE_GROUPS = {
    'Lifestyle':    ['Smoking','Passive Smoker','Alcohol use','Balanced Diet','Obesity'],
    'Environmental':['Air Pollution','Dust Allergy','OccuPational Hazards'],
    'Symptoms':     ['Fatigue','Chest Pain','Shortness of Breath','Frequent Cold','Coughing of Blood'],
    'Genetic':      ['Genetic Risk','chronic Lung Disease']
}

# Create static output directories
os.makedirs(EXPLAIN_PLOTS_DIR, exist_ok=True)

# ─── Load Data & Model ───────────────────────────────────────────
print("🔄 Loading model, scaler, and data...")
df = pd.read_csv(DATA_PATH)
df.drop(columns=['index','Patient Id'], inplace=True, errors='ignore')

X = df[TOP_15_FEATURES]
y = df['Level'].astype(int).values

scaler = pickle.load(open(SCALER_PATH,'rb'))
model  = load_model(MODEL_PATH)
X_scaled = scaler.transform(X)

# ─── Compute SHAP Values ─────────────────────────────────────────
print("⚙️ Computing SHAP values (500 samples)...")
explainer = shap.Explainer(model, X_scaled[:1000], feature_names=TOP_15_FEATURES)
shap_exp  = explainer(X_scaled[:500])
pred_probs = model.predict(X_scaled[:500])
pred_class = np.argmax(pred_probs, axis=1)

# ─── 1) Summary Bar Plot ─────────────────────────────────────────
print("📊 Generating SHAP mean‑absolute bar plot...")
plt.figure()
shap.summary_plot(
    shap_exp,
    feature_names=TOP_15_FEATURES,
    plot_type="bar",
    show=False
)
plt.savefig(
    os.path.join(EXPLAIN_PLOTS_DIR, 'shap_summary_bar.png'),
    bbox_inches='tight'
)
plt.close()

# ─── 2) Per‑Patient Waterfall (first 3) ──────────────────────────
print("💧 Generating waterfall plots for first 3 patients...")
for i in range(3):
    cls = pred_class[i]
    out_png = os.path.join(EXPLAIN_PLOTS_DIR, f'shap_waterfall_patient{i}.png')
    plt.figure(figsize=(8,5))
    try:
        shap.plots.waterfall(shap_exp[i, cls], show=False)
    except Exception:
        vals = shap_exp.values
        # If 3D: shap_exp.values[i][cls], else 2D: shap_exp.values[i]
        arr = vals[i][cls] if vals.ndim == 3 else vals[i]
        idxs = np.argsort(np.abs(arr))[-10:][::-1]
        feats = [TOP_15_FEATURES[j] for j in idxs]
        contribs = arr[idxs]
        colors = ['red' if v>0 else 'green' for v in contribs]
        plt.barh(feats, contribs, color=colors)
        plt.xlabel("SHAP value")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

# ─── 3) Grouped Feature Contributions (patient 0) ────────────────
print("🔢 Generating grouped contributions for patient 0...")
vals_array = shap_exp.values
if vals_array.ndim == 3:
    vals0 = vals_array[0, pred_class[0], :]
else:
    vals0 = vals_array[0]

group_sums = {}
for grp, feats in FEATURE_GROUPS.items():
    idxs = [TOP_15_FEATURES.index(f) for f in feats]
    group_sums[grp] = float(vals0[idxs].sum())

plt.figure(figsize=(6,4))
plt.bar(
    group_sums.keys(),
    group_sums.values(),
    color=['skyblue','salmon','gold','mediumseagreen']
)
plt.title("Grouped SHAP Contributions (Patient 0)")
plt.ylabel("Sum of SHAP values")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(
    os.path.join(EXPLAIN_PLOTS_DIR, 'shap_grouped_contributions.png')
)
plt.close()

# ─── 4) Example Multilingual Tooltip Output ──────────────────────
print("🌍 Sample multilingual tooltips (Patient 0):")
translations = {
    'Passive Smoker': {
        'en': "Passive smoking contributed {value:.2f}.",
        'te': "ప్యాసివ్ పొగతాగే అలవాటు {value:.2f} ప్రభావం చూపింది.",
        'hi': "निष्क्रिय धूम्रपान ने {value:.2f} योगदान दिया।"
    },
    'Smoking': {
        'en': "Smoking contributed {value:.2f}.",
        'te': "ధూమపానం {value:.2f} ప్రభావం చూపింది.",
        'hi': "धूम्रपान ने {value:.2f} योगदान दिया।"
    }
}
top5 = np.argsort(np.abs(vals0))[-5:][::-1]
for idx in top5:
    feat, val = TOP_15_FEATURES[idx], vals0[idx]
    if feat in translations:
        for lang, tpl in translations[feat].items():
            print(f" - {feat} [{lang}]: {tpl.format(value=val)}")

# ─── 5) Generate PDF Report ──────────────────────────────────────
print("📄 Generating PDF report with captions...")
doc = SimpleDocTemplate(PDF_OUTPUT_PATH, pagesize=letter)
styles = getSampleStyleSheet()
elements = [Paragraph("SHAP Explainability Report", styles['Title']), Spacer(1,12)]

plot_list = [
    ("Mean‑Absolute SHAP (Bar)",   'shap_summary_bar.png'),
    ("Waterfall Patient 0",        'shap_waterfall_patient0.png'),
    ("Waterfall Patient 1",        'shap_waterfall_patient1.png'),
    ("Waterfall Patient 2",        'shap_waterfall_patient2.png'),
    ("Grouped Contributions (0)",  'shap_grouped_contributions.png'),
]

for caption, fname in plot_list:
    img_path = os.path.join(EXPLAIN_PLOTS_DIR, fname)
    elements.append(Paragraph(caption, styles['Heading3']))
    elements.append(Image(img_path, width=400, height=300))
    elements.append(Spacer(1,12))

doc.build(elements)
print(f"✅ PDF report saved to {PDF_OUTPUT_PATH}")
print("✅ Finished SHAP explain generation.")
