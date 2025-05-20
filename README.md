# 🫁 Lung Cancer Risk Predictor

A Flask‑powered web application that uses a TensorFlow/Keras deep‑learning model to predict lung cancer risk levels (**Low**, **Medium**, **High**) based on the **top 15** clinical, lifestyle, and environmental factors. Trained and tested on **50,000**‑sample datasets for robust, real‑world performance.

---

## 🔍 Overview

This end‑to‑end system enables:

1. **User Input**  
   Enter patient details (Age, Gender, Smoking history, Air pollution, etc.) via a responsive web form.

2. **Preprocessing**  
   Data is scaled with `StandardScaler`, and only the **15 selected features** are fed to the model.

3. **Prediction**  
   A trained MLP model classifies risk into Low/Medium/High.

4. **Explainability**  
   SHAP visualizations (summary and waterfall plots) embedded in the results page to show feature contributions.

5. **Deployment**  
   Flask backend serves the model, HTML/CSS/JS front‑end with Lottie animations, Swiper.js testimonials, and dark/light theme.

<img src="screenshots/result_preview.png" alt="Result Page Preview" width="600"/>

---

## 🎯 Features

- ✨ **Glassmorphic** UI with dark/light toggle  
- 📽️ **Lottie** animations & **Particles.js** background  
- 📱 Fully **responsive** design  
- 🧠 Deep‑learning model (TensorFlow/Keras MLP) using **top 15** features  
- ⚖️ **SMOTE** for class imbalance  
- 🔄 **SHAP** explainability (summary + waterfall plots)  
- 📊 t‑SNE visualization of test embeddings  
- 🧪 CLI **test** script for batch predictions  
- 🔧 **Retrainable** via `train.py`  
- 🐳 Docker support (optional)

---

## ⚙️ Installation & Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/bindu2607/lung-cancer-predictor.git
cd lung-cancer-predictor

# 2. (Optional) Create a virtual environment
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the Flask app
python app.py
# ➜ Open http://127.0.0.1:5000 in your browser

# 5. (Optional) Re‑train the model
python train.py

# 6. (Optional) Run batch predictions/test
python test.py

# 7. (Optional) Generate SHAP report
python shap_explain.py


---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👩‍💻 Author

**Marpini Himabindu**  
B.Tech in Information Technology (2022–2026)  
[GitHub](https://github.com/bindu2607) | [LinkedIn](https://www.linkedin.com/in/your-linkedin/)

---

*For questions, suggestions, or contributions, please open an issue or submit a pull request!*


