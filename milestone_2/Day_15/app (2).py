
# app.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.datasets import load_iris
import joblib

st.set_page_config(page_title="Iris Classifier • Streamlit", page_icon="🌸", layout="wide")

# ---------- Load Data & Artifacts ----------
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df, iris.feature_names, iris.target_names

@st.cache_resource
def load_model(artifact_dir: Path):
    model_path = artifact_dir / "model.joblib"
    feature_names_path = artifact_dir / "feature_names.json"
    target_names_path = artifact_dir / "target_names.json"

    if model_path.exists():
        model = joblib.load(model_path)
        feature_names = json.loads(feature_names_path.read_text())
        target_names = json.loads(target_names_path.read_text())
        return model, feature_names, target_names
    else:
        # Fallback to a simple classifier trained on the fly (in case artifacts not present)
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        df, feature_names, target_names = load_data()
        X = df[feature_names]
        y = df["target"]
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
        pipe.fit(X, y)
        return pipe, feature_names, target_names

DATA_DF, FEATURE_NAMES, TARGET_NAMES = load_data()

artifacts_dir = Path("artifacts")
MODEL, model_feature_names, model_target_names = load_model(artifacts_dir)

# ---------- Sidebar Navigation ----------
st.sidebar.title("🔎 Navigation")
mode = st.sidebar.radio("Choose a mode:", ["Prediction", "Explore Data"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use the toggle above to switch between exploring the dataset and making live predictions.")

# ---------- Utilities ----------
def probability_bar_chart(proba: np.ndarray, class_names: list[str]):
    df = pd.DataFrame({"species": class_names, "probability": proba})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(x=alt.X("probability:Q").title("Probability"),
                y=alt.Y("species:N").title("Species"),
                tooltip=["species", alt.Tooltip("probability:Q", format=".2f")])
    )
    st.altair_chart(chart, use_container_width=True)

def badge(text: str, kind: str = "success"):
    if kind == "success":
        st.success(text)
    elif kind == "warning":
        st.warning(text)
    else:
        st.info(text)

# ---------- Prediction Mode ----------
if mode == "Prediction":
    st.title("🌸 Iris Species Predictor")
    st.write("Provide measurements of an iris flower and get the predicted species.")

    # Determine slider bounds from the dataset for a nice UX
    bounds = {col: (float(DATA_DF[col].min()), float(DATA_DF[col].max())) for col in FEATURE_NAMES}

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider(
            "Sepal length (cm)",
            min_value=bounds[FEATURE_NAMES[0]][0], max_value=bounds[FEATURE_NAMES[0]][1],
            value=float(DATA_DF[FEATURE_NAMES[0]].median()),
            help="Length of the sepal — the outer part of the flower that protects the bud."
        )
        sepal_width = st.slider(
            "Sepal width (cm)",
            min_value=bounds[FEATURE_NAMES[1]][0], max_value=bounds[FEATURE_NAMES[1]][1],
            value=float(DATA_DF[FEATURE_NAMES[1]].median()),
            help="Width of the sepal. Wider sepals are common in Setosa."
        )

    with col2:
        petal_length = st.slider(
            "Petal length (cm)",
            min_value=bounds[FEATURE_NAMES[2]][0], max_value=bounds[FEATURE_NAMES[2]][1],
            value=float(DATA_DF[FEATURE_NAMES[2]].median()),
            help="Length of the petal — the colorful part of the flower."
        )
        petal_width = st.slider(
            "Petal width (cm)",
            min_value=bounds[FEATURE_NAMES[3]][0], max_value=bounds[FEATURE_NAMES[3]][1],
            value=float(DATA_DF[FEATURE_NAMES[3]].median()),
            help="Width of the petal. Larger petals often indicate Virginica."
        )

    with st.expander("Show numeric inputs"):
        c1, c2, c3, c4 = st.columns(4)
        with c1: sepal_length = st.number_input("Sepal length", value=float(sepal_length), step=0.1, help="Fine tune exact value.")
        with c2: sepal_width  = st.number_input("Sepal width",  value=float(sepal_width), step=0.1)
        with c3: petal_length = st.number_input("Petal length", value=float(petal_length), step=0.1)
        with c4: petal_width  = st.number_input("Petal width",  value=float(petal_width), step=0.1)

    user_vector = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Species", use_container_width=True)

    if predict_btn:
        # Ensure feature order matches the model
        if model_feature_names and list(model_feature_names) != list(FEATURE_NAMES):
            # Reorder if needed (defensive)
            order = [FEATURE_NAMES.index(f) for f in model_feature_names]
            user_vector = user_vector[:, order]

        proba = MODEL.predict_proba(user_vector)[0]
        pred_idx = int(np.argmax(proba))
        pred_species = model_target_names[pred_idx]

        # Color-coded result
        badge(f"Prediction: **{pred_species.title()}**", kind="success")

        st.subheader("Prediction probabilities")
        probability_bar_chart(proba, model_target_names)

        with st.expander("What does this mean?"):
            st.markdown(
                "- **Setosa**: Typically smaller petals and wider sepals.\n"
                "- **Versicolor**: Intermediate measurements.\n"
                "- **Virginica**: Often has larger petals."
            )
    else:
        st.info("Adjust the sliders and click **Predict Species** to see the result.")

# ---------- Explore Data Mode ----------
else:
    st.title("📊 Explore the Iris Dataset")
    st.write("Use the controls in the sidebar to explore distributions and relationships.")

    st.subheader("Peek at the data")
    st.dataframe(DATA_DF.head())

    st.markdown("### Histogram")
    feature_for_hist = st.selectbox("Choose a feature for the histogram:", FEATURE_NAMES, index=0)
    bins = st.slider("Bins", min_value=5, max_value=50, value=20)

    hist = (
        alt.Chart(DATA_DF)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(f"{feature_for_hist}:Q", bin=alt.Bin(maxbins=bins), title=feature_for_hist),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip(f"{feature_for_hist}:Q", title=feature_for_hist), alt.Tooltip("count():Q", title="Count")]
        )
    )
    st.altair_chart(hist, use_container_width=True)

    st.markdown("### Scatter Plot")
    x_feature = st.selectbox("X-axis", FEATURE_NAMES, index=0, key="xfeat")
    y_feature = st.selectbox("Y-axis", FEATURE_NAMES, index=1, key="yfeat")
    scatter = (
        alt.Chart(DATA_DF)
        .mark_circle(size=80)
        .encode(
            x=alt.X(f"{x_feature}:Q", title=x_feature),
            y=alt.Y(f"{y_feature}:Q", title=y_feature),
            color=alt.Color("species:N", title="Species"),
            tooltip=[x_feature, y_feature, "species"]
        )
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)

    with st.expander("About this dataset"):
        st.markdown(
            "The classic **Iris** dataset contains 150 samples of iris flowers across three species — "
            "**setosa, versicolor, virginica** — with four measured features: "
            f"{', '.join([f.replace('(cm)', '').strip() for f in FEATURE_NAMES])} (all in cm)."
        )

st.caption("Built with Streamlit • scikit-learn • Altair")
