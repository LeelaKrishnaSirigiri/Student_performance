import streamlit as st
import pickle
import pandas as pd

# ---------------- LOAD FILES ---------------- #
model = pickle.load(open("model.pkl", "rb"))
scalerr = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Student Score Predictor", page_icon="🎓", layout="wide")

# ---------------- HEADER ---------------- #
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎓 Student Exam Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict student performance using Machine Learning</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- INFO NOTE ---------------- #

st.info("This model predicts the score typically below 75%")
# ---------------- INPUT SECTION ---------------- #
st.subheader("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    Hours_Studied = st.slider("📚 Hours Studied", 0, 12, 5)
    Attendance = st.slider("🏫 Attendance (%)", 0, 100, 75)
    Previous_Scores = st.slider("📊 Previous Scores", 0, 100, 70)
    Sleep_Hours = st.slider("😴 Sleep Hours", 0, 12, 7)

with col2:
    Motivation_Level = st.selectbox("🔥 Motivation Level", ["Low","Medium","High"], index=1)
    Internet_Access = st.selectbox("🌐 Internet Access", ["Yes","No"])
    Parental_Involvement = st.selectbox("👨‍👩‍👧 Parental Involvement", ["Low","Medium","High"], index=1)

st.markdown("---")

# ---------------- PREDICT ---------------- #
if st.button("🚀 Predict Score"):

    # -------- INPUT DATA -------- #
    data = {
        "Hours_Studied": Hours_Studied,
        "Attendance": Attendance,
        "Previous_Scores": Previous_Scores,
        "Sleep_Hours": Sleep_Hours,
        "Motivation_Level": Motivation_Level,
        "Internet_Access": Internet_Access,
        "Parental_Involvement": Parental_Involvement,

        # -------- AUTO-FILL -------- #
        "Access_to_Resources": "Medium",
        "Extracurricular_Activities": "Yes",
        "Tutoring_Sessions": 2,
        "Family_Income": "Medium",
        "Teacher_Quality": "Good",
        "School_Type": "Public",
        "Peer_Influence": "Neutral",
        "Physical_Activity": 3,
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "Graduate",
        "Distance_from_Home": "Near",
        "Gender": "Male"
    }

    df = pd.DataFrame([data])

    # -------- ENCODING -------- #
    for col in encoders:
        le = encoders[col]
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    # -------- ALIGN + SCALE -------- #
    df = df[model.feature_names_in_]
    df = scalerr.transform(df)

    # -------- PREDICTION -------- #
    prediction = model.predict(df)[0]

    # -------- PERFORMANCE CATEGORY -------- #
    if prediction < 50:
        performance = "⚠️ Average"
        st.warning(performance)
    elif prediction < 65:
        performance = "👍 Good"
        st.info(performance)
    else:
        performance = "🔥 Excellent"
        st.success(performance)

    # -------- DISPLAY SCORE -------- #
    st.markdown(f"### 🎯 Predicted Exam Score: **{round(prediction, 2)}**")

    # -------- PROGRESS BAR -------- #
    st.progress(min(int(prediction), 100))

    # -------- FUN EFFECT -------- #
    if prediction > 65:
        st.balloons()

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("ℹ️ About")
st.sidebar.info(
"""
This app predicts student exam scores using Machine Learning.
"""
)

st.sidebar.markdown("👨‍💻 Developed by Leela Krishna")
