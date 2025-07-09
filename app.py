# 📦Streamlit Web App: EV Charging Station Demand Predictor

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="⚡ EV Demand Predictor", layout="wide", page_icon="🔋")

# Load model and data
@st.cache_data
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("ev_data.csv")

model = load_model()
data = load_data()

# Sidebar navigation
st.sidebar.image("https://img.freepik.com/free-vector/electric-vehicle-charger-location-pins_78370-1720.jpg", use_container_width=True)
st.sidebar.title("🔌 EV Demand Predictor")
page = st.sidebar.radio("📁 Select a Page", ["🏠 Home", "📊 Dataset", "📈 Insights", "🔮 Prediction", "👩‍💻 About"])

# -------------------- HOME --------------------
if page == "🏠 Home":
    st.title("⚡ Powering Tomorrow's Drive: EV Charging Demand Predictor")
    st.markdown("""
    <style> .big-font { font-size:24px !important; font-weight:500; } </style>
    <p class="big-font">
    Curious where EV charging stations will be needed most? This smart tool helps city planners, EV businesses,
    and enthusiasts predict future station needs based on data-driven demand insights.
    </p>
    """, unsafe_allow_html=True)

    st.image("https://www.chargepoint.com/sites/default/files/pr-image/2023-06/Chargepoint-282.jpg", width=700)

    st.markdown("""
    <br>
    🚗 Whether you're a policy maker or a curious student, this tool lets you explore where EV demand is growing the fastest.
    <br>
    🌍 Let’s plan a cleaner, greener tomorrow — today.
    """, unsafe_allow_html=True)

# -------------------- DATASET --------------------
elif page == "📊 Dataset":
    st.header("🔍 Explore the EV Station Dataset")
    st.markdown("This data shows city-wise EV station count, population and vehicle density.")
    if st.checkbox("📂 Show Dataset Table"):
        st.dataframe(data)
        st.success(f"📌 Total Records: {data.shape[0]}")

# -------------------- INSIGHTS --------------------
elif page == "📈 Insights":
    st.header("📈 Visual Insights")

    st.subheader("🔹 Correlation Heatmap")
    corr = data[["Station_Count", "Population_Density", "EV_Vehicle_Density"]].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("🔹 Top 10 Cities with Most EV Stations")
    top_cities = data.sort_values(by="Station_Count", ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_cities, x="Station_Count", y="City", hue="State", ax=ax2)
    st.pyplot(fig2)

    st.subheader("📈 Line Chart of Station Counts")
    st.line_chart(data.sort_values(by="Station_Count", ascending=False).head(20).set_index("City")["Station_Count"])

# -------------------- PREDICTION --------------------
elif page == "🔮 Prediction":
    st.header("🔮 Predict Future EV Station Demand")
    st.markdown("Input your area characteristics below to estimate how many charging stations might be needed:")

    pop_density = st.slider("🏙️ Population Density (per sq.km)", 100, 5000, step=100)
    ev_density = st.slider("🚗 EV Vehicle Density (per sq.km)", 10, 300, step=10)

    if st.button("🔍 Predict Now"):
        pred = model.predict([[pop_density, ev_density]])[0]
        demand_level = "🔴 High Demand - Rapid Infrastructure Needed!" if pred > 8 else ("🟡 Medium Demand - Growing Interest" if pred > 5 else "🟢 Low Demand - Emerging Area")
        st.markdown(f"<h3 style='color: teal;'>📍 Estimated Station Count: {round(pred)}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: gray;'>{demand_level}</h4>", unsafe_allow_html=True)

# -------------------- ABOUT --------------------
elif page == "👩‍💻 About":
    st.header("👩‍💻 About Me")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135823.png", width=120)
    st.markdown("""
    **Name:** Pratiksha Gorakh Bade  
    **📩Email:** badepratiksha37@gmail.com  
    **🔗LinkedIn:** [linkedin.com/in/pratiksha-bade-2992b7306](https://linkedin.com/in/pratiksha-bade-2992b7306)
    """)

    st.markdown("---")
    st.markdown("""
    🔋 This project predicts the demand for EV charging stations using regression-based insights from population and EV vehicle density.  
    It is built with the goal of supporting smarter urban infrastructure planning and sustainable mobility solutions.
    """)


    st.markdown("""
    <style>
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
