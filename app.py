import streamlit as st
import pandas as pd
import psycopg2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Calibration Predictor", layout="centered")

# --- Connect to Supabase DB ---
@st.cache_data
def load_data():
    conn = psycopg2.connect(
        host=st.secrets["DB_HOST"],
        dbname=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASS"],
        port=st.secrets["DB_PORT"],
        sslmode="require"
    )
    df = pd.read_sql("SELECT * FROM sensor_readings", conn)
    conn.close()
    return df

# --- Load data ---
df = load_data()

st.title("🔧 AI-Powered Calibration Predictor")
st.markdown("Predict the optimal calibration factor from real sensor readings.")

st.subheader("📊 Historical Data Sample")
st.dataframe(df.head())

# --- Train model ---
X = df[['raw_value', 'ambient_temp', 'humidity']]
y = df['calibration_factor']
model = LinearRegression()
model.fit(X, y)

# --- User Input ---
st.subheader("🔍 Enter New Sensor Reading")
col1, col2, col3 = st.columns(3)

with col1:
    raw_value = st.number_input("Raw Value", min_value=80.0, max_value=120.0, value=98.5)
with col2:
    ambient_temp = st.number_input("Ambient Temp (°C)", min_value=15.0, max_value=45.0, value=27.0)
with col3:
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=40)

# --- Predict ---
input_df = pd.DataFrame([[raw_value, ambient_temp, humidity]], columns=X.columns)
predicted = model.predict(input_df)[0]
st.success(f"🔮 Predicted Calibration Factor: **{predicted:.4f}**")

# --- Visualization ---
st.subheader("📈 Actual vs Predicted (Historical Data)")
df['predicted'] = model.predict(X)

fig, ax = plt.subplots()
ax.scatter(df['calibration_factor'], df['predicted'], alpha=0.5)
ax.plot([df['calibration_factor'].min(), df['calibration_factor'].max()],
        [df['calibration_factor'].min(), df['calibration_factor'].max()],
        'r--')
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
st.pyplot(fig)
