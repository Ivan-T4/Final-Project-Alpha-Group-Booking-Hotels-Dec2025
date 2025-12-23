import os

# IMPORTS
# ======================================================
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pycountry
import datetime as dt
import calendar


# PAGE CONFIGURATION
# ======================================================
st.set_page_config(page_title="Hotel Booking Cancellation Predictor",
                   layout="wide",
                   page_icon="🏨")


# HEADER SECTION
# ======================================================
st.title("🏨 Hotel Booking Cancellation Predictor")


st.markdown("---")


# HELPERS
# ======================================================
@st.cache_data
def get_country_options():
    """Return a sorted list of ISO-3166 country names."""
    return sorted({c.name for c in pycountry.countries})

def month_to_season(month_num: int) -> str:
    """Map month number (1-12) to season. Adjust if your business uses different seasons."""
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }
    return season_map[month_num]

def compute_iso_week_number(year: int, month: int, day: int) -> int:
    """ISO week number: 1..53"""
    return dt.date(year, month, day).isocalendar().week

def split_week_vs_weekend_nights(arrival_date: dt.date, total_nights: int):
    """
    Count week vs weekend nights based on arrival date and total nights.
    Weekend nights = Saturday or Sunday.
    Returns floats to match your schema (float64).
    """
    week_nights = 0
    weekend_nights = 0
    d = arrival_date
    for _ in range(max(0, int(total_nights))):
        if d.weekday() in (5, 6):  # Saturday=5, Sunday=6
            weekend_nights += 1
        else:
            week_nights += 1
        d += dt.timedelta(days=1)
    return float(week_nights), float(weekend_nights)

def max_day_of_month(year: int, month: int) -> int:
    """Get the max valid day for the given year/month (handles leap years)."""
    return calendar.monthrange(year, month)[1]


# SIDEBAR INPUT SECTION
# ======================================================
st.sidebar.header("📋 Booking Details Input")

def user_input_features(country_options: list):
    # Month names (object) to match dataset
    MONTH_NAMES = list(calendar.month_name)[1:]  # ['January', ..., 'December']
    name_to_num = {name: i for i, name in enumerate(MONTH_NAMES, start=1)}

    # Arrival date inputs
    arrival_date_year = st.sidebar.number_input("Arrival Year", 2015, 2017, 2016)
    arrival_date_month_name = st.sidebar.selectbox("Arrival Month", options=MONTH_NAMES, index=6)  # July default
    arrival_month_num = name_to_num[arrival_date_month_name]
    dom_max = max_day_of_month(arrival_date_year, arrival_month_num)
    arrival_date_day_of_month = st.sidebar.number_input("Arrival Day of Month", 1, dom_max, min(dom_max, 15))
    arrival_dt = dt.date(arrival_date_year, arrival_month_num, arrival_date_day_of_month)

    # Derived fields
    arrival_date_week_number = compute_iso_week_number(arrival_date_year, arrival_month_num, arrival_date_day_of_month)
    season = month_to_season(arrival_month_num)

    # Core numerics
    lead_time = st.sidebar.number_input("Lead Time (days before arrival)", 0, 730, 50)
    adults = st.sidebar.number_input("Adults", 1, 10, 2)
    children = st.sidebar.number_input("Children", 0, 10, 0)
    total_nights = st.sidebar.number_input("Total Nights (length of stay)", 0, 60, 3)
    stays_in_week_nights, stays_in_weekend_nights = split_week_vs_weekend_nights(arrival_dt, total_nights)

    # Other features
    adr = st.sidebar.number_input("Average Daily Rate (USD)", 0.0, 2000.0, 100.0)
    required_car_parking_spaces = st.sidebar.number_input("Required Car Parking Spaces", 0, 5, 0)
    total_of_special_requests = st.sidebar.number_input("Total Special Requests", 0, 5, 0)
    booking_changes = st.sidebar.number_input("Number of Booking Changes", 0, 10, 0)
    is_repeated_guest = st.sidebar.selectbox("Is Repeated Guest", [0, 1])

    hotel = st.sidebar.selectbox("Hotel Type", ['City Hotel', 'Resort Hotel'])
    deposit_type = st.sidebar.selectbox("Deposit Type", ['No Deposit', 'Refundable', 'Non Refund'])
    customer_type = st.sidebar.selectbox("Customer Type", ['Transient', 'Transient-Party', 'Contract', 'Group'])

    # Complete lists matching Kaggle dataset categories
    market_segment = st.sidebar.selectbox(
        "Market Segment",
        ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups', 'Undefined', 'Aviation']
    )
    distribution_channel = st.sidebar.selectbox(
        "Distribution Channel",
        ['Direct', 'TA/TO', 'Corporate', 'GDS', 'Undefined']
    )

    group_type = st.sidebar.selectbox("Group Type", ['Solo', 'Couple', 'Family', 'Group', 'Other'])

    # Assigned room type — ideally load from your training data's uniques
    assigned_room_type = st.sidebar.selectbox(
        "Assigned Room Type", options=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'P']
    )
    meal = st.sidebar.selectbox("Meal Type", ['BB', 'HB', 'FB', 'SC', 'Undefined'])

    # Country via dropdown (ISO-3166 list)
    default_idx = country_options.index("United States") if "United States" in country_options else 0
    country_full = st.sidebar.selectbox("Country (Full Name)", country_options, index=default_idx)

    total_guests = float(adults + children)
    total_stay = float(total_nights)

    # Build dataframe with derived fields
    df = pd.DataFrame({
        'hotel': [hotel],
        'lead_time': [lead_time],
        'arrival_date_year': [arrival_date_year],
        'arrival_date_month': [arrival_date_month_name],       # month names (object)
        'arrival_date_week_number': [arrival_date_week_number],
        'arrival_date_day_of_month': [arrival_date_day_of_month],
        'stays_in_weekend_nights': [stays_in_weekend_nights],  # derived
        'stays_in_week_nights': [stays_in_week_nights],        # derived
        'adults': [adults],
        'children': [float(children)],                         # keep float64 compatibility
        'meal': [meal],
        'market_segment': [market_segment],
        'distribution_channel': [distribution_channel],
        'is_repeated_guest': [is_repeated_guest],
        'assigned_room_type': [assigned_room_type],
        'booking_changes': [booking_changes],
        'deposit_type': [deposit_type],
        'customer_type': [customer_type],
        'adr': [adr],
        'required_car_parking_spaces': [required_car_parking_spaces],
        'total_of_special_requests': [total_of_special_requests],
        'total_guests': [total_guests],      # derived
        'total_stay': [total_stay],          # derived
        'group_type': [group_type],
        'season': [season],                  # derived
        'country_full': [country_full]
    })
    
    return df

# Prepare cached country list and call the function
country_options = get_country_options()
df_input = user_input_features(country_options)
df_input.index = ["Input"]


# LOAD MODEL
# ======================================================
try:
    model = pickle.load(open('best_model_xgbr_hotel_bookings.sav', 'rb'))
except FileNotFoundError:
    st.error(" Model file not found: 'best_model_xgbr_hotel_bookings.sav'. Please upload or check the path.")
    st.stop()
except pickle.UnpicklingError as e:
    st.error(f" Could not unpickle the model: {e}")
    st.stop()



# PREDICTION SECTION
# ======================================================

st.subheader("🎯 Prediction Result")

try:
    # Because you saved the full pipeline, you can predict on raw df_input
    prediction = int(model.predict(df_input)[0])

    # XGBClassifier supports predict_proba; positive class is index 1
    proba = model.predict_proba(df_input)[0]
    prediction_proba = float(proba[1]) if len(proba) > 1 else float(proba)
except Exception as e:
    st.error(f" Prediction failed: {e}")
    st.stop()


col1, col2 = st.columns(2)

with col1:
    st.write("### Booking Information")
    st.dataframe(df_input.transpose(), width='stretch')

with col2:
    st.write("### Cancellation Prediction")
    if prediction == 1:
        st.error(f"⚠️ **Likely to Cancel** (Probability: {prediction_proba:.2%})")
    else:
        st.success(f"✅ **Likely to Stay** (Probability of Cancellation: {prediction_proba:.2%})")


# COST IMPACT SECTION
# ======================================================
st.markdown("---")
st.subheader("💰 Estimated Financial Impact")


# ADR × total_stay (days) = revenue; adjust currency label as needed
try:
    potential_loss = float(df_input['adr'][0]) * float(df_input['total_stay'][0])
except Exception:
    potential_loss = np.nan

if int(prediction) == 1:
    st.error(f"Estimated potential revenue loss if cancelled: **${potential_loss:,.2f}**")
    st.caption("Consider contacting this guest or offering flexible rebooking options.")
else:
    st.success(f"Expected revenue from this booking: **${potential_loss:,.2f}**")
    st.caption("Low cancellation risk — monitor normally.")



# FEATURE IMPORTANCE SECTION
# ======================================================


st.markdown("---")
st.subheader("📊 Model Feature Importances (XGBoost) model")

try:
    preprocess = model.named_steps['preprocess']
    clf = model.named_steps['model']

    importances = clf.feature_importances_

    try:
        feature_names = preprocess.get_feature_names_out()
    except:
        feature_names = [f"f{i}" for i in range(len(importances))]

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
    )

    # Horizontal barplot
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5*len(importance_df))))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        orient="h",
        palette="Blues",
        ax=ax,
        edgecolor="black"
    )

    ax.set_title("Top 10 Feature Importances", fontsize=12, pad=12)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

except Exception as e:
    st.warning("Could not compute feature importances.")
    st.caption(str(e))



# SHAP EXPLANATION SECTION
# ======================================================
st.markdown("---")
st.subheader("🧠 Model Explainability (SHAP Values) for the inputed data")


try:
    preprocess = model.named_steps['preprocess']
    clf = model.named_steps['model']

    X_enc = preprocess.transform(df_input)
    X_enc_dense = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc

    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_enc_dense.shape[1])]

    
    # Explainer
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_enc_dense)

    # ---- Local (first row) waterfall ----
    if isinstance(shap_values, list):
        shap_vec = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        base_val = explainer.expected_value[1] if hasattr(explainer, "expected_value") and len(shap_values) > 1 else explainer.expected_value
    else:
        shap_vec = shap_values[0]
        base_val = explainer.expected_value

    exp = shap.Explanation(
        values=shap_vec,
        base_values=base_val,
        data=X_enc_dense[0],
        feature_names=feature_names
    )

    fig2 = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(exp, max_display=10, show=False)
    st.pyplot(fig2)


except Exception as e:
    st.warning("SHAP explanation not available or not supported by this model.")
    st.caption(str(e))

# ======================================================

