import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import cv2
import easyocr
import re
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
import time
import unicodedata
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="License Plate Detection", layout="wide", page_icon="🚗")

# Custom CSS for a beautiful premium background
page_bg_css = """
<style>
/* Main background (High-end Developer Dark Mode) */
[data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    background-image: radial-gradient(circle at 50% 0%, #1a2332 0%, #0d1117 70%);
    color: #c9d1d9;
}
/* Sidebar background and text */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
/* Ensure all navigation text in the sidebar is white */
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
/* Metric cards simulating sleek IDE panels */
[data-testid="stMetric"] {
    background-color: #161b22;
    border-radius: 6px;
    padding: 15px;
    border: 1px solid #30363d;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    border-left: 4px solid #58a6ff;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-left-color 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.8);
    border-left: 4px solid #79c0ff;
}
/* Minimalist header styling with monospace feel */
h1, h2, h3 {
    color: #e6edf3 !important;
    font-family: 'Consolas', 'Courier New', monospace;
    font-weight: 600;
}
/* Highlighted metric values */
[data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-family: 'Consolas', 'Courier New', monospace;
    font-weight: 700;
}
/* Subdued metric labels */
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.85rem;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# Title
st.title("🚗 License Plate Detection Dashboard")

# Initialize easyocr reader and cascade
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en', 'hi'], gpu=False)

@st.cache_resource
def load_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    return cv2.CascadeClassifier(cascade_path)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Detection Logs", "Forecasting & Anomalies", "Live Camera", "Upload Image"])

# Helper function
def classify_chars(plate_str):
    plate_str = str(plate_str)
    has_ascii_letters = bool(re.search(r'[a-zA-Z]', plate_str))
    has_ascii_digits = bool(re.search(r'[0-9]', plate_str))
    has_hindi = bool(re.search(r'[\u0900-\u097F]', plate_str))
    has_other_special = bool(re.search(r'[^a-zA-Z0-9\u0900-\u097F]', plate_str))
    return pd.Series({
        'Has_Letters': has_ascii_letters,
        'Has_Digits': has_ascii_digits,
        'Has_Hindi': has_hindi,
        'Has_Special/Other': has_other_special
    })

if page == "Dashboard":
    col_hdr, col_btn = st.columns([8, 2])
    with col_hdr:
        st.header("📊 Analytics Dashboard")
    with col_btn:
        auto_refresh = st.checkbox("Auto-Refresh (2s)")
        if not auto_refresh:
            st.button("↻ Refresh Data")
    
    if auto_refresh:
        time.sleep(2)
        st.rerun()
    csv_file = "plate_log.csv"
    
    if not os.path.exists(csv_file):
        st.warning("No data found. Please run the detection script first.")
    else:
        df = pd.read_csv(csv_file, on_bad_lines='skip')
        if df.empty:
            st.warning("The log file is empty.")
        else:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Hour'] = df['Timestamp'].dt.hour
            df['Day of Week'] = df['Timestamp'].dt.day_name()
            
            # Metrics
            st.subheader("Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Detections", len(df))
            
            df['Length'] = df['Plate Number'].astype(str).apply(len)
            col2.metric("Avg Plate Length", f"{df['Length'].mean():.1f}")
            
            if 'Confidence' in df.columns:
                valid_conf = df.dropna(subset=['Confidence'])
                if not valid_conf.empty:
                    col3.metric("Avg Confidence", f"{valid_conf['Confidence'].mean()*100:.1f}%")
            
            st.divider()
            
            # Layout for charts
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Top 10 Most Frequent Plates")
                st.caption("Shows the 10 license plates most often detected by the system.")
                top_10 = df['Plate Number'].value_counts().head(10).reset_index()
                top_10.columns = ['Plate Number', 'Frequency']
                fig = px.bar(top_10, x='Plate Number', y='Frequency', color='Frequency', color_continuous_scale='Blues')
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.subheader("Hourly Detection Frequency")
                st.caption("Displays the number of plates detected during each hour of the day.")
                hourly = df['Hour'].value_counts().sort_index().reset_index()
                hourly.columns = ['Hour', 'Count']
                fig = px.bar(hourly, x='Hour', y='Count', color='Count', color_continuous_scale='viridis')
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            c3, c4 = st.columns(2)
            
            with c3:
                st.subheader("Cumulative Detections Over Time")
                st.caption("Visualizes the total accumulation of detected license plates over time.")
                df_sorted = df.sort_values('Timestamp').reset_index(drop=True)
                df_sorted['Cumulative'] = range(1, len(df_sorted) + 1)
                fig = px.line(df_sorted, x='Timestamp', y='Cumulative', markers=True)
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
            with c4:
                st.subheader("Character Types")
                st.caption("Breaks down the count of different character types found in the plates.")
                char_stats = df['Plate Number'].apply(classify_chars)
                char_counts = char_stats.sum().reset_index()
                char_counts.columns = ['Character Type', 'Count']
                fig = px.bar(char_counts, x='Character Type', y='Count', color='Character Type')
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

elif page == "Detection Logs":
    col_hdr, col_btn = st.columns([8, 2])
    with col_hdr:
        st.header("📋 Detection Logs")
    with col_btn:
        auto_refresh = st.checkbox("Auto-Refresh (2s)")
        if not auto_refresh:
            st.button("↻ Refresh Logs")
            
    if auto_refresh:
        time.sleep(2)
        st.rerun()
        
    csv_file = "plate_log.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, on_bad_lines='skip')
        st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
        
        st.subheader("Recent Captures")
        if not df.empty and 'Image Path' in df.columns:
            recent = df.tail(10).iloc[::-1] # Last 10, reversed
            cols = st.columns(5)
            for idx, (index, row) in enumerate(recent.iterrows()):
                col = cols[idx % 5]
                img_path = row['Image Path']
                if pd.notna(img_path) and isinstance(img_path, str) and os.path.exists(img_path):
                    with col:
                        # Use container width
                        st.image(img_path, caption=f"{row['Plate Number']}")
    else:
        st.info("No log file found.")

elif page == "Live Camera":
    st.header("🔴 Live Camera Feed")
    st.write("Process live webcam feed and detect license plates in real-time.")
    
    run = st.checkbox('Start Webcam')
    
    # We use an empty container to update the image in place
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        reader = load_reader()
        cascade = load_cascade()
        
        detected_plates = set()
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not read frame from webcam.")
                break
                
            frame_bgr = frame
            # Convert to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            plates = cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in plates:
                # --- 1. Pad the bounding box slightly ---
                pad_y = int(h * 0.15)
                pad_x = int(w * 0.15)
                y1, y2 = max(0, y - pad_y), min(frame_bgr.shape[0], y + h + pad_y)
                x1, x2 = max(0, x - pad_x), min(frame_bgr.shape[1], x + w + pad_x)
                plate_img = frame_bgr[y1:y2, x1:x2]
                
                # --- 2. Aggressive Preprocessing for OCR ---
                plate_img_resized = cv2.resize(plate_img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                plate_img_gray = cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2GRAY)
                plate_img_blur = cv2.GaussianBlur(plate_img_gray, (5, 5), 0)
                _, plate_img_thresh = cv2.threshold(plate_img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                # Force easyocr to only return alphanumeric characters
                result = reader.readtext(plate_img_thresh, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                
                if result:
                    # Draw on RGB frame for streamlit
                    cv2.rectangle(frame_rgb, (x,y), (x+w,y+h), (0,255,0), 2)
                    
                    plate_text = "".join([res[1] for res in result])
                    translation_table = str.maketrans("०१२३४५६७८९", "0123456789")
                    plate_text = plate_text.translate(translation_table)
                    plate_text = unicodedata.normalize('NFKD', plate_text).encode('ascii', 'ignore').decode('ascii')
                    plate_text = str(''.join(filter(str.isalnum, plate_text)).upper())
                    # ----------------------------------------------------
                    # Custom Post Processing: Fixed Indian License Plate format
                    # First 2 chars: Letters (e.g. MH)
                    # Next 2 chars: Numbers (e.g. 12)
                    # Next 1-2 chars: Letters (e.g. AB)
                    # Last 4 chars: Numbers (e.g. 1234)
                    # ----------------------------------------------------
                    if len(plate_text) >= 4:
                        plate_text_str = str(plate_text)
                        state_str = plate_text_str[:2]
                        # Force letters 
                        state_str = state_str.replace('1', 'T').replace('0', 'O').replace('8', 'B')
                        
                        district_str = plate_text_str[2:4]
                        # Force Numbers
                        district_str = district_str.replace('T', '1').replace('O', '0').replace('B', '8').replace('I', '1')
                        
                        plate_text = state_str + district_str + plate_text_str[4:]
                    
                    # For tricky characters like M vs N anywhere else, context is hard without strict regex mapping,
                    # but OpenCV scale resizing generally addresses it best.
                    
                    confidence = 0.0
                    if len(result) > 0:
                        total_conf = 0.0
                        for res in result:
                            total_conf += res[2]
                        confidence = total_conf / len(result)
                    
                    cv2.putText(frame_rgb, f"{plate_text} ({confidence:.2f})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    
                    if plate_text not in detected_plates and len(plate_text) >= 6:
                        detected_plates.add(plate_text)
                        
                        # Save to disk & log
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if not os.path.exists("plates"):
                            os.makedirs("plates")
                        img_path = f"plates/{plate_text}_{int(datetime.now().timestamp())}.jpg"
                        cv2.imwrite(img_path, plate_img)
                        
                        log_file = "plate_log.csv"
                        new_row = pd.DataFrame([{
                            "Plate Number": plate_text,
                            "Timestamp": timestamp_str,
                            "Image Path": img_path,
                            "Confidence": confidence
                        }])
                        if not os.path.exists(log_file):
                            new_row.to_csv(log_file, index=False)
                        else:
                            new_row.to_csv(log_file, mode='a', header=False, index=False)
                        
                        # Show notification
                        st.toast(f"🚗 Detected: {plate_text}", icon="✅")

            FRAME_WINDOW.image(frame_rgb)
        
        cap.release()
    else:
        st.info("Check 'Start Webcam' to begin real-time detection.")

elif page == "Upload Image":
    st.header("🖼️ Upload Image for Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with st.spinner('Loading OCR and Processing...'):
            reader = load_reader()
            cascade = load_cascade()
            
            image = Image.open(uploaded_file)
            frame = np.array(image.convert('RGB'))
            
            # Resize image if it's too large (essential for high-res images > 1MB)
            max_width = 1000
            if frame.shape[1] > max_width:
                scale_factor = max_width / frame.shape[1]
                new_dimensions = (max_width, int(frame.shape[0] * scale_factor))
                frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            plates = cascade.detectMultiScale(gray, 1.1, 4)
            
            detected_plates_info = []
            
            for (x, y, w, h) in plates:
                # --- 1. Pad the bounding box slightly ---
                pad_y = int(h * 0.15)
                pad_x = int(w * 0.15)
                y1, y2 = max(0, y - pad_y), min(frame_bgr.shape[0], y + h + pad_y)
                x1, x2 = max(0, x - pad_x), min(frame_bgr.shape[1], x + w + pad_x)
                plate_img = frame_bgr[y1:y2, x1:x2]
                
                # --- 2. Aggressive Preprocessing for OCR ---
                plate_img_resized = cv2.resize(plate_img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                plate_img_gray = cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2GRAY)
                plate_img_blur = cv2.GaussianBlur(plate_img_gray, (5, 5), 0)
                _, plate_img_thresh = cv2.threshold(plate_img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                # Force easyocr to only return alphanumeric characters
                result = reader.readtext(plate_img_thresh, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                
                if result:
                    plate_text = "".join([res[1] for res in result])
                    translation_table = str.maketrans("०१२३४५६७८९", "0123456789")
                    plate_text = plate_text.translate(translation_table)
                    plate_text = unicodedata.normalize('NFKD', plate_text).encode('ascii', 'ignore').decode('ascii')
                    plate_text = str(''.join(filter(str.isalnum, plate_text)).upper())
                    
                    # Attempt state-district automatic correction based on rigid positions (e.g. MH12...)
                    if len(plate_text) >= 4:
                        plate_text_str = str(plate_text)
                        state_str = plate_text_str[:2]
                        # Force letters
                        state_str = state_str.replace('1', 'T').replace('0', 'O').replace('8', 'B')
                        
                        district_str = plate_text_str[2:4]
                        # Force Numbers
                        district_str = district_str.replace('T', '1').replace('O', '0').replace('B', '8').replace('I', '1')
                        
                        plate_text = state_str + district_str + plate_text_str[4:]
                            
                    confidence = 0.0
                    if len(result) > 0:
                        total_conf = 0.0
                        for res in result:
                            total_conf += res[2]
                        confidence = total_conf / len(result)
                    
                    # Draw on RGB image for streamlit
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{plate_text} ({confidence:.2f})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    
                    # Save to log and disk
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if not os.path.exists("plates"):
                        os.makedirs("plates")
                    img_path = f"plates/{plate_text}_{int(datetime.now().timestamp())}.jpg"
                    cv2.imwrite(img_path, plate_img)
                    
                    log_file = "plate_log.csv"
                    new_row = pd.DataFrame([{
                        "Plate Number": plate_text,
                        "Timestamp": timestamp_str,
                        "Image Path": img_path,
                        "Confidence": confidence
                    }])
                    if not os.path.exists(log_file):
                        new_row.to_csv(log_file, index=False)
                    else:
                        new_row.to_csv(log_file, mode='a', header=False, index=False)
                    
                    detected_plates_info.append({"Plate": plate_text, "Confidence": confidence, "Image": plate_img})
                    
        st.image(frame, caption="Processed Image", use_container_width=True)
        
        if detected_plates_info:
            st.success(f"Detected {len(detected_plates_info)} plate(s)!")
            for info in detected_plates_info:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(cv2.cvtColor(info["Image"], cv2.COLOR_BGR2RGB), caption="Cropped Plate")
                with col2:
                    st.write(f"**Plate Number:** {info['Plate']}")
                    st.write(f"**Confidence:** {info['Confidence']:.2f}")
        else:
            st.warning("No license plates detected or OCR failed to read them.")

elif page == "Forecasting & Anomalies":
    col_hdr, col_btn = st.columns([8, 2])
    with col_hdr:
        st.header("📈 Forecasting & Anomaly Detection")
    with col_btn:
        auto_refresh = st.checkbox("Auto-Refresh (2s)")
        if not auto_refresh:
            st.button("↻ Refresh Page")
    
    st.write("Advanced ML models to predict traffic density and flag suspicious plates.")
    
    csv_file = "plate_log.csv"
    if not os.path.exists(csv_file):
        st.warning("No data found to analyze.")
    else:
        df = pd.read_csv(csv_file, on_bad_lines='skip')
        if df.empty:
            st.warning("Data file is empty.")
        else:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            tab1, tab2 = st.tabs(["🔮 Traffic Forecasting", "⚠️ Anomaly Detection (NLP)"])
            
            with tab1:
                st.subheader("Traffic Density Prediction")
                st.write("Using Holt-Winters Exponential Smoothing (from statsmodels) to predict future vehicle counts based on historical trends.")
                
                # Resample timeline to hourly counts
                ts_data = df.set_index('Timestamp').resample('1h').size().reset_index(name='Count')
                
                if len(ts_data) < 3:
                    st.info("Not enough historical data for accurate forecasting. Need at least 3 hours of accumulated data.")
                    fig = px.line(ts_data, x='Timestamp', y='Count', title="Current Traffic Density")
                    fig.update_layout(
                        template="plotly_dark", 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        legend=dict(font=dict(color='white')),
                        title=dict(font=dict(color='white'))
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                else:
                    try:
                        # Ensure steady frequency for statsmodels
                        series = ts_data.set_index('Timestamp')['Count']
                        
                        # Train Model
                        model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated").fit()
                        
                        # Predict next 12 hours
                        forecast = model.forecast(12)
                        
                        # Build DataFrame for plot
                        forecast_df = pd.DataFrame({
                            'Timestamp': pd.date_range(start=series.index[-1] + pd.Timedelta(hours=1), periods=12, freq='1h'),
                            'Count': np.round(forecast.values),
                            'Type': 'Predicted'
                        })
                        
                        history_df = ts_data.copy()
                        history_df['Type'] = 'Historical'
                        
                        combined_df = pd.concat([history_df, forecast_df])
                        
                        fig = px.line(combined_df, x='Timestamp', y='Count', color='Type', markers=True, 
                                      color_discrete_map={'Historical': '#58a6ff', 'Predicted': '#f39c12'})
                        fig.update_layout(
                            template="plotly_dark", 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            legend=dict(font=dict(color='white')),
                            title=dict(font=dict(color='white'))
                        )
                        st.plotly_chart(fig, use_container_width=True, theme=None)
                    except Exception as e:
                        st.error(f"Could not generate forecast: {e}")
                        
            with tab2:
                st.subheader("Suspicious Plate Anomaly Detection")
                st.caption("Flagging suspicious license plates based on Indian RTO formatting rules and OCR confidence.")
                
                VALID_STATE_CODES = [
                    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DN", "DD", "DL", "GA", 
                    "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", 
                    "MN", "ML", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", 
                    "TG", "TS", "TR", "UP", "UK", "WB"
                ]
                
                def check_anomaly(row):
                    # Convert to uppercase and ignore 'IND' text string
                    plate = str(row.get('Plate Number', '')).upper().replace('IND', '')
                    conf = row.get('Confidence', 1.0)
                    reasons = []
                    
                    if len(plate) < 6:
                        reasons.append("Too Short (<6 chars)")
                    elif len(plate) > 11:
                        reasons.append("Too Long (>11 chars)")
                        
                    if len(plate) >= 2 and plate[:2].upper() not in VALID_STATE_CODES:
                        reasons.append("Invalid State Code")
                        
                    if pd.notna(conf) and conf < 0.35:
                        reasons.append(f"Low OCR Confidence ({conf:.2f})")
                        
                    if len(reasons) > 0:
                        return " | ".join(reasons)
                    return "Valid Plate"
                
                df_anomalies = df.copy()
                df_anomalies['Analysis'] = df_anomalies.apply(check_anomaly, axis=1)
                
                flagged = df_anomalies[df_anomalies['Analysis'] != "Valid Plate"]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Scanned", len(df))
                c2.metric("Suspicious/Anomalous Flags", len(flagged), delta_color="inverse")
                c3.metric("Anomaly Rate", f"{(len(flagged) / len(df) * 100) if len(df) > 0 else 0:.1f}%")
                
                st.write("Recent Scan Analysis:")
                df_display = df_anomalies.iloc[::-1]  # Reverse so newest are at top
                
                def color_rows(row):
                    if row['Analysis'] == 'Valid Plate':
                        return ['background-color: rgba(46, 204, 113, 0.1)'] * len(row)
                    else:
                        return ['background-color: rgba(231, 76, 60, 0.2)'] * len(row)
                        
                st.dataframe(df_display[['Timestamp', 'Plate Number', 'Confidence', 'Analysis']].style.apply(color_rows, axis=1), use_container_width=True)
