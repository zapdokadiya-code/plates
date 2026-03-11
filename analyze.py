import pandas as pd
import re
import os
import sys

if sys.stdout.encoding != 'utf-8':
    # Use hasattr to prevent static type checkers (and older Pythons) from complaining
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

csv_file = "plate_log.csv"

if not os.path.exists(csv_file):
    print("No data found to analyze.")
    exit(0)

# Load data, handle missing columns gracefully in case of old data
df = pd.read_csv(csv_file, on_bad_lines='skip')

if df.empty:
    print("The log file is empty. Nothing to analyze.")
    exit(0)

print("="*40)
print("PLATE DETECTION ANALYSIS REPORT")
print("="*40)

# Total detections
total_plates = len(df)
print(f"Total Plates Detected: {total_plates}")

# 1. Plate Length Analysis
df['Length'] = df['Plate Number'].astype(str).apply(len)
avg_length = df['Length'].mean()
print(f"\n--- Plate Length Characteristics ---")
print(f"Average Plate Length: {avg_length:.2f} characters")
print("Frequency of Plate Lengths:")
print(df['Length'].value_counts().sort_index().to_string())

# 2. Character Set Analysis
def classify_chars(plate_str):
    plate_str = str(plate_str)
    has_ascii_letters = bool(re.search(r'[a-zA-Z]', plate_str))
    has_ascii_digits = bool(re.search(r'[0-9]', plate_str))
    
    # Simple check for Devanagari (Hindi) Unicode range \u0900-\u097F
    has_hindi = bool(re.search(r'[\u0900-\u097F]', plate_str))
    has_other_special = bool(re.search(r'[^a-zA-Z0-9\u0900-\u097F]', plate_str))
    
    return pd.Series({
        'Has_Letters': has_ascii_letters,
        'Has_Digits': has_ascii_digits,
        'Has_Hindi': has_hindi,
        'Has_Special/Other': has_other_special
    })

char_stats = df['Plate Number'].apply(classify_chars)

print(f"\n--- Character Type Findings ---")
print(f"Plates with English Letters: {char_stats['Has_Letters'].sum()} ({char_stats['Has_Letters'].mean()*100:.1f}%)")
print(f"Plates with Digits:          {char_stats['Has_Digits'].sum()} ({char_stats['Has_Digits'].mean()*100:.1f}%)")
print(f"Plates with Hindi/Devanagari:{char_stats['Has_Hindi'].sum()} ({char_stats['Has_Hindi'].mean()*100:.1f}%)")
print(f"Plates w/ Special/Other Chars:{char_stats['Has_Special/Other'].sum()} ({char_stats['Has_Special/Other'].mean()*100:.1f}%)")

# 3. OCR Reliability (Confidence Analysis)
print(f"\n--- OCR Reliability ---")
if 'Confidence' in df.columns and not df['Confidence'].isnull().all():
    # Only calculate for rows that have a confidence value
    valid_conf_df = df.dropna(subset=['Confidence'])
    avg_conf = valid_conf_df['Confidence'].mean()
    min_conf = valid_conf_df['Confidence'].min()
    max_conf = valid_conf_df['Confidence'].max()
    
    print(f"Average Confidence:  {avg_conf*100:.2f}%")
    print(f"Highest Confidence:  {max_conf*100:.2f}%")
    print(f"Lowest Confidence:   {min_conf*100:.2f}%")
else:
    print("No confidence scores available in the log. (Note: this feature was just added!)")

# 4. Suggestions / Recommendations
print(f"\n--- Recommended Further Steps ---")
print("- Some plates contain Hindi characters. You might want to filter OCR results to english alphanumeric only using the 'allowlist' parameter in EasyOCR.")
print("- Alternatively, verify if Devanagari numbers should be mapped to standard digits.")
print("- Consider ignoring plates with low confidence (e.g. < 40%) or weird characters (like '|' or ']') to reduce false positives.")
print("="*40)
print("Analysis complete.")

df_log = pd.read_csv('plate_log.csv', on_bad_lines='skip')
print(df_log)

df_log['Timestamp'] = pd.to_datetime(df_log['Timestamp'])
print("Timestamp column converted to datetime.")

df_log['Hour'] = df_log['Timestamp'].dt.hour
df_log['Day of Week'] = df_log['Timestamp'].dt.day_name()
print("Extracted 'Hour' and 'Day of Week' from Timestamp.")

hourly_counts = df_log['Hour'].value_counts().sort_index()
print("\nHourly detection counts:\n", hourly_counts)

day_of_week_counts = df_log['Day of Week'].value_counts()
print("\nDay of week detection counts:\n", day_of_week_counts)

import matplotlib.pyplot as plt
import seaborn as sns

top_10_plates = df_log['Plate Number'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_plates.index, y=top_10_plates.values, palette='viridis')
plt.title('Top 10 Most Frequent Plate Numbers')
plt.xlabel('Plate Number')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
print("Graph: Shows the 10 license plates most often detected by the system.")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=hourly_counts.index, y=hourly_counts.values, hue=hourly_counts.index, palette='Blues_d', legend=False)
plt.title('Hourly Plate Detection Frequency')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Detections')
plt.xticks(range(24))
plt.tight_layout()
print("Graph: Displays the number of plates detected during each hour of the day.")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=day_of_week_counts.index, y=day_of_week_counts.values, hue=day_of_week_counts.index, palette='viridis', legend=False)
plt.title('Daily Plate Detection Frequency')
plt.xlabel('Day of Week')
plt.ylabel('Number of Detections')
plt.tight_layout()
print("Graph: Displays the number of plates detected on each day of the week.")
plt.show()

df_log = df_log.sort_values(by='Timestamp').reset_index(drop=True)
df_log['Cumulative Detections'] = range(1, len(df_log) + 1)
print("DataFrame sorted and 'Cumulative Detections' column added.")

plt.figure(figsize=(12, 6))
sns.lineplot(x='Timestamp', y='Cumulative Detections', data=df_log)
plt.title('Cumulative Plate Detections Over Time')
plt.xlabel('Time')
plt.ylabel('Cumulative Number of Detections')
plt.grid(True)
plt.tight_layout()
print("Graph: Visualizes the total accumulation of detected license plates over time.")
plt.show()

char_type_counts = char_stats.sum()

plt.figure(figsize=(8, 5))
sns.barplot(x=char_type_counts.index, y=char_type_counts.values, hue=char_type_counts.index, palette='pastel', legend=False)
plt.title('Distribution of Character Types in Plate Numbers')
plt.xlabel('Character Type')
plt.ylabel('Total Count')
plt.tight_layout()
print("Graph: Breaks down the count of different character types found in the plates.")
plt.show()
print("Displayed bar plot of character type distribution.")