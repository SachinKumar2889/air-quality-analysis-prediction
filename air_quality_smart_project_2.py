
#Smart Air Quality Forecasting, Health Risk Assessment & Pollution Alert System using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.weightstats import ztest

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import warnings
warnings.filterwarnings('ignore')

#LOAD DATASET
df = pd.read_csv(r"C:\Users\Sachin Kumar\Downloads\archive (1)\city_day.csv")

print("Original Shape:", df.shape)

print("\n Original Data Summary:")
print(df.describe())

# DATA PREPROCESSING
df['Date'] = pd.to_datetime(df['Date'])

cities = ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Hyderabad', 'Bengaluru']
df = df[df['City'].isin(cities)]

df = df[['City','Date','PM2.5','PM10','CO','NO2','SO2','O3']]
df.columns = ['city','date','pm2_5','pm10','co','no2','so2','o3']

df = df.fillna(df.median(numeric_only=True))
df = df.ffill()

print("Cleaned Shape:", df.shape)

#FEATURE ENGINEERING
df['month'] = df['date'].dt.month

def get_season(month):

    if month in [12,1,2]:
        return "Winter"

    elif month in [3,4,5]:
        return "Summer"

    elif month in [6,7,8,9]:
        return "Monsoon"

    else:
        return "Post-Monsoon"

df['Season'] = df['month'].apply(get_season)

pollutants = ['pm10','pm2_5','co','no2','so2','o3']

#BEFORE OUTLIER REMOVAL (PM2.5 ONLY)
plt.figure(figsize=(10,6))
sns.boxplot(x=df['city'], y=df['pm2_5'])
plt.title("PM2.5 BEFORE Outlier Removal")
plt.xticks(rotation=45)
plt.show()

# REMOVE OUTLIERS
df_clean = df[df['pm2_5'] < 300]

print("Before Removal:", df.shape)
print("After Removal:", df_clean.shape)

# AFTER OUTLIER REMOVAL (PM2.5 ONLY)
plt.figure(figsize=(10,6))
sns.boxplot(x=df_clean['city'], y=df_clean['pm2_5'])
plt.title("PM2.5 AFTER Outlier Removal")
plt.xticks(rotation=45)
plt.show()

# VISUALIZATION (UNCHANGED)

df = df_clean

plt.figure(figsize=(18,10))
for i, col in enumerate(pollutants, 1):
    plt.subplot(2,3,i)
    df.groupby('city')[col].mean().plot(kind='bar')
    plt.title(f"{col.upper()} by City")
plt.tight_layout()
plt.show()

plt.figure(figsize=(18,10))
for i, col in enumerate(pollutants, 1):
    plt.subplot(2,3,i)
    df.groupby('month')[col].mean().plot(marker='o')
    plt.title(f"{col.upper()} Monthly Trend")
    plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df[pollutants].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# AQI ANALYSIS

def aqi_category(pm):
    if pm <= 50: return "Good"
    elif pm <= 100: return "Satisfactory"
    elif pm <= 200: return "Moderate"
    elif pm <= 300: return "Poor"
    elif pm <= 400: return "Very Poor"
    else: return "Severe"


def health_risk(pm):

    if pm <= 50:
        return "Low Risk"

    elif pm <= 100:
        return "Medium Risk"

    elif pm <= 200:
        return "High Risk"

    else:
        return "Very High Risk"


df['AQI_Category'] = df['pm2_5'].apply(aqi_category)

df['Health_Risk'] = df['pm2_5'].apply(health_risk)

# POLLUTION SCORE

df['pollution_score'] = (
    0.40 * df['pm2_5']
    + 0.30 * df['pm10']
    + 0.10 * (df['co'] * 100)
    + 0.10 * df['no2']
    + 0.05 * df['so2']
    + 0.05 * df['o3']
)

print("\nTop polluted records:")
print(df[['date','city','pm2_5']].sort_values(by='pm2_5', ascending=False).head())

print("\nAQI Category")
print(df['AQI_Category'].value_counts())


aqi_counts = df['AQI_Category'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(aqi_counts,
        labels=aqi_counts.index,
        autopct='%1.1f%%',
        startangle=140)

plt.title("AQI Category Distribution")
plt.show()

# POLLUTION HOTSPOT RANKING
print("\n===== POLLUTION HOTSPOT RANKING =====")

city_rank = (
    df.groupby('city')['pollution_score']
    .mean()
    .sort_values(ascending=False)
)

print(city_rank)


plt.figure(figsize=(8,5))
city_rank.plot(
    kind='bar',
    color='orange'
)

plt.title("Pollution Hotspot Ranking")
plt.ylabel("Average Pollution Score")
plt.xlabel("City")

plt.grid(axis='y')
plt.show()

# Most Polluted Day
top_day = df.loc[df['pm2_5'].idxmax()]

print("\n===== MOST POLLUTED DAY =====")

print("Date :", top_day['date'])
print("City :", top_day['city'])
print("PM2.5 :", round(top_day['pm2_5'],2))
print("AQI Category :", top_day['AQI_Category'])

# Season-wise Analysis
season_analysis = (
    df.groupby('Season')['pm2_5']
    .mean()
    .sort_values(ascending=False)
)

print("\n===== SEASON WISE POLLUTION =====")
print(season_analysis)

plt.figure(figsize=(8,5))

season_analysis.plot(
    kind='bar',
    color='green'
)

plt.title("Season-wise Average PM2.5")
plt.ylabel("Average PM2.5")
plt.xlabel("Season")
plt.grid(axis='y')
plt.show()


print("\n Statistical Summary (PM2.5):")
print(df['pm2_5'].describe())

print("\n Strong Hypothesis Testing")

threshold = 50   
alpha = 0.05

# H0: mean ≤ 50 (safe)
# H1: mean > 50 (unsafe)

sample = df['pm2_5']

#  Z-TEST (One-Tailed)
z_stat, p_value_z = ztest(sample, value=threshold)

# convert to one-tailed
p_value_z = p_value_z / 2

print("\n🔹 Z-Test (One-Tailed)")
print("Z-score:", round(z_stat,3))
print("p-value:", f"{p_value_z:.10f}")
print("alpha:", alpha)

if z_stat > 0 and p_value_z < alpha:
    print("Reject H0 → Air is Unsafe")
else:
    print("Fail to Reject H0 → Air is Safe")



# T-TEST (One-Tailed)
t_stat, p_value_t = stats.ttest_1samp(sample, threshold)

# convert to one-tailed
p_value_t = p_value_t / 2

print("\n🔹 T-Test (One-Tailed)")
print("T-statistic:", round(t_stat,3))
print("p-value:", f"{p_value_t:.10f}")
print("alpha:", alpha)

if t_stat > 0 and p_value_t < alpha:
    print("Reject H0 → Air is Unsafe")
else:
    print("Fail to Reject H0 → Air is Safe")


#  MACHINE LEARNING

X = df[['pm10','co','no2','so2','o3','month']]
y = df['pm2_5']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")

print("R² Score :", round(r2,3))
print("MSE      :", round(mse,2))
print("MAE      :", round(mae,2))

#  FINAL GRAPH

X_simple = df[['pm10']]
y_simple = df['pm2_5']

model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)

y_line = model_simple.predict(X_simple)

r2_simple = r2_score(y_simple, y_line)

sorted_idx = np.argsort(df['pm10'])

plt.figure(figsize=(12,6))
plt.scatter(df['pm10'], df['pm2_5'], label="Actual")

plt.plot(df['pm10'].iloc[sorted_idx],
         y_line[sorted_idx],
         color='red', label="Predicted")

plt.xlabel("PM10")
plt.ylabel("PM2.5")
plt.title(f"PM10 vs PM2.5 (R² = {round(r2_simple,2)})")

plt.legend()
plt.grid()
plt.show()


# USER INPUT PREDICTION

print("\n Predict PM2.5 from User Input")

try:
    pm10 = float(input("Enter PM10 (0-500): "))
    co = float(input("Enter CO (0-5): "))
    no2 = float(input("Enter NO2 (0-200): "))
    so2 = float(input("Enter SO2 (0-100): "))
    o3 = float(input("Enter O3 (0-200): "))
    month = int(input("Enter Month (1-12): "))

    #VALIDATION
    if not (0 <= pm10 <= 500 and
            0 <= co <= 5 and
            0 <= no2 <= 200 and
            0 <= so2 <= 100 and
            0 <= o3 <= 200 and
            1 <= month <= 12):

        print("\n Invalid Input! Please enter values in correct range.")

    else:
        user_data = np.array([[pm10, co, no2, so2, o3, month]])
        user_data_scaled = scaler.transform(user_data)

        prediction = model.predict(user_data_scaled)

        pred_pm = round(prediction[0],2)

        print("\n===== PREDICTION RESULT =====")

        print("Predicted PM2.5 :", pred_pm)

        aqi = aqi_category(pred_pm)
        risk = health_risk(pred_pm)

        print("AQI Category :", aqi)
        print("Health Risk :", risk)

        print("\n===== AIR QUALITY ALERT =====")

        if pred_pm <= 50:

            print("Safe Air Quality")

        elif pred_pm <= 100:

            print("Moderate Pollution")

        elif pred_pm <= 200:

            print("High Pollution Alert")

        else:

            print("SEVERE POLLUTION ALERT")

        print("\n===== HEALTH RECOMMENDATIONS =====")

        if pred_pm <= 50:

            print("- Outdoor Exercise Recommended")

        elif pred_pm <= 100:

            print("- Stay Hydrated")

        elif pred_pm <= 200:

            print("- Wear Face Mask")
            print("- Avoid Heavy Outdoor Exercise")
            print("- Keep Windows Closed")

        else:

            print("- Stay Indoors")
            print("- Use N95 Mask")
            print("- Use Air Purifier")

except:
    print("Invalid Input (Enter numbers only)")
