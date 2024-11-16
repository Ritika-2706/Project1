# PREDICTING THE RISE OF SEA LEVEL BY 2050 USING AI
### DESCRIPTION
This project seeks to predict sea levels by the year 2050, based on historical data, climate models, and machine learning techniques. The rising sea levels are resulted from three main contributors: melting of polar ice sheets, thermal expansion, and glaciers, which pose an adverse threat to coastal communities, ecosystems, and global infrastructure. Outcomes of this research could be useful directly to policymakers, urban planners, and environmentalists to guide how adaptations could be made for coastal infrastructure and global ventures aimed at combating climate change.

### PROGRAM
````
# Import necessary libraries
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

# Helper function to generate a list of month-long date ranges
def generate_monthly_date_ranges(start_date, end_date):
    """
    Generate a list of (start_date, end_date) tuples, each representing a month-long range.
    """
    ranges = []
    current_date = start_date

    while current_date < end_date:
        month_end = (current_date + timedelta(days=31)).replace(day=1) - timedelta(days=1)
        if month_end > end_date:
            month_end = end_date
        ranges.append((current_date.strftime('%Y%m%d'), month_end.strftime('%Y%m%d')))
        current_date = month_end + timedelta(days=1)

    return ranges

# Adjusted fetch function to retrieve data month by month
def fetch_noaa_sea_level_data_monthly(station_id, start_date, end_date):
    """
    Function to fetch sea level data from NOAA Tides & Currents API in monthly chunks.
    """
    all_data = []
    date_ranges = generate_monthly_date_ranges(start_date, end_date)

    for start, end in date_ranges:
        print(f"Fetching data from {start} to {end}")
        base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        params = {
            "station": station_id,
            "begin_date": start,
            "end_date": end,
            "product": "water_level",
            "datum": "MSL",
            "units": "metric",
            "time_zone": "GMT",
            "format": "json"
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                monthly_data = pd.DataFrame(data['data'])
                all_data.append(monthly_data)
            else:
                print(f"No 'data' key found for range {start} to {end}")
        else:
            print(f"Failed to fetch data for range {start} to {end}")
            print("Status Code:", response.status_code)
            print("Response Text:", response.text)

    # Combine all monthly data into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data was retrieved

# Define station, start, and end dates
station_id = "9414290"  # Example station: San Francisco
start_date = datetime.strptime("19800101", "%Y%m%d")  # Start date
end_date = datetime.strptime("20221231", "%Y%m%d")    # End date

# Fetch data month by month
df = fetch_noaa_sea_level_data_monthly(station_id, start_date, end_date)

# Proceed if data is available
if df.empty:
    print("No data available. Please check the API response for details.")
else:
    # Data preparation as outlined in the previous code
    df['t'] = pd.to_datetime(df['t'])
    df['v'] = pd.to_numeric(df['v'])
    df = df.rename(columns={"t": "Date", "v": "Sea_Level_mm"})

    print("NOAA Sea Level Data Sample:")
    print(df.head())

    # (Continue with data visualization, modeling, and predictions)

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Sea_Level_mm'], label='Sea Level (mm)', color='b', marker='o')
plt.title("Daily Sea Level Measurements (NOAA)")
plt.xlabel("Date")
plt.ylabel("Sea Level (mm)")
plt.legend()
plt.show()  # Ensure the plot renders

df['Sea_Level_mm'] = pd.to_numeric(df['Sea_Level_mm'], errors='coerce')

df['Year'] = df['Date'].dt.year
numeric_columns = df.select_dtypes(include=['float64', 'int']).columns

yearly_data = df.groupby('Year')[numeric_columns].mean()

print("Yearly Aggregated Data Sample:")
print(yearly_data.head())

X = yearly_data.index.values.reshape(-1, 1)
y = yearly_data['Sea_Level_mm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Sea Levels')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Sea Levels')
plt.title("Actual vs Predicted Sea Levels")
plt.xlabel("Year")
plt.ylabel("Sea Level (mm)")
plt.legend()
plt.show()

future_years = pd.DataFrame({"Year": range(X[-1][0] + 1, 2051)})
future_sea_levels = model.predict(future_years.values)

plt.figure(figsize=(10, 6))
plt.plot(yearly_data.index, yearly_data['Sea_Level_mm'], label='Historical Data', color='blue', marker='o')
plt.plot(future_years['Year'], future_sea_levels, label='Predicted Sea Levels (2023-2050)', color='orange', linestyle='--')
plt.title("Sea Level Predictions Until 2050")
plt.xlabel("Year")
plt.ylabel("Sea Level (mm)")
plt.legend()
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(X_test, residuals, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of Predicted Sea Levels")
plt.xlabel("Year")
plt.ylabel("Residual (Actual - Predicted Sea Level)")
plt.show()

# Subtract the predicted values from the actual values to get errors (residuals)
residuals = y_test - y_pred

import matplotlib.pyplot as plt

# Plot the errors
plt.figure(figsize=(10, 6))
plt.scatter(X_test, residuals, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of Predicted Sea Levels")
plt.xlabel("Year")
plt.ylabel("Error (Actual - Predicted Sea Level)")
plt.show()
````
### OUTPUT
![image](https://github.com/user-attachments/assets/d6e07c35-3100-47e5-9335-4038d8cb4e56)
![image](https://github.com/user-attachments/assets/a9ddc0dc-dded-414d-9c53-b73a3bb9e592)
![image](https://github.com/user-attachments/assets/5d047853-0c5a-4dd4-838c-b35ca0f51cd6)
![image](https://github.com/user-attachments/assets/ef4d53d7-3a7b-4c8f-b0df-e489fbc47daa)
![image](https://github.com/user-attachments/assets/56840bde-86df-4048-8f6c-cb547685351b)


### RESULT
The project projects increases in sea levels by 2050 associated with melting ice sheets, thermal expansion, and glacier contributions through the application of Artificial Intelligence and machine learning for accurate predictions.

