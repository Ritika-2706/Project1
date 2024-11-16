# PREDICTING THE RISE OF SEA LEVEL BY 2050 USING AI
### DESCRIPTION
This project seeks to predict sea levels by the year 2050, based on historical data, climate models, and machine learning techniques. The rising sea levels are resulted from three main contributors: melting of polar ice sheets, thermal expansion, and glaciers, which pose an adverse threat to coastal communities, ecosystems, and global infrastructure. Outcomes of this research could be useful directly to policymakers, urban planners, and environmentalists to guide how adaptations could be made for coastal infrastructure and global ventures aimed at combating climate change.

### PROGRAM
````
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the dataset
df = pd.read_csv('/content/epa-sea-level (2).csv')

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], label='Data', color='blue')

# Line of best fit for the entire dataset
slope, intercept, r_value, p_value, std_err = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])

# Predict sea level for years up to 2050
years_extended = list(range(df['Year'].min(), 2051))
sea_level_extended = [slope * year + intercept for year in years_extended]

plt.plot(years_extended, sea_level_extended, color='red', linestyle='--', label='Best fit line (1880-2050)')

# Filter data from 2000 to the most recent year
df_recent = df[df['Year'] >= 2000]

# Line of best fit for the recent data
slope_recent, intercept_recent, _, _, _ = linregress(df_recent['Year'], df_recent['CSIRO Adjusted Sea Level'])

# Predict sea level for years up to 2050
sea_level_recent = [slope_recent * year + intercept_recent for year in years_extended]

plt.plot(years_extended, sea_level_recent, color='green', linestyle='-', label='Best fit line (2000-2050)')

# Add labels, title, and legend
plt.xlabel('Year')
plt.ylabel('Sea Level (inches)')
plt.title('Rise in Sea Level')
plt.legend()

# Save and show the plot
plt.savefig('sea_level_rise.png')
plt.show()
````
### OUTPUT
![image](https://github.com/user-attachments/assets/46e40fa4-a3bf-41f9-8f1b-bdc5b9e325f1)
### RESULT
The project projects increases in sea levels by 2050 associated with melting ice sheets, thermal expansion, and glacier contributions through the application of Artificial Intelligence and machine learning for accurate predictions.

