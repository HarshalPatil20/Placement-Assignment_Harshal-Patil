import requests
import pandas as pd

# Download the data from the given link
url = "https://data.nasa.gov/resource/y77d-th95.json"
response = requests.get(url)
data = response.json()

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Save the data as a CSV file
df.to_csv("earth_meteorites.csv", index=False)

# Analyze the data and answer the questions
# Get all the Earth meteorites that fell before the year 2000
df['year'] = pd.to_datetime(df['year']).dt.year
meteorites_before_2000 = df[df['year'] < 2000]

# Get all the Earth meteorites' coordinates that fell before the year 1970
meteorites_coords_before_1970 = df[df['year'] < 1970][['reclat', 'reclong']]

# Assuming mass is in kg, get all the Earth meteorites with mass > 10000 kg
df['mass'] = df['mass'].astype(float)
meteorites_mass_gt_10000 = df[df['mass'] > 10000]

# Print the insights
print("Earth meteorites that fell before the year 2000:")
print(meteorites_before_2000)

print("\nEarth meteorites' coordinates that fell before the year 1970:")
print(meteorites_coords_before_1970)

print("\nEarth meteorites with mass > 10000 kg:")
print(meteorites_mass_gt_10000)

# Plotting example: Plotting the mass distribution of Earth meteorites
df['mass'].plot(kind='hist', bins=20, title='Mass Distribution of Earth Meteorites')

# Display the plots
plt.show()
