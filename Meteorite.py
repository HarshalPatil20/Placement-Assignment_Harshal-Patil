import requests
import csv

def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to download data.")
        return None

def parse_data(data):
    parsed_data = []
    for meteorite in data:
        parsed_meteorite = {
            "name": meteorite.get("name", ""),
            "id": meteorite.get("id", ""),
            "nametype": meteorite.get("nametype", ""),
            "recclass": meteorite.get("recclass", ""),
            "mass": meteorite.get("mass", ""),
            "year": meteorite.get("year", ""),
            "reclat": meteorite.get("reclat", ""),
            "reclong": meteorite.get("reclong", "")
        }
        parsed_data.append(parsed_meteorite)
    return parsed_data

def export_to_csv(data, filename):
    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


# Provide the link to download the data
url = "https://data.nasa.gov/resource/y77d-th95.json"

# Download the data
meteorite_data = download_data(url)

if meteorite_data is not None:
    # Parse the data
    parsed_data = parse_data(meteorite_data)

    # Export to CSV
    export_to_csv(parsed_data, "meteorite_data.csv")
    print("Data exported to 'meteorite_data.csv' successfully.")
