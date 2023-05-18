import requests
import json

def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to download data.")
        return None

def extract_data(data):
    extracted_data = {
        "id": data["id"],
        "url": data["url"],
        "name": data["name"],
        "season": data.get("season", ""),
        "number": data.get("number", ""),
        "type": data.get("type", ""),
        "airdate": data.get("airdate", ""),
        "airtime": data.get("airtime", ""),
        "runtime": data.get("runtime", ""),
        "average_rating": data["rating"]["average"],
        "summary": remove_html_tags(data["summary"]),
        "medium_image_link": data["image"]["medium"],
        "original_image_link": data["image"]["original"]
    }
    return extracted_data


def remove_html_tags(text):
    # Function to remove HTML tags from the summary
    clean_text = ""
    in_tag = False
    for char in text:
        if char == "<":
            in_tag = True
        elif char == ">":
            in_tag = False
        elif not in_tag:
            clean_text += char
    return clean_text.strip()

# Provide the API link to download the data
url = "http://api.tvmaze.com/singlesearch/shows?q=westworld&embed=episodes"

# Download the data
show_data = download_data(url)

if show_data is not None:
    # Extract the required data
    extracted_data = extract_data(show_data)

    # Print the extracted data
    for attribute, value in extracted_data.items():
        print(f"{attribute}: {value}")
