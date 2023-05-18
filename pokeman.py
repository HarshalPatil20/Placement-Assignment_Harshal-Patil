import requests
import json
import pandas as pd

def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to download data.")
        return None

def parse_data(data):
    parsed_data = []
    for pokemon in data["pokemon"]:
        parsed_pokemon = {
            "id": pokemon["id"],
            "num": pokemon["num"],
            "name": pokemon["name"],
            "img": pokemon["img"],
            "type": ",".join(pokemon["type"]),
            "height": pokemon["height"],
            "weight": pokemon["weight"],
            "candy": pokemon.get("candy", ""),
            "candy_count": pokemon.get("candy_count", ""),
            "egg": pokemon.get("egg", ""),
            "spawn_chance": pokemon.get("spawn_chance", ""),
            "avg_spawns": pokemon.get("avg_spawns", ""),
            "spawn_time": pokemon.get("spawn_time", ""),
            "weakness": ",".join(pokemon["weaknesses"]),
            "next_evolution": [evo["num"] + " " + evo["name"] for evo in pokemon.get("next_evolution", [])],
            "prev_evolution": [evo["num"] + " " + evo["name"] for evo in pokemon.get("prev_evolution", [])]
        }
        parsed_data.append(parsed_pokemon)
    return parsed_data

def export_to_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)

# Provide the link to download the data
url = "https://raw.githubusercontent.com/Biuni/PokemonGO-Pokedex/master/pokedex.json"

# Download the data
pokemon_data = download_data(url)

if pokemon_data is not None:
    # Parse the data
    parsed_data = parse_data(pokemon_data)

    # Export to Excel
    export_to_excel(parsed_data, "pokemon_data.xlsx")
    print("Data exported to 'pokemon_data.xlsx' successfully.")
