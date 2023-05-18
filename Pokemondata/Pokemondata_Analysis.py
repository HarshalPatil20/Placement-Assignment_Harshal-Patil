import requests
import pandas as pd
import matplotlib.pyplot as plt

def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to download data.")
        return None

def analyze_pokemon_data(data):
    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)

    # Get all Pokemons whose spawn rate is less than 5%
    spawn_rate_less_than_5 = df[df['spawn_chance'] < 5]
    print("Pokemons with spawn rate less than 5%:")
    print(spawn_rate_less_than_5)

    # Get all Pokemons that have less than 4 weaknesses
    weaknesses_less_than_4 = df[df['weaknesses'].apply(len) < 4]
    print("Pokemons with less than 4 weaknesses:")
    print(weaknesses_less_than_4)

    # Get all Pokemons that have no multipliers at all
    no_multipliers = df[df['multipliers'].apply(len) == 0]
    print("Pokemons with no multipliers:")
    print(no_multipliers)

    # Get all Pokemons that do not have more than 2 evolutions
    evolutions_less_than_3 = df[df['next_evolution'].apply(lambda x: len(x) if x else 0) <= 2]
    print("Pokemons with no more than 2 evolutions:")
    print(evolutions_less_than_3)

    # Get all Pokemons whose spawn time is less than 300 seconds
    df['spawn_time'] = df['spawn_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    spawn_time_less_than_300 = df[df['spawn_time'] < 300]
    print("Pokemons with spawn time less than 300 seconds:")
    print(spawn_time_less_than_300)

    # Get all Pokemon who have more than two types of capabilities
    df['num_types'] = df['type'].apply(lambda x: len(x))
    pokemon_more_than_2_types = df[df['num_types'] > 2]
    print("Pokemons with more than two types of capabilities:")
    print(pokemon_more_than_2_types)

    # Plotting
    # Bar chart: Number of Pokemons with each number of weaknesses
    weaknesses_count = df['weaknesses'].apply(len)
    weaknesses_count.value_counts().sort_index().plot(kind='bar', xlabel='Number of Weaknesses', ylabel='Count')
    plt.title('Number of Pokemons with each Number of Weaknesses')
    plt.show()

    # Pie chart: Percentage of Pokemons with each type of capability
    type_count = df['type'].apply(lambda x: len(x))
    type_count.value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Percentage of Pokemons with each Number of Types')
    plt.ylabel('')
    plt.show()

# Provide the link to download the data
url = "https://raw.githubusercontent.com/Biuni/PokemonGO-Pokedex/master/pokedex.json"

# Download the data
pokemon_data = download_data(url)

if pokemon_data is not None:
    # Analyze the Pokémon data
    analyze_pokemon_data(pokemon_data)
