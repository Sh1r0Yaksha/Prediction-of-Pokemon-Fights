import pandas as pd
from sklearn.preprocessing import OneHotEncoder 

# Read data of all pokemon from file (change it according to file location)
pokemon = pd.read_csv('Data/pokemon_data.csv')


# Making 2 lists of features 'Type 1' and 'Type 2' for doing one hot encoding
type_1_list = pokemon['Type 1']
type_2_list = pokemon['Type 2']

# One hot encoding 'Type 1' and 'Type 2'
type_1_list = pd.get_dummies(
    type_1_list,
    prefix=None,
    prefix_sep='_', 
    dummy_na=True, 
    columns=['Type 1'], 
    sparse=False, 
    drop_first=False, 
    dtype=None
)

type_2_list = pd.get_dummies(
    type_2_list,
    prefix=None,
    prefix_sep='_', 
    dummy_na=True, 
    columns=['Type 2'], 
    sparse=False, 
    drop_first=False, 
    dtype=None
)


# Adding the lists above to get a new list with all Pokemon's types
both_type = type_1_list + type_2_list

# Adding the one hot encoded features and removing excess and unwanted features
new_pokemon = pokemon.join(both_type)
new_pokemon.pop('Type 1')
new_pokemon.pop('Type 2')
new_pokemon.pop('Generation')
new_pokemon.pop('Legendary')
new_pokemon.pop('Name')

# Converting this dataset to csv format
new_pokemon.to_csv('Data/PreProcessed.csv')
