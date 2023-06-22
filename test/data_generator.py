import pandas as pd
import numpy as np

np.random.seed(42)
data = {
    'Column1': np.random.choice(['A', 'B', 'C', 'D'], size=5000),
    'Column2': np.random.choice(['X', 'Y', 'Z'], size=5000),
    'Column3': np.random.randint(1, 100, size=5000),
    'Column4': np.random.choice([True, False], size=5000),
    'Column5': pd.date_range(start='1/1/2023', periods=5000),
    'Column6': np.random.normal(0, 1, size=5000),
    'Column7': np.random.randint(1, 100, size=5000),
    'Column8': np.random.choice(['Category1', 'Category2', 'Category3'], size=5000),
    'Column9': np.random.choice([True, False], size=5000),
    'Column10': np.random.randint(1, 100, size=5000)
}

df = pd.DataFrame(data)

# Adding missing values
df.loc[np.random.choice(df.index, size=500), 'Column3'] = np.nan
df.loc[np.random.choice(df.index, size=500), 'Column7'] = np.nan

# Displaying the initial dataset
print("Initial Dataset:")
print(df.head())

df['Column10']

df.columns = ['alphabets1', 
              'alphabets2', 
              'missing_column1', 
              'boolean_column1',
              'date_column', 
              'normalisation_column', 
              'missing_column2',
              'categorical_column', 
              'boolean_column2',
              'random_numbers']

df.to_csv('data.csv', index=False)