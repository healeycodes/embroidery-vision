import csv
import scipy.spatial as sp

DMC_CSV = 'dmc.csv'

dmc_colors = []

with open(DMC_CSV, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for index, row in enumerate(reader):
        dmc_colors.append({
            'index': index,
            'floss': row['floss#'],
            'description': row['description'],
            'red': int(row['red']),
            'green': int(row['green']),
            'blue': int(row['blue']),
            'hex': '#' + row['hex'],
            'dmc_row': row['row'],
        })

rgb_colors = []
for color in dmc_colors:
    rgb_colors.append((
        color['red'], color['green'], color['blue']
    ))

def rgb_to_dmc(r, g, b):
    tree = sp.KDTree(rgb_colors)
    # don't need the Euclidean distance only the index
    _, result = tree.query((r, g, b))
    return dmc_colors[result]
