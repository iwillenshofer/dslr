from tools import stats
from pandas import read_csv, DataFrame
from matplotlib import pyplot as plt

DATASET = "datasets/dataset_train.csv"

# Defines the index column.
# may be set as the column name, 0 for no index, or None, for auto number
INDEX_COL = 'Index'

LABEL = 'Hogwarts House'

# selecting two of the possible features:
# 'Arithmancy', 'Astronomy', 'Herbology',
# 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
# 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
# 'Care of Magical Creatures', 'Charms', 'Flying'
FEATURES = [
    'Astronomy',
    'Defense Against the Dark Arts'
]


def scatterplot(ds: DataFrame, label, features):
    """creates a scatterplot taking dataframe as parameter"""
    stats.normalize_dataframe(ds)
    ds = ds[[label, features[0], features[1]]]
    labels = ds[LABEL].unique()
    for ll in labels:
        data1 = ds[ds[label] == ll][features[0]]
        data2 = ds[ds[label] == ll][features[1]]
        plt.scatter(data1, data2, s=.1)
    plt.legend(labels, loc='lower right', markerscale=10.0)
    plt.title(f"Scatterplot comparing {features[0]} and {features[1]}")
    plt.show()


try:
    df = read_csv(DATASET, index_col=INDEX_COL)
    scatterplot(df, LABEL, FEATURES)
except Exception as error:
    print(f"{type(error).__name__}: {error}")
