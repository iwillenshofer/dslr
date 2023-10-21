from tools import stats
from pandas import read_csv, DataFrame
from matplotlib import pyplot as plt

DATASET = "datasets/dataset_train.csv"

# Defines the index column.
# may be set as the column name, 0 for no index, or None, for auto number
INDEX_COL = 'Index'

LABEL = 'Hogwarts House'

# selecting one of the possible features:
# 'Arithmancy', 'Astronomy', 'Herbology',
# 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
# 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
# 'Care of Magical Creatures', 'Charms', 'Flying'
FEATURE = 'Care of Magical Creatures'


def histogram(ds: DataFrame, label: str, feature: str):
    """creates a histogram"""
    stats.normalize_dataframe(ds)
    ds = ds[[label, feature]]
    labels = ds[label].unique()
    for ll in labels:
        data = ds[ds[label] == ll][feature]
        plt.hist(data, label=ll, alpha=0.5, bins=10, linewidth=1.2)
    plt.legend(labels)
    plt.xlabel("Grades (normalized)")
    plt.ylabel("Occurrences")
    plt.title(f"Histogram for {FEATURE}")
    plt.show()


try:
    df = read_csv(DATASET, index_col=INDEX_COL)
    histogram(df, LABEL, FEATURE)
except Exception as error:
    print(f"{type(error).__name__}: {error}")
