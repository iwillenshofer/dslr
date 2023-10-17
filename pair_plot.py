from tools import stats
from pandas import read_csv, DataFrame, to_datetime
from matplotlib import pyplot as plt

DATASET = "datasets/dataset_train.csv"

# Defines the index column.
# may be set as the column name, 0 for no index, or None, for auto number\
INDEX_COL = 'Index'

LABEL = 'Hogwarts House'

FEATURES = [
    'Best Hand',
    'Birthday',
    'Arithmancy',
    'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    'Divination',
    'Muggle Studies',
    'Ancient Runes',
    'History of Magic',
    'Transfiguration',
    'Potions',
    'Care of Magical Creatures',
    'Charms',
    'Flying'
]


def pairplot(ds: DataFrame):
    """creates a scatterplot taking dataframe as parameter"""
    labels = ds[LABEL].unique()
    nb_cols = len(FEATURES)
    nb_rows = len(FEATURES)
    fig, axs = plt.subplots(nb_rows, nb_cols)
    for r, feat_row in enumerate(FEATURES):
        for c, feat_col in enumerate(FEATURES):
            for label in labels:
                data1 = ds[ds[LABEL] == label][feat_row]
                data2 = ds[ds[LABEL] == label][feat_col]
                if r != c:
                    axs[r][c].scatter(data1, data2, s=.1, alpha=.7)
                else:
                    axs[r][c].hist(data1, alpha=.5)
                if not c:
                    axs[r][c].set_ylabel(
                        f'{feat_row}', fontsize=8, rotation=30, ha='right')
                if not r:
                    axs[r][c].set_title(
                        f'{feat_col}', fontsize=8, rotation=30, ha='left')
                if c != nb_cols - 1:
                    axs[r][c].yaxis.set_tick_params(labelleft=False)
                else:
                    axs[r][c].yaxis.set_tick_params(
                        labelleft=False, labelright=True)
                if r != nb_rows - 1:
                    axs[r][c].xaxis.set_tick_params(labelbottom=False)
    fig.legend(labels, loc='lower right', markerscale=10.0)
    fig.suptitle('Pairplot comparing all courses')
    plt.show()


try:
    ds = read_csv(DATASET, index_col=INDEX_COL)
    ds.replace('Right', 0, inplace=True)
    ds.replace('Left', 1, inplace=True)
    ds["Birthday"] = to_datetime(ds["Birthday"]).dt.strftime("%Y%m%d").astype(int)
    print(ds)

    stats.normalize_dataframe(ds)
    print(ds)
    pairplot(ds)
    plt.show()
except Exception as error:
    print(f"{type(error).__name__}: {error}")
