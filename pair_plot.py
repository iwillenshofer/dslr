from tools import stats
from pandas import read_csv
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

DATASET = "datasets/dataset_train.csv"

INDEX_COL = 'Index'

LABEL = 'Hogwarts House'
try:
    df = read_csv(DATASET, index_col=INDEX_COL)
    stats.normalize_dataframe(df)
    sns.pairplot(df, hue=LABEL)
    plt.show()
except Exception as error:
    print(f"{type(error).__name__}: {error}")
