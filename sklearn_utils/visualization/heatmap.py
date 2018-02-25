import pandas as pd
import seaborn as sns

from sklearn_utils.utils import feature_importance_report


def plot_heatmap(X, y, top_n=10, metric='correlation', method='complete'):
    '''
    Plot heatmap which shows features with classes.

    :param X: list of dict
    :param y: labels
    :param top_n: most important n feature
    :param metric: metric which will be used for clustering
    :param method: method which will be used for clustering
    '''
    sns.set(color_codes=True)

    df = feature_importance_report(X, y)

    df_sns = pd.DataFrame().from_records(X)[df[:top_n].index].T
    df_sns.columns = y

    color_mapping = dict(zip(set(y), sns.mpl_palette("Set2", len(set(y)))))

    return sns.clustermap(df_sns, figsize=(22, 22), z_score=0,
                          metric=metric, method=method,
                          col_colors=[color_mapping[i] for i in y])
