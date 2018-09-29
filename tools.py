import operator
import pprint as pp
from itertools import combinations

import seaborn as sns


def drop_cols(df, cols):
    drop_cols = [c for c in df.columns if c in cols]
    df.drop(drop_cols, axis=1, inplace=True)
    return df


def print_corr(df_corr, keyword=None):
    sns.heatmap(
        df_corr,
        annot=True,
        xticklabels=df_corr.columns.values,
        yticklabels=df_corr.columns.values)

    corr_dict = {}
    for c in list(combinations(df_corr.columns, 2)):
        corr_dict[c] = df_corr.loc[c[0], c[1]]

    corr_list = []
    for i, corr in enumerate(
            sorted(
                corr_dict.items(), key=operator.itemgetter(1), reverse=True)):
        if keyword:
            if keyword in corr:
                corr_list.append("{} {}".format(i, corr))
            else:
                pass
        else:
            corr_list.append("{} {}".format(i, corr))
    pp.pprint(corr_list)
    return corr_list


def compare_corr(df_train, df_test):
    df_train_corr = df_train.corr()
    df_test_corr = df_test.corr()

    corr_dict_train = {}
    corr_dict_test = {}
    corr_dict = {}
    for c in list(combinations(df_train_corr.columns, 2)):
        corr_dict_train[c] = df_train_corr.loc[c[0], c[1]]
        corr_dict_test[c] = df_test_corr.loc[c[0], c[1]]
        corr_dict[c] = abs(corr_dict_train[c] - corr_dict_test[c])

    return corr_dict, corr_dict_train, corr_dict_test


def dict_to_sortedlist(d):
    l = []
    for i, row in enumerate(
            sorted(
                d.items(), key=operator.itemgetter(1), reverse=True)):
        l.append("{} {}".format(i, row))
    return l
