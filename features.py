import time

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from tqdm import tqdm


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


@timeit
def fea_date_time(df):
    date_org = df['date']
    df['date'] = df['date'].astype(str)
    df["date"] = df["date"].apply(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:])
    df["date"] = pd.to_datetime(df["date"])
    # df["year"] = df['date'].dt.year
    df["month"] = df['date'].dt.month
    df["day"] = df['date'].dt.day
    df['hour'] = pd.to_datetime(df['visitStartTime'], unit='s').dt.hour  # aiden

    # df["weekofmonth"] = df['day'].astype(int) // 7  # aiden

    df["weekday"] = df['date'].dt.weekday
    # df['weekofyear'] = df['date'].dt.weekofyear
    df['month_unique_user_count'] = df.groupby('month')['fullVisitorId'].transform('nunique')
    # df['month_unique_s_count'] = df.groupby('month')['sessionId'].transform('nunique')

    # df['day_unique_user_count'] = df.groupby('day')['fullVisitorId'].transform('nunique')
    # df['day_unique_s_count'] = df.groupby('day')['sessionId'].transform('nunique')

    df['weekday_unique_user_count'] = df.groupby('weekday')['fullVisitorId'].transform('nunique')
    # df['weekday_unique_s_count'] = df.groupby('weekday')['sessionId'].transform('nunique')

    df['hour_unique_user_count'] = df.groupby('hour')['fullVisitorId'].transform('nunique')  # aiden
    # df['hour_unique_s_count'] = df.groupby('hour')['sessionId'].transform('nunique')  # aiden

    df['hour_unique_user_count'] = df.groupby('hour')['fullVisitorId'].transform('nunique')

    df['user_hour_mean'] = df.groupby(['fullVisitorId'])['hour'].transform('mean')  # aiden
    df['user_hour_max'] = df.groupby(['fullVisitorId'])['hour'].transform('max')  # aiden
    df['user_hour_min'] = df.groupby(['fullVisitorId'])['hour'].transform('min')  # aiden

    df['date'] = date_org

    return df


@timeit
def fea_format(df):
    for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
        df[col] = df[col].astype(float)
    df['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    df['trafficSource_isTrueDirect'].fillna(False, inplace=True)
    return df


@timeit
def fea_device(df):
    df['browser_category'] = df['device_browser'] + '_' + df['device_deviceCategory']
    df['browser_operatingSystem'] = df['device_browser'] + '_' + df['device_operatingSystem']
    df['mean_hour_per_browser_operatingSystem'] = df.groupby('browser_operatingSystem')['hour'].transform(
        'mean')  # aiden
    df['source_country'] = df['trafficSource_source'] + '_' + df['geoNetwork_country']
    return df


@timeit
def fea_totals(df):
    df['visitNumber'] = np.log1p(df['visitNumber'])
    df['totals_hits'] = np.log1p(df['totals_hits'])
    df['totals_pageviews'] = np.log1p(df['totals_pageviews'].fillna(0))

    # df['totals_pageviews_hit_rate'] = df['totals_hits'] - df['totals_pageviews']

    # df['mean_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('mean')
    df['sum_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('sum')
    df['max_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('max')
    # df['min_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('min')
    df['var_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('var')

    df['mean_hits_per_hour'] = df.groupby(['hour'])['totals_hits'].transform('mean')  # aiden
    df['sum_hits_per_hour'] = df.groupby(['hour'])['totals_hits'].transform('sum')  # aiden
    df['max_hits_per_hour'] = df.groupby(['hour'])['totals_hits'].transform('max')  # aiden
    # df['min_hits_per_hour'] = df.groupby(['hour'])['totals_hits'].transform('min')  # aiden
    # df['var_hits_per_hour'] = df.groupby(['hour'])['totals_hits'].transform('var')  # aiden
    return df


@timeit
def fea_geo_network(df):
    # df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    # df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform(
    #     'count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform(
        'mean')
    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    # df['count_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    # df['mean_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')
    return df


@timeit
def fea_traffic_source(df):
    df['source_country'] = df['trafficSource_source'] + '_' + df['geoNetwork_country']
    df['campaign_medium'] = df['trafficSource_campaign'] + '_' + df['trafficSource_medium']
    df['medium_hits_mean'] = df.groupby(['trafficSource_medium'])['totals_hits'].transform('mean')
    df['medium_hits_max'] = df.groupby(['trafficSource_medium'])['totals_hits'].transform('max')
    df['medium_hits_min'] = df.groupby(['trafficSource_medium'])['totals_hits'].transform('min')
    df['medium_hits_sum'] = df.groupby(['trafficSource_medium'])['totals_hits'].transform('sum')
    return df


def get_features(df):
    org_cols = df.columns
    df = fea_date_time(df)
    df = fea_format(df)
    df = fea_device(df)
    df = fea_totals(df)
    df = fea_geo_network(df)
    df = fea_traffic_source(df)
    fea_cols = list(set(df.columns) - set(org_cols))
    # print(new_cols)

    return df, fea_cols


@timeit
def encode_label(df_train, df_test, categorical_feature):
    print(categorical_feature)

    df_merge = pd.concat([df_train[categorical_feature], df_test[categorical_feature]])
    train_size = df_train.shape[0]
    for c in tqdm(categorical_feature):
        # st = time.time()

        labels, _ = pd.factorize(df_merge[c].values.astype('str'))

        df_train[c] = labels[:train_size]
        df_test[c] = labels[train_size:]
        # print(c, time.time() - st)

    return df_train, df_test


@timeit
def encode_frequency(df_train, df_test, categorical_feature):
    df_merge = pd.concat([df_train[categorical_feature], df_test[categorical_feature]])

    for col in tqdm(categorical_feature):
        freq_col = '{}_Frequency'.format(col)
        df_freq = df_merge.groupby([col]).size() / df_merge.shape[0]
        df_freq = df_freq.reset_index().rename(columns={0: freq_col})

        if freq_col in df_train.columns:
            df_train.drop(freq_col, axis=1, inplace=True)
        if freq_col in df_test.columns:
            df_test.drop(freq_col, axis=1, inplace=True)

        df_train = df_train.merge(df_freq, on=col, how='left')
        df_test = df_test.merge(df_freq, on=col, how='left')

    print(df_train.shape, df_test.shape)

    return df_train, df_test


@timeit
def encode_mean_k_fold(df_train, df_test, categorical_feature, target_col):
    def _encode(col, alpha):
        target_mean_global = df_train[target_col].mean()

        nrows_cat = df_train.groupby(col)[target_col].count()
        target_means_cats = df_train.groupby(col)[target_col].mean()

        target_means_cats_adj = (target_means_cats * nrows_cat +
                                 target_mean_global * alpha) / (nrows_cat + alpha)
        # Mapping means to test data
        encoded_col_test = df_test[col].map(target_means_cats_adj)

        kfold = KFold(n_splits=5, shuffle=True, random_state=1989)
        parts = []
        for trn_inx, val_idx in kfold.split(df_train):
            df_for_estimation, df_estimated = df_train.iloc[trn_inx], df_train.iloc[val_idx]
            nrows_cat = df_for_estimation.groupby(col)[target_col].count()
            target_means_cats = df_for_estimation.groupby(col)[target_col].mean()

            target_means_cats_adj = (target_means_cats * nrows_cat +
                                     target_mean_global * alpha) / (nrows_cat + alpha)

            encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
            parts.append(encoded_col_train_part)

        encoded_col_train = pd.concat(parts, axis=0)
        encoded_col_train.fillna(target_mean_global, inplace=True)
        encoded_col_train.sort_index(inplace=True)

        return encoded_col_train, encoded_col_test

    for col in tqdm(categorical_feature):
        temp_encoded_tr, temp_encoded_te = _encode(col, 5)
        new_feat_name = 'mean_k_fold_{}'.format(col)
        df_train[new_feat_name] = temp_encoded_tr.values
        df_test[new_feat_name] = temp_encoded_te.values

    print(df_train.shape, df_test.shape)
    print(df_train.columns)

    return df_train, df_test


@timeit
def encode_lda(df_train, df_test, categorical_feature, y_categorized, lda_name='lda'):
    n_components = 10
    print('lda_{}_0to{}'.format(lda_name, n_components - 1))
    clf = LinearDiscriminantAnalysis(n_components=n_components)

    df_merge = pd.concat([df_train[categorical_feature], df_test[categorical_feature]])

    clf.fit(df_merge[categorical_feature], y_categorized)

    df_train_lda = pd.DataFrame(clf.transform(df_train[categorical_feature]))
    df_test_lda = pd.DataFrame(clf.transform(df_test[categorical_feature]))

    col_map = {i: 'lda_{}_{}'.format(lda_name, i) for i in range(n_components)}
    df_train_lda.rename(columns=col_map, inplace=True)
    df_test_lda.rename(columns=col_map, inplace=True)

    for c in col_map:
        if c in df_train.columns:
            df_train.drop(c, axis=1, inplace=True)
        if c in df_test.columns:
            df_test.drop(c, axis=1, inplace=True)

    df_train = pd.concat([df_train, df_train_lda], axis=1)
    df_test = pd.concat([df_test, df_test_lda], axis=1)

    print(df_train.shape, df_test.shape)

    return df_train, df_test
