import pandas as pd
import numpy as np
import time


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

    df["weekofmonth"] = df['day'].astype(int) // 7  # aiden

    df["weekday"] = df['date'].dt.weekday
    df['weekofyear'] = df['date'].dt.weekofyear
    df['month_unique_user_count'] = df.groupby('month')['fullVisitorId'].transform('nunique')
    df['day_unique_user_count'] = df.groupby('day')['fullVisitorId'].transform('nunique')
    df['weekday_unique_user_count'] = df.groupby('weekday')['fullVisitorId'].transform('nunique')
    df['month_unique_s_count'] = df.groupby('month')['sessionId'].transform('nunique')
    df['day_unique_s_count'] = df.groupby('day')['sessionId'].transform('nunique')
    df['weekday_unique_s_count'] = df.groupby('weekday')['sessionId'].transform('nunique')

    df['hour'] = pd.to_datetime(df['visitStartTime'], unit='s').dt.hour  # aiden
    df['hour_unique_user_count'] = df.groupby('hour')['fullVisitorId'].transform('nunique')  # aiden
    df['hour_unique_s_count'] = df.groupby('hour')['sessionId'].transform('nunique')  # aiden

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
    df['mean_hour_per_browser_operatingSystem'] = df.groupby('browser_operatingSystem')['hour'].transform('mean')  # aiden
    df['source_country'] = df['trafficSource_source'] + '_' + df['geoNetwork_country']
    return df


@timeit
def fea_totals(df):
    df['visitNumber'] = np.log1p(df['visitNumber'])
    df['totals_hits'] = np.log1p(df['totals_hits'])
    df['totals_pageviews'] = np.log1p(df['totals_pageviews'].fillna(0))

    df['totals_pageviews_hit_rate'] = df['totals_hits'] - df['totals_pageviews']

    df['mean_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('mean')
    df['sum_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('sum')
    df['max_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('max')
    df['min_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('min')
    df['var_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('var')

    df['mean_hits_per_hour'] = df.groupby(['hour'])['totals_hits'].transform('mean')  # aiden
    df['sum_hits_per_hour'] = df.groupby(['hour'])['totals_hits'].transform('sum')  # aiden
    df['max_hits_per_hour'] = df.groupby(['hour'])['totals_hits'].transform('max')  # aiden
    df['min_hits_per_hour'] = df.groupby(['hour'])['totals_hits'].transform('min')  # aiden
    df['var_hits_per_hour'] = df.groupby(['hour'])['totals_hits'].transform('var')  # aiden
    return df


@timeit
def fea_geo_network(df):
    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform(
        'count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform(
        'mean')
    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')
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
    print(set(df.columns) - set(org_cols))

    return df
