import datetime
import operator
import os
import subprocess

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

RESULT_PATH = 'results'


def get_feature_importance(model):
    fea_dict = {}
    for k, v in zip(model.feature_name(), model.feature_importance()):
        fea_dict[k] = v

    fea_list = ['<feature_importance>']
    for i, fea in enumerate(sorted(fea_dict.items(), key=operator.itemgetter(1), reverse=True)):
        fea_list.append("{} {}".format(i, fea))

    res = '\n'.join(fea_list)
    print(res)
    return res


def report(df_train_idx, df_test_idx, y_pred_train, y_pred, msg, model=None):
    time_tag = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')

    if model:
        rmse_tag = 'T{0:.3f}_V{1:.3f}_K'.format(model.best_score['train']['rmse'], model.best_score['valid']['rmse'])
    else:
        rmse_tag = 'NONE'

    result_path = os.path.join(RESULT_PATH, '{}__{}'.format(time_tag, rmse_tag))

    os.makedirs(result_path, exist_ok=True)

    # Feature importance as an image
    fig, ax = plt.subplots(figsize=(15, 15))
    lgb.plot_importance(model, ax=ax, max_num_features=30)
    fig.savefig(os.path.join(result_path, 'feature_importance.jpg'))
    # fig.savefig('path/to/save/image/to.png')  # save the figure to file
    plt.close(fig)

    fea_res = get_feature_importance(model)
    msg.insert(0, fea_res)

    # Create train set raw result file
    df_res = df_train_idx.copy()
    df_res['y_pred'] = y_pred_train
    file_name = 'reg_train_{}.csv'.format(time_tag)
    df_res.to_csv(os.path.join(result_path, file_name), index=False)
    print('raw_train:', os.path.join(result_path, file_name))

    # Create test set raw result file
    df_res = df_test_idx.copy()
    df_res['y_pred'] = y_pred
    file_name = 'reg_test_{}.csv'.format(time_tag)
    df_res.to_csv(os.path.join(result_path, file_name), index=False)
    print('raw_test:', os.path.join(result_path, file_name))

    # Create submit file
    df_test_idx['PredictedLogRevenue'] = 0
    df_test_idx.loc[:, 'PredictedLogRevenue'] = y_pred
    df_test_idx.loc[:, "PredictedLogRevenue"] = df_test_idx["PredictedLogRevenue"].apply(lambda x: 0.0 if x < 0 else x)
    df_test_idx.loc[:, "PredictedLogRevenue"] = df_test_idx["PredictedLogRevenue"].fillna(0.0)
    df_test_idx.loc[:, "PredictedLogRevenue"] = np.expm1(df_test_idx["PredictedLogRevenue"])

    df_submit = df_test_idx[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
    df_submit.loc[:, "PredictedLogRevenue"] = np.log1p(df_submit["PredictedLogRevenue"])

    file_name = 'aiden_{}.csv.tar.gz'.format(time_tag)
    df_submit.to_csv(os.path.join(result_path, file_name), index=False, compression='gzip')
    print('submit:', os.path.join(result_path, file_name))

    # Write MSG
    with open(os.path.join(result_path, 'result.log'), 'w') as f:
        f.write('\n'.join(msg))

    # Copy notebook to results for history
    cmd = """cp -f {notebook_name} {result_path}/{notebook_name}
    """.format(**{'notebook_name': 'reg_lgbm.ipynb', 'result_path': result_path})
    print(cmd)
    subprocess.call(cmd, shell=True)

    return os.path.join(result_path, file_name)
