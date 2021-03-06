import datetime
import operator
import os
import subprocess

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from subprocess import PIPE, CalledProcessError, check_call, Popen

RESULT_PATH = 'results'


def get_feature_importance(model, importance_type='split'):
    fea_dict = {}
    for k, v in zip(model.feature_name(), model.feature_importance(importance_type)):
        fea_dict[k] = v

    fea_list = ['<feature_importance:{}>'.format(importance_type)]
    for i, fea in enumerate(sorted(fea_dict.items(), key=operator.itemgetter(1), reverse=True)):
        fea_list.append("{} {}".format(i, fea))

    res = '\n'.join(fea_list)
    print(res)
    return res


def report(df_train_idx, df_test_idx, y_pred_train, y_pred, msg, model=None):
    time_tag = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')

    if model:
        rmse_tag = 'T{0:.4f}_V{1:.4f}_R{2:.4f}_K'.format(
            model.best_score['train']['rmse'],
            model.best_score['valid']['rmse'],
            model.best_score['valid']['rmse'] / model.best_score['train']['rmse'])

        fea_res = get_feature_importance(model, importance_type='split')
        msg.insert(0, fea_res)
        fea_res = get_feature_importance(model, importance_type='gain')
        msg.insert(1, fea_res)
    else:
        rmse_tag = 'NONE'

    result_path = os.path.join(RESULT_PATH, '{}__{}'.format(time_tag, rmse_tag))

    os.makedirs(result_path, exist_ok=True)

    # Feature importance as an image
    # fig, ax = plt.subplots(figsize=(15, 15))
    # lgb.plot_importance(model, ax=ax, max_num_features=30)
    # fig.savefig(os.path.join(result_path, 'feature_importance.jpg'))
    # # fig.savefig('path/to/save/image/to.png')  # save the figure to file
    # plt.close(fig)



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
    y_pred[y_pred < 0] = 0
    df_test_idx["PredictedLogRevenue"] = np.expm1(y_pred)

    df_submit = df_test_idx.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
    df_submit.columns = ["fullVisitorId", "PredictedLogRevenue"]
    df_submit["PredictedLogRevenue"] = np.log1p(df_submit["PredictedLogRevenue"])

    file_name = 'aiden_{}.csv.tar.gz'.format(time_tag)
    df_submit.to_csv(os.path.join(result_path, file_name), index=False, compression='gzip')
    print('submit:', os.path.join(result_path, file_name))

    # Write MSG
    with open(os.path.join(result_path, 'result.log'), 'w') as f:
        f.write('\n\n'.join(msg))

    # Copy notebook to results for history
    cmd = """cp -f {notebook_name} {result_path}
    """.format(**{'notebook_name': '*py*', 'result_path': result_path})
    print(cmd)
    subprocess.call(cmd, shell=True)

    return os.path.join(result_path, file_name)


def _exe_cli(cmd):
    print(cmd)
    cmd_list = cmd.strip().split(' ')

    output = ''
    err = ''
    try:
        check_call(cmd_list)
        with Popen(cmd_list, stdout=PIPE) as df:
            output, err = df.communicate()
    except CalledProcessError as e:
        print(e)

    return output, err


def _chagne_user(user):
    kaggle_path = os.path.join(os.getenv('HOME'), ".kaggle")
    print(kaggle_path)
    cmd = "rm -f {kaggle_path}/kaggle.json".format(kaggle_path=kaggle_path)
    print(_exe_cli(cmd))

    cmd = "ln -s {kaggle_path}/kaggle.json_{user} {kaggle_path}/kaggle.json". \
        format(user=user, kaggle_path=kaggle_path)
    print(_exe_cli(cmd))


def submit_to_kaggle(user, file_path, msg='msg'):
    _chagne_user(user)

    msg = msg.replace(' ', '_')
    kaggle_cmd = os.path.join(os.getenv('HOME'), ".local", 'bin')
    cmd = """{kaggle_cmd}/kaggle competitions submit -c ga-customer-revenue-prediction -f {file_path} -m "{msg}"
    """.format(**{'file_path': file_path, 'msg': msg, 'kaggle_cmd': kaggle_cmd})
    print(_exe_cli(cmd))


def show_submissions(user=None):
    if user:
        _chagne_user(user)

    kaggle_cmd = os.path.join(os.getenv('HOME'), '.local', 'bin')
    cmd = "{kaggle_cmd}/kaggle competitions submissions ga-customer-revenue-prediction".format(kaggle_cmd=kaggle_cmd)
    output, err = _exe_cli(cmd)
    print(output[:1], err)


if __name__ == '__main__':
    # submit_to_kaggle('aidensong', '', '')
    show_submissions(user='aidensong')
