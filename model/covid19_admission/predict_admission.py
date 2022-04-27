import os
import json
import cv2
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import traceback
from ..classification_pylon.predict import predict as pylon_predict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

threshold = [0.5, 0.5, 0.5]

def prepare_x(all_pred_df, record):
    # merge df
    # prepare data
    f = open(os.path.join(BASE_DIR, 'covid19_admission', 'required_columns.json'))
    required_cols = (json.load(f))['columns']
    f.close()

    img_columns = ['_'.join(col.split(' ')) for col in all_pred_df['Finding']]
    img_df = pd.DataFrame(columns=img_columns)
    img_df.loc[0] = all_pred_df['Confidence']

    record_new = {}
    # zero_cols = [] # used to fill na with zero (to replace get_dummies)
    for k, v in record.items():
        if isinstance(v, type(None)):
            record_new[k] = np.nan
        elif isinstance(v, str):
            col_name = f'{k}_{v}'
            # tmp = [c for c in required_cols if (k in c) and (c != col_name)]
            # zero_cols = zero_cols + tmp
            record_new[col_name] = 1
        else:
            record_new[k] = v

    record_df = pd.DataFrame()
    record_df = record_df.append(record_new, ignore_index=True)

    x_data = pd.concat([record_df, img_df], axis=1)
    
    # fill missing columns
    x_cols = x_data.columns
    missing_cols = []
    for col in required_cols:
        if col not in x_cols:
            missing_cols.append(col)
    
    missing_df = pd.DataFrame(columns=missing_cols)
    x_new = pd.concat([x_data, missing_df], axis=1)
    # x_new[zero_cols] = 0
    x_new = x_new[required_cols]
    # print(x_new.columns, x_new.to_numpy())
    return x_new.to_numpy()

def predict(ds, record):
    _, _, _, all_pred_df, _ = pylon_predict(ds) 

    # what to do with nan?
    x_data = prepare_x(all_pred_df, record)

    # xgboost old version / version changed -> score changed??
    clf1 = XGBClassifier()
    clf1.load_model(os.path.join(BASE_DIR, 'covid19_admission', "xgb_covid_d_1.json"))
    pred_d1 = (clf1.predict_proba(x_data))[0][1]

    clf2 = XGBClassifier()
    clf2.load_model(os.path.join(BASE_DIR, 'covid19_admission', "xgb_covid_d_2.json"))
    pred_d2 = (clf2.predict_proba(x_data))[0][1]

    clf3 = XGBClassifier()
    clf3.load_model(os.path.join(BASE_DIR, 'covid19_admission', "xgb_covid_d_3.json"))
    pred_d3 = (clf3.predict_proba(x_data))[0][1]

    pred = {
        'Finding': [
            'Admission within day 1',
            'Admission within day 2',
            'Admission within day 3'
        ],
        'Confidence': [float(pred_d1), float(pred_d2), float(pred_d3)],
        'Threshold': threshold,
        'isPositive': [
            True if pred_d1 >= threshold[0] else False,
            True if pred_d2 >= threshold[1] else False,
            True if pred_d3 >= threshold[2] else False
        ]
    }

    return pred

def main(ds, res_dir, record):
    try:
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        pred = predict(ds, record)
        if 'PatientName' in ds:
            pred['patient_name'] = ds.PatientName.given_name + " " + ds.PatientName.family_name
        with open(os.path.join(res_dir, 'prediction.txt'), 'w') as f:
            json.dump(pred, f)

        return True, ""
    except Exception as e:
        print(traceback.format_exc())
        return False, e