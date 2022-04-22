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
    img_df = pd.DataFrame(columns=all_pred_df['Finding'])
    img_df.loc[0] = all_pred_df['Confidence']

    record_df = pd.DataFrame()
    record_df = record_df.append(record, ignore_index=True)

    x_data = pd.concat([record_df, img_df], axis=1)

    obj_cols = []
    for cname, ctype in x_data.dtypes.iteritems():
        if ctype=='object' and cname!='Image file':
            obj_cols.append(cname)
    # get dummy for nominal data
    x_data = pd.get_dummies(x_data, columns=obj_cols, drop_first=True)
    
    # fill missing columns
    f = open(os.path.join(BASE_DIR, 'covid19_admission', 'required_columns.json'))
    required_cols = (json.load(f))['columns']
    f.close()

    x_cols = x_data.columns
    missing_cols = []
    for col in required_cols:
        if col not in x_cols:
            missing_cols.append(col)
    
    missing_df = pd.DataFrame(columns=missing_cols)
    x_new = pd.concat([x_data, missing_df], axis=1)
    x_new = x_new[required_cols]
    return x_new.to_numpy()

def predict(ds, record):
    _, _, _, all_pred_df, image_load = pylon_predict(ds) 

    # wrong cxr confidence
    # what to do with nan?
    x_data = prepare_x(all_pred_df, record)

    # xgboost old version
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

    return pred, image_load

def main(ds, file_dir, record):
    try:
        res_dir = os.path.join(file_dir, 'result')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        pred, image_load = predict(ds, record)
        if 'PatientName' in ds:
            pred['patient_name'] = ds.PatientName.given_name + " " + ds.PatientName.family_name
        with open(os.path.join(res_dir, 'prediction.txt'), 'w') as f:
            json.dump(pred, f)

        scale = 512/np.array(image_load).shape[0]
        original_image = cv2.resize(np.array(image_load), (0,0), fx=scale, fy=scale)
        cv2.imwrite(os.path.join(res_dir, 'original.png'), original_image)

        return True, ""
    except Exception as e:
        print(traceback.format_exc())
        return False, e