import os
import pandas as pd
import cv2

def add_info(df, dir):
    mapping1 = {
        'personA':0,
        'personB':1,
        'personC':2,
        'personD':3,
        'personE':4
    }
    mapping2 = {
        'real':0,
        'forged':1
    }
    y = []
    path = []
    for index, row in df.iterrows():
        y.append(row["image_name"][:7])
        #print(row["image_name"][:7])
        if row["image_name"].__contains__('nA_'):
            x = 'personA'
        elif row["image_name"].__contains__('nB_'):
            x = 'personB'
        elif row["image_name"].__contains__('nC_'):
            x = 'personC'
        elif row["image_name"].__contains__('nD_'):
            x = 'personD'
        elif row["image_name"].__contains__('nE_'):
            x = 'personE'
        p = os.path.join(dir, x, row["image_name"])
        path.append(p)
    df['class'] = y
    df['class'] = df['class'].map(mapping1)
    df['label'] = df['label'].map(mapping2)
    df['path'] = path

def load_images(df):
    grays = []
    rgbs = []
    for image_path in df['path']:
        image = cv2.imread(image_path)
        # (390, 527)
        res = cv2.resize(image, dsize=(527, 390), interpolation=cv2.INTER_CUBIC)
        grays.append(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY))
        rgbs.append(res)
    df['gray_images'] = grays
    df['rgb_images'] = rgbs