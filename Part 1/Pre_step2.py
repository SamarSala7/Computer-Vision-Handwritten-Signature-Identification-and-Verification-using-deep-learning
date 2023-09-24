import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

main_dir = os.getcwd()
path = 'Part 1'
dir = os.path.join(main_dir, path)

test_files, train_files = [], []
test_csvs, train_csvs = [], []
y_test, y_train = [], []

train_dir = os.path.join(dir, 'Train')
test_dir = os.path.join(dir, 'Test')

train_dirCSV = os.path.join(dir, 'Train_CSV')
test_dirCSV = os.path.join(dir, 'Test_CSV')

train_dirA = os.path.join(train_dir, 'PersonA')
test_dirA = os.path.join(test_dir, 'PersonA')
train_dirB = os.path.join(train_dir, 'PersonB')
test_dirB = os.path.join(test_dir, 'PersonB')
train_dirC = os.path.join(train_dir, 'PersonC')
test_dirC = os.path.join(test_dir, 'PersonC')
train_dirD = os.path.join(train_dir, 'PersonD')
test_dirD = os.path.join(test_dir, 'PersonD')
train_dirE = os.path.join(train_dir, 'PersonE')
test_dirE = os.path.join(test_dir, 'PersonE')

for subdir, dirs, files in os.walk(dir):
    for file in files:
        if subdir.endswith('Test'):
            test_files.append(os.path.join(subdir, file))
        elif subdir.endswith('Train'):
            train_files.append(os.path.join(subdir, file))
try:
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(train_dirCSV)
    os.makedirs(test_dirCSV)
    os.makedirs(train_dirA)
    os.makedirs(test_dirA)
    os.makedirs(train_dirB)
    os.makedirs(test_dirB)
    os.makedirs(train_dirC)
    os.makedirs(test_dirC)
    os.makedirs(train_dirD)
    os.makedirs(test_dirD)
    os.makedirs(train_dirE)
    os.makedirs(test_dirE)
    os.makedirs(train_dir)

except:
    print('Already Created')


df_test = pd.DataFrame()
df_train = pd.DataFrame()

for a in train_files:
    if a.endswith('csv'):
        df_temp = pd.read_csv(a)
        df_train = df_train.append(df_temp, ignore_index=True)
    elif a.endswith('png'):
        if a.__contains__('nA_'):
            shutil.copy(a, train_dirA)
        elif a.__contains__('nB_'):
            shutil.copy(a, train_dirB)
        elif a.__contains__('nC_'):
            shutil.copy(a, train_dirC)
        elif a.__contains__('nD_'):
            shutil.copy(a, train_dirD)
        elif a.__contains__('nE_'):
            shutil.copy(a, train_dirE)


for b in test_files:
    if b.endswith('csv'):
        df_temp = pd.read_csv(b)
        df_test = df_test.append(df_temp, ignore_index=True)
    elif b.endswith('png'):
        if b.__contains__('nA_'):
            shutil.copy(b, test_dirA)
        elif b.__contains__('nB_'):
            shutil.copy(b, test_dirB)
        elif b.__contains__('nC_'):
            shutil.copy(b, test_dirC)
        elif b.__contains__('nD_'):
            shutil.copy(b, test_dirD)
        elif b.__contains__('nE_'):
            shutil.copy(b, test_dirE)

def add_info(df, dir):
    y = []
    path = []
    for index, row in df.iterrows():
        y.append(row["image_name"][:7])
        path.append(os.path.join(dir, row["image_name"]))
    df['class'] = y
    df['path'] = path
    # df['image_path'] = os.path.join(dir,)


add_info(df_test, test_dir)
add_info(df_train, train_dir)
# df_test['class'] = df_test['image_name'].str[:7]
# df_train['class'] = df_train['image_name'].str[:7]
df_test.to_csv(os.path.join(test_dirCSV, 'out.csv'), index=False)
df_train.to_csv(os.path.join(train_dirCSV, 'out.csv'), index=False)

