from keras.models import load_model
# from keras.models import Model
import pandas as pd
import numpy as np


# Normalization
def normalization(df):
    df_x = df.loc[:, ['left_shoulder_x', 'right_shoulder_x', 'left_elbow_x', 'right_elbow_x', 'left_wrist_x',
                    'right_wrist_x', 'left_hip_x', 'right_hip_x', 'left_knee_x', 'right_knee_x', 'left_ankle_x', 'right_ankle_x']]
    df_y = df.loc[:, ['left_shoulder_y', 'right_shoulder_y', 'left_elbow_y', 'right_elbow_y', 'left_wrist_y',
                    'right_wrist_y', 'left_hip_y', 'right_hip_y', 'left_knee_y', 'right_knee_y', 'left_ankle_y', 'right_ankle_y']]

    df_x = df_x.apply(lambda iterator: ((df.max().max() - iterator)/(df.max().max() - df.min().min())).round(5))
    df_y = df_y.apply(lambda iterator: (1- (df.max().max() - iterator)/(df.max().max() - df.min().min())).round(5))
    df_out = pd.concat([df_x, df_y], axis=1)

    return df_out


def create_features(df):
    df_out = pd.DataFrame()
    df_out['left_wrist_up'] = df['left_wrist_x'] - df['left_shoulder_x']
    df_out['right_wrist_up'] = df['right_wrist_x'] - df['right_shoulder_x']
    df_out['right_elbow_dist'] = df['right_elbow_y'] - (df['right_shoulder_y'] + df['left_shoulder_y']) / 2
    df_out['right_arm_y'] = df['right_shoulder_y'] - df['right_wrist_y']
    df_out['left_arm_y'] = df['left_shoulder_y'] - df['left_wrist_y']
    df_out['wrist_y'] = df['right_wrist_y'] - df['left_wrist_y']
    df_out['shoulder_y'] = df['right_shoulder_y'] - df['left_shoulder_y']
    df_out['left_elbow_dist'] = (df['right_shoulder_y'] + df['left_shoulder_y']) / 2 - df['left_elbow_y']
    df_out['wrist_ankle_y'] = df['right_wrist_y'] - df['left_ankle_y']
    df_out['ankle_open'] = df['right_ankle_y'] - df['left_ankle_y']

    return df_out


def classification(mode):
    if mode == 'advanced':
        quick_mode = False
    else:
        quick_mode = True

    data_swing = pd.read_csv('data/CSV/pose_detection.csv').drop('index', axis=1).iloc[0:90]
    data_swing = normalization(data_swing)
    data_swing = create_features(data_swing)
    data_swing = data_swing.to_numpy()
    data_swing = data_swing.reshape((1, data_swing.shape[0], data_swing.shape[1]))

    # model_name = 'tennis_swing_model_v5.h5'
    # model = load_model('model_weight/'+model_name, compile=False)
    # print("Model: ", model_name)
    #
    # pred = model.predict([data_swing[:,:,0], data_swing[:,:,1], data_swing[:,:,2], data_swing[:,:,3],
    #                       data_swing[:,:,4], data_swing[:,:,5], data_swing[:,:,6], data_swing[:,:,7],
    #                       data_swing[:,:,8], data_swing[:,:,9]])
    #
    # print('pred: ', pred)
    #
    # pred_y = np.argmax(pred, axis=1)
    # pred_y = ['f', 'n', 'd', 'm'][pred_y[0]]
    # print('pred_y: ', pred_y)

    model_feature = load_model('model_weight/tennis_swing_model_v5_feature.h5', compile=False)
    features = model_feature.predict(
        [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
         data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
         data_swing[:, :, 8], data_swing[:, :, 9]])

    print("features.shape: ", features.shape)
    df_feature_ext = pd.read_csv('data/CSV/feature_ext_all_v5.csv').drop('Unnamed: 0', axis=1)
    l_dist = []
    for i in range(df_feature_ext.shape[0]):
        # get the distance between features
        dist = abs(np.linalg.norm(features[0, :] - df_feature_ext.iloc[i, :-1]))
        l_dist.append(dist)
    df_feature_ext['dist'] = l_dist
    df_feature_ext = df_feature_ext.sort_values('dist', ascending=True)
    df_feature_ext.loc[: , ['name', 'dist']].to_csv('data/CSV/feature_dist.csv')

    # Calculate the likelihood
    frac = 0.8
    n_samples = int(160 / frac)
    samples = df_feature_ext.loc[:, 'name'].iloc[:n_samples]
    print('samples.shape: ', samples.shape)
    num_f = sum(1 for row in samples if row[0] == 'f')
    num_n = sum(1 for row in samples if row[0] == 'n')
    num_d = sum(1 for row in samples if row[0] == 'd')
    num_m = sum(1 for row in samples if row[0] == 'm')

    pred_msg_f = '{:.2f} %'.format((num_f / n_samples) * 100)  # Federer
    pred_msg_n = '{:.2f} %'.format((num_n / n_samples) * 100)  # Nadal
    pred_msg_d = '{:.2f} %'.format((num_d / n_samples) * 100)  # Djokovic
    pred_msg_m = '{:.2f} %'.format((num_m / n_samples) * 100)  # Murray

    # kNN (k=200)
    pred_idx = pd.Series([num_f, num_n, num_d, num_m]).idxmax()
    pred_y = ['f', 'n', 'd', 'm'][pred_idx]
    print('pred_y: ', pred_y)

    for sample in df_feature_ext.loc[:, 'name']:
        if sample.startswith(pred_y):
            most_similar_sample = sample
            break

    pred_result_video = "static/videos/result_videos/" + most_similar_sample[:-4] + '.mp4'

    if not quick_mode:
        # Find nearest n sub-features
        model_sub1 = load_model('model_weight/model_sub1.h5', compile=False)
        model_sub2 = load_model('model_weight/model_sub2.h5', compile=False)
        model_sub3 = load_model('model_weight/model_sub3.h5', compile=False)
        model_sub4 = load_model('model_weight/model_sub4.h5', compile=False)
        model_sub5 = load_model('model_weight/model_sub5.h5', compile=False)
        model_sub6 = load_model('model_weight/model_sub6.h5', compile=False)
        model_sub7 = load_model('model_weight/model_sub7.h5', compile=False)
        model_sub8 = load_model('model_weight/model_sub8.h5', compile=False)
        model_sub9 = load_model('model_weight/model_sub9.h5', compile=False)
        model_sub10 = load_model('model_weight/model_sub10.h5', compile=False)

        # Extract sub-features of input data
        features_sub1_input = model_sub1.predict(
            [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
             data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
             data_swing[:, :, 8], data_swing[:, :, 9]])[0]
        features_sub2_input = model_sub2.predict(
            [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
             data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
             data_swing[:, :, 8], data_swing[:, :, 9]])[0]
        features_sub3_input = model_sub3.predict(
            [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
             data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
             data_swing[:, :, 8], data_swing[:, :, 9]])[0]
        features_sub4_input = model_sub4.predict(
            [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
             data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
             data_swing[:, :, 8], data_swing[:, :, 9]])[0]
        features_sub5_input = model_sub5.predict(
            [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
             data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
             data_swing[:, :, 8], data_swing[:, :, 9]])[0]
        features_sub6_input = model_sub6.predict(
            [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
             data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
             data_swing[:, :, 8], data_swing[:, :, 9]])[0]
        features_sub7_input = model_sub7.predict(
            [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
             data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
             data_swing[:, :, 8], data_swing[:, :, 9]])[0]
        features_sub8_input = model_sub8.predict(
            [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
             data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
             data_swing[:, :, 8], data_swing[:, :, 9]])[0]
        features_sub9_input = model_sub9.predict(
            [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
             data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
             data_swing[:, :, 8], data_swing[:, :, 9]])[0]
        features_sub10_input = model_sub10.predict(
            [data_swing[:, :, 0], data_swing[:, :, 1], data_swing[:, :, 2], data_swing[:, :, 3],
             data_swing[:, :, 4], data_swing[:, :, 5], data_swing[:, :, 6], data_swing[:, :, 7],
             data_swing[:, :, 8], data_swing[:, :, 9]])[0]
        df_feature_sub_input = pd.DataFrame(np.stack([features_sub1_input, features_sub2_input, features_sub3_input,
                                                          features_sub4_input, features_sub5_input, features_sub6_input,
                                                          features_sub7_input, features_sub8_input, features_sub9_input,
                                                          features_sub10_input]))

        df_feature_sub_all = pd.read_csv('data/CSV/feature_ext_sub_all.csv').drop('Unnamed: 0', axis=1)
        df_feature_sub_sample = df_feature_sub_all[(df_feature_sub_all['name'] == most_similar_sample)].iloc[:, :-2].astype('float32')
        print("!!!!!!!!!!!!!!!", df_feature_sub_sample.shape)

        # Calculate the distances
        l_dist = []
        for i in range(df_feature_sub_input.shape[0]):
            dist = abs(np.linalg.norm(df_feature_sub_input.iloc[i] - df_feature_sub_sample.iloc[i]))
            print(i)
            print(df_feature_sub_input.iloc[i] - df_feature_sub_sample.iloc[i])
            print(np.linalg.norm(df_feature_sub_input.iloc[i] - df_feature_sub_sample.iloc[i]))
            print(dist)
            l_dist.append(dist)
        print(df_feature_sub_input.iloc[0])
        print(df_feature_sub_sample.iloc[0])
        print(df_feature_sub_input.shape)
        print(df_feature_sub_sample.shape)
        print(l_dist)

        smallest_indices = []
        for i in range(3):
            min_index = l_dist.index(min(l_dist))
            smallest_indices.append(min_index)
            l_dist.pop(min_index)

        msg_feature_sub = ["Height of your non-dominant hand.",
                           "Height of your dominant hand.",
                           "Position of your dominant elbow",
                           "Movement of your dominant arm.",
                           "Movement of your non-dominant arm.",
                           "Distance of your right and left arms",
                           "Movement of your shoulders",
                           "Position of your non-dominant elbow",
                           "Position of your dominant hand.",
                           "Distance of your right and left feet."]

    print("!!!!!!!!!!!!")
    print(smallest_indices)

    if not quick_mode:
        return [pred_msg_f, pred_msg_n, pred_msg_d, pred_msg_m, pred_result_video,
                msg_feature_sub[smallest_indices[0]], msg_feature_sub[smallest_indices[1]], msg_feature_sub[smallest_indices[2]]]
    else:
        return [pred_msg_f, pred_msg_n, pred_msg_d, pred_msg_m, pred_result_video]