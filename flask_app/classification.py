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


def classification():
    data_swing = pd.read_csv('data/CSV/pose_detection.csv').drop('index', axis=1).iloc[0:90]
    data_swing = normalization(data_swing)
    data_swing = create_features(data_swing)
    data_swing = data_swing.to_numpy()
    data_swing = data_swing.reshape((1, data_swing.shape[0], data_swing.shape[1]))

    # model_name = 'tennis_swing_model_v5.h5'
    # model = load_model('model_weight/'+model_name, compile=False)
    # print("Model: ", model_name)

    # pred = model.predict([data_swing[:,:,0], data_swing[:,:,1], data_swing[:,:,2], data_swing[:,:,3],
    #                       data_swing[:,:,4], data_swing[:,:,5], data_swing[:,:,6], data_swing[:,:,7],
    #                       data_swing[:,:,8], data_swing[:,:,9]])[0]

    # pred_msg_f = '{:.4f} %'.format(pred[0]*100)  # Federer
    # pred_msg_n = '{:.4f} %'.format(pred[1]*100)  # Nadal
    # pred_msg_d = '{:.4f} %'.format(pred[2]*100)  # Djokovic
    # pred_msg_m = '{:.4f} %'.format(pred[3]*100)  # Murray

    # Feature extraction
    # model_feature = Model(
    #     inputs=[model.get_layer('input_11').input, model.get_layer('input_12').input, model.get_layer('input_13').input,
    #             model.get_layer('input_14').input, model.get_layer('input_15').input,
    #             model.get_layer('input_16').input, model.get_layer('input_17').input, model.get_layer('input_18').input,
    #             model.get_layer('input_19').input, model.get_layer('input_20').input],
    #     outputs=model.get_layer('dense_25').output)

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
    df_feature_ext.loc[: ,['name','dist']].to_csv('data/CSV/feature_dist.csv')

    #kNN
    k = 30
    df_knn = df_feature_ext.loc[:, 'name'].iloc[0:k]
    knn_f = sum(1 for row in df_knn if row[0] == 'f')
    knn_n = sum(1 for row in df_knn if row[0] == 'n')
    knn_d = sum(1 for row in df_knn if row[0] == 'd')
    knn_m = sum(1 for row in df_knn if row[0] == 'm')
    pred_idx = pd.Series([knn_f, knn_n, knn_d, knn_m]).idxmax()
    pred = ['f', 'n', 'd', 'm'][pred_idx]

    # Calculate the likelihood
    frac = 0.75
    n_samples = int(sum(1 for row in df_feature_ext.loc[:, 'name'] if row[0] == pred) / frac)
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
    pred_result_video = "static/videos/result_videos/" + df_feature_ext.loc[:, 'name'].iloc[0][:-4] + '.mp4'

    return [pred_msg_f, pred_msg_n, pred_msg_d, pred_msg_m, pred_result_video]

