from keras.models import load_model
import pandas as pd


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
    data_swing = create_features(data_swing)
    data_swing = data_swing.to_numpy()
    data_swing = data_swing.reshape((1, data_swing.shape[0], data_swing.shape[1]))

    model_name = 'tennis_swing_model_v01.h5'
    model = load_model('model_weight/'+model_name, compile=False)
    print("Model: ", model_name)

    pred = model.predict([data_swing[:,:,0], data_swing[:,:,1], data_swing[:,:,2], data_swing[:,:,3],
                          data_swing[:,:,4], data_swing[:,:,5], data_swing[:,:,6], data_swing[:,:,7],
                          data_swing[:,:,8], data_swing[:,:,9], ])[0]

    pred_msg_f = '{:.4f} %'.format(pred[0]*100)  # Federer
    pred_msg_n = '{:.4f} %'.format(pred[1]*100)  # Nadal
    pred_msg_d = '{:.4f} %'.format(pred[2]*100)  # Djokovic
    pred_msg_m = '{:.4f} %'.format(pred[3]*100)  # Murray

    return [pred_msg_f, pred_msg_n, pred_msg_d, pred_msg_m]

