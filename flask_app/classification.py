from keras.models import load_model
import pandas as pd


def classification():
    data_swing = pd.read_csv('data/CSV/pose_detection.csv').drop('index', axis=1).iloc[0:90]
    data_swing = data_swing.to_numpy()
    data_swing = data_swing.reshape((1, data_swing.shape[0], data_swing.shape[1]))

    model = load_model('model_weight/tennis_swing_model.h5', compile=False)

    pred = model.predict(data_swing)[0]

    pred_msg_f = '{:.4f} %'.format(pred[0]*100)  # Federer
    pred_msg_n = '{:.4f} %'.format(pred[1]*100)  # Nadal
    pred_msg_d = '{:.4f} %'.format(pred[2]*100)  # Djokovic
    pred_msg_m = '{:.4f} %'.format(pred[3]*100)  # Murray

    return [pred_msg_f, pred_msg_n, pred_msg_d, pred_msg_m]

