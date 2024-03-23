import cv2
import math

## カット開始地点のミリ秒
startSeconds = 0  ## = 3秒
## カット終了地点のミリ秒
stopSeconds = 10  ## = 10秒
 
videoCapture = cv2.VideoCapture("flask_app/static/videos/Federer_short.mp4")
## FPS : Frames per Second
fps = videoCapture.get(cv2.CAP_PROP_FPS)
print("fps:", fps)
## 総フレーム数
totalFrames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
print("total frames:", totalFrames)
## カット開始のフレームインデックス
startFrameIndex = math.ceil(fps * startSeconds)
## カット終了のフレームインデックス
stopFrameIndex = math.ceil(fps * stopSeconds)
## 範囲を越えたフレームインデックス修正
if(startFrameIndex < 0): 
    startFrameIndex = 0
if(stopFrameIndex >= totalFrames):
    stopFrameIndex = totalFrames-1
 
## 開始地点まで動画をシークする
videoCapture.set(cv2.CAP_PROP_POS_FRAMES, startFrameIndex)
## このあと使う変数
frameIndex = startFrameIndex


## 開始～終了までを画像に分割
imgArr = []
while(frameIndex <= stopFrameIndex):
    _,img = videoCapture.read()
    imgArr.append(img)
    frameIndex += 1

## 出力動画の形式（ascii4文字）
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
## 出力動画
video = None
## 出力動画のファイル名
videoFileName = "trimmed.mp4"
for img in imgArr:
    if(video is None):
        print("Writing")
        ## 動画の幅・高さ
        h,w,_ = img.shape
        ## FPS20で動画オブジェクト作成
        video = cv2.VideoWriter(videoFileName, fourcc, fps, (w,h))
    ## 動画末尾にフレーム書き出し
    video.write(img)
## 最期にリリース。忘れずに！
video.release()
