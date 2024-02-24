import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import shutil
from pathlib import Path

# YOLOモデルの読み込み
model = YOLO("yolov8n.pt")

# 画像フォルダのパス
images_dir = "images"

#ぼやけた画像のの保存先フォルダ
blurred_images_dir = "blurred_images"

# ディレクトリの中身を削除する関数
def clear_directory(directory):
    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

# blurred_images_dir と pred_faces_dir、person の中身を削除
blurred_images_dir = Path("blurred_images")
pred_faces_dir = Path("pred_faces")
new_images_dir = Path("new_images")
clear_directory(blurred_images_dir)
clear_directory(pred_faces_dir)
clear_directory(new_images_dir)

print("blurred_images と pred_facesディレクトリの中身を削除しました。")

# 画像ファイルのリストを取得
image_files = os.listdir(images_dir)

# 各画像ファイルに対して処理を行う
for image_file in image_files:
    # 画像認識の実行
    results = model(os.path.join(images_dir, image_file))

def is_blurred_face(face_image):
    """顔画像がぼやけているかを判断する関数"""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(lap_var)
    return lap_var > 500  # ラプラシアン分散のしきい値

# 画像フォルダとぼやけた画像の保存先フォルダ、検出された顔の保存先フォルダのパス
images_dir = Path("images")
blurred_images_dir = Path("blurred_images")
pred_faces_dir = Path("pred_faces")

# 存在しない場合はディレクトリを作成
blurred_images_dir.mkdir(parents=True, exist_ok=True)
pred_faces_dir.mkdir(parents=True, exist_ok=True)
new_images_dir.mkdir(parents=True, exist_ok=True)

# 顔認識のためのHaar Cascade分類器をロード
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 画像ファイルのリストを取得
image_files = [file for ext in ["*.jpg", "*.jpeg", "*.png"] for file in images_dir.glob(ext)]

# 各画像ファイルに対して処理を行う
for image_file in image_files:
    print(image_file)
    image = cv2.imread(str(image_file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔の検出
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for i, (x, y, w, h) in enumerate(faces):
        face_image = image[y:y+h, x:x+w]

        # 検出された顔をpred_facesフォルダに保存
        face_filename = f"{image_file.stem}_face{i}{image_file.suffix}"  # 新しいファイル名を生成
        cv2.imwrite(str(pred_faces_dir / face_filename), face_image)

        # ぼやけているかどうかを判断
        if is_blurred_face(face_image):
            # ぼやけている顔が検出されたら、画像をblurred_imagesフォルダに移動
            target_path = blurred_images_dir / image_file.name
            #image_file.rename(target_path) # 本のファイルを移動
            shutil.copy(image_file, target_path)  # 元のファイルをコピー
            break  # 1つのぼやけた顔を検出したらその画像は処理を終える

print("すべての処理が完了しました。")
