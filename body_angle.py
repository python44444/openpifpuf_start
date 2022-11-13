import cv2
import os
import sys
import openpifpaf
import numpy as np

cur_dir = os.path.dirname(sys.argv[0])


def main():
    # 検出の準備
    predictor = openpifpaf.Predictor(checkpoint="shufflenetv2k16")

    # 画像を取得
    capture = cv2.VideoCapture(os.path.join(cur_dir, "images/sample.jpg"))
    _, image = capture.read()

    # 検出
    predictions, _, _ = predictor.numpy_image(image)

    # 人間を選ぶfor
    for pred in predictions:
        # その人物のデータを取得するfor
        # for pt in pred.data[:, :2].astype("int"):
        #     cv2.circle(
        #         image, center=pt, radius=5, color=(255, 255, 0), thickness=-1
        #     )
        left_shoulder = pred.data[pred.keypoints.index("left_shoulder")][:2]
        right_shoulder = pred.data[pred.keypoints.index("right_shoulder")][:2]
        left_hip = pred.data[pred.keypoints.index("left_hip")][:2]
        right_hip = pred.data[pred.keypoints.index("right_hip")][:2]

        center_shoulder = np.mean([left_shoulder, right_shoulder], 0, np.int16)[:2]
        center_hip = np.mean([left_hip, right_hip], 0, np.int16)[:2]

        vec = (center_shoulder - center_hip) * [1, -1]
        angle = np.arctan2(vec[0], vec[1]) * 180 / np.pi

        cv2.putText(
            image,
            text=f"{angle:.2f}",
            org=(center_shoulder[0] - 20, center_shoulder[1] + 20),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 255, 0),
            thickness=2,
        )

    # 10秒表示
    cv2.imshow("test", image)
    cv2.waitKey(10000)


if __name__ == "__main__":
    main()
