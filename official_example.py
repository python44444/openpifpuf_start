import argparse
import openpifpaf
import cv2

parser = argparse.ArgumentParser()
openpifpaf.show.cli(parser)
args = parser.parse_args()
args.show = True
openpifpaf.show.configure(args)

capture = cv2.VideoCapture("images/sample.jpg")
_, image = capture.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor = openpifpaf.Predictor(checkpoint="shufflenetv2k16")
predictions, gt_anns, meta = predictor.numpy_image(image)

print(predictions[0].data[0])

with openpifpaf.show.Canvas.image(image) as ax:
    pass

annotation_painter = openpifpaf.show.AnnotationPainter()
with openpifpaf.show.Canvas.image(image) as ax:
    annotation_painter.annotations(ax, predictions)
