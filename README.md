# FaceStyle

This is a project combined with face detection and face neural style transfer.

Face detection stage is an implementation of YOLO v2 for face detection based on <a href="https://github.com/abars/YoloKerasFaceDetection">YoloKerasFaceDetection by abars</a>.
Face neural style transfer stage is heavily inspired from <a href="https://github.com/titu1994/Fast-Neural-Style">Fast-Neural-Style by titu1994</a>.

Example:

    python main.py -s la_muse -i ./pic/img_1289.jpg

Train a YOLO detection with FDDB:

    cd ./faceDetection/train_yolov2/
    python train.py

The detected face(s) will be saved under "face" folder and the stylized face picture will be save under "transferred" folder.
