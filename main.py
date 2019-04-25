import os, time
import argparse
from faceDetection.face_detect import detect
from styleTransfer.transform import transfer


def face_detect(src_pic):
    face_list = detect(src_pic)
    return face_list


def style_transfer(face_list, style):
    output_dir = './transfered'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    t1 = time.time()
    for face in face_list:
        pic = face.split('/')[-1]
        name = pic.split('.jpg')[0]
        blend_alpha = 0.1
        media_filter = 3
        transfer(face, style, name, output_dir, blend_alpha, media_filter)
    print("process: %s" % (time.time() - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time style transfer')

    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name without extension')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='input picture path')

    args = parser.parse_args()
    style = args.style
    src_pic = args.input
    if not os.path.exists(src_pic):
        print(src_pic, 'is not exist, Please check your picture path!')
    else:
        face_list = face_detect(src_pic)
        if face_list:
            style_transfer(face_list, style)