环境要求：
1. keras2、tensorflow、python3、opencv等。

网络介绍：
1. 人脸检测使用的是DarkNet的YOLOv2网络结构，在FDDB中进行训练，并将DarkNet模型转化为Keras模型。
2. 风格迁移使用的是在ImageNet上预训练的VGG16模型，并训练了5种风格网络模型。

文件说明：
1. face: 		人脸检测生成的结果文件夹。
2. faceDetection：	人脸检测部分核心检测和训练代码。其中训练代码train.py在faceDetection/train_yolov2/目录里。
3. pic：		原始的FDDB人脸数据集中的部分图片。
4. styleTransfer：	风格迁移部分核心代码。
5. transfered：	风格迁移后生成的结果文件夹。
6. main.py:  	入口程序，在main.py的第27行指定要检测的原始人脸图片，第28行指定要使用的风格网络。
7. train.py:	训练风格网络的代码。
8. process analyse: 	过程分析文件夹，包含Yolov2模型在FDDB数据集上训练时的Loss下降曲线。

运行方式：
1. 运行人脸检测和风格迁移：
    命令行中使用 python main.py -s 风格 -i 测试图片即可运行主程序，并生成face和transfered两个结果文件夹。

重新训练自己的数据：
1. 运行python train.py -s 风格即可重新利用styleTransfer/images/train下的风格图片训练自己的风格网络，并生成对应的h5模型。
2. 在faceDetection/train_yolov2/目录下运行python train.py即可重新使用FDDB数据集训练Yolov2模型。（该操作需要在linux环境下运行）