from styleTransfer.loss import dummy_loss
from keras.optimizers import Adam
from scipy.misc import imsave
import numpy as np
import h5py

from scipy.ndimage.filters import median_filter
from styleTransfer.img_util import preprocess_reflect_image, crop_image

import styleTransfer.nets as nets


def blend(original, stylized, alpha):
    return alpha * original + (1 - alpha) * stylized


def median_filter_all_colours(im_small, window_size):
    """
    Applies a median filer to all colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = median_filter(im_small[:,:,d], size=(window_size,window_size))
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")
    return im_conv


def load_weights(model, file_path):
    f = h5py.File(file_path)

    layer_names = [name for name in f.attrs['layer_names']]

    for i, layer in enumerate(model.layers[:31]):
        g = f[layer_names[i]]
        weights = [g[name] for name in g.attrs['weight_names']]
        layer.set_weights(weights)

    f.close()
    print('Pretrained Model weights loaded.')


def transfer(input_file, style, pic_name, output_dir, blend_alpha, media_filter):
    aspect_ratio, x = preprocess_reflect_image(input_file, size_multiple=4)

    img_width = img_height = x.shape[1]
    net = nets.image_transform_net(img_width,img_height)
    model = nets.loss_net(net.output, net.input, img_width, img_height, '', 0,0)

    model.compile(Adam(), dummy_loss)      # Dummy loss since we are learning from regularizes

    model.load_weights("./styleTransfer/pretrained/"+style+'_weights.h5', by_name=False)

    y = net.predict(x)[0] 
    y = crop_image(y, aspect_ratio)

    ox = crop_image(x[0], aspect_ratio)

    y = median_filter_all_colours(y, media_filter)

    if blend_alpha > 0:
        y = blend(ox, y, blend_alpha)

    imsave(output_dir + '/' + pic_name + '_' + style + '.png', y)
