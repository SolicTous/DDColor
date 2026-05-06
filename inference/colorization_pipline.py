import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from convert_onnx import *

import sys
sys.path.append("D:/PyTools")
from ImageChunk import *
from DebugPhoto import *
from Metrics import *
from PbLoad import *
from ChangeImage import *

class ImageColorizationPipeline(object):

    def __init__(self, model_path, input_size=256, model_size='large'):
        
        self.input_size = input_size

        # if torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        #     print('cuda')
        # else:
        #     self.device = torch.device('cpu')
        #     print('cpu')
        self.device = torch.device('cpu')

        if model_size == 'tiny':
            self.encoder_name = 'convnext-t'
        else:
            self.encoder_name = 'convnext-l'

        self.decoder_type = "MultiScaleColorDecoder"

        if self.decoder_type == 'MultiScaleColorDecoder':
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='MultiScaleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ).to(self.device)
        else:
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='SingleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=256,
            ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()

        self.convert_done = False

        # self.model.refine_net[0][0].weight_orig = torch.nn.Parameter(self.model.refine_net[0][0].weight_orig.permute(0, 2, 3, 1))

        # from torchinfo import summary
        # summary(self.model, input_size=(1, 3, 512, 512), device='cpu', depth=20, verbose=2)

        mpath = 'C:/Users/marsel/PycharmProjects/DDColor/saved_model/'
        # loaded = tf.saved_model.load(mpath, tags=['serve'])
        # print(list(loaded.signatures.keys()))
        # self.colorizer = loaded
        # # self.colorizer = loaded.signatures['serving_default']
        # print(self.colorizer.structured_outputs)

        self.inp_name = 'x'
        input_dict = {self.inp_name: (1, 512, 512, 3)}
        output_name = 'Identity'

        self.colorizer = ModelLoader(model_filepath=mpath + 'frozen.pb',
                                     inputs=input_dict, output=output_name, gpu_use=1, dtype=tf.float32)

    @torch.no_grad()
    def process(self, img):
        self.height, self.width = img.shape[:2]
        # print(self.width, self.height)
        # if self.width * self.height < 100000:
        #     self.input_size = 256
        print('start image', img.shape, type(img), 'min = ', np.min(img), 'max = ', np.max(img))

        img = (img / 255.0).astype(np.float32)
        print('before cv_bgr2lab img', img.shape, type(img),'min = ', np.min(img), 'max = ', np.max(img))
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        orig_l = orig_lab[:,:,:1]  # (h, w, 1)
        print('after cv_bgr2lab orig_l', orig_lab.shape, type(orig_lab), 'min_L = ', np.min(orig_lab[:, :, :1]), 'max_L = ', np.max(orig_lab[:, :, :1]),
              'min_AB = ', np.min(orig_lab[:, :, 1:2]), 'max_AB = ', np.max(orig_lab[:, :, 1:2]))

        # resize rgb image -> lab -> get grey -> rgb
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        print('after cv_bgr2lab img_lab_resized', img_lab.shape, type(img_lab), 'min_L = ', np.min(img_lab[:, :, :1]), 'max_L = ', np.max(img_lab[:, :, :1]),
              'min_AB = ', np.min(img_lab[:, :, 1:2]), 'max_AB = ', np.max(img_lab[:, :, 1:2]))
        img_l = img_lab[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)

        tensor_gray_rgb = tensor_gray_rgb.permute(0, 2, 3, 1)
        print('before NN tensor_gray_rgb', tensor_gray_rgb.shape, type(tensor_gray_rgb),
              'min = ', np.min(tensor_gray_rgb.cpu().numpy()), 'max = ',  np.max(tensor_gray_rgb.cpu().numpy()))

        cv2.imshow('________________________________________', (tensor_gray_rgb[0] * 255).cpu().numpy().round().astype(np.uint8))
        cv2.waitKey(0)

        # output_ab = self.colorizer(tensor_gray_rgb.cpu().numpy())
        output_ab = torch.tensor(self.colorizer.test({self.inp_name: tensor_gray_rgb.cpu().numpy()}))
        # output_ab = self.model(tensor_gray_rgb).cpu()  # (1, 2, self.height, self.width)

        # if not self.convert_done:
        #     convert_onnx(pt_model = self.model, fin_path = r"D:/Models/DDColor/default/colorizer_new.onnx", device = self.device)
        #     self.convert_done = True

        print('after NN output_ab', output_ab.shape, type(output_ab),
              'min = ', np.min(output_ab.numpy()),'max = ', np.max(output_ab.numpy()))
        output_ab = torch.tensor(output_ab.numpy()).permute(0, 3, 1, 2)
        print(output_ab.shape, type(output_ab))

        # tensor_gray_rgb = tensor_gray_rgb.cpu().numpy()
        # print(tensor_gray_rgb.shape)
        # output_ab = torch.tensor(self.colorizer(tensor_gray_rgb))

        # resize ab -> concat original l -> rgb
        output_ab = output_ab[0].float().numpy().transpose(1, 2, 0)
        output_ab_resize = cv2.resize(output_ab, (self.width, self.height))
        print(output_ab_resize.shape, type(output_ab_resize))
        print(orig_l.shape, type(orig_l))
        # output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        print('before cv_lab2rgb output_lab', output_lab.shape, type(output_lab),
              'min_L = ', np.min(output_lab[:, :, :1]), 'max_L = ', np.max(output_lab[:, :, :1]),
              'min_AB = ', np.min(output_lab[:, :, 1:2]), 'max_AB = ', np.max(output_lab[:, :, 1:2]))
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        print('after cv_bgr2lab output_bgr', output_bgr.shape, type(output_bgr), 'min = ', np.min(output_bgr), 'max = ', np.max(output_bgr))

        output_img = (output_bgr * 255.0).round().astype(np.uint8)

        print('final image', output_img.shape, type(output_img), 'min = ', np.min(output_img), 'max = ', np.max(output_img))

        return output_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='D:/Models/DDColor/default/damo/cv_ddcolor_image-colorization/pytorch_model.pt') # pretrain/net_g_200000.pth
    parser.add_argument('--input', type=str, default='T:/NeuralNetworks/Color/Images/test/') # figure/
    parser.add_argument('--output', type=str, default='T:/NeuralNetworks/Color/DDColor/default/test/', help='output folder or video path') # results
    parser.add_argument('--input_size', type=int, default=512, help='input size for model')
    parser.add_argument('--model_size', type=str, default='large', help='ddcolor model size')
    args = parser.parse_args()

    print(f'Output path: {args.output}')
    os.makedirs(args.output, exist_ok=True)
    img_list = os.listdir(args.input)
    assert len(img_list) > 0

    colorizer = ImageColorizationPipeline(model_path=args.model_path, input_size=args.input_size, model_size=args.model_size)

    for name in tqdm(img_list):
        img = cv2.imread(os.path.join(args.input, name))
        image_out = colorizer.process(img)
        image_out = np.concatenate((cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), image_out), axis=1)
        cv2.imwrite(os.path.join(args.output, name).replace('.png','.jpg'), image_out)
        # cv2.imshow(' ', image_out)
        # cv2.waitKey(0)
        break


if __name__ == '__main__':
    main()
