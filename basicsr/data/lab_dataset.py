import cv2
import random
import time
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import rgb2lab, augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.fmix import sample_mask


def augment_image_in_memory(img, jpeg_prob=0.25, jp_low=50, jp_high=100,
                            noise_prob=0.2, blur_prob=0.25):
    """
    Применяет случайные лёгкие аугментации к изображению OpenCV прямо в памяти.

    :param img: numpy массив изображения (BGR, uint8)
    :param jpeg_prob: вероятность применения JPEG-артефактов
    :param jp_low: минимальное качество JPEG
    :param jp_high: максимальное качество JPEG
    :param noise_prob: вероятность добавления шума
    :param blur_prob: вероятность добавления размытия
    :return: изменённый numpy массив
    """
    # Работаем с копией, чтобы не менять исходное изображение
    img_out = img.copy()
    h, w = img_out.shape[:2]
    is_color = len(img_out.shape) == 3

    # 1. Размытие (Blur)
    if random.random() < blur_prob:
        blur_type = random.choice(['gaussian', 'median', 'avg', 'resize'])

        if blur_type == 'gaussian':
            k = random.choice([3, 5])
            sigma = random.uniform(0.5, 1.5)
            img_out = cv2.GaussianBlur(img_out, (k, k), sigma)

        elif blur_type == 'median':
            k = random.choice([3, 5])
            img_out = cv2.medianBlur(img_out, k)

        elif blur_type == 'avg':
            k = random.choice([3, 5])
            img_out = cv2.blur(img_out, (k, k))

        elif blur_type == 'resize':
            # Увеличение и обратное уменьшение (имитация артефактов интерполяции/лёгкого блюра)
            scale = random.uniform(1.1, 1.3)
            new_h, new_w = int(h * scale), int(w * scale)
            img_up = cv2.resize(img_out, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_out = cv2.resize(img_up, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Лёгкий шум (Noise)
    if random.random() < noise_prob:
        noise_type = random.choice(['gaussian', 'salt_pepper', 'uniform'])

        if noise_type == 'gaussian':
            sigma = random.uniform(5.0, 12.0)
            noise = np.random.normal(0, sigma, img_out.shape)
            img_out = np.clip(img_out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        elif noise_type == 'salt_pepper':
            amount = random.uniform(0.005, 0.015)  # 0.5% - 1.5% пикселей
            row, col = img_out.shape[:2]
            s_vs_p = 0.5

            # Salt (белые точки)
            num_salt = int(np.ceil(amount * row * col * s_vs_p))
            coords = [np.random.randint(0, i - 1, num_salt) for i in img_out.shape[:2]]
            if is_color:
                img_out[coords[0], coords[1], :] = 255
            else:
                img_out[coords[0], coords[1]] = 255

            # Pepper (чёрные точки)
            num_pepper = int(np.ceil(amount * row * col * (1.0 - s_vs_p)))
            coords = [np.random.randint(0, i - 1, num_pepper) for i in img_out.shape[:2]]
            if is_color:
                img_out[coords[0], coords[1], :] = 0
            else:
                img_out[coords[0], coords[1]] = 0

        elif noise_type == 'uniform':
            noise = np.random.uniform(-10, 10, img_out.shape)
            img_out = np.clip(img_out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 3. JPEG артефакты (сжатие/распаковка в памяти)
    if random.random() < jpeg_prob:
        quality = random.randint(jp_low, jp_high)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc_img = cv2.imencode('.jpg', img_out, encode_param)
        img_out = cv2.imdecode(enc_img, cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE)

    return img_out


@DATASET_REGISTRY.register()
class LabDataset(data.Dataset):
    """
    Dataset used for Lab colorizaion
    """

    def __init__(self, opt):
        super(LabDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        meta_info_file = self.opt['meta_info_file']
        assert meta_info_file is not None
        if not isinstance(meta_info_file, list):
            meta_info_file = [meta_info_file]
        self.paths = []
        for meta_info in meta_info_file:
            with open(meta_info, 'r') as fin:
                self.paths.extend([line.strip() for line in fin])

        self.min_ab, self.max_ab = -128, 128
        self.interval_ab = 4
        self.ab_palette = [i for i in range(self.min_ab, self.max_ab + self.interval_ab, self.interval_ab)]
        # print(self.ab_palette)

        self.do_fmix = opt['do_fmix']
        self.fmix_p = opt['fmix_p']
        self.do_cutmix = opt['do_cutmix']
        self.cutmix_params = {'alpha':1.}
        self.cutmix_p = opt['cutmix_p']

        self.jpeg_prob = self.opt['jpeg_prob']
        self.jp_low = self.opt['jp_low']
        self.jp_high = self.opt['jp_high']
        self.noise_prob = self.opt['noise_prob']
        self.blur_prob = self.opt['blur_prob']


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        gt_size = self.opt['gt_size']
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = cv2.resize(img_gt, (gt_size, gt_size))  # TODO: 直接resize是否是最佳方案？

        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])
        
        # -------------------------------- (Optional) CutMix & FMix -------------------------------- #
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > self.fmix_p:
            with torch.no_grad():
                fmix_shape = (img_gt.shape[0], img_gt.shape[1])  # Use actual image size (H, W)
                lam, mask = sample_mask(alpha=1., decay_power=3., shape=fmix_shape, max_soft=0.0, reformulate=False)
                
                fmix_index = random.randint(0, self.__len__() - 1)
                fmix_img_path = self.paths[fmix_index]
                fmix_img_bytes = self.file_client.get(fmix_img_path, 'gt')
                fmix_img = imfrombytes(fmix_img_bytes, float32=True)
                fmix_img = cv2.resize(fmix_img, (gt_size, gt_size))

                mask = mask.transpose(1, 2, 0)  # (1, H, W) -> (H, W, 1)
                img_gt = mask * img_gt + (1. - mask) * fmix_img
                img_gt = img_gt.astype(np.float32)

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > self.cutmix_p:
            with torch.no_grad():
                cmix_index = random.randint(0, self.__len__() - 1)
                cmix_img_path = self.paths[cmix_index]
                cmix_img_bytes = self.file_client.get(cmix_img_path, 'gt')
                cmix_img = imfrombytes(cmix_img_bytes, float32=True)
                cmix_img = cv2.resize(cmix_img, (gt_size, gt_size))

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(cmix_img.shape[:2], lam)

                img_gt[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]


        # ----------------------------- Get gray lq, to tentor ----------------------------- #
        # convert to gray
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        augmented_img = augment_image_in_memory(
            img_gt,
            jpeg_prob = self.jpeg_prob,
            jp_low = self.jp_low,
            jp_high = self.jp_high,
            noise_prob = self.noise_prob,
            blur_prob = self.blur_prob
        )

        img_l, _ = rgb2lab(augmented_img)
        _, img_ab = rgb2lab(img_gt)

        target_a, target_b = self.ab2int(img_ab)

        # numpy to tensor
        img_l, img_ab = img2tensor([img_l, img_ab], bgr2rgb=False, float32=True)
        target_a, target_b = torch.LongTensor(target_a), torch.LongTensor(target_b)
        return_d = {
            'lq': img_l,
            'gt': img_ab,
            'target_a': target_a,
            'target_b': target_b,
            'lq_path': gt_path,
            'gt_path': gt_path
        }
        return return_d

    def ab2int(self, img_ab):
        img_a, img_b = img_ab[:, :, 0], img_ab[:, :, 1]
        int_a = (img_a - self.min_ab) / self.interval_ab
        int_b = (img_b - self.min_ab) / self.interval_ab

        return np.round(int_a), np.round(int_b)

    def __len__(self):
        return len(self.paths)


def rand_bbox(size, lam):
    '''cutmix 的 bbox 截取函数
    Args:
        size : tuple 图片尺寸 e.g (256,256)
        lam  : float 截取比例
    Returns:
        bbox 的左上角和右下角坐标
        int,int,int,int
    '''
    W = size[0]  # 截取图片的宽度
    H = size[1]  # 截取图片的高度
    cut_rat = np.sqrt(1. - lam)  # 需要截取的 bbox 比例
    cut_w = int(W * cut_rat)  # 需要截取的 bbox 宽度
    cut_h = int(H * cut_rat)  # 需要截取的 bbox 高度

    cx = np.random.randint(W)  # 均匀分布采样，随机选择截取的 bbox 的中心点 x 坐标
    cy = np.random.randint(H)  # 均匀分布采样，随机选择截取的 bbox 的中心点 y 坐标

    bbx1 = np.clip(cx - cut_w // 2, 0, W)  # 左上角 x 坐标
    bby1 = np.clip(cy - cut_h // 2, 0, H)  # 左上角 y 坐标
    bbx2 = np.clip(cx + cut_w // 2, 0, W)  # 右下角 x 坐标
    bby2 = np.clip(cy + cut_h // 2, 0, H)  # 右下角 y 坐标
    return bbx1, bby1, bbx2, bby2