import math
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt


def denormalization(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = np.array(mean)
    std = np.array(std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


class Visualizer(object):
    def __init__(self, root, prefix=''):
        self.root = root
        self.prefix = prefix
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, 'normal_ok'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'normal_nok'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'anomaly_ok'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'anomaly_nok'), exist_ok=True)
    
    def set_prefix(self, prefix):
        self.prefix = prefix

    def plot(self, test_imgs, scores, img_scores, gt_masks, file_names, img_types, img_threshold):
        """
        Args:
            test_imgs (ndarray): shape (N, 3, h, w)
            scores (ndarray): shape (N, h, w)
            img_scores (ndarray): shape (N, )
            gt_masks (ndarray): shape (N, 1, h, w)
        """
        vmax = scores.max() * 255.
        vmin = scores.min() * 255. + 10
        vmax = vmax - 220
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(len(scores)):
            img = test_imgs[i]
            img = denormalization(img)
            gt_mask = gt_masks[i].squeeze()
            score = scores[i]
            #score = gaussian_filter(score, sigma=4)
            
            heat_map = score * 255
            fig_img, ax_img = plt.subplots(1, 3, figsize=(9, 3))

            fig_img.subplots_adjust(wspace=0.05, hspace=0)
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)

            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Input image')
            ax_img[1].imshow(gt_mask, cmap='gray')
            ax_img[1].title.set_text('GroundTruth')
            ax_img[2].imshow(heat_map, cmap='jet', norm=norm, interpolation='none')
            ax_img[2].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
            ax_img[2].title.set_text('Segmentation')
            
            if img_types[i] == 'good':
                if img_scores[i] <= img_threshold:
                    fig_img.savefig(os.path.join(self.root, 'normal_ok', img_types[i] + '_' + file_names[i]), dpi=300)
                else:
                    fig_img.savefig(os.path.join(self.root, 'normal_nok', img_types[i] + '_' + file_names[i]), dpi=300)
            else:
                if img_scores[i] > img_threshold:
                    fig_img.savefig(os.path.join(self.root, 'anomaly_ok', img_types[i] + '_' + file_names[i]), dpi=300)
                else:
                    fig_img.savefig(os.path.join(self.root, 'anomaly_nok', img_types[i] + '_' + file_names[i]), dpi=300)
            
            #fig_img.savefig(os.path.join(self.root, str(i) + '.png'), dpi=1000)
              
            plt.close()


def visualize_correlations(attn, img, file_name, img_type, args):
    L = attn.shape[0]
    H = W = int(math.sqrt(L))
    attn = attn.reshape(H, W, H, W).cpu().numpy()
    # downsampling factor for the feature level
    factor = 16
    
    # let's select 4 reference points for visualization
    idxs = [(60, 60), (110, 110), (130, 130), (140, 160)]
    
    # here we create the canvas
    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # and we add one plot per reference point
    gs = fig.add_gridspec(2, 4)
    axs = [fig.add_subplot(gs[0, 0]),
           fig.add_subplot(gs[1, 0]),
           fig.add_subplot(gs[0, -1]),
           fig.add_subplot(gs[1, -1])]
    
    # for each one of the reference points, let's plot the self-attention for that point
    for idx_o, ax in zip(idxs, axs):
        idx = (idx_o[0] // factor, idx_o[1] // factor)
        ax.imshow(attn[idx[0], idx[1], ...], cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'global-correlation{idx_o}')
    
    # and now let's add the central image, with the reference points as red circles
    img = denormalization(img.squeeze(0))
    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(img)
    for (y, x) in idxs:
        scale = img.shape[0] / img.shape[0]
        x = ((x // factor) + 0.5) * factor
        y = ((y // factor) + 0.5) * factor
        fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), factor // 4 , color='r'))
        fcenter_ax.axis('off')  
    
    os.makedirs(os.path.join('vis_results', 'attn', args.class_name), exist_ok=True)
    fig.savefig(os.path.join('vis_results', 'attn', args.class_name, img_type + '_' + file_name + '.png'), dpi=1000)  