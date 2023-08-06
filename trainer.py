import os
import time
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from utils.utils import *
from utils.visualizer import Visualizer
from datasets.mvtec import MVTecDataset, MVTEC_CLASS_NAMES
from datasets.btad import BTADDataset, BTAD_CLASS_NAMES
from datasets.mvtec_3d import MVTec3DDataset, MVTEC3D_CLASS_NAMES
from models.model import FOD
from losses import kl_loss, entropy_loss


class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        if args.class_name in MVTEC_CLASS_NAMES:
            train_dataset = MVTecDataset(args, is_train=True)
            test_dataset  = MVTecDataset(args, is_train=False)
        elif args.class_name in BTAD_CLASS_NAMES:
            train_dataset = BTADDataset(args.data_path, classname=args.class_name, resize=self.args.inp_size, cropsize=self.args.inp_size, is_train=True)
            test_dataset  = BTADDataset(args.data_path, classname=args.class_name, resize=self.args.inp_size, cropsize=self.args.inp_size, is_train=False)
        elif args.class_name in MVTEC3D_CLASS_NAMES:
            train_dataset = MVTec3DDataset(args.data_path, classname=args.class_name, resize=self.args.inp_size, cropsize=self.args.inp_size, is_train=True)
            test_dataset  = MVTec3DDataset(args.data_path, classname=args.class_name, resize=self.args.inp_size, cropsize=self.args.inp_size, is_train=False)
        else:
            raise ValueError('Invalid Class Name: {}'.format(args.class_name))
        
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)

        self.build_model()
        self.l2_criterion = nn.MSELoss()
        self.cos_criterion = nn.CosineSimilarity(dim=-1)

    def build_model(self):
        encoder = timm.create_model(self.args.backbone_arch, features_only=True, 
                out_indices=[2, 3], pretrained=True)
        self.encoder = encoder.to(self.args.device).eval()
        
        feat_dims = encoder.feature_info.channels()
        print("Feature Dimensions:", feat_dims)
        
        models = []
        self.seq_lens = [1024, 256]
        self.ws = [32, 16]  # feature map height/width
        for seq_len, in_channels, d_model in zip(self.seq_lens, feat_dims, [256, 512]):
            model = FOD(seq_len=seq_len,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        d_model=d_model,
                        n_heads=8,
                        n_layers=3,
                        args=self.args)
            print('One Model...Done')
            models.append(model.to(self.args.device))
        self.models = models
        print('Creating Models...Done')
        params = list(models[0].parameters())
        for l in range(1, self.args.feature_levels):
            params += list(models[l].parameters())  
        self.optimizer = torch.optim.Adam(params, lr=self.args.lr)
        self.avg_pool = torch.nn.AvgPool2d(3, 1, 1)

    def train(self):
        path = os.path.join(self.args.save_path, self.args.save_prefix)
        if not os.path.exists(path):
            os.makedirs(path)

        start_time = time.time()
        train_steps = len(self.train_loader)
        best_img_auc, best_pix_auc = 0.0, 0.0
        for epoch in range(self.args.num_epochs):
            print("======================TRAIN MODE======================")
            iter_count = 0
            loss_rec_list, loss_intra_entropy_list, loss_inter_entropy_list = [], [], []
            loss_corr_list, loss_target_list = [], []

            epoch_time = time.time()
            for model in self.models:
                model.train()
            for i, (images, _, _, _, _) in enumerate(self.train_loader):
                iter_count += 1
                images = images.float().to(self.args.device)  # (N, 3, H, W)
                
                with torch.no_grad():
                    features = self.encoder(images)
                
                for fl in range(self.args.feature_levels):
                    m = torch.nn.AvgPool2d(3, 1, 1)
                    input = m(features[fl])
                    N, D, _, _ = input.shape
                    input = input.permute(0, 2, 3, 1).reshape(N, -1, D)
                
                    # output: reconstructed features, (N, L, dim)
                    # intra_corrs: intra correlations, list[(N, num_heads, L, L)]
                    # intra_targets: intra target correlations, list[(N, num_heads, L, L)]
                    # inter_corrs: inter correlations, list[(N, num_heads, L, L)]
                    # inter_targets: inter target correlations, list[(N, num_heads, L, L)]
                    # len of list is attention layers of transformer
                    model = self.models[fl]
                    output, intra_corrs, intra_targets, inter_corrs, inter_targets = model(input)
               
                    if self.args.with_intra:
                        loss_intra1, loss_intra2, loss_intra_entropy = 0.0, 0.0, 0.0
                        for l in range(len(intra_targets)):
                            L = intra_targets[l].shape[-1]
                            norm_targets = (intra_targets[l] / torch.unsqueeze(torch.sum(intra_targets[l], dim=-1), dim=-1).repeat(1, 1, 1, L)).detach()
                            # optimizing intra correlations
                            loss_intra1 += torch.mean(kl_loss(norm_targets, intra_corrs[l])) + torch.mean(kl_loss(intra_corrs[l], norm_targets))
                            
                            norm_targets = intra_targets[l] / torch.unsqueeze(torch.sum(intra_targets[l], dim=-1), dim=-1).repeat(1, 1, 1, L)
                            loss_intra2 += torch.mean(kl_loss(norm_targets, intra_corrs[l].detach())) + torch.mean(kl_loss(intra_corrs[l].detach(), norm_targets))
                            
                            loss_intra_entropy += torch.mean(entropy_loss(intra_corrs[l]))
                        
                        loss_intra1 = loss_intra1 / len(intra_targets)
                        loss_intra2 = loss_intra2 / len(intra_targets)
                        loss_intra_entropy = loss_intra_entropy / len(intra_targets)

                    if self.args.with_inter:
                        loss_inter1, loss_inter2, loss_inter_entropy = 0.0, 0.0, 0.0
                        for l in range(len(inter_targets)):
                            L = inter_targets[l].shape[-1]
                            norm_targets = (inter_targets[l] / torch.unsqueeze(torch.sum(inter_targets[l], dim=-1), dim=-1).repeat(1, 1, 1, L)).detach()
                            # optimizing inter correlations
                            loss_inter1 += torch.mean(kl_loss(norm_targets, inter_corrs[l])) + torch.mean(kl_loss(inter_corrs[l], norm_targets))
                            
                            norm_targets = inter_targets[l] / torch.unsqueeze(torch.sum(inter_targets[l], dim=-1), dim=-1).repeat(1, 1, 1, L)
                            loss_inter2 += torch.mean(kl_loss(norm_targets, inter_corrs[l].detach())) + torch.mean(kl_loss(inter_corrs[l].detach(), norm_targets))

                            loss_inter_entropy += torch.mean(entropy_loss(inter_corrs[l]))
                        
                        loss_inter1 = loss_inter1 / len(inter_targets)
                        loss_inter2 = loss_inter2 / len(inter_targets)
                        loss_inter_entropy = loss_inter_entropy / len(inter_targets)
                    
                    loss_rec = self.l2_criterion(output, input) + torch.mean(1 - self.cos_criterion(output, input)) # mse + cosine
                    
                    if self.args.with_intra and self.args.with_inter:  # patch-wise reconstruction + intra correlation + inter correlation
                        loss1 = loss_rec + self.args.lambda1 * loss_intra2 - self.args.lambda1 * loss_inter2 
                        loss2 = loss_rec - self.args.lambda1 * loss_intra1 - self.args.lambda2 * loss_intra_entropy + self.args.lambda1 * loss_inter1 + self.args.lambda2 * loss_inter_entropy 
                    elif self.args.with_intra:  # patch-wise reconstruction + intra correlation
                        loss1 = loss_rec + self.args.lambda1 * loss_intra2  
                        loss2 = loss_rec - self.args.lambda1 * loss_intra1 - self.args.lambda2 * loss_intra_entropy 
                    elif self.args.with_inter:  # patch-wise reconstruction + inter correlation
                        loss1 = loss_rec - self.args.lambda1 * loss_inter2  
                        loss2 = loss_rec + self.args.lambda1 * loss_inter1 + self.args.lambda2 * loss_inter_entropy 
                    else:  # only patch-wise reconstruction
                        loss = loss_rec
                        
                    loss_rec_list.append(loss_rec.item())
                    if self.args.with_intra and self.args.with_inter:
                        loss_target_list.append((loss_intra2 - loss_inter2).item())
                        loss_corr_list.append((-loss_intra1 + loss_inter1).item())
                        loss_intra_entropy_list.append(loss_intra_entropy.item())
                        loss_inter_entropy_list.append(loss_inter_entropy.item())
                    elif self.args.with_intra:
                        loss_target_list.append((loss_intra2).item())
                        loss_corr_list.append((-loss_intra1).item())
                        loss_intra_entropy_list.append(loss_intra_entropy.item())
                    elif self.args.with_inter:
                        loss_target_list.append((-loss_inter2).item())
                        loss_corr_list.append((loss_inter1).item())
                        loss_inter_entropy_list.append(loss_inter_entropy.item())

                    self.optimizer.zero_grad()
                    if not self.args.with_intra and not self.args.with_inter:  # only patch-wise reconstruction
                        loss.backward()
                    else:
                        # Two-stage optimization
                        loss1.backward(retain_graph=True)
                        loss2.backward()
                    self.optimizer.step()

            speed = (time.time() - start_time) / iter_count
            left_time = speed * ((self.args.num_epochs - epoch) * train_steps - i)
            print("Epoch: {} cost time: {}s | speed: {:.4f}s/iter | left time: {:.4f}s".format(epoch + 1, time.time() - epoch_time, speed, left_time))
            iter_count = 0
            start_time = time.time()

            if self.args.with_intra and self.args.with_inter:
                print(
                    "Epoch: {0}, Steps: {1} | Rec Loss: {2:.7f} | Target Loss: {3:.7f} | Corr Loss: {4:.7f} | Intra Entropy: {5:.7f} | Inter Entropy: {6:.7f}".format(
                        epoch + 1, train_steps, np.average(loss_rec_list), np.average(loss_target_list), np.average(loss_corr_list), np.average(loss_intra_entropy_list), np.average(loss_inter_entropy_list)))
            elif self.args.with_intra:
                print(
                    "Epoch: {0}, Steps: {1} | Rec Loss: {2:.7f} | Target Loss: {3:.7f} | Corr Loss: {4:.7f} | Intra Entropy: {5:.7f}".format(
                        epoch + 1, train_steps, np.average(loss_rec_list), np.average(loss_target_list), np.average(loss_corr_list), np.average(loss_intra_entropy_list)))
            elif self.args.with_inter:
                print(
                    "Epoch: {0}, Steps: {1} | Rec Loss: {2:.7f} | Target Loss: {3:.7f} | Corr Loss: {4:.7f} | Inter Entropy: {5:.7f}".format(
                        epoch + 1, train_steps, np.average(loss_rec_list), np.average(loss_target_list), np.average(loss_corr_list), np.average(loss_inter_entropy_list)))
            else:
                print(
                    "Epoch: {0}, Steps: {1} | Rec Loss: {2:.7f}".format(epoch + 1, train_steps, np.average(loss_rec_list)))

            img_auc, pix_auc = self.test(vis=False)

            print("Epoch: {0}, Class Name: {1}, Image AUC: {2:.7f} | Pixel AUC: {3:.7f}".format(epoch + 1, self.args.class_name, img_auc, pix_auc))
            
            if img_auc > best_img_auc:
                best_img_auc = img_auc
                state = {'state_dict': [model.state_dict() for model in self.models]}
                torch.save(state, os.path.join(path, self.args.class_name + '-img.pth'))
            if pix_auc > best_pix_auc:
                best_pix_auc = pix_auc 
                state = {'state_dict': [model.state_dict() for model in self.models]}
                torch.save(state, os.path.join(path, self.args.class_name + '-pix.pth'))
        
        return best_img_auc, best_pix_auc
    
    def test(self, vis=False, checkpoint_path=None):
        if checkpoint_path is not None:
            checkpoint = torch.load(os.path.join(checkpoint_path, self.args.class_name + '-pix.pth'))
            state_dict = checkpoint['state_dict']
            for i, model in enumerate(self.models):
                model.load_state_dict(state_dict[i])
        for model in self.models:
            model.eval()
        temperature = 1

        print("======================TEST MODE======================")

        l2_criterion = nn.MSELoss(reduction='none')
        cos_criterion = nn.CosineSimilarity(dim=-1)

        scores_list = [list() for _ in range(self.args.feature_levels)]
        test_imgs, gt_label_list, gt_mask_list, file_names, img_types = [], [], [], [], []
        for i, (image, label, mask, file_name, img_type) in enumerate(self.test_loader):
            test_imgs.append(image.cpu().numpy())
            gt_label_list.extend(label)
            gt_mask_list.extend(mask.numpy())
            file_names.extend(file_name)
            img_types.extend(img_type)
            
            image = image.float().to(self.args.device)

            with torch.no_grad():
                features = self.encoder(image)
            
                for fl in range(self.args.feature_levels):
                    m = torch.nn.AvgPool2d(3, 1, 1)
                    input = m(features[fl])
                    N, D, _, _ = input.shape
                    input = input.permute(0, 2, 3, 1).reshape(N, -1, D)
                    
                    model = self.models[fl]
                    output, intra_corrs, intra_targets, inter_corrs, inter_targets = model(input, train=False)

                    rec_score = torch.mean(l2_criterion(input, output), dim=-1) + 1 - cos_criterion(input, output)

                    if self.args.with_intra:
                        correlations1, correlations2, entropys = 0.0, 0.0, 0.0
                        for l in range(len(intra_targets)):
                            L = intra_targets[l].shape[-1]
                            norm_targets = intra_targets[l] / torch.unsqueeze(torch.sum(intra_targets[l], dim=-1), dim=-1).repeat(1, 1, 1, L)
                            correlations1 += kl_loss(intra_corrs[l], norm_targets) * temperature
                            correlations2 += kl_loss(norm_targets, intra_corrs[l]) * temperature
                            
                            entropys += entropy_loss(intra_corrs[l])
                        
                        corrs = (correlations1 + correlations2) / len(intra_targets)
                        intra_score = torch.softmax((-corrs), dim=-1)
                        # entropys = entropys / len(intra_targets)
                        # ent_score = torch.softmax(-entropys, dim=-1)
                    if self.args.with_inter:
                        correlations1, correlations2, entropys = 0.0, 0.0, 0.0
                        for l in range(len(inter_targets)):
                            L = inter_targets[l].shape[-1]
                            norm_targets = inter_targets[l] / torch.unsqueeze(torch.sum(inter_targets[l], dim=-1), dim=-1).repeat(1, 1, 1, L)
                            correlations1 += kl_loss(inter_corrs[l], norm_targets) * temperature
                            correlations2 += kl_loss(norm_targets, inter_corrs[l]) * temperature
                            
                            entropys += entropy_loss(inter_corrs[l])
                
                        corrs = (correlations1 + correlations2) / len(inter_targets)
                        inter_score = torch.softmax((-corrs), dim=-1)
                        inter_score = torch.max(inter_score) - inter_score
                        # entropys = entropys / len(inter_targets)
                        # ent_score = torch.softmax(-entropys, dim=-1)
                        # ent_score = torch.max(ent_score) - ent_score

                    if self.args.with_intra and self.args.with_inter:
                        # we find that only use inter_score can get slightly better results, 
                        # but in training the intra-correlations learning is still necessary for achieving the best results
                        score = rec_score * inter_score
                    elif self.args.with_intra:
                        score = rec_score * intra_score
                    elif self.args.with_inter:
                        score = rec_score * inter_score
                    else:
                        score = rec_score
                    score = score.detach()  # (N, L)
                    score = score.reshape(score.shape[0], self.ws[fl], self.ws[fl])
                    score = F.interpolate(score.unsqueeze(1),
                        size=self.args.inp_size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
                    scores_list[fl].append(score)
        
        lvl_scores = []
        for l in range(self.args.feature_levels):
            lvl_score = np.stack(scores_list[l], axis=0)  # (N, 256, 256)
            lvl_scores.append(lvl_score)
            
        scores = np.zeros_like(lvl_scores[0])
        for l in range(self.args.feature_levels):
            scores += lvl_scores[l]
        scores = scores / self.args.feature_levels
        
        # scores = np.ones_like(lvl_scores[0])
        # for l in range(self.args.feature_levels):
        #     scores *= lvl_scores[l]
        
        gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
        pix_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        
        for i in range(scores.shape[0]):
            scores[i] = gaussian_filter(scores[i], sigma=4)

        # image and pixel level auroc
        img_scores = np.max(scores, axis=(1, 2))
        gt_label = np.asarray(gt_label_list, dtype=np.bool)
        img_auc = roc_auc_score(gt_label, img_scores)
        
        if vis:
            precision, recall, thresholds = precision_recall_curve(gt_label, img_scores)
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            img_threshold = thresholds[np.argmax(f1)]
            
            visulizer = Visualizer(f'vis_results/{self.args.save_prefix}/{self.args.class_name}')
            max_score = np.max(scores)
            min_score = np.min(scores)
            scores = (scores - min_score) / (max_score - min_score)
            test_imgs = np.concatenate(test_imgs, axis=0)
            visulizer.plot(test_imgs, scores, img_scores, gt_mask, file_names, img_types, img_threshold)

        return img_auc, pix_auc
