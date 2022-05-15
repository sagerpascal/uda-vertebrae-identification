import numpy as np
import torch
import torch.nn as nn
from skimage.segmentation import felzenszwalb


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)


class IdentificationLoss(nn.Module):
    def __init__(self):
        super(IdentificationLoss, self).__init__()

    def forward(self, inputs, targets):
        ignored = targets > 0
        return torch.sum(torch.abs(inputs - targets.unsqueeze(1)) * ignored.unsqueeze(1)) / torch.sum(ignored)


class RegionProposalLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.id_loss = IdentificationLoss()

    def get_segmentation_mask(self, tgt_img, detection):

        img = tgt_img[4].cpu().detach().numpy()
        detection = detection.cpu().detach().numpy()

        if np.any(detection != 0):
            spine_area = np.where(detection != 0)
            dim0_min, dim0_max = np.min(spine_area[0]), np.max(spine_area[0])
            dim1_min, dim1_max = np.min(spine_area[1]), np.max(spine_area[1])
            tgt_img_cropped = img[dim0_min:dim0_max + 1, dim1_min:dim1_max + 1]
            detection_cropped = detection[dim0_min:dim0_max + 1, dim1_min:dim1_max + 1]
            spine_cropped = tgt_img_cropped * detection_cropped

            spine_cropped_enhanced = (spine_cropped > 1.42) * spine_cropped
            segments_felzenszwalb = felzenszwalb(spine_cropped_enhanced, scale=60, sigma=0.8, min_size=150)

            for i in range(np.max(segments_felzenszwalb) + 1):
                positions = np.where(segments_felzenszwalb == i)
                length = np.max(positions[1]) - np.min(positions[1])
                if length > 50:
                    segments_felzenszwalb[segments_felzenszwalb == i] = 0

            unique = np.unique(segments_felzenszwalb)
            positions = {}
            for u in unique:
                if u == 0:
                    continue
                p = np.where(segments_felzenszwalb == u)
                positions[u] = (np.min(p[1]), np.max(p[1]))

            for u1, p1 in positions.items():
                for u2, p2 in positions.items():
                    if u1 == u2:
                        continue
                    if ((p1[0] - 4) <= p2[0] and (p1[1] + 4) >= p2[1]):
                        segments_felzenszwalb[segments_felzenszwalb == u2] = u1

            segments_felzenszwalb = np.pad(segments_felzenszwalb, (
                (dim0_min, img.shape[0] - dim0_max - 1), (dim1_min, img.shape[1] - dim1_max - 1)), mode="constant",
                                           constant_values=0)

            return torch.as_tensor(segments_felzenszwalb).cuda()

        else:
            return torch.zeros_like(tgt_img[4])

    def forward(self, tgt_img, y_tgt_pr, mask):
        loss = torch.tensor(0., device='cuda')
        weak_labels = torch.zeros_like(y_tgt_pr)
        for b in range(tgt_img.shape[0]):
            proposed_mask_b = self.get_segmentation_mask(tgt_img[b], mask[b])
            y_tgt_pr_b = y_tgt_pr[b].squeeze(0)

            # # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
            # # ax[0, 0].imshow(tgt_img[b, 4].cpu().numpy())
            # # ax[0, 1].imshow(tgt_img[b, 4].cpu().numpy())
            # # ax[0, 1].imshow(proposed_mask_b.cpu(), cmap='jet', alpha=0.5)
            # # ax[1, 0].imshow(proposed_mask_b.cpu().numpy())
            # # ax[1, 1].imshow(proposed_mask_b.cpu().numpy() == 2)
            # # plt.tight_layout()
            # # plt.show()

            proposed_masks = list(torch.unique(proposed_mask_b))[1:]
            for i in proposed_masks:
                loss += (len(torch.unique(y_tgt_pr_b[proposed_mask_b == i])) - 1)

            if len(proposed_masks) > 0:
                loss /= len(proposed_masks)

            y_tgt_pr_b = torch.round(y_tgt_pr_b)
            for value in torch.unique(proposed_mask_b):
                if value == 0:
                    continue
                med = torch.median(y_tgt_pr_b[proposed_mask_b == value])
                proposed_mask_b[proposed_mask_b == value] = float(med)
                weak_labels[b] = proposed_mask_b

            # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
            # ax[0, 0].imshow(tgt_img[b, 4].cpu().numpy())
            # ax[0, 1].imshow(tgt_img[b, 4].cpu().numpy())
            # mask_01 = ax[0, 1].imshow(weak_labels[b].squeeze().cpu(), cmap='jet', alpha=0.5)
            # ax[1, 0].imshow(proposed_mask_b.cpu().numpy())
            # ax[1, 1].imshow(weak_labels[b].squeeze().cpu().numpy())
            # fig.colorbar(mask_01, ax=ax[0, 1])
            # plt.tight_layout()
            # plt.show()

        # return self.id_loss(y_tgt_pr, weak_labels)
        return loss


class DescendingLoss(nn.Module):

    def forward(self, y_pr, mask):
        """ Vertebrae should be descending (e.g. T12 -> T11 -> ... and not T12 -> T13 -> ...)"""
        pred = torch.round(y_pr).squeeze(1)
        invalid = 0

        for shift in range(1, 30):
            pred_shifted = torch.zeros_like(pred)
            mask_shifted = torch.zeros_like(mask)
            pred_shifted[:, :, :-shift] = pred[:, :, shift:]
            mask_shifted[:, :, :-shift] = mask[:, :, shift:] * mask[:, :, :-shift]
            invalid += torch.sum((pred * mask - pred_shifted * mask_shifted) < 0)  # TODO: Min x,0 instead of <0
        return invalid / pred.numel()


class VerticalEqualLoss(nn.Module):

    def forward(self, y_pr, mask):
        pred = torch.round(y_pr)
        pred_masked = pred * mask.unsqueeze(1)
        pred_masked[pred_masked == 0] = float('nan')
        col_median = torch.nanmedian(pred_masked, axis=2)
        col_median = torch.nan_to_num(col_median[0]).unsqueeze(2).repeat(1, 1, y_pr.shape[2], 1)
        invalid = torch.sum(((pred - col_median) * mask) != 0)  # TODO: Torch.abs instead of != 0
        return invalid / pred.numel()


class CenterDistLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.center_dists = {
            2: 18,
            3: 18,
            4: 18.5,
            5: 19,
            6: 19.5,
            7: 20,
            8: 20,
            9: 20,
            10: 20.5,
            11: 21,
            12: 21.5,
            13: 22,
            14: 22.5,
            15: 23,
            16: 24.5,
            17: 24.5,
            18: 26.5,
            19: 28.5,
            20: 29.5,
            21: 33,
            22: 33,
            23: 33,
            24: 33,
            25: 33,
            26: 33,
        }

    def forward(self, y_pr, mask):
        pred = torch.round(y_pr)
        pred_masked = pred * mask.unsqueeze(1)

        loss = 0

        for b in range(pred.shape[0]):
            pred_masked_slice = pred_masked[b]
            y_center_prev, x_center_prev = None, None
            # for i in range(1, int(torch.max(pred_masked_slice))+1):
            for i in range(int(torch.min(pred_masked_slice)) + 1, int(torch.max(pred_masked_slice)) + 1):  # -> 36
                if torch.any(pred_masked_slice == i):
                    positions = torch.where(pred_masked_slice == i)
                    y_center = torch.mean(positions[1].double())
                    x_center = torch.mean(positions[2].double())

                    if x_center_prev is not None:
                        dist = torch.sqrt((x_center - x_center_prev) ** 2 + (y_center - y_center_prev) ** 2)
                        if i in self.center_dists:
                            mean_dist = self.center_dists[i]
                        elif i > 26:
                            mean_dist = 30
                        else:
                            mean_dist = 14
                        loss += torch.abs(dist - mean_dist)

                    x_center_prev = x_center
                    y_center_prev = y_center

                else:
                    x_center_prev = None
                    y_center_prev = None

        return loss


class NoneLoss(nn.Module):

    def forward(self, *args, **kwargs):
        return 0


class VertebraeCharacteristicsLoss(nn.Module):

    def __init__(self, use_descending_loss, use_vertical_equal_loss, use_center_dist_loss, use_region_proposal_loss):
        super().__init__()
        self.descending_loss = DescendingLoss() if use_descending_loss else NoneLoss()
        self.vertical_equal_loss = VerticalEqualLoss() if use_vertical_equal_loss else NoneLoss()
        self.center_dist_loss = CenterDistLoss() if use_center_dist_loss else NoneLoss()
        self.region_proposal_loss = RegionProposalLoss() if use_region_proposal_loss else NoneLoss()

    def forward(self, targets, predictions, mask=None):
        loss = self.descending_loss(predictions, mask) * 20
        loss += self.vertical_equal_loss(predictions, mask)
        loss += self.center_dist_loss(predictions, mask) / 1000
        loss += self.region_proposal_loss(targets, predictions, mask) / 100

        loss.requires_grad = True
        return loss
