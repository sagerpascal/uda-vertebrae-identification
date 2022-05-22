import torch
import torch.nn as nn


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

    def forward(self, tgt_img, y_tgt_pr, mask, weak_mask):
        loss = torch.tensor(0., device='cuda')
        weak_labels = torch.zeros_like(y_tgt_pr)
        for b in range(tgt_img.shape[0]):
            proposed_mask_b = weak_mask[b]
            y_tgt_pr_b = y_tgt_pr[b].squeeze(0)

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
            invalid += torch.sum((pred * mask - pred_shifted * mask_shifted) < 0)
        return invalid / pred.numel()


class VerticalEqualLoss(nn.Module):

    def forward(self, y_pr, mask):
        pred = torch.round(y_pr)
        pred_masked = pred * mask.unsqueeze(1)
        pred_masked[pred_masked == 0] = float('nan')
        col_median = torch.nanmedian(pred_masked, axis=2)
        col_median = torch.nan_to_num(col_median[0]).unsqueeze(2).repeat(1, 1, y_pr.shape[2], 1)
        invalid = torch.sum(((pred - col_median) * mask) != 0)
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

    def forward(self, targets, predictions, detection_mask=None, weak_mask=None):
        loss = self.descending_loss(predictions, detection_mask) * 20
        loss += self.vertical_equal_loss(predictions, detection_mask)
        loss += self.center_dist_loss(predictions, detection_mask) / 1000
        loss += self.region_proposal_loss(targets, predictions, detection_mask, weak_mask) / 100

        loss.requires_grad = True
        return loss
