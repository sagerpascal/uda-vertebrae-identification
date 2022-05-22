import sys

import torch
from torch.autograd import Variable
from tqdm import tqdm

from metrics import AverageValueMeter


class Epoch:

    def __init__(self, model_g, model_head, criterion, criterion_vert, metrics, n_classes, stage_name, device='cpu',
                 verbose=True, use_domain_adaptation=True, with_detections=False):
        self.model_g = model_g
        self.model_head = model_head
        self.criterion = criterion
        self.criterion_vert = criterion_vert
        self.metrics = metrics
        self.n_classes = n_classes
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.use_domain_adaptation = use_domain_adaptation
        self.with_detections = with_detections
        self._to_device()

    def _to_device(self):
        self.model_g.to(self.device)
        self.model_head.to(self.device)
        self.criterion.to(self.device)
        if self.criterion_vert is not None:
            self.criterion_vert.to(self.device)
        for metric in self.metrics.values():
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update_da(self, src_imgs, src_lbls, tgt_imgs, tgt_detections, weak_mask):
        raise NotImplementedError

    def batch_update(self, src_imgs, src_lbls):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, epoch):

        self.on_epoch_start()

        logs = {}
        target_loss_per_epoch = 0.
        source_loss_per_epoch = 0.

        source_loss_meter, classifier_loss_meter, target_loss_meter = AverageValueMeter(), AverageValueMeter(), AverageValueMeter()
        metric_meters = {k: AverageValueMeter() for k in self.metrics.keys()}

        with tqdm(dataloader, desc=f"{self.stage_name} (epoch {epoch})", file=sys.stdout, disable=not self.verbose) as iterator:
            for ind, data in enumerate(iterator):

                if self.use_domain_adaptation:
                    (source, target) = data
                    assert len(source['labeling'].shape) > 1 and len(target['detection'].shape) > 1

                    src_imgs, src_lbls = Variable(source['sample']), Variable(source['labeling'])
                    tgt_imgs, tgt_detections = Variable(target['sample']), Variable(target['detection'])
                    weak_mask = Variable(target['weak_mask'])

                    if torch.cuda.is_available():
                        src_imgs, src_lbls = src_imgs.cuda(), src_lbls.cuda()
                        tgt_imgs, tgt_detections = tgt_imgs.cuda(), tgt_detections.cuda()
                        weak_mask = weak_mask.cuda()

                    outputs, target_loss, source_loss = self.batch_update_da(src_imgs, src_lbls, tgt_imgs,
                                                                             tgt_detections, weak_mask)

                    source_loss = source_loss.item()
                    source_loss_per_epoch += source_loss
                    source_loss_meter.add(source_loss)
                    target_loss = target_loss.item()
                    target_loss_meter.add(target_loss)
                    target_loss_per_epoch += target_loss

                    logs.update({
                        self.stage_name + "/" + 'Source Loss': source_loss_meter.mean,
                        self.stage_name + "/" + 'Target Loss': target_loss_meter.mean,
                    })

                else:
                    source = data
                    assert len(source['labeling'].shape) > 1
                    src_imgs, src_lbls = Variable(source['sample']), Variable(source['labeling'])

                    if torch.cuda.is_available():
                        src_imgs, src_lbls = src_imgs.cuda(), src_lbls.cuda()

                    loss, outputs = self.batch_update(src_imgs, src_lbls)
                    source_loss_meter.add(loss.item())

                    logs.update({
                        self.stage_name + "/" + 'Source Loss': source_loss_meter.mean,
                    })

                for metric_name, metric_fn in self.metrics.items():
                    val_c1 = metric_fn(outputs, src_lbls)
                    metric_meters[metric_name].add(val_c1.item())

                metrics_logs_c1 = {self.stage_name + "/" + k: v.mean for k, v in metric_meters.items()}
                logs.update(metrics_logs_c1)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model_g, model_head, criterion, criterion_vert, metrics, optimizer, n_classes,
                 device='cpu', verbose=True, use_domain_adaptation=True, with_detections=False):
        super().__init__(
            model_g=model_g,
            model_head=model_head,
            criterion=criterion,
            criterion_vert=criterion_vert,
            metrics=metrics,
            n_classes=n_classes,
            stage_name='train',
            device=device,
            verbose=verbose,
            use_domain_adaptation=use_domain_adaptation,
            with_detections=with_detections,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model_g.train()
        self.model_head.train()

    def batch_update(self, src_imgs, src_lbls):
        """ update generator and classifiers by source samples """
        self.optimizer.zero_grad()

        outputs = self.model_g(src_imgs)
        outputs = self.model_head(outputs)

        source_loss = self.criterion(outputs, src_lbls)
        source_loss.backward()

        self.optimizer.step()

        return source_loss, outputs

    def batch_update_da(self, src_imgs, src_lbls, tgt_imgs, tgt_detections, weak_mask):
        self.optimizer.zero_grad()

        src_outputs = self.model_g(src_imgs)
        src_outputs = self.model_head(src_outputs)

        source_loss = self.criterion(src_outputs, src_lbls)
        source_loss.backward()

        if self.criterion_vert is not None:
            tgt_outputs = self.model_g(tgt_imgs)
            tgt_outputs = self.model_head(tgt_outputs)
            target_loss = self.criterion_vert(tgt_imgs, tgt_outputs, tgt_detections, weak_mask)
            target_loss.backward()
        else:
            target_loss = 0

        self.optimizer.step()

        return src_outputs, target_loss, source_loss


class ValidEpoch(Epoch):

    def __init__(self, model_g, model_head, criterion, criterion_vert, metrics, n_classes, device='cpu', verbose=True,
                 use_domain_adaptation=True, stage_name='valid', with_detections=False):
        super().__init__(
            model_g=model_g,
            model_head=model_head,
            criterion=criterion,
            criterion_vert=criterion_vert,
            metrics=metrics,
            n_classes=n_classes,
            stage_name=stage_name,
            device=device,
            verbose=verbose,
            use_domain_adaptation=use_domain_adaptation,
            with_detections=with_detections,
        )

    def on_epoch_start(self):
        self.model_g.eval()
        self.model_head.eval()

    def batch_update(self, src_imgs, src_lbls):
        """ update generator and classifiers by source samples """
        with torch.no_grad():
            outputs = self.model_g(src_imgs)

            outputs = self.model_head(outputs)

            source_loss = self.criterion(outputs, src_lbls)

        return source_loss, outputs

    def batch_update_da(self, src_imgs, src_lbls, tgt_imgs, tgt_detections, weak_mask):
        with torch.no_grad():
            src_outputs = self.model_g(src_imgs)
            src_outputs = self.model_head(src_outputs)

            source_loss = self.criterion(src_outputs, src_lbls)

            if self.criterion_vert is not None:
                tgt_outputs = self.model_g(tgt_imgs)
                tgt_outputs = self.model_head(tgt_outputs)
                target_loss = self.criterion_vert(tgt_imgs, tgt_outputs, tgt_detections, weak_mask)

            else:
                target_loss = 0

        return src_outputs, target_loss, source_loss
