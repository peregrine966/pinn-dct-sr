import os
import math
from decimal import Decimal
import utility
import torch_dct as dct
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import torch.nn as nn
#import pytorch_ssim
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        #self.physic_loss= dct_loss()
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def dct_loss(self, res, true):
        """
        计算基于三维离散余弦变换（DCT）后的损失函数
        参数:
        res (torch.Tensor): 预测结果张量，形状为 [batch_size, 3, size, size]
        true (torch.Tensor): 真实标签张量，形状为 [batch_size, 3, size, size]
        返回:
        torch.Tensor: 计算得到的损失值
        """
        batch_size = res.size(0)
        channels = res.size(1)
        size = res.size(2)

        # 判断是否使用GPU，如果可用则将数据移动到GPU上
        if torch.cuda.is_available():
            res = res.cuda()
            true = true.cuda()

        # 用于存储res经过dct_3d变换后的结果
        res_dct_result = torch.zeros_like(res)
        # 用于存储true经过dct_3d变换后的结果
        true_dct_result = torch.zeros_like(true)

        for b in range(batch_size):
            # 对res每个batch、每个通道的数据进行dct_3d变换
            res_dct_result[b, :, :, :] = dct.dct_3d(res[b, :, :, :])
            # 对true每个batch、每个通道的数据进行dct_3d变换
            true_dct_result[b, :, :, :] = dct.dct_3d(true[b, :, :, :])

        # 使用类的属性self.loss_fn来计算损失
        loss = self.loss(res_dct_result, true_dct_result)
        loss = loss / batch_size
        return loss
    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            #loss = self.loss(sr, hr)
            physic_loss=self.dct_loss(sr,hr)
            loss=physic_loss/100
            #loss = 0.85*loss+0.15*physic_loss  #增加物理损失
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )

        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        ssim_count={}
        ssim_sum={}
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                key = (d.dataset.name, scale)  # 以(数据集名, 尺度)作为字典的键
                ssim_sum[key] = 0
                ssim_count[key] = 0
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]

                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    """

                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_ssim(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
"""
                    # 计算SSIM
                    # pyt_ssim = pytorch_ssim.SSIM(window_size=11)
                    # ssim_value = pyt_ssim(sr, hr)
                    # ssim_sum[key] += ssim_value
                    # ssim_count[key] += 1
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                #avg_ssim = ssim_sum[key] / ssim_count[key] if ssim_count[key] > 0 else 0
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    #'[{} x{}]\SSIM: {:.3f} (Best: {:.3f} @ epoch {})'.format(
                    #'[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})\tpytorch_SSIM: {:.4f}'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

