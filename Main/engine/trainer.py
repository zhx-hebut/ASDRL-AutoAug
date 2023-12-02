import os
import time
import numpy as np
from pathlib import Path
import torch
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
from engine.checkpointer import Checkpointer
from network.customized_evaluate import LoggedMeasure
from network.customized_loss import LoggedLoss
from preprocess.tools import mkdir
from torchvision.utils import save_image
stage_dict = {
    'train': 2,
    'continue': 2,
    'test': 1,
    'val': 1,
    'pred': -1
}


class Trainer(object):
    def __init__(self, cfg):
        print('Constructing components...')
        import network.inits as inits

        # basic settings
        self.cfg = cfg
        self.epoch = cfg.trainer.epoch 
        self.stage = cfg.trainer.stage 
        self.gpus = cfg.trainer.gpus
        self.output_path = Path(cfg.trainer.output_path)
        self.test_epoch = cfg.trainer.test_epoch 
        self.start_eval = cfg.trainer.start_eval 
        self.start_save_intermediate_model = cfg.trainer.start_save_intermediate_model 
        self.output_path.mkdir(exist_ok=True)

        # seed and stage
        seed = cfg.trainer.seed 
        self.set_seed(seed)

        # To cuda
        print('GPUs id:{}'.format(self.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus

        # components
        self.net = inits.get_network(cfg)
        self.opt = inits.get_solver(self.net, cfg) 
        self.loss_func = inits.get_loss_func(cfg)  
        self.eval_func = inits.get_eval_func(cfg)
        self.train_loader, self.val_loader, self.test_loader = inits.get_dataloader(cfg, ['train',
                                                                                          'val',
                                                                                          'test'])

        self.scheduler = inits.get_scheduler(self.opt, self.train_loader, cfg)

        # log and checkpoint
        self.checkpointer = Checkpointer(self.output_path, self.net, self.opt, self.scheduler)

        # for loading pretrained model
        load_from = Path(cfg.trainer.load_from)  
        if load_from.is_file():
            self.checkpointer.load_model_from_path(load_from)

        # to record log
        self.logger = inits.get_logger(cfg, 'Base')
        self.logger.info('')
        # self.component_state = ComponentState(self.net, self.opt, self.loss_func, self.eval_func,
        #                                       (self.train_loader, self.val_loader, self.test_loader), self.scheduler)

        self.start_epoch = 0
        if cfg.trainer.to_cuda:
            if len(self.gpus.split(',')) > 1:
                self.net = DataParallel(self.net)
                cudnn.benchmark = True
            self.net = self.to_cuda(self.net)

        self.loss_func = self.to_cuda(self.loss_func)
        self.opt = self.to_cuda(self.opt)

        self.set_training_stage(self.stage)

        self.best_train_accuracy = 0
        self.best_val_accuracy = 0
        pass

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed) 

    @staticmethod
    def to_cuda(t):
        if isinstance(t, (torch.nn.Module, torch.Tensor)):
            return t.cuda()
        elif isinstance(t, (list, tuple)):
            l = []
            for i in t:
                l.append(i.cuda())
            return l
        elif isinstance(t, torch.optim.Optimizer):
            for state in t.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            return t
        else:
            return t

    def set_training_stage(self, stage):
        stage = stage.strip().lower()
        self.stage = stage_dict[stage]
        if stage == 'continue':
            start_model = self.get_load_name(self.test_epoch)
            self.start_epoch = self.checkpointer.load_model(start_model) + 1
        if stage == 'pred':
            self.pred = True
        else:
            self.pred = False

    def get_load_name(self, epoch=-1):
        if epoch == -1:
            model_name = 'best'
        elif epoch == -2:
            model_name = None
        else:
            model_name = str(epoch)
        return model_name

    def inference(self, name, input, label):
        input = self.to_cuda(input)
        label = self.to_cuda(label)
        logits = self.net(input) 

        from torch.nn import functional as F
        # from preprocess import visualize *-*
        # print('Logits size:{}'.format(logits.size()))
        # size = logits.size()
        # size[1] = 1
        # pred = torch.argmax(F.softmax(logits, dim=1), dim=1).view(*size)
        # visualize.torch_im_show(input, label, logits) *-*

        if torch.isinf(logits[0]).any():
            raise Exception("Nan when validating, data : {}".format(name))

        if hasattr(self.net, 'post_process'): 
            self.net.post_process(name, input, label, logits)

        # calculate loss by user
        loss = self.loss_func(logits, label)

        # get accuracy with dice_loss
        with torch.no_grad():
            self.eval_func(logits, label)

        return loss, logits

    def _train_net(self, epoch):
        """
        Train current net with dataloader
        :param dataloader:
        :return:
        """
        # prepare for calculating loss and accuracy
        self.eval_func.clear_cache()
        self.loss_func.clear_cache()
        self.net.train()
        self.loss_func.train()
        self.eval_func.train()

        losses = 0.
        nTotal = 0
        nProcessed = 0
        length = len(self.train_loader.dataset) 
        start_time = time.time()
        for step, (name, batch_x, batch_y) in enumerate(self.train_loader):
            # forward
            loss, logits = self.inference(name, batch_x, batch_y)
            if not os.path.exists(os.path.join(self.output_path,'pre','epoch-{}'.format(epoch+1))):
                os.makedirs(os.path.join(self.output_path,'pre','epoch-{}'.format(epoch+1)))
            save_image(logits,os.path.join(self.output_path,'pre','epoch-{}'.format(epoch+1),'image-{}.png'.format(step)))
            if isinstance(logits, int):
                continue
            # reset grad
            self.opt.zero_grad()
            # backward
            loss.backward()
            # update params
            self.opt.step()
            # update lr
            lr = self.scheduler.step(epoch, step)
            losses += float(loss)
            nTotal += 1
            nProcessed += batch_x.size(0)
            print('Epoch:[{}, {:.2f}%], loss:{},lr:{}'.format(epoch, 100 * nProcessed / length,  loss,lr))
        losses /= nTotal
        iou, dice = self.eval_func.get_last()
        end_time = time.time()
        return losses, iou, dice, end_time-start_time

    def _validate_net(self, dataloader):
        """
        Validate current model
        :param dataloader:  a  dataloader, it should be set to run in validate mode
        :return:
        """
        self.eval_func.clear_cache()
        self.loss_func.clear_cache()
        self.net.eval()
        self.loss_func.eval()
        self.eval_func.eval()
        losses = 0.
        nTotal = 0
        start_time = time.time()
        for step, (name, batch_x, batch_y) in enumerate(dataloader):
            # forward
            loss, logits = self.inference(name, batch_x, batch_y)
            losses += loss
            nTotal += 1
        # Log
        losses /= nTotal
        iou, dice = self.eval_func.get_last()
        end_time = time.time()
        return losses, iou, dice, end_time-start_time

    def train(self):
        best_val_accuracy = 0

        if self.stage >= 2:
            # start train_net log
            self.logger.info('  Epoch    train_loss |    iou    dice          val_loss |     iou     dice')
            for epoch in range(self.start_epoch, self.epoch):
                # TODO: adjust learning rate
                # train_net model
                train_loss, train_iou, train_dice, train_time = self._train_net(epoch)

                # validate model
                if self.val_loader is not None and len(self.val_loader) > 0 and epoch >= self.start_eval:
                    with torch.no_grad():
                        val_loss, val_iou, val_dice, val_time = self._validate_net(self.val_loader)

                self.logger.info('[ {}/ {}]     {:.4f} | {:.4f} {:.4f}           {:.4f} | {:.4f} {:.4f}     '
                                 'time : train {:.2f}, val {:.2f}'.format(epoch, self.epoch,
                                                                  train_loss, train_iou, train_dice,
                                                                  val_loss, val_iou, val_dice,
                                                                  train_time, val_time))

                # save the best model in validation,
                if epoch >= self.start_eval:
                    # has validation acc
                    if best_val_accuracy <= val_dice:
                        best_val_accuracy = val_dice
                        self.checkpointer.save_model('best', epoch)

                # save model of every epoch
                # if epoch >= self.start_save_intermediate_model:
                #     self.checkpointer.save_model(str(epoch), epoch)

        # test model
        if self.stage >= 1 and self.test_loader is not None and len(self.test_loader) > 0:
            self.logger.info('  Test    loss |    iou    dice')
            if self.stage == 2:
                # if just finished training, test the best model(according to val or train loss)
                epoch = -1
            else:
                # if not training, test the model user specified, default -1 the best model
                epoch = self.test_epoch
            # if has full_image_loader, test it, it is more important than patch based test
            test_loss, test_iou, test_dice, test_time = self.test(epoch)
            self.logger.info('          {:.4f} | {:.4f} {:.4f}    time : {:.2f}'.
                             format(test_loss, test_iou, test_dice, test_time))

        if self.stage >= -1 and self.pred:
            self.predict(epoch=self.test_epoch)

    def test(self, epoch=-1):
        # use best model to predict
        self.eval_func.clear_cache()
        if self.eval_func.get_max_len() < len(self.test_loader):
            self.eval_func.set_max_len(len(self.test_loader))
        with torch.no_grad():
            self.checkpointer.load_model(self.get_load_name(epoch))
            val_loss, val_iou, val_dice, val_time = self._validate_net(self.test_loader)
        return val_loss, val_iou, val_dice, val_time

    def predict(self, output_dir=None, epoch=-1):
        print("Predicting cases ...")
        from network import inits
        (pred_loader,) = inits.get_dataloader(self.cfg, ['pred'])
        if pred_loader is None:
            print("Your should specify 'pred_root' and 'pred_dataset' in your config file to test full image!!")
            return

        if output_dir is None:
            output_dir = self.output_path / 'predictions'
            mkdir(output_dir)

        self.checkpointer.load_model(self.get_load_name(epoch))

        size = self.train_loader.dataset[0][-2].size()
        z_step = None  

        self.net.eval()
        with torch.no_grad():
            for step, (fnames, batch_x) in enumerate(pred_loader):
                batch_x = self.to_cuda(batch_x)
                logits = self.test_full_image_3d(batch_x, z_step)
                if hasattr(self.net, 'get_pred'):
                    self.net.get_pred(fnames, batch_x, logits, output_dir)