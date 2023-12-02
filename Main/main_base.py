from configs.parser import BaseOptions
from engine.trainer import Trainer

if __name__ == '__main__':
    cfg = BaseOptions().opts
    trainer = Trainer(cfg)
    trainer.train()
