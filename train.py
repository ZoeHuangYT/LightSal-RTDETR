import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import torch
torch.cuda.empty_cache()


if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/lightsal_rtdetr.yaml')
    state_dict = torch.load('vis_sal_priors.pt')
    pretrained_weight = state_dict['model.model.0.conv.weight']
    # 只需前 3 个输入通道
    pretrained_weight = pretrained_weight[:, :3, :, :]
    state_dict['model.model.0.conv.weight'] = pretrained_weight
    model.load_state_dict(state_dict, strict=False)

    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=16,
                device='0',
               # resume='runs/train/exp80/weights/last.pt', # last.pt path
                project='runs/train',
                name='exp',
                patience=50,

                )
