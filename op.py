import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from resizer import Resizer
from op_seg import ModelBuilder
from op_edge import Sobel
import kornia
from torchvision.transforms.functional import to_pil_image
from functools import partial

class SuperResolutionOperator(nn.Module):
    def __init__(self, in_shape, scale_factor):
        super(SuperResolutionOperator, self).__init__()
        self.scale_factor = scale_factor
        self.down_sample = Resizer(in_shape, 1/scale_factor)
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)

    def forward(self, x, keep_shape=False):
        x = (x + 1.0) / 2.0
        y = self.down_sample(x)
        y = (y - 0.5) / 0.5
        if keep_shape:
            y = F.interpolate(y, scale_factor=self.scale_factor, mode='bicubic')
        return y

    def transpose(self, y):
        return self.up_sample(y)

    def y_channel(self):
        return 3
    
    def to_pil(self, y):
        y = (y[0] + 1.0) / 2.0
        y = torch.clip(y, 0, 1)
        y = to_pil_image(y, 'RGB')
        return y
    
class AEDSegOperator(nn.Module):
    def __init__(self):
        super(AEDSegOperator, self).__init__()
        self.encoder = ModelBuilder.build_encoder(arch="mobilenetv2dilated",fc_dim=320,weights="./models/models_seg/mobilenetv2-c1/encoder_epoch_20.pth")
        self.decoder = ModelBuilder.build_decoder(arch="c1_deepsup",fc_dim=320,num_class=150,weights="./models/models_seg/mobilenetv2-c1/decoder_epoch_20.pth",use_softmax=True)
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False
        for _, param in self.decoder.named_parameters():
            param.requires_grad = False
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def forward(self, x, **kwargs):

        x = (x + 1) / 2.0
        x = self.transform(x)
        y = self.decoder(self.encoder(x, return_feature_maps=True), segSize=(512,512))
        assert 'mode' in kwargs
        if kwargs['mode'] == 'init':
            return torch.argmax(y, dim=1, keepdim=True)
        else:
            return y

    def y_channel(self):
        return 3


class EdgeOperator(nn.Module):

    def __init__(self):
        super(EdgeOperator, self).__init__()
        # self.sobel = Sobel()
        self.canny = kornia.filters.Canny()

    def forward(self, data, **kwargs):
        # out = self.sobel(torch.mean((data + 1.0) / 2.0, dim=1, keepdim=True))
        out = self.canny((data + 1.0) / 2.0)[0]

        return out

    def y_channel(self):
        return 1

    def to_pil(self, y):
        y = torch.cat([y, y, y], dim=1)[0]
        y = y / torch.max(y)
        y = to_pil_image(y, 'RGB')
        return y

if __name__ == "__main__":
    from dataset import ImageDataset
    from op_seg import visualize_result
    DATA_ROOT = "./ffhq_512"
    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    f = EdgeOperator()
    # f = AEDSegOperator()
    f = f.to(device="cuda")

    dataset = ImageDataset(root=DATA_ROOT, transform=test_transforms, return_path=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (x, x_path) in enumerate(dataloader):
        x = x.to(device="cuda")
        x.requires_grad = True
        y = f(x, mode='non-init')
        y = f.to_pil(y)
        save_image(x, 'x.png', normalize=True, value_range=(-1,1))
        y.save('y.png')
        assert(0)

