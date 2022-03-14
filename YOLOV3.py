import torch
import torch.nn as nn
import yaml
import logging

def set_logging(name=None, verbose=True):
    logging.basicConfig(format="%(message)s", level=logging.INFO if verbose else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)  # define globally (used in train.py, val.py, detect.py, etc.)

def autopad(kernel, padding=None):
    if padding is None:
        p = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__():
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if act is True else nn.Identity
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut = True, e = 0.5):
        super().__init__()
        c_= int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Concat(nn.Module):
    #Concat a list of tensors (along specified dimension)
    def __init__(self, dimension = 1):
        super().__init__()
        self.d = dimension 
    
    def forward(self, x):
        return torch.cat(x, self.d)

class Detect(nn.Module):
    def __init__(self, nc, anchors, ch):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.num_det = len(anchors)
        self.num_anch = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.num_det
        self.anchor_grid = [torch.zeros(1)] * self.num_det
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.num_det, -1, 2))

        #to convert x filter into self.no filters
        self.m = nn.MoudleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def _make_grid(self, nx, ny, i):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv,yv), 2).expand((1,self.num_anch, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.num_anch, 1, 1, 2)).expand((1, self.num_anch, ny, nx, 2)).float()
        return grid, anchor_grid
    
    def forward(self, x):
        z = []
        for i in range(self.num_det):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.num_anch, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid.shape[2:4] != x.shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny).to(x.device)
                x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * self.stride[i]
                x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid[i]
                x[...,4:] = torch.sigmoid()
                x = x.view(bs, -1, self.no)
            
            return x if self.training else (torch.cat(z, 1), x)
            




def parse_cfg(cfg, ch):
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc = cfg['anchors'], cfg['nc']
    # each anchor has 2 coordinates
    na = len(anchors[0]) // 2
    # na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers, save, c2 =[], [], ch[-1]
    for i, (f, n, m, args) in enumerate(cfg['backbone]'] + cfg['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
    
        if m in [Conv, Bottleneck]:
            c1, ch2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
        elif m is Concat:
            ch2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]
        
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        num_params = sum(x.numel() for x in m_.parameters)
        m_.i, m_.f, m_.name, m_.num_params = i, f, t, num_params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{num_params:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0: 
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

        


class Model(nn.Module):
    def __init__(self, cfg, ch=None, nc=None, anchors=None):
        super().__init__():

        with open(cfg, encoding='ascii', errors='ignore') as f:
            self.yaml = yaml.safe_load(f)
        # if ch and ch != self.yaml['ch']:
        #     LOGGER.info(f"Overriding model.yaml ch={self.yaml['ch']} with ch={ch}")
        # if nc and nc != self.yaml['ch']:
        #     LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        # if ch and ch != self.yaml['ch']:
        #     LOGGER.info(f"Overriding model.yaml ch={self.yaml['ch']} with anchors={anchors}")
        self.model = parse_cfg(self.yaml)

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1,ch, s, s))])
            self.stride = m.stride
            #TODO note check if you need to divide anchors!!!
            #TODO initialize
    
    def forward(self, x):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


        



