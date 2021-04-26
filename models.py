from torch import nn
import torch
import torch.nn.functional as F
class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
        
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
    
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
    
class RDCAB(nn.Module):

            
    def __init__(self, in_channels, growth_rate):
        super(RDCAB, self).__init__()
        distillation_rate=0.25
        gc = growth_rate
        fc = 48
        #Feature Distillation(Refine)
        self.layer1 = nn.Conv2d(in_channels + 0 * gc, gc, 3, padding=1, bias=True)
        self.layer2 = nn.Conv2d(in_channels + 1 * gc, gc, 3, padding=1, bias=True)
        self.layer3 = nn.Conv2d(in_channels + 2 * gc, gc, 3, padding=1, bias=True)
        self.layer4 = nn.Conv2d(in_channels + 3 * gc, gc, 3, padding=1, bias=True)
        self.layer5 = nn.Conv2d(in_channels + 4 * gc, gc, 3, padding=1, bias=True)
        self.layer6 = nn.Conv2d(in_channels + 5 * gc, gc, 3, padding=1, bias=True)
        self.layer7 = nn.Conv2d(in_channels + 6 * gc, gc, 3, padding=1, bias=True)
        self.layer8 = nn.Conv2d(in_channels + 7 * gc, 32, 3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        
        #Feature Distillation(Retain)
        self.layer1_sub = nn.Conv2d(gc,32,1)
        self.layer2_sub = nn.Conv2d(gc,32,1)
        self.layer3_sub = nn.Conv2d(gc,32,1)
        self.layer4_sub = nn.Conv2d(gc,32,1)
        self.layer5_sub = nn.Conv2d(gc,32,1)
        self.layer6_sub = nn.Conv2d(gc,32,1)
        self.layer7_sub = nn.Conv2d(gc,32,1)
        
        #Local Feature_fusion
        self.lff = nn.Conv2d(gc*4, gc, kernel_size=1)
        
        #Contrast Channle Attention 
        self.contrast = stdv_channels
        # feature channel downscale and upscale --> channel weight
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(64, 64 // 16, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // 16, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        

    def forward(self, x): 
        Local_Residual = x
        
        layer1 = self.act(self.layer1(x)+x) #64->64
        layer1_sub = self.layer1_sub(x) # 64->32
        
        layer2 = self.act(self.layer2(torch.cat((x, layer1), 1))+layer1) # 128->64
        layer2_sub = self.layer2_sub(layer2) # 64->32
        
        layer3 = self.act(self.layer3(torch.cat((x, layer1,layer2), 1))+layer2) # 192->64
        layer3_sub = self.layer3_sub(layer3) # 64->32
        
        layer4 = self.act(self.layer4(torch.cat((x, layer1,layer2,layer3), 1))+layer3) #256->64
        layer4_sub = self.layer4_sub(layer4) # 64->32
        
        layer5 = self.act(self.layer5(torch.cat((x, layer1,layer2,layer3,layer4), 1))+layer4) #320->64
        layer5_sub = self.layer5_sub(layer5) # 64->32
        
        layer6 = self.act(self.layer6(torch.cat((x, layer1,layer2,layer3,layer4,layer5), 1))+layer5) #384->64
        layer6_sub = self.layer6_sub(layer6) # 64->32
        
        layer7 = self.act(self.layer7(torch.cat((x, layer1,layer2,layer3,layer4,layer5,layer6), 1))+layer6) #448->64
        layer7_sub = self.layer7_sub(layer7) # 64->32
        
        layer8 = self.layer8(torch.cat((x, layer1,layer2,layer3,layer4,layer5,layer6,layer7),1)) #512->32
        
        
        out = torch.cat([layer1_sub,layer2_sub,layer3_sub,layer4_sub,layer5_sub,layer6_sub,layer7_sub,layer8], dim=1) 
        x = self.lff(out)
        
        y =self.contrast(x)+self.avg_pool(x)
        y = self.conv_du(y)
        x = x*y
        x = x+Local_Residual
        return x


class RecursiveBlock(nn.Module):
    def __init__(self,num_channels, num_features, growth_rate, B, U):
        super(RecursiveBlock, self).__init__()
        self.U = U
        self.G0 = num_features
        self.G = growth_rate
        self.rdbs = RDCAB(self.G0, self.G) #residual dense channel attention block & Feature Distillation 
        
    def forward(self, sfe2):
        global cocnat_LF
        x=sfe2
        local_features = []
        for i in range(self.U):
            x = self.rdbs(x)
            local_features.append(x)
        cocnat_LF = torch.cat(local_features, 1)
        return x
        
class DRRDB(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, B, U):
        super(DRRDB, self).__init__()
        self.B = B
        self.G0 = num_features
        self.G = growth_rate
        self.U = U
        self.scale_factor=scale_factor
        self.num_channels=num_channels
        
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
        
        self.recursive_SR = nn.Sequential(*[RecursiveBlock(num_channels if i==0 else num_features,
        num_features,
        growth_rate, 
        B, 
        U) for i in range(B)])
        
        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.U * self.B, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        
        
        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )
        # middle point wise 64->32
        self.middle_pointwise=nn.Conv2d(self.G0,32,1)
        # information refinement block
        self.convIRB_1 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.convIRB_2 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.convIRB_3 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.convIRB_4 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.convIRB = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,padding=1,groups=1,bias=False))

    def forward(self, x):
        x_up = F.interpolate(x, mode='bicubic',scale_factor=self.scale_factor)
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        
        local_global_features=[]
        
        x= self.recursive_SR(sfe2)
        
        x = self.gff(concat_LF) + sfe1
        x = self.upscale(x)
        x = self.middle_pointwise(x)
        x = self.convIRB_1(x)
        x = self.convIRB_2(x)
        x = self.convIRB_3(x)
        x = self.convIRB_4(x)
        x = self.convIRB(x)+x_up
        
        return x
