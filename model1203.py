import torch
import torch.nn as nn




class ConvLSTMCell(nn.Module):
    # 这里面全都是数，衡量后面输入数据的维度/通道尺寸
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 卷积核为一个数组
        self.kernel_size = kernel_size
        # 填充为高和宽分别填充的尺寸
        self.padding_size = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2
        self.bias = bias
        self.conv = nn.Conv3d(self.input_dim + self.hidden_dim,
                              4 * self.hidden_dim,  # 4* 是因为后面输出时要切4片
                              self.kernel_size,
                              padding=self.padding_size,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat((input_tensor, h_cur), dim=1)
        combined_conv = self.conv(combined)
        cc_f, cc_i, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # torch.sigmoid(),激活函数--
        # nn.functional中的函数仅仅定义了一些具体的基本操作，
        # 不能构成PyTorch中的一个layer
        # torch.nn.Sigmoid()(input)等价于torch.sigmoid(input)
        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 这里的乘是矩阵对应元素相乘，哈达玛乘积
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width, depth = image_size
        # 返回两个是因为cell的尺寸与h一样
        return (torch.zeros(batch_size, self.hidden_dim, height, width, depth,device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, depth,device=self.conv.weight.device))

class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1,1)
        x = inputs * x
        return x

class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm3d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm3d(in_c),
            nn.ReLU(),
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm3d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm3d(in_c[0]),
            nn.ReLU(),
            nn.Conv3d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool3d((2, 2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm3d(in_c[1]),
            nn.ReLU(),
            nn.Conv3d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm3d(in_c[1]),
            nn.ReLU(),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y

class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d

class build_resunetplusplus(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = Stem_Block(3, 16, stride=1)
        self.c2 = ResNet_Block(16, 32, stride=2)
        self.c3 = ResNet_Block(32, 64, stride=2)
        self.c4 = ResNet_Block(64, 128, stride=2)

        # self.b1 = ASPP(128, 256)
        self.b1 = ConvLstm(input_dim=128, hidden_dim=[256], kernel_size=[3, 3, 3], num_layers=1, batch_first=True,
                               return_all_layers=False)
        

        self.d1 = Decoder_Block([64, 256], 128)
        self.d2 = Decoder_Block([32, 128], 64)
        self.d3 = Decoder_Block([16, 64], 32)

        # self.aspp = ASPP(32, 16)
        self.b2 = ConvLstm(input_dim=32, hidden_dim=[32,16], kernel_size=[3, 3, 3], num_layers=2, batch_first=True,
                               return_all_layers=False)
        self.output = nn.Conv3d(16, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)

        
        cl1 = c4.repeat(10, 1, 1, 1, 1, 1)#10 2 128 4 4 2
        cl2 = self.b1(cl1)# 2 10 256 4 4 2
        ls_d3_n= []

        for i in range(10):
            d1_i = self.d1(c3, cl2[:,i,:])#batch,time_step,chennal
            d2_i = self.d2(c2, d1_i)
            d3_i = self.d3(c1, d2_i)
            ls_d3_n.append(d3_i)

        # output = self.aspp(d3_n)
        cl3 = torch.stack(ls_d3_n)# 10 2 32 32 32 16
        cl4 = self.b2(cl3)#2 10 16  32 32 16
        output = self.output(cl4[:,0,:])
        for i in range(1,10):
            output = torch.cat(
                                 (output,self.output(cl4[:,i,:])),
                                 dim=1
                                 )
        return output

class ConvLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=False,
                return_all_layers=False):
        super(ConvLstm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []  # 为了储存每一层的参数尺寸
        for i in range(0, num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]  # 注意这里利用lstm单元得出到了输出h，h再作为下一层的输入，依次得到每一层的数据维度并储存
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                        hidden_dim=self.hidden_dim[i],
                                        kernel_size=self.kernel_size,
                                        bias=self.bias
                                        ))
        # 将上面循环得到的每一层的参数尺寸/维度，储存在self.cell_list中，后面会用到
        # 注意这里用了ModuLelist函数，模块化列表
        self.cell_list = nn.ModuleList(cell_list)
    # 这里forward有两个输入参数，input_tensor 是一个五维数据
    # （t时间步,b输入batch_ize,c输出数据通道数--维度,h,w图像高乘宽）
    # hidden_state=None 默认刚输入hidden_state为空，等着后面给初始化
    def forward(self, input_layer , hidden_state=None): #16 32 32 80

        # input_layer = conv4.repeat(10, 1, 1, 1, 1, 1)

        # 取出图片的数据，供下面初始化使用
        if self.batch_first:
            input_layer = input_layer.permute(1, 0, 2, 3, 4, 5)
        # 先调整一下输出数据的排列
        b, _, _, h, w, d = input_layer.size()
        # 初始化hidd_state,利用后面和lstm单元中的初始化函数
        hidden_state = self._init_hidden(b, (h, w, d))
        # 储存输出数据的列表
        layer_output_list = []
        seq_len = input_layer.size(1)
        # 初始化输入数据
        cur_layer_input =input_layer
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # 每一个时间步都更新 h,c
                # 注意这里self.cell_list是一个模块(容器)
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :, :], [h, c])
                # 储存输出，注意这里 h 就是此时间步的输出
                output_inner.append(h)
            # 这一层的输出作为下一次层的输入,
            layer_output = torch.stack(output_inner,1)
            cur_layer_input = layer_output
            layer_output_list=output_inner
        layer_output = torch.stack(layer_output_list, 1)
        return layer_output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

class IdxLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pre_y, y):
        error = pre_y - y
        error_squared = torch.pow(error, 2)
        mean_error_squared = torch.mean(error_squared)
        loss = torch.sqrt_(mean_error_squared)
        return loss


