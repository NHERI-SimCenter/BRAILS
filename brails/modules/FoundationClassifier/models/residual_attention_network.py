from models.attention_module import *
from models.basic_layer import ResidualBlock, ResidualBlock1D

class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size, last_layer=False, dropout_p=0.0):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.dp = nn.Dropout(dropout_p)
        self.bn = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        self.last_layer = last_layer

        # initialize weights
        # x = float(1/128)
        # torch.nn.init.uniform_(self.fc.weight,-x,x)
        # torch.nn.init.uniform_(self.fc.bias,-x,x)



    def forward(self, x):
        out = self.fc(x)
        out = self.dp(out)
        if not self.last_layer:
            out = self.bn(out)
            out = self.relu(out)
        return out


class MLP(nn.Module):
    # Lesson learned: Dropout only on the last layer seems to be superior
    # hidden size 1024 or at least double that is best
    def __init__(self, input_size, hidden_size, num_classes, num_dense_layers = 6, dropout_p=0.0):
        super(MLP, self).__init__()
        self.dense_blocks = []
        self.dense_blocks.append(DenseBlock(input_size, hidden_size*2,dropout_p=0.0))
        self.dense_blocks.append(DenseBlock(hidden_size*2, hidden_size*3,dropout_p=0.0))
        self.dense_blocks.append(DenseBlock(hidden_size*3, hidden_size,dropout_p=0.0))
        for i in range(num_dense_layers-1):
            self.dense_blocks.append(DenseBlock(hidden_size, hidden_size,dropout_p=0.0))
        self.dense_blocks.append(DenseBlock(hidden_size, num_classes, last_layer=True,dropout_p=dropout_p))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)
        # self.test_a = ResidualBlock1D(1,32)
        # self.test_b = ResidualBlock1D(32,128)
        # self.test_c = ResidualBlock1D(128,64)
        # self.test_d = ResidualBlock1D(64,8)
        # self.test_e = nn.Linear(1024,1)

    def forward(self, x):
        x = x.squeeze()
        out = self.dense_blocks(x)

        # out = self.test_a(x)
        # out = self.test_b(out)
        # out = self.test_c(out)
        # out = self.test_d(out)
        # out = out.flatten(start_dim=1)
        # out = self.test_e(out)

        return out


class ResidualAttentionModel_92(nn.Module):
    # for input size 224
    def __init__(self, output_dim_ffe, dropout=False):
        super(ResidualAttentionModel_92, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dropout = dropout
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.ResidualBlock1 = ResidualBlock(64, 256)
        if self.dropout:
            self.dp_1 = nn.Dropout(0.2)
            self.dp_2 = nn.Dropout(0.2)
            self.dp_3 = nn.Dropout(0.2)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.ResidualBlock2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.ResidualBlock3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.ResidualBlock4 = ResidualBlock(1024, 2048, 2)
        self.ResidualBlock5 = ResidualBlock(2048, 2048)
        self.ResidualBlock6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc_1 = nn.Linear(2048, output_dim_ffe)
        #self.fc_2 = nn.Linear(2048, output_dim_nos)
        #self.softmax = nn.Softmax(dim=1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.xavier_uniforml_(m.bias.data)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        if self.dropout:
            out = self.dp_1(out)

        out = self.ResidualBlock1(out)
        out = self.attention_module1(out)
        out = self.ResidualBlock2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        if self.dropout:
            out = self.dp_2(out)
        out = self.ResidualBlock3(out)
                # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.ResidualBlock4(out)
        out = self.ResidualBlock5(out)
        out = self.ResidualBlock6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        if self.dropout:
            out = self.dp_3(out)

        out_ffe = self.fc_1(out)
        #out_ffe = torch.nn.functional.normalize(out_ffe, 2) # For npid
        #out_nos = self.fc_2(out)
        # Not necessary actually
        #out = self.softmax(out)


        return out_ffe#, out_nos

class ResidualAttentionModel_92_Small(nn.Module):
    # for input size 224
    def __init__(self,output_dim=6, dropout=False):
        super(ResidualAttentionModel_92_Small, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dropout = dropout
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.ResidualBlock1 = ResidualBlock(64, 256)
        if self.dropout:
            self.dp_1 = nn.Dropout(0.2)
            self.dp_2 = nn.Dropout(0.0)
            self.dp_3 = nn.Dropout(0.0)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.ResidualBlock2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.ResidualBlock3 = ResidualBlock(512, 1024, 3)

        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=3)
        )
        self.fc = nn.Linear(1024,output_dim)
        #self.softmax = nn.Softmax(dim=1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.xavier_uniforml_(m.bias.data)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        if self.dropout:
            out = self.dp_1(out)
        # print(out.data)
        out = self.ResidualBlock1(out)
        out = self.attention_module1(out)
        out = self.ResidualBlock2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        if self.dropout:
            out = self.dp_2(out)
        out = self.ResidualBlock3(out)
                # print(out.data)

        out = self.mpool2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        if self.dropout:
            out = self.dp_3(out)

        out = self.fc(out)
        #out = torch.nn.functional.normalize(out, 2) # For npid
        # Not necessary actually
        #out = self.softmax(out)


        return out

class ResidualAttentionModel_92_Small_1D(nn.Module):
    # for input size 224
    def __init__(self,output_dim=6, dropout=False):
        start_size = 64 # 256
        mid_size = 64 # 512
        end_size = 64 # 1024

        super(ResidualAttentionModel_92_Small_1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=2, stride=2, bias = False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.dropout = dropout
        self.mpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.ResidualBlock1 = ResidualBlock1D(64, start_size)
        if self.dropout:
            self.dp_1 = nn.Dropout(0.4)
            self.dp_2 = nn.Dropout(0.4)
            self.dp_3 = nn.Dropout(0.4)
        #self.attention_module1 = AttentionModule_stage1(256, 256)
        self.ResidualBlock2 = ResidualBlock1D(start_size, mid_size, 2)
        #self.attention_module2 = AttentionModule_stage2(512, 512)
        #self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add

        self.ResidualBlock3 = ResidualBlock1D(mid_size, end_size, 3)

        self.mpool2 = nn.Sequential(
            nn.BatchNorm1d(end_size),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=4, stride=3)
        )
        self.fc = nn.Linear(end_size,output_dim)
        #self.softmax = nn.Softmax(dim=1)

        for m in self.children():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.xavier_uniforml_(m.bias.data)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        if self.dropout:
            out = self.dp_1(out)
        # print(out.data)
        out = self.ResidualBlock1(out)
        #out = self.attention_module1(out)
        out = self.ResidualBlock2(out)
        #out = self.attention_module2(out)
        #out = self.attention_module2_2(out)
        if self.dropout:
            out = self.dp_2(out)
        out = self.ResidualBlock3(out)
                # print(out.data)

        out = self.mpool2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        if self.dropout:
            out = self.dp_3(out)

        out = self.fc(out)
        # Not necessary actually
        #out = self.softmax(out)


        return out


class ResidualAttentionModel_92_32input(nn.Module):
    # for input size 32
    def __init__(self, output_dim):
        super(ResidualAttentionModel_92_32input, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16*16
        self.ResidualBlock1 = ResidualBlock(32, 128)  # 16*16
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128)  # 16*16
        self.ResidualBlock2 = ResidualBlock(128, 256, 2)  # 8*8
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256)  # 8*8
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256)  # 8*8 # tbq add
        self.ResidualBlock3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 4*4
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 4*4 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 4*4 # tbq add
        self.ResidualBlock4 = ResidualBlock(512, 1024)  # 4*4
        self.ResidualBlock5 = ResidualBlock(1024, 1024)  # 4*4
        self.ResidualBlock6 = ResidualBlock(1024, 1024)  # 4*4
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=4, stride=1)
        )
        self.fc = nn.Linear(1024,output_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.ResidualBlock1(out)
        out = self.attention_module1(out)
        out = self.ResidualBlock2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.ResidualBlock3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.ResidualBlock4(out)
        out = self.ResidualBlock5(out)
        out = self.ResidualBlock6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResidualAttentionModel_92_32input_update(nn.Module):
    # for input size 32
    def __init__(self, output_dim):
        super(ResidualAttentionModel_92_32input_update, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32
        # self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16*16
        self.ResidualBlock1 = ResidualBlock(32, 128)  # 32*32
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128, size1=(32, 32), size2=(16, 16))  # 32*32
        self.ResidualBlock2 = ResidualBlock(128, 256, 2)  # 16*16
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16 # tbq add
        self.ResidualBlock3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.ResidualBlock4 = ResidualBlock(512, 1024)  # 8*8
        self.ResidualBlock5 = ResidualBlock(1024, 1024)  # 8*8
        self.ResidualBlock6 = ResidualBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(1024, output_dim)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.mpool1(out)
        # print(out.data)
        out = self.ResidualBlock1(out)
        out = self.attention_module1(out)
        out = self.ResidualBlock2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.ResidualBlock3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.ResidualBlock4(out)
        out = self.ResidualBlock5(out)
        out = self.ResidualBlock6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out