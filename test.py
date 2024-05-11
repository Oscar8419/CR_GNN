import numpy as np
import torch
from torch_geometric.data import Data
import math
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Tanh, Sigmoid, BatchNorm1d as BN
import time


#################################性能评估函数##########################################
def obj_IA_sum_rate(H, p, var_noise, K):  # 最大化速率和的计算
    y = 0.0
    # (11,11)
    for i in range(1, K+1):
        s = var_noise  # 噪声功率
        for j in range(1, K+1):
            if j != i:
                s = s+H[i, j]**2*p[j-1]
        y = y+math.log2(1+H[i, i]**2*p[i-1]/s)  # 和速率
    return y

def np_sum_rate(H, p, N, var_noise=1):
    num_sample = H.shape[2]
    pyrate = np.zeros(num_sample)
    for i in range(num_sample):
        pyrate[i] = obj_IA_sum_rate(H[:, :, i], p[:, i], var_noise, N)
    sum_rate = sum(pyrate)/num_sample
    # np.savetxt('./GNN/wmmse_rate.txt', pyrate)
    return sum_rate

def np_sum_rate1(H, p, N, var_noise=1):   # relu
    num_sample = H.shape[2]
    nnrate = np.zeros(num_sample)
    for i in range(num_sample):
        nnrate[i] = obj_IA_sum_rate(H[:, :, i], p[:, i], var_noise, N) #-0.2  # 修正
    sum_rate = sum(nnrate)/num_sample
    # np.savetxt('./GNN/GNN_rate.txt', nnrate)
    return sum_rate


################################图数据生成##########################################
def get_cg(n):  # 生成边
    adj = []
    for i in range(0, n):
        for j in range(0, n):
            if (not (i == j)):
                adj.append([i, j])
    return adj


def build_graph(adj):  # 建立图  每个样本的
    x = np.ones((K, 1))
    x1 = np.ones((K, 20))
    edge_attr = np.ones((420, 2))
    H = np.concatenate((x, x1), axis=1)

    x = torch.tensor(x, dtype=torch.float)  # 节点特征
    edge_index = torch.tensor(adj, dtype=torch.long)  #  边关系
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)        # (50x132)x2
    y = torch.tensor(np.expand_dims(H, axis=0), dtype=torch.float)  # 50x12x12
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)
    return data



def proc_data():
    data_list = []
    cg = get_cg(K)   # zhu+ci
    for i in range(sample):
        data = build_graph(cg)
        data_list.append(data)
    return data_list
###########################################################################


##############################网络模型#############################################
class IGConv(MessagePassing):  # MPGNN网络层的建立
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)#3聚合

        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)   # 32+1=33
        return torch.cat([x[:, :2], comb], dim=1)   # 每一层的输出

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)  # 1+2=2
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias = True), ReLU())#, BN(channels[i]))
        for i in range(1, len(channels))
    ])


class IGCNet(torch.nn.Module):
    def __init__(self):
        super(IGCNet, self).__init__()

        self.mlp1 = MLP([3, 16, 36])
        self.mlp2 = MLP([37, 16])
        self.mlp2 = Seq(*[self.mlp2,Seq(Lin(16, 1, bias = True), Sigmoid())])
        self.conv = IGConv(self.mlp1,self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        out = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        # x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        # out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        return out




def sr_loss(data, out, K=21, var=1):   # 计算速率和  4800x2
    power = out[:, 1]
    power = torch.reshape(power, (-1, K, 1))
    power = power[:, 1:su_num+1, :]
    abs_H = data.y
    abs_H = abs_H[:, 1:su_num+1, 1:su_num+1]
    abs_H_2 = torch.pow(abs_H, 2)
    rx_power = torch.mul(abs_H_2, power)  # abs_H_2*power
    mask = torch.eye(su_num)  # 对角线全为1的矩阵
    mask = mask.to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)   # 提取特征
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
    rate = torch.log(1 + torch.div(valid_rx_power, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    loss = torch.neg(sum_rate)  # 取反
    # print(loss)
    return loss



def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  #  out输出的为功率
        loss = sr_loss(data, out)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / num_train


def test():
    model.eval()

    total_loss = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            start_time = time.time()
            loss = sr_loss(data, out)
            finish_time = time.time()
            total_time = finish_time - start_time
            total_loss += loss.item() * data.num_graphs     # 为什么需要乘
    return total_loss / num_test, total_time

###########################################################################

i = 0
pu_num = 1  # 主用户
su_num = 20  # 次用户个数
num_train = 1
num_test = 1
sample = 1
epoch = 10
K = pu_num + su_num
########################################################################################
# 训练集
# Xtrain = np.random.random((su_num+1, su_num+1, num_train))
# PUtrain = np.zeros((1, num_train))
#
# # 测试集
# X = np.random.random((su_num+1, su_num+1, num_test))
# PU = np.zeros((1, num_test))

train_data_list = proc_data()
test_data_list = proc_data()
perf_data_list = proc_data()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
model = IGCNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
train_loader = DataLoader(train_data_list, batch_size=num_test, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=num_test, shuffle=False)
perf_lodaer= DataLoader(perf_data_list, batch_size=num_test, shuffle=False)



for epoch in range(epoch):
    loss1 = train()
    loss2, time1 = test()
    print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
        epoch, loss1, loss2))
    scheduler.step()



model.eval()
total_loss = 0
for data in perf_lodaer:
    data = data.to(device)
    with torch.no_grad():
        tout = model(data)
        loss = sr_loss(data, tout)
        total_loss += loss.item() * data.num_graphs     # 为什么需要乘


power = tout[:, 1]
power = torch.reshape(power, (-1, K, 1))
power = power[:, 1:su_num+1, 0] #
power = power.transpose(1, 0)
X = np.random.random((su_num+1, su_num+1, num_test))
t1 = np_sum_rate1(X, power, su_num)
print(t1)
print(time1)

