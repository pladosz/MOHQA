#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
import time

class NatureConvBody(nn.Module):
    def __init__(self, in_channels = 1):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 16
        self.conv1 = layer_init(nn.Conv2d(in_channels, 16, kernel_size=4, stride=4))
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=2, stride=2))
        self.conv3 = layer_init(nn.Conv2d(32, 32, kernel_size=2, stride=1))
        self.fc4 = layer_init(nn.Linear(288, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        features_detached = self.fc4(y).cpu().detach().numpy()
        #print(y.shape)
        y = F.tanh(self.fc4(y))
        return y,features_detached

class ApnnConvBody(nn.Module):
    """The body used for APNN. The only difference between this and NatureConvBody
       is lack of tanh in output of forward() method."""
    def __init__(self, in_channels = 1):
        super(ApnnConvBody, self).__init__()
        self.feature_dim = 16
        self.conv1 = layer_init(nn.Conv2d(in_channels, 16, kernel_size=4, stride=4))
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=2, stride=2))
        self.conv3 = layer_init(nn.Conv2d(32, 32, kernel_size=2, stride=1))
        self.fc4 = layer_init(nn.Linear(288, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        self.conv_layers_output = y.detach()
        y = self.fc4(y)
        return y


class NatureConvBody_lstm(nn.Module):
    def __init__(self, in_channels = 1):
        super(NatureConvBody_lstm, self).__init__()
        self.feature_dim = 16
        self.rnn_input_dim = 256
        self.conv1 = layer_init(nn.Conv2d(in_channels, 4, kernel_size = 5, stride = 1))
        self.conv2 = layer_init(nn.Conv2d(4, 8, kernel_size = 3, stride = 1))
        self.conv3 = layer_init(nn.Conv2d(8, 16, kernel_size = 3, stride = 1))
        self.lstm = nn.LSTM(self.rnn_input_dim, self.feature_dim)
        self.fc4 = layer_init(nn.Linear(self.feature_dim, 16))
        self.hidden = self.init_hidden()
        self.reset_flag = False
        self.count = 0

    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(1, 1, self.feature_dim)).cuda(),
                torch.autograd.Variable(torch.zeros(1, 1, self.feature_dim)).cuda())

    def forward(self, x):
        self.count = self.count+1
        if self.reset_flag:
            #print("repackagin!")
            self.hidden = self.repackage_hidden(self.hidden)
            self.reset_flag = False
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.tanh(self.conv3(y))
        #print(y.shape)
        y = y.view(y.size(0), -1)
        y = y.view(-1, 1, self.rnn_input_dim)
        y, self.hidden = self.lstm(y, self.hidden)
        #y = F.tanh(y)
        y = torch.squeeze(y, 1)
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

    def repackage_hidden(self,h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

class CombinedNet(nn.Module, BaseNet):
    ''' not sure what I'm doing here (AS) need to review'''
    def __init__(self, bodyPredict):
        super(CombinedNet, self).__init__()
#        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
#        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.bodyPredict = bodyPredict
        self.to(Config.DEVICE)

    def returnFeatures(self, x, to_numpy = False):
        phi = self.bodyPredict(tensor(x))
        if to_numpy:
            return phi.cpu().detach().numpy()
        return phi
    #        value = self.fc_value(phi)
#        advantange = self.fc_advantage(phi)
#        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim = True).expand_as(advantange))
#        if to_numpy:
#            return q.cpu().detach().numpy()
#        return q

class DDPGConvBody(nn.Module):
    def __init__(self, in_channels = 4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size = 3, stride = 2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size = 3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units = (64, 64), gate = F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units = (64, 64), gate = F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim = 1)))
        return phi

class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate = F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim = 1))
        return phi

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class Mod3LNatureConvBody_directTH(nn.Module):
    '''direct neuromodulation with tanh so that plasticity is modulated in the range [-1,1].
    19/12/18 does not work, to delete'''
    def __init__(self, in_channels = 4):
        super(Mod3LNatureConvBody_directTH, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size = 8, stride = 4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size = 8, stride = 4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size = 4, stride = 2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size = 4, stride = 2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size = 3, stride = 1))
        self.conv3_mem_features = layer_init(nn.Conv2d(64, 64, kernel_size = 3, stride = 1))

        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        y_mem0 = F.relu(self.conv1_mem_features(x_mem))
        y_mod0 = torch.tanh(y_mem0)
        y = y0 * y_mod0

        y1 = F.relu(self.conv2(y))
        y_mem1 = F.relu(self.conv2_mem_features(y_mem0))
        y_mod1 = torch.tanh(y_mem1)
        y = y1 * y_mod1

        y2 = F.relu(self.conv3(y))
        y_mem2 = F.relu(self.conv3_mem_features(y_mem1))
        y_mod2 = torch.tanh(y_mem2)
        y = y2 * y_mod2

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y
