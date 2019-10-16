#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
import math
import time
import sys

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy = False):
        phi,self.features_detached = self.body(tensor(x))
        print(type(phi))
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class ApnnNet(nn.Module, BaseNet):
    """The main implementation of the apnn head"""
    def __init__(self, output_dim, body,apnn_conf_data):
        super(ApnnNet, self).__init__()
        self.apnn_conf_data = apnn_conf_data
        self.output_dim = output_dim
        self.apnn_head_output = nn.Linear(body.feature_dim, output_dim)
        self.apnn_head_output.weight.data.fill_(0.0000)
        self.apnn_head_output.bias.data.fill_(0.000)
        for p in self.apnn_head_output.parameters():
            p.requires_grad = False
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        # APNN parameters
        self.alpha = self.apnn_conf_data["alpha"]
        self.beta = self.apnn_conf_data["beta"]
        self.alpha_matrix = self.alpha*torch.ones(output_dim,body.feature_dim).cuda()
        self.beta_matrix = self.beta*torch.ones(output_dim,body.feature_dim).cuda()
        self.zeta_matrix = -0.005*torch.ones(output_dim,body.feature_dim).cuda()
        self.zeros_matrix = torch.zeros(output_dim,body.feature_dim).cuda()
        self.Theta = torch.zeros(output_dim,body.feature_dim).cuda()
        self.Eligbility_traces = torch.zeros(output_dim,body.feature_dim).cuda()
        self.modulation = 0
        self.tau_E = self.apnn_conf_data["tau_E"]
        self.baseline_modulation = self.apnn_conf_data["baseline_modulation"]
        self.step_previous = -1
        self.theta_low = self.apnn_conf_data["theta_low"]
        self.theta_high = self.apnn_conf_data["theta_high"]
        self.no_top_features = int(round(self.apnn_conf_data["percent_of_top_features"]*body.feature_dim))
        self.no_bottom_features = int(round(self.apnn_conf_data["percent_of_bottom_features"]*body.feature_dim))
        self.features_running_sum = torch.zeros(1,body.feature_dim).cuda()
        self.running_average = torch.zeros(1,body.feature_dim).detach().cuda()
        self.step = 0
        self.running_avg_counter = 0
        self.action_storage = []
        self.mu = self.apnn_conf_data["averging mu"]
        self.number_stored_activations = self.apnn_conf_data["number_of_stored_activations"]
        self.stored_activations = torch.zeros(body.feature_dim*self.number_stored_activations).float().cuda()
        self.to(Config.DEVICE)

    def compute_EMA(self,mu,x,last_average):
        """Function computes exponential decaying moving average (EMA)
        inputs: mu- parameter determining how much current data is affecting the average (i.e. the lower the more effect current data has)
                x - current data point
        last_average - past average to be updated
        outputs: new_average - updated average"""
        new_average = (1-self.mu)*x + self.mu*last_average
        return new_average.detach()

    def predict(self, x, steps, to_numpy = False):
        """Function is used for prediction of next action agent should take
        input: x - current state
               steps - current step
               to_numpy - flag deciding if output tensor should be casted as a numpy array
        outputs: y - Q-values from DQN
        """
        features = self.body(tensor(x))
        apnn_features = features.detach()
        random_mask = torch.cuda.FloatTensor(apnn_features.shape).uniform_()>0.8
        apnn_features = apnn_features+((random_mask.float()*(torch.randn(apnn_features.shape).cuda()*0.1)))
        if to_numpy:
            self.running_average = self.compute_EMA(self.mu,features,self.running_average)
        if type(self.running_average) == np.ndarray:
            self.running_average = torch.stack(self.running_average.tolist()).squeeze()
        self.y_apnn_in = F.tanh(apnn_features.detach().cuda()-self.running_average)
        self.y_apnn_out = F.sigmoid(self.apnn_head_output(self.y_apnn_in).detach())
        y_dqn_out = F.sigmoid(self.fc_head(F.relu(features)))
        self.y_dqn_out_detached = y_dqn_out.detach()
        y = y_dqn_out
        self.y_apnn_out.detach_()
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

    def predict_apnn(self, x , to_numpy = False):
        """Function used for deubbing where only MOHA head is used """
        features = tensor(x)
        apnn_features = features.detach()
        random_mask = torch.cuda.FloatTensor(apnn_features.shape).uniform_()>0.9
        apnn_features = apnn_features+(random_mask.float()*torch.randn(apnn_features.shape).cuda()*0.05)
        if to_numpy:
            self.running_average = self.compute_EMA(self.mu,features,self.running_average)
        if type(self.running_average) == np.ndarray:
            self.running_average = torch.stack(self.running_average.tolist()).squeeze()
        self.y_apnn_in = F.tanh(features.detach().unsqueeze(0).cuda()-self.running_average)
        self.y_apnn_out = self.apnn_head_output(self.y_apnn_in).detach()
        y = self.y_apnn_out
        self.y_apnn_out.detach_()
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

    def running_matrix_activation_storage(self,activation_multiplication):
        """This function can be used to use multiple activations to decide most correlated features/actions. Currently disused"""
        upper_limit_keep = (self.body.feature_dim*(self.number_stored_activations-1))
        self.stored_activations = self.stored_activations[0:upper_limit_keep]
        self.stored_activations = torch.cat((activation_multiplication,self.stored_activations),0)
        return 0

    def compute_eligbility_traces_action(self,value,action):
        """Function which computes eligbility traces based on action agent takes
        inputs: value - value as output by the network. Currently value from MOHA head is used, although either can be used
        action - vector indicating action take by an agent
        """
        value_tensor = torch.from_numpy(value).cuda().unsqueeze(0)
        activation_multiplication_value = torch.transpose(value_tensor,0,1)
        mask = torch.ones(activation_multiplication_value.shape).byte()
        mask[action] = 0
        activation_multiplication_value[mask] = 0
        Activation_multiplication = torch.mm(activation_multiplication_value,self.y_apnn_in)
        self.running_matrix_activation_storage(Activation_multiplication[action,:])
        positive_feature_number = torch.sum(self.stored_activations!= 0).item()
        self.no_top_features = int(math.ceil(self.apnn_conf_data["percent_of_top_features"]*positive_feature_number))
        self.no_bottom_features = int(math.ceil(self.apnn_conf_data["percent_of_bottom_features"]*positive_feature_number))
        delete,top_features_indecies = torch.topk(self.stored_activations,self.no_top_features)
        activation_multiplication_bottom_features = self.stored_activations.clone()
        activation_multiplication_bottom_features[activation_multiplication_bottom_features == 0] = 50
        delete,bottom_features_indecies = torch.topk(activation_multiplication_bottom_features,self.no_bottom_features,largest = False)
        activation_mask = torch.zeros(Activation_multiplication.shape).cuda()
        bottom_features_indecies = torch.masked_select(bottom_features_indecies,bottom_features_indecies<self.body.feature_dim)
        top_features_indecies = torch.masked_select(top_features_indecies,top_features_indecies<self.body.feature_dim)
        activation_mask[action,top_features_indecies] = 1
        activation_mask[action,bottom_features_indecies] = -1
        Theta_alpha = torch.where(activation_mask>self.theta_high,self.alpha_matrix,self.zeros_matrix)
        Theta_beta = torch.where(activation_mask<self.theta_low,self.beta_matrix,self.zeros_matrix)
        indexes = activation_mask[action] == 0
        activation_mask[action,indexes] = 50
        Theta_zeta = torch.where(activation_mask == 50,self.zeta_matrix,self.zeros_matrix)
        self.Theta = Theta_alpha+Theta_beta+Theta_zeta
        self.compute_eligbility_traces(1,0)

    def compute_eligbility_traces(self,delta_step,index):
        """this function updates elgbility traces once appropriate theta has been computed
        input: delta_stepa - step increase from previous time elgibilkity traces have been computed. Note: code tested with value of 1 only
        index - only used for update of multiple elgibility traces, will be removed in future versions.
        """
        if type(self.Eligbility_traces) is np.ndarray:
            self.Eligbility_traces[index] = self.Eligbility_traces[index]*math.exp(-delta_step/self.tau_E)+self.Theta[index]
        else:
            temp_elgibility_trace = self.Eligbility_traces
            self.Eligbility_traces = self.Eligbility_traces*math.exp(-delta_step/self.tau_E)+self.Theta

    def update(self,rewards,steps,episodes):
        """Here weights are updated for MOHA
        input: rewards - reward agent scored just before given update
        steps - steps since last updates
        episodes - current episode, note this parameter is currently disused
        """
        episode = episodes
        reward = rewards
        self.step = steps
        self.modulation = reward+self.baseline_modulation
        weight_change = self.modulation*self.Eligbility_traces
        self.apnn_head_output.weight.data = (self.apnn_head_output.weight.data+weight_change).clamp(-1,1)
        self.step_previous = self.step
    def reset(self):
        self.step_previous = -1
        self.Eligbility_traces = torch.zeros(self.output_dim,self.body.feature_dim).cuda()
        self.modulation = 0

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.reset_flag = False
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy = False):
        #self.body.reset_flag = self.reset_flag
        phi,_ = self.body(tensor(x))
        print(type(phi))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim = True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy = False):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim = -1)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob

class CategoricalNetMod(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNetMod, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, x_mem, to_numpy = False):
        phi = self.body(tensor(x), tensor(x_mem))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim = -1)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy = False):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        if to_numpy:
            quantiles = quantiles.cpu().detach().numpy()
        return quantiles

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim = -1)
        return q, beta, log_pi

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body = None,
                 actor_body = None,
                 critic_body = None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def predict(self, obs, to_numpy = False):
        phi = self.feature(obs)
        action = self.actor(phi)
        if to_numpy:
            return action.cpu().detach().numpy()
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))

class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body = None,
                 actor_body = None,
                 critic_body = None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.to(Config.DEVICE)

    def predict(self, obs, action = None, to_numpy = False):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim = 1, keepdim = True)
        return action, log_prob, tensor(np.zeros((log_prob.size(0), 1))), v

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body = None,
                 actor_body = None,
                 critic_body = None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def predict(self, obs, action = None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits = logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1), v
