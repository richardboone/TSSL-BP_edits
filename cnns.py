import torch
import torch.nn as nn
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.linear as linear
import functions.loss_f as f


class Network(nn.Module):
    def __init__(self, network_config, layers_config, input_shape):
        super(Network, self).__init__()
        self.layers = []
        self.network_config = network_config
        self.layers_config = layers_config

        self.n_steps = self.network_config['n_steps']
        self.tau_s = self.network_config['tau_s']
        self.syn_a = torch.zeros(1, 1, 1, 1, self.n_steps).cuda()
        self.syn_a[..., 0] = 1
        for t in range(self.n_steps-1):
            self.syn_a[..., t+1] = self.syn_a[..., t] - self.syn_a[..., t] / self.tau_s 
        self.syn_a /= self.tau_s

        parameters = []
        print("Network Structure:")
        for key in layers_config:
            c = layers_config[key]
            if c['type'] == 'conv':
                self.layers.append(conv.ConvLayer(network_config, c, key, input_shape))
                input_shape = self.layers[-1].out_shape
                parameters.append(self.layers[-1].get_parameters())
            elif c['type'] == 'linear':
                self.layers.append(linear.LinearLayer(network_config, c, key, input_shape))
                input_shape = self.layers[-1].out_shape
                parameters.append(self.layers[-1].get_parameters())
            elif c['type'] == 'pooling':
                self.layers.append(pooling.PoolLayer(network_config, c, key, input_shape))
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'dropout':
                self.layers.append(dropout.DropoutLayer(c, key))
            else:
                raise Exception('Undefined layer type. It is: {}'.format(c['type']))
        self.my_parameters = nn.ParameterList(parameters)
        print("-----------------------------------------")

    def forward(self, spike_input, epoch, is_train):
        spikes = f.psp(spike_input, self.network_config)
        skip_spikes = {}
        assert self.network_config['model'] == "LIF"
        
        for l in self.layers:
            if l.type == "dropout":
                if is_train:
                    spikes = l(spikes)
            elif self.network_config["rule"] == "TSSLBP":
                spikes = l.forward_pass(spikes, epoch, n_steps, tau_s, syn_a)
            else:
                raise Exception('Unrecognized rule type. It is: {}'.format(self.network_config['rule']))
        return spikes

    def get_parameters(self):
        return self.my_parameters

    def weight_clipper(self):
        for l in self.layers:
            l.weight_clipper()

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()
