
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, n_features, n_input, n_output,  **kwargs):
        super(ResidualBlock, self).__init__()

        self.layer_1 = nn.Linear(n_input, n_features)
        self.layer_out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self.layer_1.weight,
                                gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.layer_out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, value, state, action):
        state_action_value = torch.cat((state.float(), action.float(), value.float()), dim=1)

        Q = value
        features1 = F.relu(self.layer_1(state_action_value))
        rho = self.layer_out(features1)
        Q += rho

        return Q


class Encoder(nn.Module):
    def __init__(self, n_features, encoder_output=10,  **kwargs):
        super(Encoder, self).__init__()
        n_features2 = int(n_features/2)
        n_features3 = int(n_features/4)

        self.layer_1 = nn.Linear(n_features, n_features2)
        self.layer_2 = nn.Linear(n_features2, n_features3)
        self.layer_out = nn.Linear(n_features3, encoder_output)

        nn.init.xavier_uniform_(self.layer_1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_out.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, features):


        features1 = F.relu(self.layer_1(features))
        features2 = F.relu(self.layer_2(features1))
        features3 = F.relu(self.layer_out(features2))

        return features3

class QResnetReuse(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        encoder_output = 10
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        self._residual = ResidualBlock(n_features, n_input + encoder_output, n_output)
        self._encoder = Encoder(n_features, encoder_output)



    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        q_0 = self._out(features3)

        encoder_out = self._encoder(features3)
        q_1 = self._residual(q_0, state, action, encoder_out)

        return torch.squeeze(q_1)


class QResnetReuseFeatures(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        encoder_output = 10
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        self._residual = ResidualBlock(n_features, n_input + n_features, n_output)



    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        q_0 = self._out(features3)

        q_1 = self._residual(q_0, state, action, features3)

        return torch.squeeze(q_1)


class QResnetReuseq(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        self._residual = ResidualBlock(n_features, n_input + n_output, n_output)



    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        q_0 = self._out(features3)

        q_1 = self._residual(q_0, state, action)

        return torch.squeeze(q_1)

class QResnetReuseFeaturesReduction(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        encoder_output = 10
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._reduction = nn.Linear(n_features, encoder_output)

        self._out = nn.Linear(n_features, n_output)
        self._residual = ResidualBlock(n_features, n_input + encoder_output, n_output)



    def forward(self, state, action):

        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))
        q_0 = self._out(features3)
        reduces_features = self._reduction(features3)
        q_1 = self._residual(q_0, state, action, reduces_features)

        return torch.squeeze(q_1)

