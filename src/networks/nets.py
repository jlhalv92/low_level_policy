import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, 0.01)
        nn.init.constant_(self._h3.bias, 0.0)

    def forward(self, state, **kwargs):
        features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.elu(self._h2(features1))
        a = self._h3(features2)

        return a


class ValueNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, 1.0)
        nn.init.constant_(self._h3.bias, 0.0)

    def forward(self, state, **kwargs):
        features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.elu(self._h2(features1))
        a = self._h3(features2)

        return a



class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)

class PolicyNetworkStudent(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, 2*n_features)
        self._h2 = nn.Linear(2*n_features, n_features)
        self._h3 = nn.Linear(n_features, int(n_features/2))
        self._h4 = nn.Linear(int(n_features/2), n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, np.sqrt(2))
        nn.init.constant_(self._h3.bias, 0.0)
        nn.init.orthogonal_(self._h4.weight, 0.01)
        nn.init.constant_(self._h4.bias, 0.0)

    def forward(self, state, **kwargs):
        features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.elu(self._h2(features1))
        features3 = F.elu(self._h3(features2))
        a = self._h4(features3)

        return a

class ActorBiNetwork(nn.Module):
    def __init__(self, input_shape,
                 output_shape,
                 n_features,
                 size_propioception=4,
                 size_exteroception=None, **kwargs):
        super(ActorBiNetwork, self).__init__()

        self.size_propioception = size_propioception
        if size_exteroception:
            self.size_exteroception = size_exteroception
        else:
            self.size_exteroception = input_shape[0] - size_propioception
        n_output = output_shape[0]

        # Intrinsic Processing (Expand 4 → 10)
        self._h_intrinsic = nn.Linear(self.size_propioception, 10)
        nn.init.xavier_uniform_(self._h_intrinsic.weight, gain=nn.init.calculate_gain('relu'))

        # Extrinsic Processing (54 → n_features → 20)
        self._h_extrinsic_1 = nn.Linear(self.size_exteroception, n_features)
        self._h_extrinsic_2 = nn.Linear(n_features, 20)  # Increased from 10 → 20

        nn.init.xavier_uniform_(self._h_extrinsic_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h_extrinsic_2.weight, gain=nn.init.calculate_gain('relu'))

        # State Embedding (Intrinsic: 10 + Extrinsic: 20 = 30)
        self._h1 = nn.Linear(30, n_features)
        self._h2 = nn.Linear(n_features, n_features)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))

        # Output Layer
        self._out = nn.Linear(n_features, n_output)
        # nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain('linear'))

        nn.init.uniform_(self._out.weight, -0.05, 0.05)  # Small uniform values
        nn.init.zeros_(self._out.bias)  # Initialize bias to zero

    def forward(self, state):
        state_intrinsic = state[:,:self.size_propioception]
        state_extrinsic_privilege = state[:,self.size_propioception:]

        # Intrinsic Processing
        intrinsic_transformed = F.relu(self._h_intrinsic(state_intrinsic.float()))

        # Extrinsic Processing
        features_extrinsic_1 = F.relu(self._h_extrinsic_1(torch.squeeze(state_extrinsic_privilege, 1).float()))
        features_extrinsic_2 = F.relu(self._h_extrinsic_2(features_extrinsic_1))

        # Concatenate Intrinsic and Extrinsic Features
        state_embedding = torch.cat((intrinsic_transformed, features_extrinsic_2.float()), dim=1)

        # Fully Connected Layers
        features1 = F.relu(self._h1(torch.squeeze(state_embedding, 1).float()))
        features2 = F.relu(self._h2(features1))

        # Output Action
        a = self._out(features2)

        return a

class VisualActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(VisualActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._exteroception = models.resnet18(pretrained=True)
        self._exteroception.fc = torch.nn.Linear(self._exteroception.fc.in_features, 20)

        self._h1 = nn.Linear(n_input + 20, n_features)
        self._h2 = nn.Linear(n_features, n_features)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))

        self._out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, state_img):
        features_img = self._exteroception(torch.squeeze(state_img, 1).float())
        all_features = torch.cat((torch.squeeze(state, 1).float(), features_img), dim=1)

        features1 = F.relu(self._h1(all_features))
        features2 = F.relu(self._h2(features1))

        a = self._out(features2)

        return a

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))

        self._out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):

        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._out(features2)

        return a



class ActorObstacleNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorObstacleNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))

        self._out = nn.Linear(n_features, n_output)

        nn.init.uniform_(self._out.weight, -0.5, 0.5)  # Small uniform values
        nn.init.zeros_(self._out.bias)  # Initialize bias to zero

    def forward(self, state):

        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._out(features2)

        return a

class ActorNetworkObjectDect(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetworkObjectDect, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input-300, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features+10, n_features)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))


        self._h_object_detection_1 = nn.Linear(300, n_features)
        self._h_object_detection_2 = nn.Linear(n_features, n_features)
        self.object_detection_out = nn.Linear(n_features, 10)

        nn.init.xavier_uniform_(self._h_object_detection_1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h_object_detection_2.weight,
                                gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.object_detection_out.weight,
                                gain=nn.init.calculate_gain('linear'))

        self._out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):

        state_control = state[:,:11]
        features1 = F.relu(self._h1(torch.squeeze(state_control, 1).float()))
        features2 = F.relu(self._h2(features1))

        state_obstacles = state[:,11:]
        features_object_detection_1 = F.relu(self._h_object_detection_1(torch.squeeze(state_obstacles, 1).float()))
        features_object_detection_2 = F.relu(self._h_object_detection_2(features_object_detection_1))
        emb_object_detection = self.object_detection_out(features_object_detection_2)

        state_emb = torch.cat((features2.float(), emb_object_detection.float()), dim=1)

        features3 = F.relu(self._h3(state_emb))

        a = self._out(features3)

        return a





class CriticGO1Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, np.sqrt(2))
        nn.init.constant_(self._h3.bias, 0.0)
        nn.init.orthogonal_(self._out.weight, 0.01)
        nn.init.constant_(self._out.bias, 0.0)

    def forward(self, state,  **kwargs):
        features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.elu(self._h2(features1))
        features_3 =  F.elu(self._h3(features2))
        q = self._out(features_3)

        return torch.squeeze(q)

class ActorGO1Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorGO1Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input,  n_features)
        self._h2 = nn.Linear( n_features,  n_features)
        self._h3 = nn.Linear( n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, np.sqrt(2))
        nn.init.constant_(self._h3.bias, 0.0)
        nn.init.orthogonal_(self._out.weight, 0.01)
        nn.init.constant_(self._out.bias, 0.0)

    def forward(self, state):
        features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.elu(self._h2(features1))
        features_3 = F.elu(self._h3(features2))
        a = self._out(features_3)

        return a


# class CriticParkourNetwork(nn.Module):
#     def __init__(self, input_shape, output_shape, n_features, **kwargs):
#         super().__init__()
#
#         n_input = input_shape[-1]
#         n_output = output_shape[0]
#
#         self._h1 = nn.Linear(n_input, n_features)
#         self._h2 = nn.Linear(n_features, n_features)
#         self._h3 = nn.Linear(n_features, n_output)
#
#         nn.init.xavier_uniform_(self._h1.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h2.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h3.weight,
#                                 gain=nn.init.calculate_gain('linear'))
#
#     def forward(self, state, action):
#         state_action = torch.cat((state.float(), action.float()), dim=1)
#         features1 = F.relu(self._h1(state_action))
#         features2 = F.relu(self._h2(features1))
#         q = self._h3(features2)
#
#         return torch.squeeze(q)


class ActorParkourNetworkGO1(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, map_features=137, out_env_encoder=10, **kwargs):
        super(ActorParkourNetworkGO1, self).__init__()

        n_input = input_shape[-1]

        n_output = output_shape[0]
        out_env_encoder = out_env_encoder
        self._map_features_index = map_features - n_input
        self._prop_idx = n_input - map_features


        self._map_features = map_features
        self._h1_ex = nn.Linear( self._map_features, 2*self._map_features)
        self._h2_ex = nn.Linear(2*self._map_features, int(n_features/2))
        self._h3_ex = nn.Linear(int(n_features/2), out_env_encoder)

        self._prop_input = n_input - map_features + out_env_encoder
        self._h1 = nn.Linear(self._prop_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

        nn.init.xavier_uniform_(self._h1_ex.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2_ex.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3_ex.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):

        state_e = state[:,-self._map_features:]

        features_e_1 = F.relu(self._h1_ex(torch.squeeze(state_e, 1).float()))
        features_e_2 = F.relu(self._h2_ex(features_e_1))
        emb_0 = self._h3_ex(features_e_2)
        s_p = state[:,:self._prop_idx]

        s = torch.cat((s_p.float(), emb_0.float()), dim=1)

        features1 = F.relu(self._h1(torch.squeeze(s, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a

class CriticParkourNetworkGO1(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, map_features=137, out_env_encoder=10, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]
        actions = 12
        out_env_encoder = out_env_encoder
        self._map_features = map_features
        self._map_features_index = map_features - n_input
        self._prop_idx = n_input - map_features

        self._h1_ex = nn.Linear( self._map_features, 2*self._map_features)
        self._h2_ex = nn.Linear(2*self._map_features, int(n_features/2))
        self._h3_ex = nn.Linear(int(n_features/2), out_env_encoder)

        self._prop_input = n_input - map_features + out_env_encoder + actions
        self._h1 = nn.Linear(self._prop_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

        nn.init.xavier_uniform_(self._h1_ex.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2_ex.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3_ex.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):

        state_e = state[:,-self._map_features:]
        features_e_1 = F.relu(self._h1_ex(torch.squeeze(state_e, 1).float()))
        features_e_2 = F.relu(self._h2_ex(features_e_1))
        emb_0 = self._h3_ex(features_e_2)

        s_p = state[:,:self._prop_idx]
        state = torch.cat((s_p.float(), emb_0.float()), dim=1)
        state_action = torch.cat((state, action.float()), dim=1)



        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)