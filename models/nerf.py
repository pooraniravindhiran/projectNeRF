import torch
import torch.nn as nn

class NeRF(nn.Module):
    """
    Input is of shape - (h * w * num_samples, (2 * num_posencoding_functions * 3)+(2 * num_direncoding_functions * 3))
    Output is of shape - (h * w * num_samples, 4) 
    where the 4-dim vector represents the RGB information and density of that respective 3D sample point.

    """
    def __init__(self, num_pos_encoding_func, num_dir_encoding_func, use_viewdirs):
        super(NeRF, self).__init__()
        self.num_layers = 8
        self.skip_layer = 4
        self.filter_size = 256
        self.output_size = 4
        self.use_viewdirs = use_viewdirs
        self.input_pos_dim = 2 * num_pos_encoding_func * 3
        self.input_dir_dim = 2 * num_dir_encoding_func * 3

        # Define the set of fully connected layers with input layer
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.input_pos_dim, self.filter_size, dtype=torch.float64))

        # Define the hidden layers in the set with one layer having a residual connection with input
        for layer in range(self.num_layers-1):
            if layer == self.skip_layer-1:
                self.fc_layers.append(nn.Linear(self.input_pos_dim + self.filter_size, self.filter_size, dtype=torch.float64))
            else:
                self.fc_layers.append(nn.Linear(self.filter_size, self.filter_size, dtype=torch.float64))

        # TODO: check why is this 
        if self.use_viewdirs:
            # Append separate output layers for RGB and density
            self.alpha_layer = nn.Linear(self.filter_size, 1, dtype=torch.float64)
            self.fc_dir1_layer = nn.Linear(self.filter_size, self.filter_size, dtype=torch.float64)
            self.fc_dir2_layer = nn.Linear(self.input_dir_dim + self.filter_size, self.filter_size//2, dtype=torch.float64)
            self.rgb_layer = nn.Linear(self.filter_size//2, self.output_size-1, dtype=torch.float64)

        # Append output fully connected layer
        else:
            self.out_layer = nn.Linear(self.filter_size, self.output_size, dtype=torch.float64)

    def forward(self, x):
        # TODO: isn't input 6D? why are they saying 5D?
        input_pos, input_dir = x[...,:self.input_pos_dim], x[..., self.input_pos_dim:]

        x = input_pos
        for i, layer in enumerate(self.fc_layers):
            if i == self.skip_layer:
                x = self.fc_layers[i](torch.cat((input_pos, x), dim=-1))
            else:
                x = self.fc_layers[i](x)
        x = nn.functional.relu(x)

        if self.use_viewdirs:
            alpha = self.alpha_layer(x)
            feature = self.fc_dir1_layer(x)
            x = self.fc_dir2_layer(torch.cat((input_dir, feature), dim=-1))
            x = nn.functional.relu(x) # TODO: Isn't this sigmoid in paper
            rgb = self.rgb_layer(x)
            x = torch.cat((rgb, alpha), dim=-1)
        else:
            x = self.out_layer(x)

        return x