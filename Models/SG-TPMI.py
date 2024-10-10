import torch
import torch.nn as nn

class SGTPMI(nn.Module):
    def __init__(
            self,
            tcr_padding_len,
            peptide_padding_len,
            map_num,
            dropout_prob,
            hidden_channel,
    ):
        super(SGTPMI, self).__init__()
        self.map_num = map_num
        self.dropout_prob = dropout_prob
        self.aa_in_channels = [map_num, map_num, map_num, map_num]
        self.aa_out_channels = [hidden_channel, hidden_channel, hidden_channel, hidden_channel]
        self.aa_kernel_size = [[3, 3], [5, 5], [9, 9], [11, 11]]
        self.in_channels = [map_num, map_num, map_num]
        self.out_channels = [hidden_channel, hidden_channel, hidden_channel]
        self.b_batch_norm = nn.BatchNorm1d(tcr_padding_len * peptide_padding_len)
        self.a_batch_norm = nn.BatchNorm1d(tcr_padding_len * peptide_padding_len)

        self.encoder = [
            nn.Sequential(
                self.CNN2D(
                    input_channel=self.aa_in_channels[index],
                    output_channel=self.aa_out_channels[index],
                    kernel_size=kernel_size,
                    act_fn=nn.ReLU(),
                    batch_norm=True,
                    dropout2d=self.dropout_prob,
                ),
                self.CNN2D(
                    input_channel=hidden_channel,
                    output_channel=hidden_channel // 2,
                    kernel_size=kernel_size,
                    act_fn=nn.ReLU(),
                    batch_norm=True,
                    # max_pooling=False,
                    dropout2d=dropout_prob,
                ),
                self.CNN2D(
                    input_channel=hidden_channel // 2,
                    output_channel=1,
                    kernel_size=kernel_size,
                    act_fn=nn.ReLU(),
                    batch_norm=True,
                    # max_pooling=True,
                    dropout2d=dropout_prob,
                ),
            ) for index, kernel_size in enumerate(self.aa_kernel_size)]


        self.aencoder = [
            nn.Sequential(
                self.CNN2D(
                    input_channel=self.aa_in_channels[index],
                    output_channel=self.aa_out_channels[index],
                    kernel_size=kernel_size,
                    act_fn=nn.ReLU(),
                    batch_norm=True,
                    dropout2d=self.dropout_prob,
                ),
                self.CNN2D(
                    input_channel=hidden_channel,
                    output_channel=hidden_channel // 2,
                    kernel_size=kernel_size,
                    act_fn=nn.ReLU(),
                    batch_norm=True,
                    # max_pooling=False,
                    dropout2d=dropout_prob,
                ),
                self.CNN2D(
                    input_channel=hidden_channel // 2,
                    output_channel=1,
                    kernel_size=kernel_size,
                    act_fn=nn.ReLU(),
                    batch_norm=True,
                    # max_pooling=True,
                    dropout2d=dropout_prob,
                ),
            ) for index, kernel_size in enumerate(self.aa_kernel_size)]

        self.task2_fc = nn.Sequential(nn.Linear(tcr_padding_len*peptide_padding_len*2, 1),
                                      nn.Sigmoid())

    def cnnencoder(self, data, encoder):
        for index, layer in enumerate(encoder):
            if index == 0:
                interaction_maps = layer(data.permute(0, 3, 1, 2))
            else:
                interaction_maps += layer(data.permute(0, 3, 1, 2))
        interaction_maps = torch.squeeze(interaction_maps, axis=1)
        return interaction_maps

    def forward(self, data):
        bpembed = self.cnnencoder(data['b3p_map'], self.encoder)
        bpembed = self.b_batch_norm(bpembed.reshape(bpembed.shape[0], -1))
        apembed = self.cnnencoder(data['a3p_map'], self.aencoder)
        apembed = self.a_batch_norm(apembed.reshape(apembed.shape[0], -1))
        cdr3 = torch.concat([bpembed, apembed], dim=1)
        out = self.fc(cdr3)
        return out

    def CNN2D(
            self,
            input_channel,
            output_channel,
            kernel_size,
            act_fn=nn.ReLU(),
            batch_norm=False,
            max_pooling=False,
            dropout2d=0.0,
    ):
        return nn.Sequential(
            nn.Conv2d(
                input_channel,
                output_channel,
                kernel_size,
                padding=[kernel_size[0] // 2, kernel_size[1] // 2]  # pad for valid conv.
            ),
            act_fn,
            nn.Dropout2d(p=dropout2d),
            nn.MaxPool2d(kernel_size=(2, 2)) if max_pooling else nn.Identity(),
            nn.BatchNorm2d(output_channel) if batch_norm else nn.Identity()
        )