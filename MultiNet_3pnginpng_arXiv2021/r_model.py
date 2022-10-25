import torch
import torch.nn as nn
import torch.nn.functional as F


class RevealNetwork1(nn.Module):
    def __init__(self):
        super(RevealNetwork1, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=3, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv4 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv7 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv9 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv10 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv11 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv12 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv13 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv14 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv15 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.final_layer = nn.Conv2d(
            in_channels=65, out_channels=3, kernel_size=(3, 3), stride=1, padding=1
        )

    def forward(self, hidden_image):

        output_1 = F.relu(self.conv1(hidden_image))
        output_2 = F.relu(self.conv2(hidden_image))
        output_3 = F.relu(self.conv3(hidden_image))
        concat_1 = torch.cat([output_1, output_2, output_3], dim=1)

        output_4 = F.relu(self.conv4(concat_1))
        output_5 = F.relu(self.conv5(concat_1))
        output_6 = F.relu(self.conv6(concat_1))
        concat_2 = torch.cat([output_4, output_5, output_6], dim=1)

        output_7 = F.relu(self.conv7(concat_2))
        output_8 = F.relu(self.conv8(concat_2))
        output_9 = F.relu(self.conv9(concat_2))
        concat_3 = torch.cat([output_7, output_8, output_9], dim=1)

        output_10 = F.relu(self.conv10(concat_3))
        output_11 = F.relu(self.conv11(concat_3))
        output_12 = F.relu(self.conv12(concat_3))
        concat_4 = torch.cat([output_10, output_11, output_12], dim=1)

        output_13 = F.relu(self.conv13(concat_4))
        output_14 = F.relu(self.conv14(concat_4))
        output_15 = F.relu(self.conv15(concat_4))
        concat_5 = torch.cat([output_13, output_14, output_15], dim=1)

        output_revealed_image = F.relu(self.final_layer(concat_5))

        return output_revealed_image


class RevealNetwork2(nn.Module):
    def __init__(self):
        super(RevealNetwork2, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=3, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv4 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv7 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv9 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv10 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv11 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv12 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv13 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv14 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv15 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.final_layer = nn.Conv2d(
            in_channels=65, out_channels=3, kernel_size=(3, 3), stride=1, padding=1
        )

    def forward(self, hidden_image):

        output_1 = F.relu(self.conv1(hidden_image))
        output_2 = F.relu(self.conv2(hidden_image))
        output_3 = F.relu(self.conv3(hidden_image))
        concat_1 = torch.cat([output_1, output_2, output_3], dim=1)

        output_4 = F.relu(self.conv4(concat_1))
        output_5 = F.relu(self.conv5(concat_1))
        output_6 = F.relu(self.conv6(concat_1))
        concat_2 = torch.cat([output_4, output_5, output_6], dim=1)

        output_7 = F.relu(self.conv7(concat_2))
        output_8 = F.relu(self.conv8(concat_2))
        output_9 = F.relu(self.conv9(concat_2))
        concat_3 = torch.cat([output_7, output_8, output_9], dim=1)

        output_10 = F.relu(self.conv10(concat_3))
        output_11 = F.relu(self.conv11(concat_3))
        output_12 = F.relu(self.conv12(concat_3))
        concat_4 = torch.cat([output_10, output_11, output_12], dim=1)

        output_13 = F.relu(self.conv13(concat_4))
        output_14 = F.relu(self.conv14(concat_4))
        output_15 = F.relu(self.conv15(concat_4))
        concat_5 = torch.cat([output_13, output_14, output_15], dim=1)

        output_revealed_image = F.relu(self.final_layer(concat_5))

        return output_revealed_image


class RevealNetwork3(nn.Module):
    def __init__(self):
        super(RevealNetwork3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=3, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv4 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv7 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv9 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv10 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv11 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv12 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.conv13 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv14 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv15 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2
        )

        self.final_layer = nn.Conv2d(
            in_channels=65, out_channels=3, kernel_size=(3, 3), stride=1, padding=1
        )

    def forward(self, hidden_image):
        output_1 = F.relu(self.conv1(hidden_image))
        output_2 = F.relu(self.conv2(hidden_image))
        output_3 = F.relu(self.conv3(hidden_image))
        concat_1 = torch.cat([output_1, output_2, output_3], dim=1)

        output_4 = F.relu(self.conv4(concat_1))
        output_5 = F.relu(self.conv5(concat_1))
        output_6 = F.relu(self.conv6(concat_1))
        concat_2 = torch.cat([output_4, output_5, output_6], dim=1)

        output_7 = F.relu(self.conv7(concat_2))
        output_8 = F.relu(self.conv8(concat_2))
        output_9 = F.relu(self.conv9(concat_2))
        concat_3 = torch.cat([output_7, output_8, output_9], dim=1)

        output_10 = F.relu(self.conv10(concat_3))
        output_11 = F.relu(self.conv11(concat_3))
        output_12 = F.relu(self.conv12(concat_3))
        concat_4 = torch.cat([output_10, output_11, output_12], dim=1)

        output_13 = F.relu(self.conv13(concat_4))
        output_14 = F.relu(self.conv14(concat_4))
        output_15 = F.relu(self.conv15(concat_4))
        concat_5 = torch.cat([output_13, output_14, output_15], dim=1)

        output_revealed_image = F.relu(self.final_layer(concat_5))

        return output_revealed_image
