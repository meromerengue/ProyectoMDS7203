from tqdm import tqdm
from IPython import display
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

def make_gen_block(input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
    if not final_layer:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            nn.Tanh()
        )

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.embed = nn.Embedding(10, z_dim)
        self.gen = nn.Sequential(
            make_gen_block(z_dim, hidden_dim * 4, kernel_size=3, stride=2),
            make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2),
            make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=2, final_layer=True),
        )

    def vec_reshape(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def get_emb(self, labels):
        return self.vec_reshape(self.embed(labels))

    def forward(self, noise, labels=None):
        x = self.vec_reshape(noise)
        if labels is not None: x = x * self.get_emb(labels)
        return self.gen(x)
    
    def intepolation(self, label1, label2, n_samples=10, device='cpu'):
        if not isinstance(label1, torch.Tensor): label1 = torch.tensor(label1, device=device).view(1, 1)
        if not isinstance(label2, torch.Tensor): label2 = torch.tensor(label2, device=device).view(1, 1)
        noise = get_noise(n_samples, self.z_dim, device=device)
        label1 = self.get_emb(label1).squeeze()
        label2 = self.get_emb(label2).squeeze()
        inter_vecs = torch.zeros(n_samples, self.z_dim, device=device)
        for i in range(n_samples):
            inter_vecs[i] = label1 + (label2 - label1) * i / (n_samples - 1)
        return self.forward(inter_vecs * noise)
    
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), save=False, name=None, epoch=None):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)

    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    if save:
        plt.savefig('{}_epoch_{}.png'.format(name, epoch))
    plt.show()
    
def get_sample(labels, device = 'cpu'):
    gen = Generator(z_dim=64)
    gen.load_state_dict(torch.load("proyecto/wcgangp.unknown", map_location = torch.device(device)))
    noise = get_noise(len(labels), 64)
    return show_tensor_images(gen(noise, labels), len(labels))