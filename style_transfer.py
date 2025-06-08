import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy

# Image loader
def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

# Gram Matrix
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        features = input.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

# Normalization Module
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Style Transfer function
def run_style_transfer(content_img_path, style_img_path, output_img_path,
                       num_steps=300, style_weight=1e6, content_weight=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imsize = 512 if torch.cuda.is_available() else 256

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_img = image_loader(content_img_path, imsize).to(device)
    style_img = image_loader(style_img_path, imsize).to(device)
    input_img = content_img.clone()

    # Loss layers
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    content_losses = []
    style_losses = []

    # Build the model
    model = nn.Sequential(normalization)
    gram = GramMatrix()

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_losses.append((name, target))

        if name in style_layers:
            target_feature = model(style_img).detach()
            target_gram = gram(target_feature)
            style_losses.append((name, target_gram))

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    mse_loss = nn.MSELoss()
    run = [0]

    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model_output = {}
            x = input_img
            for name, layer in model._modules.items():
                x = layer(x)
                model_output[name] = x

            content_score = 0
            style_score = 0

            # Compute content loss
            for name, target in content_losses:
                content_score += content_weight * mse_loss(model_output[name], target)

            # Compute style loss
            for name, target_gram in style_losses:
                current_gram = gram(model_output[name])
                style_score += style_weight * mse_loss(current_gram, target_gram)

            loss = content_score + style_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}/{num_steps} | Content Loss: {content_score.item():.4f} | Style Loss: {style_score.item():.4f}")

            return loss

        optimizer.step(closure)

    # Save output image
    input_img.data.clamp_(0, 1)
    unloader = transforms.ToPILImage()
    image = input_img.cpu().clone().squeeze(0)
    image = unloader(image)
    image.save(output_img_path)
