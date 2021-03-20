import argparse
import torch
import utils
import os
from torchvision import transforms
from models import transformer

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser  = argparse.ArgumentParser(description='Args of Train')
parser.add_argument('--style_path', type=str, default='pretrained/transformer_weight.pth', help='load model path')
parser.add_argument('--frame_folder', type=str, default='data/images/test/', help='image source')
parser.add_argument('--save_folder', type=str, default='results/images/', help='result save')
parser.add_argument('--PRESERVE_COLOR', type=str2bool, default=False, help='preserve original color')
parser.add_argument('--cuda', type=str2bool, default=False, help='enables CUDA training')
opts    = parser.parse_args()
device  = ("cuda:0" if opts.cuda else "cpu")

#STYLE_TRANSFORM_PATH = "transforms/udnie_aggressive.pth"
#PRESERVE_COLOR = False
def stylize_folder(style_path, frame_folder, save_folder, batch_size=1):
    # Device
    #device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Image loader
    '''transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    image_dataset   = utils.ImageFolderWithPaths(frame_folder, transform=transform)
    image_loader    = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size)'''

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(style_path))
    net = net.to(device)

    # Stylize batches of images
    images = [img for img in os.listdir(frame_folder) if img.endswith(".jpg")]
    with torch.no_grad():
        #for content_batch, _, path in image_loader:
        for image_name in images:
            #torch.cuda.empty_cache() # Free-up unneeded cuda memory
            torch.cuda.empty_cache() if opts.cuda else None

            ''''# Generate image
            generated_tensor = net(content_batch.to(device)).detach()
            # Save images
            for i in range(len(path)):
                generated_image = utils.ttoi(generated_tensor[i])
                if (opts.PRESERVE_COLOR):
                    generated_image = utils.transfer_color(content_image, generated_image)
                image_name = os.path.basename(path[i])
                utils.saveimg(generated_image, save_folder + image_name)'''

            # Load content image
            content_image = utils.load_image(frame_folder + image_name)
            content_tensor = utils.itot(content_image).to(device)

            # Generate image
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())

            if (opts.PRESERVE_COLOR):
                generated_image = utils.transfer_color(content_image, generated_image)
            # Save image
            utils.saveimg(generated_image, save_folder + image_name)

if __name__ == '__main__':
    if not os.path.exists(opts.save_folder):
        os.makedirs(opts.save_folder)

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(opts.style_path))
    net = net.to(device)

    # Stylize every frame
    images = [img for img in os.listdir(opts.frame_folder) if img.endswith(".jpg")]
    with torch.no_grad():
        for image_name in images:
            #torch.cuda.empty_cache() # Free-up unneeded cuda memory
            torch.cuda.empty_cache() if opts.cuda else None

            # Load content image
            content_image = utils.load_image(opts.frame_folder + image_name)
            content_tensor = utils.itot(content_image).to(device)

            # Generate image
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            if (opts.PRESERVE_COLOR):
                generated_image = utils.transfer_color(content_image, generated_image)
            # Save image
            utils.saveimg(generated_image, opts.save_folder + image_name)