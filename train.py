import argparse
import os
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import utils
from torchvision import datasets, transforms
from models import transformer, vgg

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser  = argparse.ArgumentParser(description='Args of Train')
parser.add_argument('--NUM_EPOCHS', type=int, default=10, help='train epoch numbers')
parser.add_argument('--BATCH_SIZE', type=int, default=4, help='input batch size')
parser.add_argument('--CONTENT_WEIGHT', type=float, default=17.0, help='CONTENT WEIGHT')
parser.add_argument('--STYLE_WEIGHT', type=float, default=50.0, help='STYLE WEIGHT')
parser.add_argument('--ADAM_LR', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--SAVE_MODEL_EVERY', type=int, default=100, help='how many processes number to save')
parser.add_argument('--SAVE_MODEL_PATH', type=str, default='checkpoints/', help='save model')
parser.add_argument('--SAVE_IMAGE_PATH', type=str, default='checkpoints/results/', help='save result')
parser.add_argument('--STYLE_IMAGE_PATH', type=str, default='images/style/mosaic.jpg', help='style image file')
parser.add_argument('--SAVE_FINAL_PATH', type=str, default='pretrained/', help='save the lastest model')
parser.add_argument('--cuda', type=str2bool, default=False, help='enables CUDA training')
opts    = parser.parse_args()
device  = ("cuda:0" if opts.cuda else "cpu")
kwargs  = {'num_workers': 1, 'pin_memory': True} if opts.cuda else {}

def NormalizeImg(img):
    nimg = (img - img.min()) / (img.max() - img.min())
    return nimg

def show_MNIST(img):
    grid    = torchvision.utils.make_grid(img)
    trimg   = grid.numpy().transpose(1, 2, 0)
    plt.imshow(trimg)
    plt.title('Batch from dataloader')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # GLOBAL SETTINGS
    TRAIN_IMAGE_SIZE    = 256
    #DATASET_PATH        = "dataset"
    #STYLE_IMAGE_PATH    = "styleimages/mosaic.jpg"
    #NUM_EPOCHS         = 1
    #BATCH_SIZE          = 4
    #CONTENT_WEIGHT      = 17
    #STYLE_WEIGHT        = 50
    #ADAM_LR             = 0.001
    #SAVE_MODEL_PATH     = "models/"
    #SAVE_IMAGE_PATH     = "images/out/"
    #SAVE_MODEL_EVERY    = 500  # 2,000 Images with batch size 4
    SEED                = 35
    PLOT_LOSS           = 1

    if not os.path.isdir(opts.SAVE_MODEL_PATH):
        os.mkdir(opts.SAVE_MODEL_PATH)
        os.mkdir(opts.SAVE_IMAGE_PATH)

    # Seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Dataset and Dataloader
    transform = transforms.Compose([transforms.Resize(TRAIN_IMAGE_SIZE),
                                    transforms.CenterCrop(TRAIN_IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])

    train_dataset   = datasets.ImageFolder('./dataset', transform=transform)
    train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=opts.BATCH_SIZE, shuffle=True, **kwargs)

    # Load networks
    TransformerNetwork  = transformer.TransformerNetwork().to(device)
    VGG                 = vgg.VGG16().to(device)

    # Get Style Features
    imagenet_neg_mean   = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32).reshape(1,3,1,1).to(device)
    style_image         = utils.load_image(opts.STYLE_IMAGE_PATH)
    style_tensor        = utils.itot(style_image).to(device)
    style_tensor        = style_tensor.add(imagenet_neg_mean)
    B, C, H, W          = style_tensor.shape
    style_features      = VGG(style_tensor.expand([opts.BATCH_SIZE, C, H, W]))

    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = utils.gram(value)

    # Optimizer settings
    optimizer = optim.Adam(TransformerNetwork.parameters(), lr=opts.ADAM_LR)

    # Loss trackers
    content_loss_history    = []
    style_loss_history      = []
    total_loss_history      = []
    batch_content_loss_sum  = 0
    batch_style_loss_sum    = 0
    batch_total_loss_sum    = 0

    # Optimization/Training Loop
    batch_count = 1
    start_time = time.time()
    for epoch in range(opts.NUM_EPOCHS):
        print("========Epoch {}/{}========".format(epoch+1, opts.NUM_EPOCHS))

        for data_batch, _ in train_loader:
            torch.cuda.empty_cache() if opts.cuda else None

            # Get current batch size in case of odd batch sizes
            curr_batch_size = data_batch.shape[0]

            # Free-up unneeded cuda memory
            # torch.cuda.empty_cache()

            # Zero-out Gradients
            #optimizer.zero_grad()

            # Generate images and get features
            data_batch          = data_batch[:,[2,1,0]].to(device)
            transform_batch     = TransformerNetwork(data_batch)
            data_features       = VGG(data_batch.add(imagenet_neg_mean))
            transform_features  = VGG(transform_batch.add(imagenet_neg_mean))

            '''nimga = NormalizeImg(data_batch[0]).detach().cpu()
            nimgb = NormalizeImg(transform_batch[0]).detach().cpu()
            show_MNIST(nimga)
            show_MNIST(nimgb)'''

            # Content Loss
            MSELoss = nn.MSELoss().to(device)
            content_loss = opts.CONTENT_WEIGHT * MSELoss(transform_features['relu2_2'], data_features['relu2_2'])
            batch_content_loss_sum += content_loss

            # Style Loss
            style_loss = 0
            for key, value in transform_features.items():
                s_loss = MSELoss(utils.gram(value), style_gram[key][:curr_batch_size])
                style_loss += s_loss
            style_loss *= opts.STYLE_WEIGHT
            batch_style_loss_sum += style_loss.item()

            # Total Loss
            total_loss = content_loss + style_loss
            batch_total_loss_sum += total_loss.item()

            # Zero-out Gradients
            optimizer.zero_grad()
            # Backprop and Weight Update
            total_loss.backward()
            optimizer.step()

            # Save Model and Print Losses
            if (((batch_count-1) % opts.SAVE_MODEL_EVERY == 0) or (batch_count==opts.NUM_EPOCHS*len(train_loader))):
                # Print Losses
                print("========Iteration {}/{}========".format(batch_count, opts.NUM_EPOCHS*len(train_loader)))
                print("\tContent Loss:\t{:.2f}".format(batch_content_loss_sum/batch_count))
                print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum/batch_count))
                print("\tTotal Loss:\t{:.2f}".format(batch_total_loss_sum/batch_count))
                print("Time elapsed:\t{} seconds".format(time.time()-start_time))

                # Save Model
                checkpoint_path = opts.SAVE_MODEL_PATH + "checkpoint_" + str(batch_count-1) + ".pth"
                torch.save(TransformerNetwork.state_dict(), checkpoint_path)
                print("Saved TransformerNetwork checkpoint file at {}".format(checkpoint_path))

                # Save sample generated image
                sample_tensor       = transform_batch[0].clone().detach().unsqueeze(dim=0)
                sample_image        = utils.ttoi(sample_tensor.clone().detach())
                sample_image_path   = opts.SAVE_IMAGE_PATH + "sample0_" + str(batch_count-1) + ".png"
                utils.saveimg(sample_image, sample_image_path)
                print("Saved sample tranformed image at {}".format(sample_image_path))

                # Save loss histories
                content_loss_history.append(batch_total_loss_sum/batch_count)
                style_loss_history.append(batch_style_loss_sum/batch_count)
                total_loss_history.append(batch_total_loss_sum/batch_count)

            # Iterate Batch Counter
            batch_count+=1

    stop_time = time.time()
    # Print loss histories
    print("Done Training the Transformer Network!")
    print("Training Time: {} seconds".format(stop_time-start_time))
    print("========Content Loss========")
    print(content_loss_history)
    print("========Style Loss========")
    print(style_loss_history)
    print("========Total Loss========")
    print(total_loss_history)

    # Save TransformerNetwork weights
    TransformerNetwork.eval()
    #TransformerNetwork.cpu()
    final_path = opts.SAVE_FINAL_PATH + "transformer_weight.pth"
    print("Saving TransformerNetwork weights at {}".format(final_path))
    torch.save(TransformerNetwork.state_dict(), final_path)
    print("Done saving final model")

    # Plot Loss Histories
    if (PLOT_LOSS):
        utils.plot_loss_hist(content_loss_history, style_loss_history, total_loss_history)
#train()
