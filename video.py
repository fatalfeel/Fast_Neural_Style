import argparse
import os
import time
import cv2
from stylize import stylize_folder

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser  = argparse.ArgumentParser(description='Args of Train')
parser.add_argument('--BATCH_SIZE', type=int, default=20, help='input batch size')
parser.add_argument('--style_path', type=str, default='pretrained/transformer_weight.pth', help='load model path')
parser.add_argument('--video_path', type=str, default='video/test/source.mp4', help='video file')
parser.add_argument('--VIDEO_EXTRACT_FOLDER', type=str, default='video/extract/', help='video folder')
parser.add_argument('--FRAME_TRANSFORM_FOLDER', type=str, default='video/transform/', help='video folder')
parser.add_argument('--STYLE_VIDEO_SAVE_PATH', type=str, default='video/results/', help='video folder')
parser.add_argument('--STYLE_VIDEO_SAVE_NAME', type=str, default='target.avi', help='video folder')
parser.add_argument('--cuda', type=str2bool, default=False, help='enables CUDA training')
opts    = parser.parse_args()
device  = ("cuda:0" if opts.cuda else "cpu")

#video_name = "dance.mp4"
#FRAME_TRANSFORM_FOLDER = "frames/"
#FRAME_CONTENT_FOLDER = "content_folder/"
FRAME_BASE_FILE_NAME    = "frame"
FRAME_BASE_FILE_TYPE    = ".jpg"
#STYLE_VIDEO_SAVE_PATH  = "video/results"
#STYLE_VIDEO_SAVE_NAME  = "helloworld.mp4"
#STYLE_PATH = "transforms/mosaic_aggressive.pth"
#BATCH_SIZE = 20

def getInfo(video_path):
    """
    Extracts the height, width,
    and fps of a video
    """
    vidcap = cv2.VideoCapture(video_path)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  vidcap.get(cv2.CAP_PROP_FPS)
    return height, width, fps

def getFrames(video_path):
    """
    Extracts the frames of a video
    and saves in specified path
    """
    vidcap          = cv2.VideoCapture(video_path)
    success, image  = vidcap.read()
    count           = 1
    success         = True
    while success:
        #cv2.imwrite("{}{}{}{}".format(VIDEO_EXTRACT_FOLDER+FRAME_CONTENT_FOLDER, FRAME_BASE_FILE_NAME, count, FRAME_BASE_FILE_TYPE), image)
        cv2.imwrite("{}{}{}{}".format(opts.VIDEO_EXTRACT_FOLDER, FRAME_BASE_FILE_NAME, count, FRAME_BASE_FILE_TYPE), image)
        success, image = vidcap.read()
        count += 1
    print("Done extracting all frames")
    
def makeVideo(frames_path, save_name, fps, height, width):    
    # Extract image paths. Natural sorting of directory list. Python does not have a native support for natural sorting :(
    base_name_len = len(FRAME_BASE_FILE_NAME)
    filetype_len = len(FRAME_BASE_FILE_TYPE)
    images = [img for img in sorted(os.listdir(frames_path), key=lambda x : int(x[base_name_len:-filetype_len])) if img.endswith(".jpg")]
    
    # Define the codec and create VideoWrite object
    fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    vout    = cv2.VideoWriter(save_name, fourcc, fps, (width,height))

    # Write the video
    for image_name in images:
        vout.write(cv2.imread(os.path.join(frames_path, image_name)))

    print("Done writing video")

#video_transfer(video_name, STYLE_PATH)
# def video_transfer(video_path, style_path):
if __name__ == '__main__':
    if not os.path.isdir(opts.VIDEO_EXTRACT_FOLDER):
        os.mkdir(opts.VIDEO_EXTRACT_FOLDER)
    
    if not os.path.isdir(opts.FRAME_TRANSFORM_FOLDER):
        os.mkdir(opts.FRAME_TRANSFORM_FOLDER)
    
    if not os.path.isdir(opts.STYLE_VIDEO_SAVE_PATH):
        os.mkdir(opts.STYLE_VIDEO_SAVE_PATH)

    #print("OpenCV {}".format(cv2.__version__))
    starttime = time.time()
    # Extract video info
    H, W, fps = getInfo(opts.video_path)
    print("Height: {} Width: {} FPS: {}".format(H, W, fps))

    # Extract all frames
    print("Extracting video frames")
    getFrames(opts.video_path)

    # Stylize a directory
    print("Performing style transfer on frames")
    stylize_folder(opts.style_path, opts.VIDEO_EXTRACT_FOLDER, opts.FRAME_TRANSFORM_FOLDER, batch_size=opts.BATCH_SIZE)

    # Combine all frames
    print("Combining style frames into one video")
    makeVideo(opts.FRAME_TRANSFORM_FOLDER, opts.STYLE_VIDEO_SAVE_PATH + opts.STYLE_VIDEO_SAVE_NAME, fps, int(H), int(W))
    print("Elapsed Time: {}".format(time.time() - starttime))
