import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image, draw_labels_and_boxes, draw_depth                                    

import PIL.Image as pil
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import networks

FLAGS = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path',
        type=str,
        default='./yolov3-coco/',
        help='The directory where the model weights and \
              configuration files are.')

    parser.add_argument('-w', '--weights',
        type=str,
        default='./backup/yolo-obj_30000.weights',
        help='Path to the file which contains the weights \
                 for YOLOv3.')

    parser.add_argument('-cfg', '--config',
        type=str,
        default='./cfg/yolo-obj.cfg',
        help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-i', '--image-path',
        type=str,
        help='The path to the image file')

    parser.add_argument('-v', '--video-path',
        type=str,
        help='The path to the video file')


    parser.add_argument('-vo', '--video-output-path',
        type=str,
        default='./output.avi',
        help='The path of the output video file')

    parser.add_argument('-l', '--labels',
        type=str,
        default='./cfg/obj.names',
        help='Path to the file having the \
                    labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.5')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.75,
        help='The threshold to use when applying the \
                Non-Max Suppresion')

    parser.add_argument('--download-model',
        type=bool,
        default=False,
        help='Set to True, if the model weights and configurations \
                are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
        type=bool,
        default=False,
        help='Show the time taken to infer each image.')

    FLAGS, unparsed = parser.parse_known_args()

    ## Init depth model ###########################################################
    model_name = "mono_640x192"
    encoder_path = os.path.join("weights", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("weights", model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']

    ############################################################################

    # Download the YOLOv3 models if needed
    if FLAGS.download_model:
        subprocess.call(['./yolov3-coco/get_model.sh'])

    # Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
    # If both image and video files are given then raise error
    if FLAGS.image_path is None and FLAGS.video_path is None:
        print ('Neither path to an image or path to video provided')
        print ('Starting Inference on Webcam')

    # Do inference with given image
    if FLAGS.image_path:
        # Read the image
        try:
            img = cv.imread(FLAGS.image_path)
            height, width = img.shape[:2]
        except:
            raise 'Image cannot be loaded!\n\
                               Please check the path provided!'

        finally:
            img, _, _, _, _ = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)
            show_image(img)

    elif FLAGS.video_path:
        # Read the video
        try:
            vid = cv.VideoCapture(FLAGS.video_path)
            height, width = None, None
            writer = None
        except:
            raise 'Video cannot be loaded!\n\
                               Please check the path provided!'

        finally:
            while True:
                grabbed, frame = vid.read()

                # Checking if the complete video is read
                if not grabbed:
                    break

                if width is None or height is None:
                    height, width = frame.shape[:2]

                # Depth Estimation
                with torch.no_grad():
                    rgb_resized = cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2RGB), (feed_width, feed_height), interpolation=cv.INTER_LANCZOS4)
                    rgb_resized = rgb_resized.transpose((2, 0, 1)) / 255.0
                    input_image_pytorch = torch.from_numpy(rgb_resized).float().unsqueeze(0)
                    features = encoder(input_image_pytorch)
                    outputs = depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(disp,
                (height, width), mode="bilinear", align_corners=False)

                # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()

                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)
                # Draw labels and boxes on the image
                frame_wboxes = draw_labels_and_boxes(frame, boxes, confidences, classids, idxs, colors, labels)

                vmax = np.percentile(disp_resized_np, 95)
                plt.imsave('tmp/tmp.png', disp_resized_np, cmap='magma', vmax=vmax)
                disp_img = cv.imread('tmp/tmp.png')
                disp_img, _ = draw_depth(disp_img, disp_resized_np, boxes, classids, idxs, colors, labels)

                # print(disp_img.shape, frame_wboxes.shape)

                out_frame = cv.hconcat([frame_wboxes, disp_img])

                if writer is None:
                    # Initialize the video writer
                    fourcc = cv.VideoWriter_fourcc(*"MJPG")
                    writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
                                    (frame.shape[1]*2, frame.shape[0]), True)

                writer.write(out_frame)

            print ("[INFO] Cleaning up...")
            writer.release()
            vid.release()


    else:
        # Infer real-time on webcam
        count = 0

        vid = cv.VideoCapture(0)
        while True:
            _, frame = vid.read()
            height, width = frame.shape[:2]

            if count == 0:
                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
                                    height, width, frame, colors, labels, FLAGS)
                count += 1
            else:
                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
                                    height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
                count = (count + 1) % 6

            cv.imshow('webcam', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv.destroyAllWindows()