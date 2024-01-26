import depthai as dai
import numpy as np
import cv2
import time
import argparse
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run sengmentation model on real Oak-D camera')
    parser.add_argument('--model', type=pathlib.Path, required=True,
                            help='Path to blob model')
    args = parser.parse_args()

    pipeline = dai.Pipeline()

    # Define source and output
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(320, 240)
    cam_rgb.setInterleaved(False)
    xout_video = pipeline.create(dai.node.XLinkOut)
    xout_video.setStreamName("video")
    cam_rgb.preview.link(xout_video.input)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(args.model)
    #nn.input.setBlocking(True)
    #nn.input.setQueueSize(10)
    #nn.setNumInferenceThreads(2) # By default 2 threads are used
    #nn.setNumNCEPerInferenceThread(2) # By default, 1 NCE is used per thread
    cam_rgb.preview.link(nn.input)
    nn_xout = pipeline.create(dai.node.XLinkOut)
    nn_xout.setStreamName("nn")
    #nn_xout.setFpsLimit(4)
    nn.out.link(nn_xout.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        video_queue = device.getOutputQueue(name="video", maxSize=1, blocking=False)
        nn_queue = device.getOutputQueue(name="nn", maxSize=1, blocking=False)
        while True:
            start_time = time.time()
            video_in = video_queue.get()  # Video frame
            video_time = time.time()
            nn_in = nn_queue.get()  # NN output
            nn_time = time.time()

            img = video_in.getCvFrame()
            img_time = time.time()
            img = cv2.resize(img, (640, 480))
            nn_output = nn_in.getLayerFp16('output')
            first_layer_time = time.time()
            mask = np.array(nn_output).reshape((2, 240, 320))
            mask = mask.argmax(0).astype(np.uint8)
            mask = cv2.resize(mask, (640, 480))
            height, width = mask.shape
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
            colored_mask[mask == 1] = [0, 0, 255]
            overlay = cv2.addWeighted(img, 1, colored_mask, 0.7, 0)
            process_time = time.time()
                
            cv2.imshow("OAK-D Segmentation", overlay)
            show_time = time.time()
            print(f"Total:{show_time-start_time:4f}, video:{video_time-start_time:4f}, nn:{nn_time-video_time:4f}, img:{img_time-nn_time:4f}, first_layer:{first_layer_time-img_time:4f}, process:{process_time-first_layer_time:4f}, show:{show_time-process_time:4f}")
            cv2.waitKey(1)
