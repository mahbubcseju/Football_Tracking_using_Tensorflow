from load_pretrained_model import load_label,load_pretrained_graph
import os
from PIL import Image
from utils import utils
from utils import visualization_utils as vis_util
import numpy as np
from tensorflow_detection import run_inference_for_single_image
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def detect_object_on_video(test_video_path):
    detection_graph = load_pretrained_graph()
    category_index = load_label()
    import cv2

    cap = cv2.VideoCapture(test_video_path)
    cap.set(cv2.CAP_PROP_FPS, 10)
    fps = int(cap.get(5))
    try:
        with detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                cnt = 0
                while True:
                    ret, image_np = cap.read()
                    cnt= cnt + 1

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    if cnt % 1==0:
                        #image_np_expanded = np.expand_dims(image_np, axis=0)
                        # Actual detection.
                        output_dict = run_inference_for_single_image(tensor_dict,sess,image_np, detection_graph)
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks'),
                            use_normalized_coordinates=True,
                            line_thickness=8)
                    cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
    except Exception as e:
        print(e)
        cap.release()