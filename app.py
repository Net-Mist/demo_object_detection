import logging
import os
import threading
import time

import cv2
import coloredlogs
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, redirect, url_for

from flask_cors import CORS

# Global var before starting the API
camera = None
inference_engine = None
image_size = None

app = Flask(__name__)
CORS(app)


class Camera(threading.Thread):
  def __init__(self):
    super().__init__()
    self.camera_port = 0
    self.fps = 30
    self.video_capture = None

    self.running = False

    self.lock = threading.Lock()
    self.frame = None  # jpeg encoded to bytes
    self.numpy_frame = None

    self.video_capture = cv2.VideoCapture(self.camera_port)

  def run(self) -> None:
    self.running = True
    previous_time = time.time()
    while self.running:
      current_time = time.time()
      if current_time < previous_time + 1 / self.fps:
        time.sleep(previous_time + 1 / self.fps - current_time)
      success, self.numpy_frame = self.video_capture.read()
      try:
        ret, jpeg_encoded = cv2.imencode('.jpg', self.numpy_frame)
      except Exception as e:
        print(e)
        continue
      with self.lock:
        self.frame = jpeg_encoded.tobytes()
      previous_time = current_time

    self.video_capture.release()

  def get_frame(self):
    with self.lock:
      return self.frame


class TfDetection:
  '''
  Main class to load, make inference and process outputs based on tensorflow object detection models.
  '''

  def __init__(self, path_to_frozen_graph):
    self.detection_thres = 0.3
    # Load label map
    self.id_to_labelname = {}
    with open("mscoco_label_map.pbtxt", 'r') as f:
      for line in f.readlines():
        id = int(line.split(" ")[0])
        label = " ".join(line[:-1].split(" ")[1:])
        self.id_to_labelname[id] = label
    # Load detection model and start session
    tensor_names = ['image_tensor:0', 'detection_boxes:0', 'detection_scores:0', 'detection_classes:0']
    with tf.gfile.GFile(path_to_frozen_graph, 'rb') as f:
      frozen_graph_def = tf.GraphDef()
      frozen_graph_def.ParseFromString(f.read())
    tensors = tf.import_graph_def(frozen_graph_def, return_elements=tensor_names)
    self.input_tensor = tensors[0]
    self.output_tensors = tensors[1:]
    self.session = tf.Session()

  def infer(self, rgb_image):
    '''
    Apply model on rgb image
    '''
    boxes, scores, label_ids = self.session.run(self.output_tensors, feed_dict={self.input_tensor: np.expand_dims(rgb_image, axis=0)})
    boxes, scores, label_ids = boxes[0], scores[0], label_ids[0]

    line_thikness = int(round(0.002 * max(rgb_image.shape[0:2])))
    # line_thikness = 5
    vis_image = rgb_image.copy()
    for i in range(label_ids.shape[0]):
      obj_id = label_ids[i]
      obj_score = scores[i]
      boxe = boxes[i]
      if obj_score > self.detection_thres:
        obj_id = int(obj_id)
        if obj_id in self.id_to_labelname.keys():
          vis_text = self.id_to_labelname[obj_id]
          xmin = int(boxe[1] * vis_image.shape[1])
          ymin = int(boxe[0] * vis_image.shape[0])
          xmax = int(boxe[3] * vis_image.shape[1])
          ymax = int(boxe[2] * vis_image.shape[0])
          vis_image = cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
          cv2.putText(vis_image, text=vis_text, org=(xmin, ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=1, color=(255, 255, 255), thickness=line_thikness)
        else:
          xmin = int(boxe[1] * vis_image.shape[1])
          ymin = int(boxe[0] * vis_image.shape[0])
          xmax = int(boxe[3] * vis_image.shape[1])
          ymax = int(boxe[2] * vis_image.shape[0])
          vis_image = cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    return vis_image


def image_preprocess(image, resized_shape):
  """
  Preprocess the image before feeding the neural network:
    - convert from BGR (opencv) to RGB
    - resize
    - center the pixel values between -1 and 1
    - add a new dimension for the batch
  """
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (640, 480))
  image = image / 127.5 - 1.
  image = image[np.newaxis, :, :, :].astype(np.float32)
  return image


def image_postprocess(image):
  """
  postprocess the image after the neural network part
    - remove the image from the batch
    - set pixel values between 0 and 255
    - convert from RGB to BGR for opencv
  """
  image = image[0, :, :, :]
  image = ((image + 1) * 127.5).astype(np.uint8)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  return image


class SavedModelInference:
  def __init__(self, inference_model_path):
    """
    Load the tensorflow saved model
    """
    self.loaded = tf.saved_model.load(inference_model_path)
    logging.info(list(self.loaded.signatures.keys()))
    self.interpreter = self.loaded.signatures["serving_default"]
    logging.info(self.interpreter)
    logging.info(self.interpreter.inputs)
    logging.info(self.interpreter.structured_outputs)
    logging.info(list(self.interpreter.structured_outputs.keys())[0])
    self.output_label = list(self.interpreter.structured_outputs.keys())[0]

  def infer(self, image):
    image = image_preprocess(image, (1920, 1080))
    decoded_image = self.interpreter(tf.constant(image))[self.output_label].numpy()
    return image_postprocess(decoded_image)[4:-4, :, :]


def inference(image):
  image = inference_engine.infer(image)
  ret, jpeg_encoded = cv2.imencode('.jpg', image)
  frame = jpeg_encoded.tobytes()
  return frame


@app.route("/")
def index():
  return redirect(url_for('static', filename='index.html'))


@app.route("/video_feed")
def video_feed():
  def gen_image():
    last_get_frame_time = 0
    while True:
      current_time = time.time()
      if current_time - last_get_frame_time < 1 / camera.fps:
        time.sleep(last_get_frame_time + 1 / camera.fps - current_time)
      last_get_frame_time = current_time

      frame = camera.get_frame()
      yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
  return Response(gen_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/inference_feed")
def inference_feed():
  def gen_inference():
    last_get_frame_time = 0
    while True:
      current_time = time.time()
      if current_time - last_get_frame_time < 1 / camera.fps:
        time.sleep(last_get_frame_time + 1 / camera.fps - current_time)
      last_get_frame_time = current_time

      numpy_frame = camera.numpy_frame

      yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + inference(numpy_frame) + b'\r\n')

  return Response(gen_inference(), mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
  global camera, inference_engine
  camera = Camera()
  camera.start()
  print("load TF")
  inference_engine = TfDetection("model/frozen_inference_graph.pb")
  print("TF loaded")
  app.run(host='0.0.0.0', threaded=True, port=80)


if __name__ == '__main__':
  main()
