# -*- coding: utf-8 -*-
# Created time : 2024/12/22 21:40 
# Auther : ygh
# File   : 测试2.py
# Description :
#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from a_person_face_landmark_mask_generator_comfyui_by_my import get_a_person_mask_generator_model_path
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

# STEP 1: Import the necessary modules.


# STEP 2: Create an FaceLandmarker object.
model_asset_path=get_a_person_mask_generator_model_path()
base_options = python.BaseOptions(model_asset_path=model_asset_path)#'face_landmarker_v2_with_blendshapes.task'
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(r"D:\AI\SD\face_id\caiwenjie.png")

# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(image)
print(detection_result.face_landmarks[0])
print(len(detection_result.face_landmarks[0]))
def format_landmarks(landmarks):
    formatted_landmarks = []
    for landmark in landmarks:
        formatted_landmarks.append(landmark)
    return formatted_landmarks

# # 获取格式化后的地标点列表
# formatted_landmarks = format_landmarks(detection_result.face_landmarks[0])
#
# # 打印格式化后的地标点
# for point in formatted_landmarks:
#     print(point)
#
# # 如果你需要直接访问 x, y, z 坐标，可以使用原始的 landmarks 对象
# for landmark in detection_result.face_landmarks[0]:
#     print(f"x: {landmark.x}, y: {landmark.y}, z: {landmark.z}")

new_=format_landmarks(detection_result.face_landmarks[0])
print(new_)
for i in new_:
    print(i.x)
print(new_)

# print(format_landmarks(detection_result.face_landmarks[0]))
# STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))