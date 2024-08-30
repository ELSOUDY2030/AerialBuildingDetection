import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from model.download_model import download_model


def run_inference(model_path, image_path, slice_height, slice_width, confidence_threshold):
  #Read the image
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  #Load the Model
  detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path,
    confidence_threshold=confidence_threshold,
    device='cuda:0'
  )

  result = get_sliced_prediction(image,
  detection_model,
  slice_height=slice_height,
  slice_width=slice_width,
  overlap_height_ratio=0.3,
  overlap_width_ratio=0.3,
  postprocess_class_agnostic=True)
  
  if not result.object_prediction_list:
    print("No objects detected in the image.")
  
  else:
    building_count = 0
    for object_prediction in result.object_prediction_list:
      if object_prediction.score.value < 0.4:
        continue
      
      building_count +=1
      cv2.rectangle(image,
        (object_prediction.bbox.minx, object_prediction.bbox.miny),
        (object_prediction.bbox.maxx, object_prediction.bbox.maxy),
        (0,0,0),2
      )
      
      if hasattr(object_prediction, 'mask') and object_prediction.mask is not None:
        mask = object_prediction.mask.bool_mask
        if mask.dtype != bool:
          mask = mask.astype(bool)

        image[mask] = (image[mask] * 0.5 + np.array([0, 255, 255]) * 0.5).astype(np.uint8)

  # Display the image
  print(f"Number of buildings detected: {building_count}")
  cv2.imwrite('output/output_image.jpg', image)
  print(f"Save output_image.jpg")


def main():
  parser = argparse.ArgumentParser(description="Run building detection on an image using a pre-trained YOLOv8 model.")
  parser.add_argument('--model_id', type=str, default='odil111', help="Model identifier on HuggingFace. Valid options include: 'flkennedy', 'Bruno64', 'odil111', 'keremberke_nano', 'keremberke_small', 'keremberke_medium'. Default is 'odil111'.")
  parser.add_argument('--image_path', type=str, required=True, help="File path to the input image for detection.")
  parser.add_argument('--slice_height', type=int, default=500, help="Height of each slice for the detection process. Default is 500 pixels.")
  parser.add_argument('--slice_width', type=int, default=500, help="Width of each slice for the detection process. Default is 500 pixels.")
  parser.add_argument('--confidence_threshold', type=float, default=0.5, help="Minimum confidence threshold for detecting objects. Default is 0.5.")
  
  args = parser.parse_args()
  download_model(args.model_id)
  model_path = f"model/{args.model_id}.pt"
  run_inference(model_path, args.image_path, args.slice_height, args.slice_width, args.confidence_threshold)

  
if __name__ == "__main__":
    main()




