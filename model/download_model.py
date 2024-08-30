import requests
import os 

model_urls = {
    'flkennedy': 'https://huggingface.co/flkennedy/YOLOv8s-building-segmentation-demo/resolve/main/best.pt',
    'Bruno64': 'https://huggingface.co/Bruno64/YOLOv8-building-detect/resolve/main/YOLOv8_20240124_bruno.pt',
    'odil111': 'https://huggingface.co/odil111/yolov8m-seg-fine-tuned-on-spacenetv2/resolve/main/yolov8m_inst_seg_2024-06-11--15-57-15/weights/best.pt',
    'keremberke_nano': 'https://huggingface.co/keremberke/yolov8n-building-segmentation/resolve/main/best.pt',
    'keremberke_small': 'https://huggingface.co/keremberke/yolov8s-building-segmentation/resolve/main/best.pt',
    'keremberke_medium': 'https://huggingface.co/keremberke/yolov8m-building-segmentation/resolve/main/best.pt'
}

def download_model(name_model_url):
  output_path = f"{name_model_url}.pt"
  if name_model_url not in model_urls:
    raise ValueError("Invalid model_id provided. Choose from 'flkennedy', 'Bruno64', 'odil111', 'keremberke_nano', 'keremberke_small', 'keremberke_medium'")
  if not os.path.exists(output_path):
    print(f"Downloading model to {output_path}...")
    response = requests.get(model_urls[name_model_url], stream=True)  
    if response.status_code == 200:
      with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)
      print(f"Model downloaded successfully: {output_path}")
    else:
      print(f"Failed to download the model. Status code: {response.status_code}")
  else:
    print(f"Model already exists at {output_path}, no download needed.")



