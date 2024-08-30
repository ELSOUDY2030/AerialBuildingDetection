# Aerial Building Detection

This project uses YOLOv8 models to detect and segment buildings in aerial imagery.

## Features

- Supports multiple pre-trained YOLOv8 models for building detection and segmentation
- Uses SAHI (Slicing Aided Hyper Inference) for improved detection on large images
- Visualizes detected buildings with bounding boxes and segmentation masks
- Counts and reports the number of buildings detected

## Requirements

- Python (version not specified, recommend adding)
- OpenCV
- NumPy
- Matplotlib
- SAHI
- PyTorch (implied by use of CUDA)
- Requests

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ELSOUDY2030/AerialBuildingDetection.git
   cd AerialBuildingDetection
   ```

2. Install the required packages:
   ```
   pip install opencv-python numpy matplotlib sahi torch requests
   ```

## Quick Start

To quickly run the building detection on an image, use the following command:

```
python inference.py --model_id flkennedy --image_path /content/Untitled.jpg --slice_height 600 --slice_width 600 --confidence_threshold 0.5
```

This command uses the 'flkennedy' model to detect buildings in the image '/content/Untitled.jpg', with slice dimensions of 600x600 pixels and a confidence threshold of 0.5.

## Usage

For more detailed usage, you can customize the parameters as follows:

```
python inference.py --model_id MODEL_ID --image_path PATH_TO_IMAGE [--slice_height SLICE_HEIGHT] [--slice_width SLICE_WIDTH] [--confidence_threshold CONFIDENCE_THRESHOLD]
```

Arguments:
- `--model_id`: Model identifier on HuggingFace. Options include 'flkennedy', 'Bruno64', 'odil111', 'keremberke_nano', 'keremberke_small', 'keremberke_medium'. Default is 'odil111'.
- `--image_path`: File path to the input image for detection.
- `--slice_height`: Height of each slice for the detection process. Default is 500 pixels.
- `--slice_width`: Width of each slice for the detection process. Default is 500 pixels.
- `--confidence_threshold`: Minimum confidence threshold for detecting objects. Default is 0.5.

## Model Download

The script automatically downloads the specified model if it's not already present in the `model/` directory.

## Output

The script will output:
1. The number of buildings detected
2. An image with bounding boxes and segmentation masks saved as `output/output_image.jpg`

## Acknowledgements

This project uses pre-trained models from various contributors on HuggingFace.
