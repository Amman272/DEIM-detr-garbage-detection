# Local Object Detection with Streamlit and ONNX

This is a local Streamlit application that runs object detection on a folder of images using an ONNX model (RetinaNet ResNet50-FPN-V2). It is designed to run entirely locally using CPU, without needing cloud services or PyTorch/Torchvision dependencies.

## Setup Instructions

Follow these exact steps to set up and run the app on your Windows, Mac, or Linux PC:

### 1. Create a Virtual Environment

Open your terminal or command prompt and run:
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the required CPU-only Python packages:
```bash
pip install -r requirements.txt
```

### 4. Prepare the Model

Make sure your ONNX model file `retinanet_waste.onnx` is placed in the exact same folder as `app.py`. If you want to use a model from a different location, you can modify the path in the Streamlit app sidebar later.

### 5. Run the Application

Launch the Streamlit server:
```bash
streamlit run app.py
```
This will open a new tab in your default web browser where you can use the app. Just enter the local path to your image folder and explore the detections!
