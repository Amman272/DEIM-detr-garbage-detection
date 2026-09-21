import streamlit as st
import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pathlib
import os

LABEL_TO_CLASS = {
    0: 'Bottle', 1: 'Bottle cap', 2: 'Can', 3: 'Cigarette', 
    4: 'Cup', 5: 'Lid', 6: 'Other', 7: 'Plastic bag and wrapper', 
    8: 'Pop tab', 9: 'Straw'
}

COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB"
]

@st.cache_resource
def load_model(model_path):
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def process_image(image_path, model, conf_thresh, nms_thresh=0.4):
    original_img = Image.open(image_path).convert("RGB")
    
    try:
        resample_filter = Image.Resampling.BILINEAR
    except AttributeError:
        resample_filter = Image.BILINEAR
        
    img_resized = original_img.resize((640, 640), resample_filter)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    img_np = img_np.transpose((2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: img_np})
    output_names = [o.name for o in model.get_outputs()]
    
    if "pred_logits" in output_names and "pred_boxes" in output_names:
        # DETR / DEIM family model
        idx_logits = output_names.index("pred_logits")
        idx_boxes = output_names.index("pred_boxes")
        
        logits = outputs[idx_logits][0]  # shape (300, 10)
        boxes_out = outputs[idx_boxes][0]  # shape (300, 4)
        
        # Convert logits to probabilities (sigmoid)
        scores_prob = 1 / (1 + np.exp(-logits))
        
        # Max score and labels
        max_scores = np.max(scores_prob, axis=-1)
        labels_out = np.argmax(scores_prob, axis=-1)
        
        mask = max_scores >= conf_thresh
        filtered_boxes = boxes_out[mask]
        scores = max_scores[mask]
        labels = labels_out[mask]
        
        # Convert cx, cy, w, h (normalized) to x1, y1, x2, y2
        boxes = []
        for box in filtered_boxes:
            cx, cy, w, h = box
            # Multiply by image size (640)
            cx, cy, w, h = cx * 640, cy * 640, w * 640, h * 640
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append([x1, y1, x2, y2])
        boxes = np.array(boxes)
        
        # DETR usually doesn't need NMS, but we'll set indices to all to match the rest of the code
        if len(boxes) > 0:
            indices = np.arange(len(boxes))
        else:
            indices = []
            
    else:
        # RetinaNet / Faster R-CNN family model
        boxes = outputs[0]
        scores = outputs[1]
        labels = outputs[2]
        
        mask = scores >= conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        if len(boxes) == 0:
            return img_resized, {}, 0
            
        boxes_xywh = []
        for box in boxes:
            x1, y1, x2, y2 = box
            boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_thresh, nms_thresh)
        
        if len(indices) > 0:
            indices = indices.flatten()
        else:
            indices = []
        
    draw = ImageDraw.Draw(img_resized)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        
    class_counts = {}
    
    for idx in indices:
        x1, y1, x2, y2 = boxes[idx]
        score = scores[idx]
        label_idx = int(labels[idx])
        class_name = LABEL_TO_CLASS.get(label_idx, f"Unknown ({label_idx})")
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        color = COLORS[label_idx % len(COLORS)]
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        text = f"{class_name} {score:.2f}"
        
        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(bbox, fill=color)
        except AttributeError:
            # Pillow < 8.0.0 fallback
            w, h = draw.textsize(text, font=font)
            draw.rectangle([x1, y1, x1 + w, y1 + h], fill=color)
            
        draw.text((x1, y1), text, fill="white", font=font)
        
    return img_resized, class_counts, len(indices)

def main():
    st.set_page_config(page_title="Local Object Detection", layout="wide")
    st.title("RetinaNet Object Detection (ONNX)")
    
    st.sidebar.header("Configuration")
    model_path = st.sidebar.text_input("Model Path", value="retinanet_waste.onnx")
    folder_path = st.sidebar.text_input("Input Image Folder")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model not found at '{model_path}'. Please place the file in the same folder as app.py or provide the absolute path.")
        st.stop()
        
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load ONNX model: {e}")
        st.stop()
        
    if folder_path:
        if not os.path.isdir(folder_path):
            st.error("Invalid folder path.")
            return
            
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(list(pathlib.Path(folder_path).glob(ext)))
            image_files.extend(list(pathlib.Path(folder_path).glob(ext.upper())))
            
        if not image_files:
            st.warning("No images found in the specified folder.")
            return
            
        st.write(f"Found {len(image_files)} images.")
        
        output_folder = st.sidebar.text_input("Output Folder", value="./predictions")
        save_btn = st.sidebar.button("Save all annotated images")
        
        if save_btn:
            os.makedirs(output_folder, exist_ok=True)
            
        cols = st.columns(3)
        
        for i, img_path in enumerate(image_files):
            annotated_img, class_counts, total_dets = process_image(str(img_path), model, conf_thresh)
            
            with cols[i % 3]:
                st.image(annotated_img, caption=img_path.name, use_column_width=True)
                if total_dets > 0:
                    st.write(f"**Total Detections:** {total_dets}")
                    for k, v in class_counts.items():
                        st.write(f"- {k}: {v}")
                else:
                    st.write("No detections.")
                    
            if save_btn:
                out_path = os.path.join(output_folder, img_path.name)
                annotated_img.save(out_path)
                
        if save_btn:
            st.sidebar.success(f"Saved {len(image_files)} images to '{output_folder}'")

if __name__ == "__main__":
    main()
