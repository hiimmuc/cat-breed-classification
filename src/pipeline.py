"""
Pipeline for cat breed and emotion detection with YOLO-based cat detection and flexible model support.

UNIFIED PIPELINE FLOW (Applied to ALL input types):
======================================================

1. INPUT PROCESSING
   - Image: Single image file
   - Video: Frame-by-frame processing
   - Webcam: Real-time frame processing

2. CAT DETECTION/TRACKING
   - YOLO (if available): Fast and accurate cat detection using YOLOv8
   - Fallback: Whole image assumed to contain cat

3. BOUNDING BOX EXTRACTION
   - Extract detected cat bounding boxes from YOLO results
   - Filter for cats above confidence threshold
   - Validate bbox coordinates within image bounds

4. FRAME CROPPING
   - Crop image regions based on detected bboxes
   - Each detected cat gets its own cropped frame

5. CLASSIFICATION (Two modes supported):
   A. MULTITASK MODE (--multitask flag):
      - Single model produces both breed and emotion predictions
      - Single forward pass for both tasks
   B. SINGLE MODEL MODE (default):
      - Separate models for breed and emotion classification
      - Two forward passes (one per task)

6. RESULT VISUALIZATION
   - Draw bounding boxes around detected cats
   - Add refined labels (inside/above/below bbox as appropriate)
   - Semi-transparent backgrounds for optimal readability

This ensures consistent processing regardless of input type and model configuration.
All images/frames follow the same 6-step pipeline flow with flexible model support.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from test import CatBreedPredictor
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from PIL import Image

# Import local modules
from model import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO imports from ultralytics
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
    logger.info("YOLO from ultralytics is available")
except ImportError:
    logger.warning("YOLO from ultralytics not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False


def load_pipeline_config(config_path: str = "src/configs/pipeline.yaml") -> Dict:
    """Load pipeline configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Pipeline configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def load_class_names_from_config(
    config: Dict, class_type: str, use_multitask: bool = True
) -> List[str]:
    """Load class names from JSON file specified in config."""
    model_type = "multitask" if use_multitask else "single"
    class_names_path = config["class_names"][model_type][f"{class_type}_classes"]

    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            return json.load(f)
    else:
        logger.warning(f"Class names file not found: {class_names_path}")
        return []


class CatAnalysisPipeline:
    """
    Pipeline for detecting, tracking and analyzing cats in images/videos with flexible model support.

    Pipeline Flow:
    1. Input Processing (Image/Video/Webcam)
    2. Cat Detection/Tracking (YOLO or fallback to full image)
    3. Bounding Box Extraction (from YOLO detection results)
    4. Frame Cropping (based on detected bboxes)
    5. Classification (Multitask model OR separate breed/emotion models)
    6. Result Visualization (with refined label positioning)

    This ensures consistent processing across all input types with flexible model configuration.
    Supports both multitask models (single model for both tasks) and separate models.
    """

    def __init__(
        self,
        config: Dict,
        breed_classes: Optional[List[str]] = None,
        emotion_classes: Optional[List[str]] = None,
        use_multitask: bool = True,
    ):
        """
        Initialize the pipeline from configuration.

        Args:
            config: Pipeline configuration dictionary loaded from YAML
            breed_classes: List of breed class names (loaded from config if None)
            emotion_classes: List of emotion class names (loaded from config if None)
            use_multitask: Whether to use multitask model or separate models
        """
        # Load configuration parameters
        pipeline_config = config["pipeline"]

        self.device = pipeline_config["device"]
        self.img_size = pipeline_config["img_size"]
        self.confidence_threshold = pipeline_config["confidence_threshold"]
        self.use_multitask = use_multitask

        # Load class names from config if not provided
        if breed_classes is None:
            breed_classes = load_class_names_from_config(config, "breed", use_multitask)
        if emotion_classes is None:
            emotion_classes = load_class_names_from_config(config, "emotion", use_multitask)

        self.breed_classes = breed_classes
        self.emotion_classes = emotion_classes

        # Load models based on mode
        if use_multitask:
            self._load_multitask_model(config)
        else:
            self._load_single_models(config)

        # Store YOLO model path from config
        self.yolo_model_path = config["models"]["yolo_model"]
        self.yolo_config = config["yolo"]

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Initialize YOLO detection model if available and enabled
        if YOLO_AVAILABLE and self.yolo_config["enabled"]:
            self._init_yolo_model()
        else:
            self.yolo_model = None

    def _load_multitask_model(self, config: Dict):
        """Load multitask model for joint breed and emotion classification."""
        model_config = config["model_config"]["multitask"]
        multitask_model_path = config["models"]["multitask_model"]

        logger.info(f"Loading multitask model from {multitask_model_path}")
        logger.info(f"Model config: {model_config}")

        self.model = load_model(
            path=multitask_model_path,
            num_classes=len(self.breed_classes),
            model_type="multitask",
            model_config=model_config,
            num_emotion_classes=len(self.emotion_classes),
        )
        self.model.eval()
        self.model.to(self.device)

        # For multitask, we don't need separate models
        self.breed_model = None
        self.emotion_model = None

    def _load_single_models(self, config: Dict):
        """Load separate breed and emotion models."""
        breed_model_config = config["model_config"]["single"]["breed"]
        emotion_model_config = config["model_config"]["single"]["emotion"]

        # Load breed classification model
        breed_model_path = config["models"]["breed_model"]
        logger.info(f"Loading breed model from {breed_model_path}")
        logger.info(f"Breed model config: {breed_model_config}")

        self.breed_model = load_model(
            path=breed_model_path,
            num_classes=len(self.breed_classes),
            model_type="breed",
            model_config=breed_model_config,
        )
        self.breed_model.eval()
        self.breed_model.to(self.device)

        # Load emotion classification model
        emotion_model_path = config["models"]["emotion_model"]
        logger.info(f"Loading emotion model from {emotion_model_path}")
        logger.info(f"Emotion model config: {emotion_model_config}")

        self.emotion_model = load_model(
            path=emotion_model_path,
            num_classes=len(self.emotion_classes),
            model_type="emotion",
            model_config=emotion_model_config,
        )
        self.emotion_model.eval()
        self.emotion_model.to(self.device)

        # For single models, we don't have a combined model
        self.model = None

    def _init_yolo_model(self):
        """Initialize YOLO model for cat detection."""
        try:
            if not YOLO_AVAILABLE:
                logger.warning("YOLO not available")
                self.yolo_model = None
                return

            # Initialize YOLO model from config path
            logger.info(f"Loading YOLO model from {self.yolo_model_path}")
            self.yolo_model = YOLO(self.yolo_model_path)

            # COCO dataset class IDs for cats (from config)
            self.cat_class_id = self.yolo_config["cat_class_id"]

            logger.info("YOLO model initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO model: {e}")
            self.yolo_model = None

    def detect_animals(self, image: np.ndarray) -> List[Dict]:
        """
        Step 2: Detect/Track cats in the image using YOLO or fallback.

        This is the core detection step that runs for ALL input types:
        - For images: Single detection per image
        - For videos: Detection per frame
        - For webcam: Real-time detection per frame

        Args:
            image: Input image as numpy array

        Returns:
            List of detection dictionaries with bounding boxes and scores
            Format: [{"bbox": [x1, y1, x2, y2], "score": float, "label": str}]
        """
        if not YOLO_AVAILABLE or self.yolo_model is None:
            # Fallback: assume the whole image contains a cat
            h, w = image.shape[:2]
            return [{"bbox": [0, 0, w, h], "score": 1.0, "label": "cat"}]

        try:
            # Convert BGR to RGB for YOLO (YOLO expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run YOLO inference
            results = self.yolo_model(rgb_image, verbose=False)

            detections = []

            # Process YOLO results
            for result in results:
                # Get detection boxes, scores, and class IDs
                boxes = result.boxes
                if boxes is not None:
                    # Filter for cats only (class ID 15 in COCO)
                    for i, class_id in enumerate(boxes.cls):
                        if int(class_id) == self.cat_class_id:
                            confidence = float(boxes.conf[i])

                            # Filter by YOLO confidence threshold from config
                            if confidence >= self.yolo_config["confidence_threshold"]:
                                # Get bounding box coordinates (xyxy format)
                                bbox = boxes.xyxy[i].cpu().numpy()
                                x1, y1, x2, y2 = bbox

                                # Ensure coordinates are within image bounds
                                h, w = image.shape[:2]
                                x1 = max(0, min(int(x1), w - 1))
                                y1 = max(0, min(int(y1), h - 1))
                                x2 = max(x1 + 1, min(int(x2), w))
                                y2 = max(y1 + 1, min(int(y2), h))

                                detections.append(
                                    {"bbox": [x1, y1, x2, y2], "score": confidence, "label": "cat"}
                                )

            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x["score"], reverse=True)

            return detections

        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
            # Fallback: assume the whole image contains a cat
            h, w = image.shape[:2]
            return [{"bbox": [0, 0, w, h], "score": 1.0, "label": "cat"}]

    def classify_breed_and_emotion(self, image_crop: np.ndarray) -> Tuple[str, float, str, float]:
        """
        Steps 5-6: Classify breed and emotion from a cropped cat image.

        This runs on EVERY detected bounding box from the detection step.
        The same classification logic applies to all input types.
        Supports both multitask and single model modes.

        Args:
            image_crop: Cropped cat image as numpy array (from detected bbox)

        Returns:
            Tuple of (breed_label, breed_confidence, emotion_label, emotion_confidence)
        """
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_multitask:
                # Use multitask model for both predictions
                breed_logits, emotion_logits = self.model(input_tensor, task="both")
            else:
                # Use separate models for breed and emotion
                breed_logits = self.breed_model(input_tensor)
                emotion_logits = self.emotion_model(input_tensor)

            # Process breed predictions
            breed_probs = F.softmax(breed_logits, dim=1)
            breed_confidence, breed_idx = torch.max(breed_probs, 1)
            breed_label = self.breed_classes[breed_idx.item()]
            breed_conf = breed_confidence.item()

            # Process emotion predictions
            emotion_probs = F.softmax(emotion_logits, dim=1)
            emotion_confidence, emotion_idx = torch.max(emotion_probs, 1)
            emotion_label = self.emotion_classes[emotion_idx.item()]
            emotion_conf = emotion_confidence.item()

        return breed_label, breed_conf, emotion_label, emotion_conf

    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Complete pipeline processing for a single image/frame.

        This method implements the full pipeline flow:
        1. Input: Single image/frame
        2. Detection: Animal detection/tracking
        3. Bbox Extraction: Get bounding boxes from detections
        4. Cropping: Extract cat regions based on bboxes
        5. Classification: Breed and emotion classification (multitask OR single models)
        6. Visualization: Draw results with refined label positioning

        Used by: image processing, video frame processing, webcam frame processing

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (annotated_image, detection_results)
        """
        # Step 2: Detect animals using MMPose or fallback
        detections = self.detect_animals(image)

        results = []
        annotated_image = image.copy()

        # Step 3: Process each detected bounding box
        for detection in detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            # Step 3a: Ensure bbox is within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Step 4: Crop the detected region for classification
            crop = image[y1:y2, x1:x2]

            if crop.size > 0:
                # Step 5: Classification on cropped frame (breed and emotion)
                breed_label, breed_conf, emotion_label, emotion_conf = (
                    self.classify_breed_and_emotion(crop)
                )

                # Store results for this detection
                result = {
                    "bbox": [x1, y1, x2, y2],
                    "detection_score": detection["score"],
                    "breed": breed_label,
                    "breed_confidence": breed_conf,
                    "emotion": emotion_label,
                    "emotion_confidence": emotion_conf,
                }
                results.append(result)

                # Step 6: Visualization - Draw bounding box and refined labels
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Create label text
                breed_text = f"{breed_label} ({breed_conf:.2f})"
                emotion_text = f"{emotion_label} ({emotion_conf:.2f})"

                # Calculate text sizes for proper positioning
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                breed_size = cv2.getTextSize(breed_text, font, font_scale, thickness)[0]
                emotion_size = cv2.getTextSize(emotion_text, font, font_scale, thickness)[0]

                # Calculate label background dimensions
                max_text_width = max(breed_size[0], emotion_size[0])
                line_height = max(breed_size[1], emotion_size[1])
                padding = 8
                label_height = (2 * line_height) + (3 * padding)  # Two lines + padding
                label_width = max_text_width + padding

                # Get image and bounding box dimensions
                img_height, img_width = annotated_image.shape[:2]
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                # Determine optimal label position
                if bbox_width >= label_width + 10 and bbox_height >= label_height + 10:
                    # Labels fit inside the bounding box - place at top-left
                    label_x = x1 + 5
                    label_y = y1 + 5
                    bg_color = (0, 0, 0)  # Black background for inside box
                    text_color = (255, 255, 255)  # White text for inside box
                    use_transparency = True

                    # Draw semi-transparent background
                    overlay = annotated_image.copy()
                    cv2.rectangle(
                        overlay,
                        (label_x, label_y),
                        (label_x + label_width, label_y + label_height),
                        bg_color,
                        -1,
                    )
                    cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)

                elif y1 >= label_height + 10:
                    # Place labels above the bounding box
                    label_x = x1
                    label_y = y1 - label_height - 5
                    bg_color = (0, 255, 0)  # Green background for outside box
                    text_color = (0, 0, 0)  # Black text for outside box
                    use_transparency = False

                    # Ensure label doesn't go outside image bounds
                    if label_x + label_width > img_width:
                        label_x = img_width - label_width
                    if label_x < 0:
                        label_x = 0

                    # Draw solid background
                    cv2.rectangle(
                        annotated_image,
                        (label_x, label_y),
                        (label_x + label_width, label_y + label_height),
                        bg_color,
                        -1,
                    )

                else:
                    # Place labels below the bounding box if not enough space above
                    label_x = x1
                    label_y = y2 + 5
                    bg_color = (0, 255, 0)  # Green background for outside box
                    text_color = (0, 0, 0)  # Black text for outside box
                    use_transparency = False

                    # Ensure label doesn't go outside image bounds
                    if label_x + label_width > img_width:
                        label_x = img_width - label_width
                    if label_x < 0:
                        label_x = 0
                    if label_y + label_height > img_height:
                        label_y = img_height - label_height

                    # Draw solid background
                    cv2.rectangle(
                        annotated_image,
                        (label_x, label_y),
                        (label_x + label_width, label_y + label_height),
                        bg_color,
                        -1,
                    )

                # Draw text labels
                text_x = label_x + padding // 2
                breed_y = label_y + line_height + padding
                emotion_y = breed_y + line_height + padding

                cv2.putText(
                    annotated_image,
                    breed_text,
                    (text_x, breed_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )
                cv2.putText(
                    annotated_image,
                    emotion_text,
                    (text_x, emotion_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )

        return annotated_image, results

    def process_video(
        self, video_path: str, output_path: Optional[str] = None, display: bool = True
    ) -> Optional[str]:
        """
        Process a video file using the same pipeline flow as single images.

        Pipeline Flow (applied to each frame):
        1. Input: Video frame
        2-6. Same as process_image() for each frame

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display the video while processing

        Returns:
            Path to output video if saved, None otherwise
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set up video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame using the same 6-step pipeline flow as single images
                annotated_frame, results = self.process_image(frame)

                # Write frame if output is specified
                if writer:
                    writer.write(annotated_frame)

                # Display frame if requested
                if display:
                    cv2.imshow("Cat Analysis Pipeline", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if frame_count % 30 == 0:  # Log progress every 30 frames
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% | FPS: {fps_current:.1f}")

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        logger.info(f"Video processing completed in {elapsed:.2f}s")

        return output_path

    def process_webcam(self, camera_id: int = 0):
        """
        Process webcam feed in real-time using the same pipeline flow.

        Pipeline Flow (applied to each frame):
        1. Input: Webcam frame
        2-6. Same as process_image() for each frame in real-time

        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return

        logger.info("Starting webcam processing. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame using the same 6-step pipeline flow (real-time)
                annotated_frame, results = self.process_image(frame)

                # Display frame
                cv2.imshow("Cat Analysis Pipeline - Webcam", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """
    Main function for running the unified cat analysis pipeline.

    ALL commands (image, video, webcam) use the same 6-step pipeline flow:
    YOLO Detection → Bbox Extraction → Cropping → Classification → Visualization

    Classification can be done in two modes:
    - Multitask mode (--multitask): Single model for both breed and emotion
    - Single model mode (default): Separate models for breed and emotion

    The only difference between commands is the input source:
    - image: Single image file processing
    - video: Frame-by-frame video processing
    - webcam: Real-time webcam frame processing
    """
    parser = argparse.ArgumentParser(description="Cat Analysis Pipeline")

    # Required arguments
    parser.add_argument("command", choices=["image", "video", "webcam"], help="Processing mode")

    # Optional arguments
    parser.add_argument(
        "--config",
        default="src/configs/pipeline.yaml",
        help="Path to pipeline configuration YAML file",
    )
    parser.add_argument("--input", help="Input image/video path (required for image/video modes)")
    parser.add_argument("--output", help="Output path for processed video")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera ID for webcam mode")

    # Override arguments (optional - will override config file values)
    parser.add_argument("--device", help="Device to use (overrides config)")
    parser.add_argument("--img-size", type=int, help="Input image size (overrides config)")
    parser.add_argument(
        "--confidence", type=float, help="Detection confidence threshold (overrides config)"
    )
    parser.add_argument(
        "--multitask",
        action="store_true",
        help="Use multitask model instead of separate breed and emotion models",
    )

    args = parser.parse_args()

    # Load pipeline configuration
    try:
        config = load_pipeline_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Override config values with command line arguments if provided
    if args.device:
        config["pipeline"]["device"] = args.device
    if args.img_size:
        config["pipeline"]["img_size"] = args.img_size
    if args.confidence:
        config["pipeline"]["confidence_threshold"] = args.confidence

    # Load class names from config using multitask flag
    breed_classes = load_class_names_from_config(config, "breed", args.multitask)
    emotion_classes = load_class_names_from_config(config, "emotion", args.multitask)

    if not breed_classes or not emotion_classes:
        logger.error("Failed to load class names from configuration")
        sys.exit(1)

    logger.info(
        f"Loaded {len(breed_classes)} breed classes and {len(emotion_classes)} emotion classes"
    )
    logger.info(f"Using {'multitask' if args.multitask else 'single'} model mode")

    # Initialize pipeline with configuration
    pipeline = CatAnalysisPipeline(
        config=config,
        breed_classes=breed_classes,
        emotion_classes=emotion_classes,
        use_multitask=args.multitask,
    )

    # Run processing based on command
    if args.command == "image":
        if not args.input or not os.path.exists(args.input):
            logger.error("Input image path is required and must exist")
            sys.exit(1)

        # Load and process image
        image = cv2.imread(args.input)
        if image is None:
            logger.error(f"Failed to load image: {args.input}")
            sys.exit(1)

        annotated_image, results = pipeline.process_image(image)

        # Save or display result
        if args.output:
            cv2.imwrite(args.output, annotated_image)
            logger.info(f"Result saved to {args.output}")
        else:
            w, h = annotated_image.shape[1], annotated_image.shape[0]

            cv2.imshow("Cat Analysis Result", cv2.resize(annotated_image, (w // 2, h // 2)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Print results
        for i, result in enumerate(results):
            logger.info(
                f"Detection {i+1}: Breed={result['breed']} ({result['breed_confidence']:.2f}), "
                f"Emotion={result['emotion']} ({result['emotion_confidence']:.2f})"
            )

    elif args.command == "video":
        if not args.input or not os.path.exists(args.input):
            logger.error("Input video path is required and must exist")
            sys.exit(1)

        output_path = pipeline.process_video(
            video_path=args.input, output_path=args.output, display=True
        )

        if output_path:
            logger.info(f"Processed video saved to {output_path}")

    elif args.command == "webcam":
        pipeline.process_webcam(camera_id=args.camera_id)


if __name__ == "__main__":
    main()
