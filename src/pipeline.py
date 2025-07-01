import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from PIL import Image

# Import local modules
from engine.model import load_model

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
    Pipeline for tracking, detecting and analyzing cats in images/videos with flexible model support.

    Pipeline Flow:
    1. Input Processing (Image/Video/Webcam)
    2. Cat Tracking/Detection (YOLO tracking with persistent IDs or fallback to full image)
    3. Bounding Box Extraction (from YOLO tracking results)
    4. Frame Cropping (based on tracked bboxes)
    5. Classification (Multitask model OR separate breed/emotion models)
    6. Result Visualization (with tracking lines, refined label positioning, and track IDs)

    This ensures consistent processing across all input types with flexible model configuration.
    Supports both multitask models (single model for both tasks) and separate models.
    Tracking maintains persistent IDs across frames for videos and webcam feeds.
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

        # Initialize tracking state
        self.track_history = defaultdict(lambda: [])
        self.max_track_length = 30  # Retain 30 points for track visualization

        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Initialize YOLO tracking model if available and enabled
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
        """Initialize YOLO model for cat tracking."""
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

            logger.info("YOLO tracking model initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO model: {e}")
            self.yolo_model = None

    def track_animals(self, image: np.ndarray) -> List[Dict]:
        """
        Step 2: Track cats in the image using YOLO tracking or fallback.

        This is the core tracking step that runs for ALL input types:
        - For images: Single detection per image (no tracking needed)
        - For videos: Tracking per frame with persistent IDs
        - For webcam: Real-time tracking per frame with persistent IDs

        Args:
            image: Input image as numpy array

        Returns:
            List of detection dictionaries with bounding boxes, scores, and track IDs
            Format: [{"bbox": [x1, y1, x2, y2], "score": float, "label": str, "track_id": int}]
        """
        if not YOLO_AVAILABLE or self.yolo_model is None:
            # Fallback: assume the whole image contains a cat
            h, w = image.shape[:2]
            return [{"bbox": [0, 0, w, h], "score": 1.0, "label": "cat", "track_id": 0}]

        try:
            # Convert BGR to RGB for YOLO (YOLO expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run YOLO tracking inference (persist=True maintains tracks between frames)
            results = self.yolo_model.track(rgb_image, persist=True, verbose=False)
            # results = self.yolo_model(rgb_image, verbose=False)

            detections = []

            # Process YOLO tracking results
            for result in results:
                # Get detection boxes, scores, class IDs, and track IDs
                boxes = result.boxes
                if boxes is not None and boxes.is_track:
                    # Get track IDs
                    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []

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

                                # Get track ID if available
                                track_id = track_ids[i] if i < len(track_ids) else 0

                                # Update track history for visualization
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                track = self.track_history[track_id]
                                track.append((float(center_x), float(center_y)))

                                # Limit track history length
                                if len(track) > self.max_track_length:
                                    track.pop(0)

                                detections.append(
                                    {
                                        "bbox": [x1, y1, x2, y2],
                                        "score": confidence,
                                        "label": "cat",
                                        "track_id": track_id,
                                    }
                                )

            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x["score"], reverse=True)

            return detections

        except Exception as e:
            logger.warning(f"YOLO tracking failed: {e}")
            # Fallback: assume the whole image contains a cat
            h, w = image.shape[:2]
            return [{"bbox": [0, 0, w, h], "score": 1.0, "label": "cat", "track_id": 0}]

    def classify_breed_and_emotion(
        self, image_crop: np.ndarray
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Steps 5-6: Classify breed and emotion from a cropped cat image.

        This runs on EVERY detected bounding box from the detection step.
        The same classification logic applies to all input types.
        Supports both multitask and single model modes.

        Args:
            image_crop: Cropped cat image as numpy array (from detected bbox)

        Returns:
            Tuple of (breed_top3, emotion_top3) where each is a list of (label, confidence) tuples
        """
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_multitask:
                # Use multitask model for both predictions
                breed_logits = self.model(input_tensor, task="breed")
                emotion_logits = self.model(input_tensor, task="emotion")
            else:
                # Use separate models for breed and emotion
                breed_logits = self.breed_model(input_tensor)
                emotion_logits = self.emotion_model(input_tensor)

            # Process breed predictions - get top 3
            breed_probs = F.softmax(breed_logits, dim=1)
            breed_top3_probs, breed_top3_indices = torch.topk(breed_probs, k=3, dim=1)
            breed_top3 = [
                (self.breed_classes[idx.item()], prob.item())
                for idx, prob in zip(breed_top3_indices[0], breed_top3_probs[0])
            ]

            # Process emotion predictions - get top 3
            emotion_probs = F.softmax(emotion_logits, dim=1)
            emotion_top3_probs, emotion_top3_indices = torch.topk(emotion_probs, k=3, dim=1)
            emotion_top3 = [
                (self.emotion_classes[idx.item()], prob.item())
                for idx, prob in zip(emotion_top3_indices[0], emotion_top3_probs[0])
            ]

        return breed_top3, emotion_top3

    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Complete pipeline processing for a single image/frame.

        This method implements the full pipeline flow:
        1. Input: Single image/frame
        2. Tracking: Animal tracking/detection
        3. Bbox Extraction: Get bounding boxes from tracking results
        4. Cropping: Extract cat regions based on bboxes
        5. Classification: Breed and emotion classification (multitask OR single models)
        6. Visualization: Draw results with refined label positioning and track visualization

        Used by: image processing, video frame processing, webcam frame processing

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (annotated_image, detection_results)
        """
        # Calculate FPS
        fps = self._calculate_fps()

        # Step 2: Track animals using YOLO tracking or fallback
        detections = self.track_animals(image)

        results = []
        annotated_image = image.copy()

        # Step 3: Process each detected/tracked bounding box
        for detection in detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            track_id = detection.get("track_id", 0)

            # Step 3a: Ensure bbox is within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Step 4: Crop the detected region for classification
            crop = image[y1:y2, x1:x2]

            if crop.size > 0:
                # Step 5: Classification on cropped frame (breed and emotion) - now returns top-3
                breed_top3, emotion_top3 = self.classify_breed_and_emotion(crop)

                # Store results for this detection
                result = {
                    "bbox": [x1, y1, x2, y2],
                    "detection_score": detection["score"],
                    "breed_top3": breed_top3,
                    "emotion_top3": emotion_top3,
                    "track_id": track_id,
                }
                results.append(result)

                # Step 6: Visualization - Draw tracking lines first
                if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                    # Draw tracking lines
                    track_points = self.track_history[track_id]
                    points = np.array(track_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_image,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=3,
                    )

                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw detection confidence on bounding box (top-left corner)
                confidence_text = f"cat: {detection['score']:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                # Get text size for background
                text_size = cv2.getTextSize(confidence_text, font, font_scale, thickness)[0]

                # Draw background rectangle for confidence text
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - text_size[1] - 10),
                    (x1 + text_size[0] + 10, y1),
                    (0, 255, 0),  # Green background
                    -1,
                )

                # Draw confidence text
                cv2.putText(
                    annotated_image,
                    confidence_text,
                    (x1 + 5, y1 - 5),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text
                    thickness,
                )

        # Draw information panel at top-left
        annotated_image = self._draw_info_panel(annotated_image, results, fps)

        return annotated_image, results

    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time

        if elapsed >= 1.0:  # Update FPS every second
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = current_time

        return self.current_fps

    def _draw_info_panel(self, image: np.ndarray, results: List[Dict], fps: float) -> np.ndarray:
        """
        Draw information panel at top-left with FPS and classification results.
        Panel size is proportional to the frame size (approx. 20% of width).

        Args:
            image: Input image
            results: Detection results with breed and emotion top-3 predictions
            fps: Current FPS value

        Returns:
            Image with information panel drawn
        """
        if not results:
            return image

        # Get image dimensions
        h, w = image.shape[:2]

        # Make panel proportional to frame size
        target_panel_width_ratio = 0.20  # 20% of frame width

        # Panel configuration
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Scale font based on image width
        font_scale = max(0.4, min(0.7, w / 1920))  # Scale between 0.4-0.7 based on width
        thickness = 1
        line_height = int(20 * (w / 1280))  # Scale line height based on width
        padding = int(10 * (w / 1280))  # Scale padding based on width
        panel_x = padding
        panel_y = padding

        # Prepare text lines
        info_lines = []
        info_lines.append(f"FPS: {fps:.1f}")
        info_lines.append("")  # Empty line for spacing

        # Add results for each detection
        for i, result in enumerate(results):
            track_id = result.get("track_id", "N/A")
            info_lines.append(f"Detection {i+1} (ID: {track_id}):")

            # Breed top-3
            info_lines.append("Breed:")
            for j, (breed, conf) in enumerate(result["breed_top3"]):
                info_lines.append(f"  {j+1}. {breed} {conf*100:.2f}%")

            # Emotion top-3
            info_lines.append("Emotion:")
            for j, (emotion, conf) in enumerate(result["emotion_top3"]):
                info_lines.append(f"  {j+1}. {emotion} {conf*100:.2f}%")

            if i < len(results) - 1:  # Add spacing between detections
                info_lines.append("")

        # Target panel width (20% of frame width)
        target_panel_width = int(w * target_panel_width_ratio)

        # Calculate initial panel dimensions
        max_text_width = 0
        for line in info_lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            max_text_width = max(max_text_width, text_size[0])

        # Adjust font scale if text is too wide for target panel width
        if max_text_width > (target_panel_width - 2 * padding):
            scale_factor = (target_panel_width - 2 * padding) / max_text_width
            font_scale *= scale_factor

            # Recalculate max text width with new font scale
            max_text_width = 0
            for line in info_lines:
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                max_text_width = max(max_text_width, text_size[0])

        # Set panel width to target or text width, whichever is larger
        panel_width = max(target_panel_width, max_text_width + (2 * padding))
        panel_height = len(info_lines) * line_height + (2 * padding)

        # Ensure panel doesn't exceed frame dimensions
        panel_width = min(panel_width, w - 2 * padding)
        panel_height = min(panel_height, h - 2 * padding)

        # Draw transparent black background
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),  # Black color
            -1,
        )
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Draw text lines
        for i, line in enumerate(info_lines):
            if line.strip():  # Skip empty lines
                text_y = panel_y + padding + (i + 1) * line_height

                # Check if text fits within panel width
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                if text_size[0] > panel_width - 2 * padding:
                    # Truncate text if it's too long
                    while text_size[0] > panel_width - 2 * padding and len(line) > 3:
                        line = line[:-4] + "..."
                        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]

                cv2.putText(
                    image,
                    line,
                    (panel_x + padding, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    thickness,
                )

        return image

    def process_video(
        self, video_path: str, output_path: Optional[str] = None, display: bool = True
    ) -> Optional[str]:
        """
        Process a video file using the same pipeline flow as single images with tracking.

        Pipeline Flow (applied to each frame):
        1. Input: Video frame
        2-6. Same as process_image() for each frame with persistent tracking

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
        Process webcam feed in real-time using the same pipeline flow with tracking.

        Pipeline Flow (applied to each frame):
        1. Input: Webcam frame
        2-6. Same as process_image() for each frame in real-time with persistent tracking

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
    Main function for running the unified cat analysis pipeline with tracking.

    ALL commands (image, video, webcam) use the same 6-step pipeline flow:
    YOLO Tracking → Bbox Extraction → Cropping → Classification → Visualization

    Classification can be done in two modes:
    - Multitask mode (--multitask): Single model for both breed and emotion
    - Single model mode (default): Separate models for breed and emotion

    The only difference between commands is the input source:
    - image: Single image file processing (no persistent tracking)
    - video: Frame-by-frame video processing with persistent tracking
    - webcam: Real-time webcam frame processing with persistent tracking

    Tracking features:
    - Persistent track IDs across frames for videos and webcam
    - Visual tracking lines showing movement paths
    - Track ID display on bounding boxes
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

            # cv2.imshow("Cat Analysis Result", cv2.resize(annotated_image, (w // 2, h // 2)))
            cv2.imshow("Cat Analysis Result", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Print results
        for i, result in enumerate(results):
            breed_top3 = result["breed_top3"]
            emotion_top3 = result["emotion_top3"]

            breed_str = ", ".join([f"{label} {conf*100:.2f}%" for label, conf in breed_top3])
            emotion_str = ", ".join([f"{label} {conf*100:.2f}%" for label, conf in emotion_top3])

            logger.info(
                f"Detection {i+1} (Track ID: {result.get('track_id', 'N/A')}): "
                f"Breed=[{breed_str}], Emotion=[{emotion_str}]"
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
