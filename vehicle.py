import cv2
import numpy as np
from ultralytics import YOLO
import math
import torch

class VehicleDetector:
    def __init__(self):
        # Check for CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name()}")
            print(f"Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
            print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
        
        # Initialize YOLO model with GPU
        self.model = YOLO('yolov8n.pt')
        self.model.to(self.device)
        
        # Define class names dictionary for vehicles
        self.vehicle_classes = {
            2: 'Car',
            5: 'Bus',
            7: 'Truck'
        }
        
        # Known parameters
        self.KNOWN_WIDTH = 1.8  # Average car width in meters
        self.FOCAL_LENGTH = 1000  # This needs to be calibrated for your camera
        
        # Define color for all visual elements (BGR format)
        self.COLOR = (0, 0, 255)  # Red in BGR
        
    def estimate_distance(self, pixel_width):
        """
        Estimate distance using the triangle similarity principle
        Distance = (Known width x Focal length) / Pixel width
        """
        distance = (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / pixel_width
        return distance
    
    def process_video(self, video_path, batch_size=4):
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties for output
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        output_path = 'output_detection.mp4'
        out = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, 
                            (frame_width, frame_height))
        
        frames_buffer = []
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frames_buffer.append(frame)
            
            # Process in batches to better utilize GPU
            if len(frames_buffer) >= batch_size:
                # Run YOLOv8 inference on batch
                results = self.model(frames_buffer, device=self.device)
                
                for frame, result in zip(frames_buffer, results):
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get confidence and class
                        conf = float(box.conf)
                        cls = int(box.cls)
                        
                        # Filter for vehicles (car, truck, bus)
                        if cls in self.vehicle_classes and conf > 0.5:
                            # Get vehicle type
                            vehicle_type = self.vehicle_classes[cls]
                            
                            # Calculate distance
                            pixel_width = x2 - x1
                            distance = self.estimate_distance(pixel_width)
                            
                            # Draw red bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR, 2)
                            
                            # Create label with vehicle type and confidence
                            label = f"{vehicle_type} ({conf:.2f})"
                            
                            # Calculate text size for background rectangle
                            (label_width, label_height), baseline = cv2.getTextSize(
                                label, 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                2
                            )
                            
                            # Draw background rectangle for label
                            cv2.rectangle(
                                frame,
                                (x1, y1 - label_height - baseline - 10),
                                (x1 + label_width, y1),
                                self.COLOR,
                                -1
                            )
                            
                            # Add vehicle label (white text on red background)
                            cv2.putText(
                                frame,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),  # White text
                                2
                            )
                            
                            # Display distance in red
                            distance_text = f"Distance: {distance:.2f}m"
                            cv2.putText(
                                frame,
                                distance_text,
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                self.COLOR,
                                2
                            )
                    
                    # Write frame to output video
                    out.write(frame)
                    
                    # Display frame (optional)
                    cv2.imshow('Vehicle Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        out.release()
                        cv2.destroyAllWindows()
                        return
                
                frames_buffer = []
        
        # Process any remaining frames
        if frames_buffer:
            results = self.model(frames_buffer, device=self.device)
            for frame, result in zip(frames_buffer, results):
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf)
                    cls = int(box.cls)
                    
                    if cls in self.vehicle_classes and conf > 0.5:
                        vehicle_type = self.vehicle_classes[cls]
                        pixel_width = x2 - x1
                        distance = self.estimate_distance(pixel_width)
                        
                        # Draw red bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR, 2)
                        
                        # Add vehicle label with confidence
                        label = f"{vehicle_type} ({conf:.2f})"
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            2
                        )
                        
                        # Draw background rectangle for label
                        cv2.rectangle(
                            frame,
                            (x1, y1 - label_height - baseline - 10),
                            (x1 + label_width, y1),
                            self.COLOR,
                            -1
                        )
                        
                        # Add vehicle label (white text on red background)
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),  # White text
                            2
                        )
                        
                        # Display distance in red
                        distance_text = f"Distance: {distance:.2f}m"
                        cv2.putText(
                            frame,
                            distance_text,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            self.COLOR,
                            2
                        )
                
                out.write(frame)
                cv2.imshow('Vehicle Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    # Initialize detector
    detector = VehicleDetector()
    
    # Process video
    video_path = 'D:\Hexylon\Vehicle_Detection\SampleVideo.mp4'  # Replace with your video path
    detector.process_video(video_path)

if __name__ == "__main__":
    main()