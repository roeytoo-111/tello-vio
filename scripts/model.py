from djitellopy import Tello
from ultralytics import YOLO
import cv2

# 1. Load the model (use the Nano model for speed, or your custom weights)
model = YOLO("yolov8n.pt") 

# 2. Connect to the Tello
tello = Tello()

try:
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    # 3. Start the video stream and get the background frame reader
    tello.streamon()
    frame_read = tello.get_frame_read()

    while True:
        # Fetch the most recent frame directly from the background thread
        frame = frame_read.frame
        
        # --- UPDATE 1: Prevent crash from empty frames before stream fully starts ---
        if frame is None or frame.size == 0:
            continue
        
        # Run YOLO inference
        # 'stream=True' is a memory-efficient generator, 'verbose=False' hides terminal spam
        results = model(frame, stream=True, verbose=False)
        
        # Parse results and draw accurate BBs
        for r in results:
            
            # --- UPDATE 2: Uncomment the line below to debug if YOLO works but window doesn't show (WSL issue) ---
            # print(f"Detected {len(r.boxes)} objects")
            
            for box in r.boxes:
                # Extract coordinates, class ID, and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Draw the Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate and draw the center point (crucial for tracking error)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Add label
                cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        # Show the live feed
        cv2.imshow("Tello Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- UPDATE 3: Guaranteed cleanup to prevent 'Address already in use' error ---
    print("Closing connection and releasing UDP port...")
    tello.streamoff()
    cv2.destroyAllWindows()
    tello.end()