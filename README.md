# Virtual Exam Monitoring System
A real-time AI-based proctoring system that enhances academic integrity by tracking student gaze direction, head orientation, absence, and multiple faces using webcam feed and facial landmarks.

## Features
- Eye gaze direction detection (left, right, up, down)
- Head pose estimation (tilt, turn)
- Face presence & multiple face detection
- Screenshots captured on violation
- Logs violations with timestamps in CSV
- Real-time webcam monitoring using OpenCV

## Tech Stack
- **Programming Language**: Python
- **Libraries Used**: OpenCV, MediaPipe (Face Mesh, Iris), NumPy, Pandas
  
## How It Works
1. Accesses the webcam via OpenCV
2. Detects face mesh landmarks using MediaPipe
3. Classifies gaze direction and head tilt
4. Logs and captures screenshots on rule violations:
   - Student absent
   - Multiple people detected
   - Looking away or tilting head





