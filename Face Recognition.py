import cv2
import os

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def detect_faces():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    face_shake_count = 0
    save_dir = 'saved_videos'
    create_directory(save_dir)

   
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_path = os.path.join(save_dir, 'saved_video.avi')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            cv2.putText(frame, "No source", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

          
                out.write(frame)

                if face_shake_count >= 10:
                    print(f"Video saved: {save_path}")
                    face_shake_count = 0
                else:
                    face_shake_count += 1

        cv2.imshow('Face Recognition', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    detect_faces()

if __name__ == "__main__":
    main()
