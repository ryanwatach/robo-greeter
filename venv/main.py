import face_recognition
import cv2
import threading

# --- Threaded camera class to avoid buffer lag ---
class Camera:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()

# --- Load known face ---
print("Loading known face...")
my_photo = face_recognition.load_image_file("ryan.jpg")
my_encoding = face_recognition.face_encodings(my_photo)[0]

known_encodings = [my_encoding]
known_names = ["Ryan"]

print("Done! Starting camera...")

# --- Start threaded camera ---
video = Camera("rtsp://admin:admin@192.168.1.108/cam/realmonitor?channel=1&subtype=0")

process_this_frame = True

while True:
    ret, frame = video.read()
    if not ret:
        print("Can't reach camera")
        break

    # Only process every other frame to save CPU
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                name = known_names[matches.index(True)]
            results.append((face_location, name))

    process_this_frame = not process_this_frame

    # Draw boxes
    for face_location, name in results if 'results' in dir() else []:
        top, right, bottom, left = face_location
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.stop()
cv2.destroyAllWindows()