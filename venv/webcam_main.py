import face_recognition
import cv2

# --- Load your known face ---
print("Loading known face...")
my_photo = face_recognition.load_image_file("ryan.jpg")
my_encoding = face_recognition.face_encodings(my_photo)[0]

known_encodings = [my_encoding]
known_names = ["Ryan"]

print("Done! Starting camera...")

# --- Open webcam ---
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        print("Can't reach camera")
        break

    # Shrink frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]

        # Scale box back up
        top, right, bottom, left = face_location
        top *= 4; right *= 4; bottom *= 4; left *= 4

        # Draw box and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
