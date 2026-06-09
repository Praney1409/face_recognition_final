import streamlit as st
import cv2
import numpy as np
import os

# ---------------- SYSTEM PATHS ----------------

DATASET_DIR = "dataset"
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.npy"

os.makedirs(DATASET_DIR, exist_ok=True)

# ---------------- FACE DETECTION + MODEL ----------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

# ---------------- IMAGE ENHANCEMENT ----------------

def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

# ---------------- CAPTURE PERSON DATA ----------------

def register_person(name):

    person_folder = os.path.join(DATASET_DIR, name)
    os.makedirs(person_folder, exist_ok=True)

    cam = cv2.VideoCapture(0)

    img_count = 0

    st.info(f"Registering identity for {name}. Capturing face samples...")

    while True:

        ret, frame = cam.read()

        if not ret:
            st.error("Camera access failed.")
            break

        frame = enhance_frame(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100,100)
        )

        for (x,y,w,h) in faces:

            img_count += 1

            face_img = cv2.resize(gray[y:y+h, x:x+w], (200,200))

            cv2.imwrite(
                os.path.join(person_folder,f"{name}_{img_count}.jpg"),
                face_img
            )

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("Security Registration Camera",frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or img_count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()

    st.success(f"{img_count} face samples captured for {name}")

# ---------------- TRAIN MODEL ----------------

def train_recognition_model():

    faces = []
    labels = []

    label_map = {}
    current_label = 0

    for person in os.listdir(DATASET_DIR):

        person_path = os.path.join(DATASET_DIR,person)

        if not os.path.isdir(person_path):
            continue

        for img_file in os.listdir(person_path):

            img_path = os.path.join(person_path,img_file)

            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img,(200,200))

            if person not in label_map:

                label_map[person] = current_label
                current_label += 1

            faces.append(img)
            labels.append(label_map[person])

    if len(faces) == 0:

        st.warning("No registered identities found.")
        return

    recognizer.train(faces,np.array(labels))

    recognizer.save(MODEL_FILE)

    np.save(LABELS_FILE,label_map)

    st.success("AI recognition model updated successfully.")

# ---------------- SURVEILLANCE MODE ----------------

def start_surveillance():

    if not os.path.exists(MODEL_FILE):

        st.error("Recognition model not found. Train the system first.")
        return

    recognizer.read(MODEL_FILE)

    label_map = np.load(LABELS_FILE,allow_pickle=True).item()

    reverse_map = {v:k for k,v in label_map.items()}

    cam = cv2.VideoCapture(0)

    st.info("Live surveillance activated. Press 'q' to terminate.")

    while True:

        ret, frame = cam.read()

        if not ret:
            break

        frame = enhance_frame(frame)

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100,100)
        )

        for (x,y,w,h) in faces:

            face_img = cv2.resize(gray[y:y+h,x:x+w],(200,200))

            label, confidence = recognizer.predict(face_img)

            if confidence < 80:

                name = reverse_map[label]
                display_text = f"{name}  | Confidence {int(confidence)}%"
                color = (0,255,0)

            else:

                display_text = "⚠ Unauthorized Individual"
                color = (0,0,255)

                # save alert snapshot
                cv2.imwrite("security_alert.jpg",frame)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            cv2.putText(
                frame,
                display_text,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        cv2.imshow("AI Security Surveillance",frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    st.success("Surveillance session ended.")

# ---------------- STREAMLIT UI ----------------

st.set_page_config(
    page_title="AI Surveillance System",
    layout="centered"
)

st.title("🔐 AI Surveillance & Identity Verification System")

st.caption(
"Real-time biometric monitoring platform designed for secure environments"
)

menu = st.sidebar.radio(
    "Security Control Panel",
    [
        "Register Authorized Personnel",
        "Update Recognition Model",
        "Activate Live Surveillance",
        "Authorized Personnel Database"
    ]
)

# ---------------- MENU FUNCTIONS ----------------

if menu == "Register Authorized Personnel":

    name = st.text_input("Enter personnel name")

    if st.button("Register Identity"):

        if name.strip():

            register_person(name.strip())

            st.info("Updating AI model with new identity...")

            train_recognition_model()

        else:

            st.warning("Please enter a valid name.")

elif menu == "Update Recognition Model":

    if st.button("Update AI Model"):

        train_recognition_model()

elif menu == "Activate Live Surveillance":

    if st.button("Start Surveillance"):

        start_surveillance()

elif menu == "Authorized Personnel Database":

    if os.path.exists(DATASET_DIR):

        people = os.listdir(DATASET_DIR)

        if people:

            st.subheader("Authorized Personnel Records")

            for person in people:

                person_dir = os.path.join(DATASET_DIR,person)

                images = [
                    os.path.join(person_dir,img)
                    for img in os.listdir(person_dir)[:3]
                ]

                st.write(
                    f"👤 {person}  | {len(os.listdir(person_dir))} registered samples"
                )

                st.image(images,width=150)

        else:

            st.warning("No personnel registered yet.")