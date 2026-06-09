# Face Recognition Security System

## Overview

This project is a Face Recognition Security System built using Python, OpenCV, and Streamlit. It identifies authorized users through facial recognition and can trigger a security alert when an unknown person is detected.

The application provides a simple web interface using Streamlit and utilizes a trained face recognition model for real-time detection.

## Features

* Real-time face detection
* Face recognition using trained facial data
* Authorized user identification
* Unknown person detection
* Security alert image generation
* Streamlit-based user interface
* Easy dataset management and model retraining

## Technologies Used

* Python
* OpenCV
* NumPy
* Streamlit

## Project Structure

```text
face_app/
│
├── app.py                 # Main Streamlit application
├── trainer.yml            # Trained face recognition model
├── labels.npy             # Label mappings
├── dataset/               # Face image dataset
├── security_alert.jpg     # Alert image
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Praney1409/face_recognition_final.git
cd face_recognition_final
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

Open the URL shown in the terminal, typically:

```text
http://localhost:8501
```

## How It Works

1. The camera captures live video frames.
2. Faces are detected using OpenCV.
3. The trained model compares detected faces with stored facial data.
4. Authorized users are identified and displayed.
5. Unknown users trigger a security alert.

## Future Improvements

* Email notifications for intruder detection
* Attendance logging system
* Database integration
* Cloud deployment
* Face registration through web interface
* Multi-camera support

## Author

**Praney Sanotra**

B.Tech Electronics and Communication Engineering (ECE)

## License

This project is created for educational and learning purposes.
