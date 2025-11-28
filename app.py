import streamlit as st
import cv2
import face_recognition
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import dlib
from scipy.spatial import distance as dist
import os
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def load_encodings():
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None

def initialize_attendance_file():
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    attendance_file = f'Attendance_{today_date_str}.csv'
    
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=['Name', 'Timestamp'])
        df.to_csv(attendance_file, index=False)
    
    return attendance_file

def load_attendance_data():
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    attendance_file = f'Attendance_{today_date_str}.csv'
    
    try:
        df = pd.read_csv(attendance_file)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['Name', 'Timestamp'])

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'marked_today' not in st.session_state:
    st.session_state.marked_today = []
if 'liveness_info' not in st.session_state:
    st.session_state.liveness_info = {}
if 'camera' not in st.session_state:
    st.session_state.camera = None

# Main App
def main():
    st.markdown('<h1 class="main-header">👤 Smart Attendance System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Menu")
        menu_option = st.radio(
            "Select Option",
            ["Live Attendance", "Encode Faces", "View Attendance", "About"]
        )
        
        st.markdown("---")
        st.markdown("### System Status")
        
        # Check if encodings exist
        if load_encodings() is not None:
            st.success("✅ Face encodings loaded")
        else:
            st.error("❌ No face encodings found")
    
    # Main Content
    if menu_option == "Live Attendance":
        live_attendance_page()
    elif menu_option == "Encode Faces":
        encode_faces_page()
    elif menu_option == "View Attendance":
        view_attendance_page()
    elif menu_option == "About":
        about_page()

def live_attendance_page():
    st.header("📹 Live Attendance Tracking")
    
    # Check prerequisites
    encodings = load_encodings()
    if encodings is None:
        st.error("⚠️ Please encode faces first before starting attendance!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("Today's Attendance")
        attendance_placeholder = st.empty()
        
        st.subheader("Controls")
        
        if not st.session_state.running:
            if st.button("▶️ Start Attendance", key="start_btn"):
                st.session_state.running = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Attendance", key="stop_btn"):
                st.session_state.running = False
                if st.session_state.camera is not None:
                    st.session_state.camera.release()
                    st.session_state.camera = None
                st.rerun()
        
        # Display instructions
        st.markdown("---")
        st.markdown("### Instructions")
        st.info("""
        1. Click 'Start Attendance'
        2. Face the camera
        3. Wait for automatic recognition
        4. Attendance marked instantly!
        """)
    
    # Main attendance loop
    if st.session_state.running:
        run_attendance_system(video_placeholder, status_placeholder, attendance_placeholder, encodings)

def run_attendance_system(video_placeholder, status_placeholder, attendance_placeholder, encodings):
    # No liveness detection - direct recognition
    attendance_file = initialize_attendance_file()
    
    # Initialize camera if not already done
    if st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0)
        time.sleep(2)  # Give camera time to initialize
    
    video_capture = st.session_state.camera
    
    # Check if camera opened successfully
    if not video_capture.isOpened():
        status_placeholder.error("❌ Failed to open camera. Please check if camera is connected and not being used by another application.")
        st.session_state.running = False
        st.session_state.camera = None
        return
    
    status_placeholder.success("✅ Camera initialized successfully!")
    
    frame_count = 0
    
    try:
        while st.session_state.running:
            ret, frame = video_capture.read()
            
            if not ret:
                status_placeholder.error("Failed to capture frame from camera")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Face recognition on every 3rd frame for performance
            if frame_count % 3 == 0:
                # Resize frame for faster processing
                rgb_frame_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_frame_small)
                face_encodings = face_recognition.face_encodings(rgb_frame_small, face_locations)
                
                names = []
                
                for encoding in face_encodings:
                    name = "Unknown"
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(encodings["encodings"], encoding, tolerance=0.6)
                    
                    if True in matches:
                        matched_idxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        for idx in matched_idxs:
                            name_match = encodings["names"][idx]
                            counts[name_match] = counts.get(name_match, 0) + 1
                        name = max(counts, key=counts.get)
                    
                    # Mark attendance immediately when recognized
                    if name != "Unknown" and name not in st.session_state.marked_today:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        new_entry = pd.DataFrame([[name, timestamp]], columns=['Name', 'Timestamp'])
                        new_entry.to_csv(attendance_file, mode='a', header=False, index=False)
                        st.session_state.marked_today.append(name)
                        status_placeholder.success(f"✅ Marked {name} at {timestamp}")
                    
                    names.append(name)
                
                # Scale face locations back to original frame size
                boxes = [(int(top*2), int(right*2), int(bottom*2), int(left*2)) 
                        for (top, right, bottom, left) in face_locations]
            else:
                # Use previous detection results
                if 'boxes' not in locals():
                    boxes = []
                if 'names' not in locals():
                    names = []
            
            # Draw boxes and labels on frame
            for i, (top, right, bottom, left) in enumerate(boxes):
                name = names[i] if i < len(names) else "Unknown"
                
                # Color based on recognition status
                if name != "Unknown":
                    color = (0, 255, 0)  # Green for recognized
                else:
                    color = (0, 0, 255)  # Red for unknown
                
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", width="stretch")
            
            # Update attendance display every 30 frames
            if frame_count % 30 == 0:
                df = load_attendance_data()
                attendance_placeholder.dataframe(df, width="stretch")
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.03)
    
    except Exception as e:
        status_placeholder.error(f"Error during attendance tracking: {str(e)}")
    
    finally:
        # Don't release camera here, let the stop button handle it
        pass

def encode_faces_page():
    st.header("🔐 Encode Faces")
    
    st.info("📁 Place images in the 'dataset' folder with subfolders for each person (e.g., dataset/John/, dataset/Jane/)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Status")
        
        if os.path.exists('dataset'):
            people = [name for name in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', name))]
            if people:
                st.success(f"✅ Found {len(people)} people in dataset")
                for person in people:
                    person_dir = os.path.join('dataset', person)
                    num_images = len([f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))])
                    st.write(f"- **{person}**: {num_images} images")
            else:
                st.warning("⚠️ Dataset folder is empty")
        else:
            st.error("❌ Dataset folder not found")
    
    with col2:
        st.subheader("Encoding Actions")
        
        if st.button("🚀 Start Encoding"):
            if not os.path.exists('dataset'):
                st.error("❌ Dataset folder not found!")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            dataset_path = 'dataset'
            known_encodings = []
            known_names = []
            
            people = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
            
            if not people:
                st.error("❌ No people found in dataset!")
                return
            
            total_images = sum([len(os.listdir(os.path.join(dataset_path, name))) for name in people])
            processed = 0
            
            for name in people:
                person_dir = os.path.join(dataset_path, name)
                
                for filename in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, filename)
                    
                    try:
                        image = face_recognition.load_image_file(image_path)
                        boxes = face_recognition.face_locations(image, model='hog')
                        
                        if len(boxes) > 0:
                            encoding = face_recognition.face_encodings(image, boxes)[0]
                            known_encodings.append(encoding)
                            known_names.append(name)
                            status_text.success(f"✅ Processed {filename} for {name}")
                        else:
                            status_text.warning(f"⚠️ No face found in {filename}, skipping.")
                    except Exception as e:
                        status_text.error(f"❌ Error processing {filename}: {str(e)}")
                    
                    processed += 1
                    progress_bar.progress(processed / total_images)
            
            # Save encodings
            data = {"encodings": known_encodings, "names": known_names}
            with open("encodings.pickle", "wb") as f:
                f.write(pickle.dumps(data))
            
            st.success(f"🎉 Successfully encoded {len(known_encodings)} faces from {len(set(known_names))} people!")

def view_attendance_page():
    st.header("📊 View Attendance Records")
    
    # Date selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_date = st.date_input("Select Date", datetime.now())
    
    date_str = selected_date.strftime('%Y-%m-%d')
    attendance_file = f'Attendance_{date_str}.csv'
    
    # Display attendance
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        
        if not df.empty:
            st.success(f"✅ {len(df)} attendance records found for {date_str}")
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Present", len(df))
            with col2:
                st.metric("Unique People", df['Name'].nunique())
            with col3:
                if not df.empty:
                    first_time = df['Timestamp'].iloc[0]
                    st.metric("First Entry", first_time)
            
            # Attendance table
            st.subheader("Attendance Records")
            st.dataframe(df, width="stretch")
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=attendance_file,
                mime="text/csv"
            )
        else:
            st.info(f"No attendance records for {date_str}")
    else:
        st.warning(f"⚠️ No attendance file found for {date_str}")
    
    # View all attendance files
    st.subheader("All Attendance Files")
    attendance_files = [f for f in os.listdir('.') if f.startswith('Attendance_') and f.endswith('.csv')]
    
    if attendance_files:
        for file in sorted(attendance_files, reverse=True):
            with st.expander(f"📄 {file}"):
                df = pd.read_csv(file)
                st.dataframe(df, width="stretch")
    else:
        st.info("No attendance files found")

def about_page():
    st.header("ℹ️ About Smart Attendance System")
    
    st.markdown("""
    ### Features
    - 👤 **Face Recognition**: Accurate face detection and recognition
    - ⚡ **Instant Marking**: Automatic attendance without delays
    - 📊 **Automatic Logging**: Records attendance with timestamps
    - 🎨 **User-Friendly Interface**: Easy-to-use Streamlit interface
    
    ### How to Use
    1. **Encode Faces**: Add images to the `dataset` folder and encode them
    2. **Start Attendance**: Run live attendance tracking with your webcam
    3. **View Records**: Check attendance records by date
    
    ### Requirements
    - Python 3.7+
    - Webcam
    
    ### Technology Stack
    - OpenCV for video processing
    - face_recognition for face detection
    - Streamlit for UI
    - pandas for data management
    
    ### Setup Instructions
    1. Install requirements: `pip install -r requirements.txt`
    2. Create dataset folder with person subfolders
    3. Run: `streamlit run streamlit_app.py`
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

if __name__ == "__main__":
    main()