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
import plotly.express as px
import plotly.graph_objects as go

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
    .role-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
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

def get_all_attendance_files():
    """Get all attendance CSV files"""
    return [f for f in os.listdir('.') if f.startswith('Attendance_') and f.endswith('.csv')]

def calculate_student_statistics(student_name):
    """Calculate attendance statistics for a student"""
    attendance_files = get_all_attendance_files()
    
    total_days = len(attendance_files)
    present_days = 0
    attendance_dates = []
    
    for file in attendance_files:
        df = pd.read_csv(file)
        if student_name in df['Name'].values:
            present_days += 1
            date = file.replace('Attendance_', '').replace('.csv', '')
            attendance_dates.append(date)
    
    percentage = (present_days / total_days * 100) if total_days > 0 else 0
    
    return {
        'total_days': total_days,
        'present_days': present_days,
        'absent_days': total_days - present_days,
        'percentage': percentage,
        'attendance_dates': attendance_dates
    }

def get_all_students():
    """Get list of all students from encodings"""
    encodings = load_encodings()
    if encodings:
        return list(set(encodings['names']))
    return []

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'marked_today' not in st.session_state:
    st.session_state.marked_today = []
if 'camera' not in st.session_state:
    st.session_state.camera = None

# Main App
def main():
    st.markdown('<h1 class="main-header">👤 Smart Attendance System</h1>', unsafe_allow_html=True)
    
    # Sidebar - Role Selection
    with st.sidebar:
        st.header("🔐 Select Role")
        role = st.radio(
            "Choose your role:",
            ["👨‍💼 Admin", "👨‍🏫 Faculty", "👨‍🎓 Student"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Role-based menu
        if role == "👨‍💼 Admin":
            st.markdown("### 👨‍💼 Admin Panel")
            menu_option = st.radio(
                "Select Option",
                ["Encode Faces", "System Info"],
                label_visibility="collapsed"
            )
        
        elif role == "👨‍🏫 Faculty":
            st.markdown("### 👨‍🏫 Faculty Panel")
            menu_option = st.radio(
                "Select Option",
                ["Live Attendance", "View Attendance"],
                label_visibility="collapsed"
            )
        
        elif role == "👨‍🎓 Student":
            st.markdown("### 👨‍🎓 Student Panel")
            menu_option = "Student Dashboard"
        
        st.markdown("---")
        st.markdown("### System Status")
        
        # Check if encodings exist
        if load_encodings() is not None:
            st.success("✅ Face encodings loaded")
        else:
            st.error("❌ No face encodings found")
    
    # Route to appropriate page based on role and menu
    if role == "👨‍💼 Admin":
        if menu_option == "Encode Faces":
            encode_faces_page()
        elif menu_option == "System Info":
            system_info_page()
    
    elif role == "👨‍🏫 Faculty":
        if menu_option == "Live Attendance":
            live_attendance_page()
        elif menu_option == "View Attendance":
            view_attendance_page()
    
    elif role == "👨‍🎓 Student":
        student_dashboard_page()

def encode_faces_page():
    st.markdown('<h2 class="role-header">👨‍💼 Admin - Encode Faces</h2>', unsafe_allow_html=True)
    
    st.info("📁 Place images in the 'dataset' folder with subfolders for each person (e.g., dataset/ADITTYA SAHA/, dataset/BITTU KUMAR AZAD/)")
    
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

def system_info_page():
    st.markdown('<h2 class="role-header">👨‍💼 Admin - System Information</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Features
    - 👤 **Face Recognition**: Accurate face detection and recognition
    - ⚡ **Instant Marking**: Automatic attendance without delays
    - 📊 **Automatic Logging**: Records attendance with timestamps
    - 🎨 **User-Friendly Interface**: Easy-to-use Streamlit interface
    - 📈 **Student Analytics**: Detailed attendance statistics and visualizations
    
    ### How to Use
    1. **Admin**: Encode faces by adding images to the dataset folder
    2. **Faculty**: Run live attendance tracking and view records
    3. **Student**: View personal attendance statistics and graphs
    
    ### Technology Stack
    - OpenCV for video processing
    - face_recognition for face detection
    - Streamlit for UI
    - pandas for data management
    - Plotly for visualizations
    """)
    
    # System statistics
    st.subheader("📊 System Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    encodings = load_encodings()
    attendance_files = get_all_attendance_files()
    
    with col1:
        if encodings:
            st.metric("Enrolled Students", len(set(encodings['names'])))
        else:
            st.metric("Enrolled Students", 0)
    
    with col2:
        st.metric("Attendance Days", len(attendance_files))
    
    with col3:
        if attendance_files:
            total_records = sum([len(pd.read_csv(f)) for f in attendance_files])
            st.metric("Total Records", total_records)
        else:
            st.metric("Total Records", 0)

def live_attendance_page():
    st.markdown('<h2 class="role-header">👨‍🏫 Faculty - Live Attendance</h2>', unsafe_allow_html=True)
    
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
    attendance_file = initialize_attendance_file()
    
    # Initialize camera if not already done
    if st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0)
        time.sleep(2)
    
    video_capture = st.session_state.camera
    
    if not video_capture.isOpened():
        status_placeholder.error("❌ Failed to open camera. Please check if camera is connected.")
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
            
            if frame_count % 3 == 0:
                rgb_frame_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame_small)
                face_encodings = face_recognition.face_encodings(rgb_frame_small, face_locations)
                
                names = []
                
                for encoding in face_encodings:
                    name = "Unknown"
                    matches = face_recognition.compare_faces(encodings["encodings"], encoding, tolerance=0.6)
                    
                    if True in matches:
                        matched_idxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        for idx in matched_idxs:
                            name_match = encodings["names"][idx]
                            counts[name_match] = counts.get(name_match, 0) + 1
                        name = max(counts, key=counts.get)
                    
                    if name != "Unknown" and name not in st.session_state.marked_today:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        new_entry = pd.DataFrame([[name, timestamp]], columns=['Name', 'Timestamp'])
                        new_entry.to_csv(attendance_file, mode='a', header=False, index=False)
                        st.session_state.marked_today.append(name)
                        status_placeholder.success(f"✅ Marked {name} at {timestamp}")
                    
                    names.append(name)
                
                boxes = [(int(top*2), int(right*2), int(bottom*2), int(left*2)) 
                        for (top, right, bottom, left) in face_locations]
            else:
                if 'boxes' not in locals():
                    boxes = []
                if 'names' not in locals():
                    names = []
            
            for i, (top, right, bottom, left) in enumerate(boxes):
                name = names[i] if i < len(names) else "Unknown"
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            if frame_count % 30 == 0:
                df = load_attendance_data()
                attendance_placeholder.dataframe(df, use_container_width=True)
            
            time.sleep(0.03)
    
    except Exception as e:
        status_placeholder.error(f"Error during attendance tracking: {str(e)}")

def view_attendance_page():
    st.markdown('<h2 class="role-header">👨‍🏫 Faculty - View Attendance</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_date = st.date_input("Select Date", datetime.now())
    
    date_str = selected_date.strftime('%Y-%m-%d')
    attendance_file = f'Attendance_{date_str}.csv'
    
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        
        if not df.empty:
            st.success(f"✅ {len(df)} attendance records found for {date_str}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Present", len(df))
            with col2:
                st.metric("Unique People", df['Name'].nunique())
            with col3:
                if not df.empty:
                    first_time = df['Timestamp'].iloc[0]
                    st.metric("First Entry", first_time)
            
            st.subheader("Attendance Records")
            st.dataframe(df, use_container_width=True)
            
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
    
    st.subheader("All Attendance Files")
    attendance_files = get_all_attendance_files()
    
    if attendance_files:
        for file in sorted(attendance_files, reverse=True):
            with st.expander(f"📄 {file}"):
                df = pd.read_csv(file)
                st.dataframe(df, use_container_width=True)
    else:
        st.info("No attendance files found")

def student_dashboard_page():
    st.markdown('<h2 class="role-header">👨‍🎓 Student - Attendance Dashboard</h2>', unsafe_allow_html=True)
    
    # Get all students
    students = get_all_students()
    
    if not students:
        st.warning("⚠️ No students found in the system. Please ask admin to encode faces.")
        return
    
    # Student selection
    selected_student = st.selectbox("Select Student Name", sorted(students))
    
    if selected_student:
        stats = calculate_student_statistics(selected_student)
        
        st.markdown("---")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Days", stats['total_days'])
        with col2:
            st.metric("Present Days", stats['present_days'], delta=None)
        with col3:
            st.metric("Absent Days", stats['absent_days'])
        with col4:
            st.metric("Attendance %", f"{stats['percentage']:.1f}%")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for Present vs Absent
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Present', 'Absent'],
                values=[stats['present_days'], stats['absent_days']],
                marker=dict(colors=['#2ecc71', '#e74c3c']),
                hole=0.4
            )])
            fig_pie.update_layout(
                title="Attendance Distribution",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Gauge chart for attendance percentage
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=stats['percentage'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Attendance Percentage"},
                delta={'reference': 75},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffebee"},
                        {'range': [50, 75], 'color': "#fff9c4"},
                        {'range': [75, 100], 'color': "#e8f5e9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Attendance timeline
        if stats['attendance_dates']:
            st.subheader("📅 Attendance Timeline")
            
            # Create timeline data
            all_files = sorted(get_all_attendance_files())
            dates = [f.replace('Attendance_', '').replace('.csv', '') for f in all_files]
            status = ['Present' if date in stats['attendance_dates'] else 'Absent' for date in dates]
            
            df_timeline = pd.DataFrame({
                'Date': dates,
                'Status': status
            })
            
            # Bar chart for timeline
            fig_timeline = px.bar(
                df_timeline,
                x='Date',
                y=[1]*len(dates),
                color='Status',
                color_discrete_map={'Present': '#2ecc71', 'Absent': '#e74c3c'},
                title="Daily Attendance Status"
            )
            fig_timeline.update_layout(
                showlegend=True,
                yaxis_title="",
                yaxis_showticklabels=False,
                height=300
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Detailed attendance records
            with st.expander("📋 View Detailed Attendance Records"):
                records = []
                for date in stats['attendance_dates']:
                    file = f'Attendance_{date}.csv'
                    if os.path.exists(file):
                        df = pd.read_csv(file)
                        student_record = df[df['Name'] == selected_student]
                        if not student_record.empty:
                            records.append({
                                'Date': date,
                                'Time': student_record['Timestamp'].iloc[0]
                            })
                
                if records:
                    df_records = pd.DataFrame(records)
                    st.dataframe(df_records, use_container_width=True)
        else:
            st.info("No attendance records found for this student.")
        
        # Performance message
        st.markdown("---")
        if stats['percentage'] >= 75:
            st.success("🎉 Great job! Your attendance is above 75%")
        elif stats['percentage'] >= 50:
            st.warning("⚠️ Your attendance is below 75%. Please maintain regular attendance.")
        else:
            st.error("❌ Your attendance is critically low. Please attend regularly!")

if __name__ == "__main__":
    main()
   
