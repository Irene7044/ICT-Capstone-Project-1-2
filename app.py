from PySide6.QtWidgets import (
    QApplication, QPushButton, QFileDialog, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QDialog, QTextEdit, QMessageBox, QProgressBar
)
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
import sys
import os
import shutil
import subprocess
import cv2

#Setup folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
REPORT_FOLDER = "reports"
INPUT_FOLDER = "testing resource"
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True) 

#Video detection 
class VideoDetectionThread(QThread):
    progress = Signal(int)
    finished = Signal(str)

    def __init__(self, input_path, output_path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        import detect  
        model = detect.model

        cap = cv2.VideoCapture(self.input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect._run_models_on_frame(frame)
            out.write(frame)
            frame_count += 1
            self.progress.emit(int((frame_count / total_frames) * 100))

        cap.release()
        out.release()
        self.finished.emit(self.output_path)

# Image Detection
def detect_image(image_path):
    import detect
    model = detect.model

    processing_label.setText("Processing...")
    QApplication.processEvents()

    img = cv2.imread(image_path)
    if img is None:
        QMessageBox.warning(window, "Error", "Failed to read image.")
        processing_label.setText("")
        progress_bar.setValue(0)
        return

    img = detect._run_models_on_frame(img)

    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, img)
    progress_bar.setValue(100)
    processing_label.setText("")
    progress_bar.setValue(0) 
    QMessageBox.information(window, "Result Saved", f"Image result saved to {result_path}")

# Video Detection
def detect_video(video_path):
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(video_path))
    global thread
    processing_label.setText("Processing...")
    QApplication.processEvents()
    thread = VideoDetectionThread(video_path, result_path)
    thread.progress.connect(update_progress)
    thread.finished.connect(video_finished)
    thread.start()

def update_progress(value):
    progress_bar.setValue(value)
    QApplication.processEvents()

def video_finished(result_path):
    progress_bar.setValue(100)
    processing_label.setText("")
    progress_bar.setValue(0) 
    QMessageBox.information(window, "Detection Complete", f"Video result saved to {result_path}")

#Upload functions
def upload_image():
    start_dir = os.path.abspath(INPUT_FOLDER)
    file_path, _ = QFileDialog.getOpenFileName(
        window,
        "Select Image",
        start_dir,
        "Images (*.jpg *.png *.jpeg)"
    )
    if file_path:
        dest_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file_path))
        shutil.copy(file_path, dest_path)
        QMessageBox.information(window, "Uploaded", f"Image uploaded to {dest_path}")
        detect_image(dest_path)

def upload_video():
    start_dir = os.path.abspath(INPUT_FOLDER)
    file_path, _ = QFileDialog.getOpenFileName(
        window,
        "Select Video",
        start_dir,
        "Videos (*.mp4 *.avi *.mov)"
    )
    if file_path:
        dest_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file_path))
        shutil.copy(file_path, dest_path)
        QMessageBox.information(window, "Uploaded", f"Video uploaded to {dest_path}")
        detect_video(dest_path)

#View folders / preview files
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
TEXT_EXTENSIONS = (".txt", ".csv", ".json", ".log")

def show_image_preview(file_path, title="Image Preview"):
    dialog = QDialog(window)
    dialog.setWindowTitle(title)
    dialog.resize(900, 650)

    layout = QVBoxLayout()

    image_label = QLabel()
    image_label.setAlignment(Qt.AlignCenter)

    pixmap = QPixmap(file_path)
    if pixmap.isNull():
        QMessageBox.warning(window, "Error", "Could not load image preview.")
        return

    image_label.setPixmap(
        pixmap.scaled(850, 550, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    )
    layout.addWidget(image_label)

    close_btn = QPushButton("Close")
    close_btn.setFixedSize(100, 40)
    close_btn.setStyleSheet(
        "QPushButton {background-color:white;color:black;border-radius:8px;}"
        "QPushButton:hover{background-color:#f0f0f0}"
    )
    close_btn.clicked.connect(dialog.close)
    layout.addWidget(close_btn, alignment=Qt.AlignCenter)

    dialog.setLayout(layout)
    dialog.exec()

def show_text_preview(file_path, title="File Preview"):
    dialog = QDialog(window)
    dialog.setWindowTitle(title)
    dialog.resize(900, 650)

    layout = QVBoxLayout()

    text_box = QTextEdit()
    text_box.setReadOnly(True)
    text_box.setStyleSheet("background-color: white; color: black;")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text_box.setText(f.read())
    except Exception as e:
        text_box.setText(f"Could not open file:\n{file_path}\n\n{e}")

    layout.addWidget(text_box)

    close_btn = QPushButton("Close")
    close_btn.setFixedSize(100, 40)
    close_btn.setStyleSheet(
        "QPushButton {background-color:white;color:black;border-radius:8px;}"
        "QPushButton:hover{background-color:#f0f0f0}"
    )
    close_btn.clicked.connect(dialog.close)
    layout.addWidget(close_btn, alignment=Qt.AlignCenter)

    dialog.setLayout(layout)
    dialog.exec()

def show_video_preview(file_path, title="Video Preview"):
    from PySide6.QtWidgets import QSlider
    from PySide6.QtCore import QTimer

    dialog = QDialog(window)
    dialog.setWindowTitle(title)
    dialog.resize(900, 650)

    layout = QVBoxLayout()

    video_widget = QVideoWidget()
    layout.addWidget(video_widget)

    # ── Seek slider ──────────────────────────────────────────
    seek_slider = QSlider(Qt.Horizontal)
    seek_slider.setRange(0, 0)        # range updated once video loads
    seek_slider.setStyleSheet("""
        QSlider::groove:horizontal {
            height: 6px;
            background: #cccccc;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: white;
            border: 1px solid #999;
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QSlider::sub-page:horizontal {
            background: #4a90d9;
            border-radius: 3px;
        }
    """)
    layout.addWidget(seek_slider)

    # ── Time label ───────────────────────────────────────────
    time_label = QLabel("0:00 / 0:00")
    time_label.setAlignment(Qt.AlignCenter)
    time_label.setStyleSheet("color: black; font-size: 12px;")
    layout.addWidget(time_label)
    layout.setStretchFactor(video_widget, 1)

    # ── Control buttons ──────────────────────────────────────
    controls = QHBoxLayout()
    play_btn  = QPushButton("Play")
    pause_btn = QPushButton("Pause")
    close_btn = QPushButton("Close")

    for btn in [play_btn, pause_btn, close_btn]:
        btn.setFixedSize(100, 40)
        btn.setStyleSheet(
            "QPushButton {background-color:white;color:black;border-radius:8px;}"
            "QPushButton:hover{background-color:#f0f0f0}"
        )

    controls.addWidget(play_btn)
    controls.addWidget(pause_btn)
    controls.addWidget(close_btn)
    controls.setAlignment(Qt.AlignCenter)
    layout.addLayout(controls)

    # ── Media player setup ───────────────────────────────────
    player       = QMediaPlayer(dialog)
    audio_output = QAudioOutput(dialog)
    player.setAudioOutput(audio_output)
    player.setVideoOutput(video_widget)
    player.setSource(QUrl.fromLocalFile(os.path.abspath(file_path)))

    # Keep references alive so Python doesn't garbage collect them
    dialog.player       = player
    dialog.audio_output = audio_output

    # ── Helper: format milliseconds → "m:ss" ─────────────────
    def ms_to_str(ms):
        if ms < 0:
            return "0:00"
        total_seconds = ms // 1000
        minutes       = total_seconds // 60
        seconds       = total_seconds % 60
        return f"{minutes}:{seconds:02d}"

    # ── Update slider + time label as video plays ─────────────
    # is_seeking flag prevents a feedback loop where updating
    # the slider position triggers another seek
    is_seeking = [False]

    def on_position_changed(position):
        if not is_seeking[0]:
            seek_slider.setValue(position)
        duration = player.duration()
        time_label.setText(f"{ms_to_str(position)} / {ms_to_str(duration)}")

    def on_duration_changed(duration):
        seek_slider.setRange(0, duration)

    # ── Seek when user moves the slider ──────────────────────
    def on_slider_pressed():
        is_seeking[0] = True

    def on_slider_released():
        player.setPosition(seek_slider.value())
        is_seeking[0] = False

    def on_slider_moved(position):
        # Update time label while dragging even before releasing
        duration = player.duration()
        time_label.setText(f"{ms_to_str(position)} / {ms_to_str(duration)}")

    # ── Connect everything ────────────────────────────────────
    player.positionChanged.connect(on_position_changed)
    player.durationChanged.connect(on_duration_changed)

    seek_slider.sliderPressed.connect(on_slider_pressed)
    seek_slider.sliderReleased.connect(on_slider_released)
    seek_slider.sliderMoved.connect(on_slider_moved)

    play_btn.clicked.connect(player.play)
    pause_btn.clicked.connect(player.pause)
    close_btn.clicked.connect(dialog.close)

    dialog.setLayout(layout)
    player.play()
    dialog.exec()

def browse_folder(folder_path, title, file_filter="All Files (*)"):
    os.makedirs(folder_path, exist_ok=True)

    dialog = QFileDialog(window, title, folder_path, file_filter)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setViewMode(QFileDialog.Detail)

    if dialog.exec():
        selected_file = dialog.selectedFiles()[0]
        lower_path = selected_file.lower()

        if lower_path.endswith(IMAGE_EXTENSIONS):
            show_image_preview(selected_file, title)
        elif lower_path.endswith(VIDEO_EXTENSIONS):
            show_video_preview(selected_file, title)
        elif lower_path.endswith(TEXT_EXTENSIONS):
            show_text_preview(selected_file, title)
        else:
            QMessageBox.information(
                window,
                "Selected File",
                f"Preview is not supported for this file type yet.\n\n{selected_file}"
            )

def view_uploads():
    browse_folder(UPLOAD_FOLDER, "View Uploads")

def view_results():
    browse_folder(RESULT_FOLDER, "View Results")

def view_reports():
    browse_folder(REPORT_FOLDER, "View Reports")

#Clear folders
def clear_uploads():
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))
    QMessageBox.information(window, "Uploads Cleared", "All files in uploads folder have been deleted.")

def clear_results():
    for f in os.listdir(RESULT_FOLDER):
        os.remove(os.path.join(RESULT_FOLDER, f))
    QMessageBox.information(window, "Results Cleared", "All files in results folder have been deleted.")

#User Guide
def show_user_guide():
    dialog = QDialog(window)
    dialog.setWindowTitle("📖 User Guide")
    dialog.setGeometry(150, 150, 650, 500)
    layout = QVBoxLayout()
    guide_image = QLabel()
    pixmap = QPixmap("instruction.jpeg")
    pixmap = pixmap.scaled(600, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    guide_image.setPixmap(pixmap)
    guide_image.setAlignment(Qt.AlignCenter)
    layout.addWidget(guide_image)
    text = QTextEdit()
    text.setReadOnly(True)
    text.setFont(QFont("Arial", 12))
    text.setStyleSheet("background-color: white; color: black;")
    text.setText(
        "Welcome to RoadSight!\n\n"
        "RoadSight detects roadway elements including:\n"
        "- Road signs\n"
        "- Poles\n"
        "- Traffic Lights\n"
        "- Fences\n"
        "- Trees\n"
        "- Road lanes\n"
        "- Bicyle lanes\n"
        "- Footpaths\n"
        "- Roadside barriers\n"

        "Instructions:\n"
        "1. Store the system somewhere in a folder.\n"
        "2. When the system is first launched, it will automatically create three folders: 'uploads', 'results', and 'reports' if you do not already have one.\n"
        "3. The image button is used for uploading an image file for analysis, acceptable image file types include *.jpg, *.png, *.jpeg.\n"
        "4. The video button is used for uploading a video file for analysis, acceptable video file types include *.mp4, *.avi, *.mov.\n"
        "5. All uploaded files will be saved in the 'uploads' folder, you can click the view uploads button to view the folder\n"
        "6. After analysis, the output annotated images or video files will be saved in the 'results' folder, you can click the view results button to view the folder\n"
        "7. The view reports button is used to open the 'reports' folder, where you can find the generated report files.\n"
        "8. Clear uploads and clear results buttons are used to clear the files in the 'uploads' and 'results' folders respectively.\n"
    )
    layout.addWidget(text)
    close_btn = QPushButton("Close")
    close_btn.setFixedSize(100, 40)
    close_btn.setStyleSheet("QPushButton {background-color: white;color:black;border-radius:8px;} QPushButton:hover{background-color:#f0f0f0}")
    close_btn.clicked.connect(dialog.close)
    layout.addWidget(close_btn, alignment=Qt.AlignCenter)
    dialog.setLayout(layout)
    dialog.exec()

#Application
app = QApplication(sys.argv)

EDGE_MARGIN = 8
class MainWindow(QWidget):
    # Make background responsive according to window size
    def resizeEvent(self, event):
        super().resizeEvent(event)
        background.setGeometry(0, 0, self.width(), self.height())
    
window = MainWindow()
window.setWindowTitle("RoadSight Detection System")
window.setGeometry(100, 100, 700, 500)


#Background
background = QLabel(window)
bg_pixmap = QPixmap("background.jpeg")
background.setPixmap(bg_pixmap)
background.setScaledContents(True)
background.setGeometry(0, 0, 700, 500)
background.lower()

#Layout
main_layout = QVBoxLayout()

title = QLabel("Welcome to RoadSight")
title.setFont(QFont("Arial", 25, QFont.Bold))
title.setAlignment(Qt.AlignCenter)
title.setStyleSheet("color: white;")
main_layout.addWidget(title)

subtitle = QLabel(
    "Upload videos or images to extract road network information\n"
)
subtitle.setFont(QFont("Arial", 15))
subtitle.setAlignment(Qt.AlignCenter)
subtitle.setStyleSheet("color: white;")
subtitle.setWordWrap(True)
main_layout.addWidget(subtitle)

#Progress bar
progress_bar = QProgressBar()
progress_bar.setValue(0)
progress_bar.setFixedWidth(400)
progress_bar.setStyleSheet("background-color: white;")
main_layout.addWidget(progress_bar, alignment=Qt.AlignCenter)

#Processing text label
processing_label = QLabel("")
processing_label.setFont(QFont("Arial", 12))
processing_label.setAlignment(Qt.AlignCenter)
processing_label.setStyleSheet("color: white;")
main_layout.addWidget(processing_label, alignment=Qt.AlignCenter)

#Buttons
img_btn = QPushButton("🖼 Image")
img_btn.setFixedSize(130, 40)
img_btn.clicked.connect(upload_image)

video_btn = QPushButton("🎥 Video")
video_btn.setFixedSize(130, 40)
video_btn.clicked.connect(upload_video)

guide_btn = QPushButton("📖 User Guide")
guide_btn.setFixedSize(130, 40)
guide_btn.clicked.connect(show_user_guide)

report_btn = QPushButton("📊 View Report")
report_btn.setFixedSize(130, 40)
report_btn.clicked.connect(view_reports)

view_upload_btn = QPushButton("📂 View Uploads")
view_upload_btn.setFixedSize(130, 40)
view_upload_btn.clicked.connect(view_uploads)

view_result_btn = QPushButton("📂 View Results")
view_result_btn.setFixedSize(130, 40)
view_result_btn.clicked.connect(view_results)

clear_upload_btn = QPushButton("🗑 Clear Uploads")
clear_upload_btn.setFixedSize(130, 40)
clear_upload_btn.clicked.connect(clear_uploads)

clear_result_btn = QPushButton("🗑 Clear Results")
clear_result_btn.setFixedSize(130, 40)
clear_result_btn.clicked.connect(clear_results)

button_style = "QPushButton {background-color:white;color:black;border-radius:8px;} QPushButton:hover{background-color:#f0f0f0}"
for btn in [img_btn, video_btn, report_btn, guide_btn, view_upload_btn, view_result_btn, clear_upload_btn, clear_result_btn]:
    btn.setStyleSheet(button_style)

#First row: Image, Video, Report
row1 = QHBoxLayout()
row1.addWidget(img_btn)
row1.addWidget(video_btn)
row1.addWidget(report_btn)
row1.addWidget(guide_btn)
row1.setAlignment(Qt.AlignCenter)

#Second row: View/Clear folders
row2 = QHBoxLayout()
row2.addWidget(view_upload_btn)
row2.addWidget(view_result_btn)
row2.addWidget(clear_upload_btn)
row2.addWidget(clear_result_btn)
row2.setAlignment(Qt.AlignCenter)

main_layout.addLayout(row1)
main_layout.addLayout(row2)

footer = QLabel("© 2026 RoadSight | Group 26-1")
footer.setFont(QFont("Arial", 12))
footer.setAlignment(Qt.AlignCenter)
footer.setStyleSheet("color: white;")
main_layout.addWidget(footer)

window.setLayout(main_layout)
window.show()
sys.exit(app.exec())

