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
import cv2
import csv
import time


from ultralytics import YOLO

#tree_model = YOLO("models/tree.pt")
traffic_model = YOLO("models/yolov8n.pt")
trunk_model = YOLO("models/trunk.pt")
pole_model = YOLO("models/pole.pt")


UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
REPORT_FOLDER = "reports"
INPUT_FOLDER = "testing resource"

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


CONF = {
    #"tree": 0.75,
    "traffic light": 0.4,
    "trunk": 0.4,
    "pole": 0.4
}


traffic_seen = []
trunk_seen = []
pole_seen = []


def is_new_global(box, seen_list, threshold=0.5):
    for s in seen_list:
        if iou(box, s) > threshold:
            return False
    seen_list.append(box)
    return True


def remove_duplicates(boxes, threshold=0.5):
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    filtered = []
    for box in boxes:
        keep = True
        for f in filtered:
            if iou(box[:4], f[:4]) > threshold:
                keep = False
                break
        if keep:
            filtered.append(box)

    return filtered



def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = max(0, (x2 - x1)) * max(0, (y2 - y1))
    area2 = max(0, (x2_ - x1_)) * max(0, (y2_ - y1_))

    denom = area1 + area2 - inter_area
    return inter_area / denom if denom > 0 else 0.0



def create_report_folder(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    report_dir = os.path.join(REPORT_FOLDER, base)
    frames_dir = os.path.join(report_dir, "frames")

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    return report_dir, frames_dir










class VideoDetectionThread(QThread):
    progress = Signal(int)
    finished = Signal(str)

    def __init__(self, input_path, output_path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    def run(self):

        global traffic_seen, trunk_seen, pole_seen

        cap = cv2.VideoCapture(self.input_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
        report_dir = os.path.join(
            REPORT_FOLDER,
            os.path.splitext(os.path.basename(self.input_path))[0]
        )
        frames_dir = os.path.join(report_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)


        traffic_seen = []
        trunk_seen = []
        pole_seen = []

        traffic_count = 0
        trunk_count = 0
        pole_count = 0

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

            traffic_boxes = []
            trunk_boxes = []
            pole_boxes = []


            traffic_results = traffic_model(frame, conf=CONF["traffic light"])
            for r in traffic_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = traffic_model.names[cls]
                    conf = float(box.conf[0])

                    if label.lower() == "traffic light":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        b = (x1, y1, x2, y2)

                        if is_new_global(b, traffic_seen):
                            traffic_count += 1

                        traffic_boxes.append((x1, y1, x2, y2, conf))


            pole_results = pole_model(frame, conf=CONF["pole"])
            for r in pole_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = pole_model.names[cls]
                    conf = float(box.conf[0])

                    if label.lower() == "pole":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        b = (x1, y1, x2, y2)

                        if is_new_global(b, pole_seen):
                            pole_count += 1

                        pole_boxes.append((x1, y1, x2, y2, conf))


            trunk_results = trunk_model(frame, conf=CONF["trunk"])
            for r in trunk_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = trunk_model.names[cls]
                    conf = float(box.conf[0])

                    if label.lower() == "trunk":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        b = (x1, y1, x2, y2)

                        if is_new_global(b, trunk_seen):
                            trunk_count += 1

                        trunk_boxes.append((x1, y1, x2, y2, conf))

            # =====================
            # draw
            # =====================
            for (x1, y1, x2, y2, conf) in traffic_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"traffic {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            for (x1, y1, x2, y2, conf) in trunk_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(frame, f"trunk {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

            for (x1, y1, x2, y2, conf) in pole_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
                cv2.putText(frame, f"pole {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

            out.write(frame)

            frame_count += 1
            interval = max(1, int(fps * 0.2))

            if fps > 0 and frame_count % interval == 0:
               sec = round(frame_count / fps, 2)
               cv2.imwrite(os.path.join(frames_dir, f"{sec}s.jpg"), frame)

            if total_frames > 0:
                self.progress.emit(int(frame_count / total_frames * 100))

        cap.release()
        out.release()


        csv_path = os.path.join(report_dir, "detection_report.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "count"])
            writer.writerow(["traffic light", traffic_count])
            writer.writerow(["trunk", trunk_count])
            writer.writerow(["pole", pole_count])

        self.finished.emit(self.output_path)



def detect_image(image_path):
    processing_label.setText("Processing...")
    QApplication.processEvents()

    img = cv2.imread(image_path)
    if img is None:
        QMessageBox.warning(window, "Error", "Failed to read image.")
        return

    report_dir, frames_dir = create_report_folder(image_path)
    csv_path = os.path.join(report_dir, "detection_report.csv")

    traffic_results = traffic_model(img, conf=CONF["traffic light"])
    trunk_results = trunk_model(img, conf=CONF["trunk"])
    pole_results = pole_model(img, conf=CONF["pole"])

    traffic_boxes, trunk_boxes, pole_boxes = [], [], []

    traffic_count = 0
    trunk_count = 0
    pole_count = 0

    # -------- traffic --------
    for r in traffic_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = traffic_model.names[cls]
            conf = float(box.conf[0])

            if label.lower() == "traffic light":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                traffic_boxes.append((x1, y1, x2, y2, conf))
                traffic_count += 1


    for r in trunk_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = trunk_model.names[cls]
            conf = float(box.conf[0])

            if label.lower() == "trunk":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                trunk_boxes.append((x1, y1, x2, y2, conf))
                trunk_count += 1

    # -------- pole --------
    for r in pole_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = pole_model.names[cls]
            conf = float(box.conf[0])

            if label.lower() == "pole":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pole_boxes.append((x1, y1, x2, y2, conf))
                pole_count += 1

    traffic_boxes = remove_duplicates(traffic_boxes)
    trunk_boxes = remove_duplicates(trunk_boxes)
    pole_boxes = remove_duplicates(pole_boxes)


    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count"])
        writer.writerow(["traffic light", traffic_count])
        writer.writerow(["trunk", trunk_count])
        writer.writerow(["pole", pole_count])

    # draw
    for (x1, y1, x2, y2, conf) in traffic_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.putText(img, f"traffic {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    for (x1, y1, x2, y2, conf) in trunk_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(img, f"trunk {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)

    for (x1, y1, x2, y2, conf) in pole_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(img, f"pole {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, img)

    QMessageBox.information(window, "Done", f"Saved: {result_path}")

















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
    dialog = QDialog(window)
    dialog.setWindowTitle(title)
    dialog.resize(900, 650)

    layout = QVBoxLayout()

    video_widget = QVideoWidget()
    layout.addWidget(video_widget)

    controls = QHBoxLayout()

    play_btn = QPushButton("Play")
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

    player = QMediaPlayer(dialog)
    audio_output = QAudioOutput(dialog)

    player.setAudioOutput(audio_output)
    player.setVideoOutput(video_widget)
    player.setSource(QUrl.fromLocalFile(os.path.abspath(file_path)))

    # Keep references alive
    dialog.player = player
    dialog.audio_output = audio_output

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




def clear_folder(folder):
    if os.path.exists(folder):
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")








def show_user_guide():
    dialog = QDialog(window)
    dialog.setWindowTitle("📖 User Guide")
    dialog.setGeometry(150, 150, 650, 500)

    layout = QVBoxLayout()


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(BASE_DIR, "userguide", "description.png")
    txt_path = os.path.join(BASE_DIR, "userguide", "guide.txt")

    btn_layout = QHBoxLayout()

    download_img_btn = QPushButton("Download Image")
    preview_img_btn = QPushButton("Preview Image")

    download_txt_btn = QPushButton("Download Guide")
    preview_txt_btn = QPushButton("Preview Guide")

    buttons = [
        download_img_btn, preview_img_btn,
        download_txt_btn, preview_txt_btn
    ]

    for btn in buttons:
        btn.setFixedSize(160, 40)
        btn.setStyleSheet(
            "QPushButton {background-color:white;color:black;border-radius:8px;}"
            "QPushButton:hover{background-color:#f0f0f0}"
        )

    btn_layout.addWidget(download_img_btn)
    btn_layout.addWidget(preview_img_btn)
    btn_layout.addWidget(download_txt_btn)
    btn_layout.addWidget(preview_txt_btn)
    btn_layout.setAlignment(Qt.AlignCenter)

    layout.addLayout(btn_layout)

  
    def download_file(source_file):
        if not os.path.exists(source_file):
            QMessageBox.warning(dialog, "Error", f"File not found:\n{source_file}")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            dialog,
            "Save File",
            os.path.basename(source_file)
        )

        if save_path:
            shutil.copy(source_file, save_path)
            QMessageBox.information(dialog, "Downloaded", f"Saved to:\n{save_path}")

   
    def preview_image(file_path):
        if not os.path.exists(file_path):
            QMessageBox.warning(dialog, "Error", "Image not found.")
            return

        preview = QDialog(dialog)
        preview.setWindowTitle("Image Preview")
        preview.resize(800, 600)

        layout_p = QVBoxLayout()

        label = QLabel()
        pixmap = QPixmap(file_path)

        label.setPixmap(
            pixmap.scaled(750, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        label.setAlignment(Qt.AlignCenter)

        layout_p.addWidget(label)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(preview.close)
        layout_p.addWidget(close_btn, alignment=Qt.AlignCenter)

        preview.setLayout(layout_p)
        preview.exec()

    def preview_text(file_path):
        if not os.path.exists(file_path):
            QMessageBox.warning(dialog, "Error", "File not found.")
            return

        preview = QDialog(dialog)
        preview.setWindowTitle("Guide Preview")
        preview.resize(800, 600)

        layout_p = QVBoxLayout()

        text_box = QTextEdit()
        text_box.setReadOnly(True)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_box.setText(f.read())
        except Exception as e:
            text_box.setText(str(e))

        layout_p.addWidget(text_box)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(preview.close)
        layout_p.addWidget(close_btn, alignment=Qt.AlignCenter)

        preview.setLayout(layout_p)
        preview.exec()

 
    download_img_btn.clicked.connect(lambda: download_file(img_path))
    download_txt_btn.clicked.connect(lambda: download_file(txt_path))

    preview_img_btn.clicked.connect(lambda: preview_image(img_path))
    preview_txt_btn.clicked.connect(lambda: preview_text(txt_path))

  
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
        "- Bicycle lanes\n"
        "- Footpaths\n"
        "- Roadside barriers\n\n"

        "Instructions:\n"
        "1. Store system in a folder.\n"
        "2. First launch creates 'uploads', 'results', 'reports'.\n"
        "3. Upload image (*.jpg, *.png, *.jpeg).\n"
        "4. Upload video (*.mp4, *.avi, *.mov).\n"
        "5. Uploaded files → uploads folder.\n"
        "6. Results → results folder.\n"
        "7. Reports → reports folder.\n"
        "8. Use clear buttons to delete files.\n"
    )

    layout.addWidget(text)

  
    close_btn = QPushButton("Close")
    close_btn.setFixedSize(100, 40)
    close_btn.setStyleSheet(
        "QPushButton {background-color: white;color:black;border-radius:8px;}"
        "QPushButton:hover{background-color:#f0f0f0}"
    )
    close_btn.clicked.connect(dialog.close)

    layout.addWidget(close_btn, alignment=Qt.AlignCenter)

    dialog.setLayout(layout)
    dialog.exec()







#Application
app = QApplication(sys.argv)

EDGE_MARGIN = 8
class MainWindow(QWidget):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        background.setGeometry(0, 0, self.width(), self.height())
    
    def closeEvent(self, event):
        reply = QMessageBox.question(
        self,
        "Exit",
        "Do you want to exit and clear all uploads and results?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
        )

        if reply == QMessageBox.Yes:
           clear_folder(UPLOAD_FOLDER)
           clear_folder(RESULT_FOLDER)
           event.accept()
        else:
           event.ignore()
    
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

