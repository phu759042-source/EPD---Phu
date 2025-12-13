import sys
import cv2
import json
import time
import datetime
import numpy as np
import mediapipe as mp
import traceback
import requests
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QFrame, QSplitter, QMessageBox, QInputDialog, 
                             QFileDialog, QScrollArea, QGridLayout,QMenu)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtMultimedia import QMediaDevices

SERVER_URL = "https://epd-p9ox.onrender.com/log_incident/"

def send_to_server(data, class_name, mode):
    try:
        now = datetime.datetime.now()
        end_str = now.strftime('%H:%M:%S')
        date_str = now.strftime('%Y-%m-%d') # <--- [MỚI] Lấy ngày hiện tại (Năm-Tháng-Ngày)

        payload = {
            "class_id": class_name,
            "zone_id": str(data['student_id']),
            "issue_type": data['behavior'],
            "start_time": data['timestamp'],
            "end_time": end_str,
            "duration_seconds": data['duration'],
            "date": date_str, # <--- [MỚI] Gửi thêm ngày
            "scan_mode": mode
        }
        
        # Gửi đi
        requests.post(SERVER_URL, json=payload, timeout=2)
        print(f"✅ Đã gửi lên Web (Mode: {mode}) - ({date_str}): HS-{data['student_id']} - {data['behavior']}")
    except requests.exceptions.Timeout:
        print("❌ Gửi lên web bị Timeout (Quá 2 giây).")
    except Exception as e:
        print(f"❌ Lỗi khi gửi dữ liệu lên Web: {e}")

class BehaviorTracker:
    def __init__(self, threshold_seconds=5):
        self.threshold = threshold_seconds
        self.tracking_data = {} 
        
    def update(self, student_id, current_status):
        now = time.time()
        
        # Nếu học sinh này chưa từng bị theo dõi
        if student_id not in self.tracking_data:
            # Chỉ bắt đầu theo dõi nếu hành vi là XẤU
            if self.is_bad_behavior(current_status):
                self.tracking_data[student_id] = {'status': current_status, 'start_time': now}
            return None

        # Nếu đang theo dõi
        data = self.tracking_data[student_id]
        
        # Nếu trạng thái thay đổi (đang Ngủ -> Tỉnh, hoặc Ngủ -> Mất tập trung)
        if current_status != data['status']:
            duration = now - data['start_time']
            last_status = data['status']
            start_ts = data['start_time']
            
            # Xóa trạng thái cũ đi
            del self.tracking_data[student_id]
            
            # Nếu hành vi mới cũng xấu -> Bắt đầu theo dõi cái mới ngay
            if self.is_bad_behavior(current_status):
                self.tracking_data[student_id] = {'status': current_status, 'start_time': now}

            # QUAN TRỌNG: Kiểm tra xem hành vi cũ đã kéo dài đủ lâu chưa để báo cáo
            if duration >= self.threshold:
                return self.create_report(student_id, last_status, duration, start_ts)
                
        return None

    def is_bad_behavior(self, status):
        # Cập nhật từ khóa khớp với AIProcessor
        bad_keywords = ["Buồn ngủ", "Mất tập trung", "Căng thẳng", "Buồn", "Mệt mỏi", "Thu mình"]
        return any(keyword in status for keyword in bad_keywords)

    def create_report(self, student_id, status, duration, start_timestamp):
        return {
            "student_id": student_id,
            "behavior": status,
            "duration": round(duration, 1),
            "timestamp": datetime.datetime.fromtimestamp(start_timestamp).strftime('%H:%M:%S')
        }
    
    # --- HÀM MỚI: CHỐT SỔ KHI TẮT APP ---
    def finalize_all(self):
        """Trả về danh sách tất cả các hành vi đang diễn ra chưa kết thúc"""
        reports = []
        now = time.time()
        for sid, data in self.tracking_data.items():
            duration = now - data['start_time']
            if duration >= self.threshold:
                reports.append(self.create_report(sid, data['status'], duration, data['start_time']))
        self.tracking_data.clear()
        return reports

# ======================================================
# 1. AI PROCESSOR (NÂNG CẤP CHO CHỦ ĐỀ TÂM LÝ)
# ======================================================
class AIProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # --- HÀM TÍNH TOÁN CŨ (GIỮ NGUYÊN) ---
    def calculate_ear(self, landmarks, indices):
        try:
            p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
            p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
            p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
            p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
            p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
            p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])
            vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
            horizontal = np.linalg.norm(p1 - p4) * 2.0
            return vertical / horizontal
        except: return 0.0

    def calculate_mar(self, landmarks):
        try:
            top = np.array([landmarks[13].x, landmarks[13].y])
            bottom = np.array([landmarks[14].x, landmarks[14].y])
            left = np.array([landmarks[61].x, landmarks[61].y])
            right = np.array([landmarks[291].x, landmarks[291].y])
            return np.linalg.norm(top - bottom) / np.linalg.norm(left - right)
        except: return 0.0

    def get_head_pose(self, landmarks):
        try:
            nose = landmarks[1].x
            left_ear = landmarks[234].x
            right_ear = landmarks[454].x
            ratio = (nose - left_ear) / (right_ear - nose + 0.0001)

            if ratio < 0.5: return "Né tránh (Trái)" # Đổi tên cho hợp tâm lý
            if ratio > 2.0: return "Né tránh (Phải)"
            
            if (landmarks[1].y - landmarks[10].y) < 0.03: 
                return "Cúi đầu"
            return "Thẳng"
        except: return "KĐ"

    # --- HÀM MỚI: PHÁT HIỆN CẢM XÚC (LOGIC HÌNH HỌC) ---
    def detect_emotion(self, landmarks):
        try:
            # 1. Lấy tọa độ
            # Khóe miệng trái (61) và phải (291)
            mouth_corner_y = (landmarks[61].y + landmarks[291].y) / 2
            
            # Trung tâm môi (lấy điểm giữa môi trên 13 và môi dưới 14)
            mouth_center_y = (landmarks[13].y + landmarks[14].y) / 2
            
            # Môi trên (0) dùng để check nụ cười
            upper_lip_y = landmarks[0].y
            
            # Lông mày để check stress
            brow_left = np.array([landmarks[107].x, landmarks[107].y])
            brow_right = np.array([landmarks[336].x, landmarks[336].y])
            face_width = np.linalg.norm(np.array([landmarks[234].x, landmarks[234].y]) - 
                                        np.array([landmarks[454].x, landmarks[454].y]))
            brow_dist = np.linalg.norm(brow_left - brow_right)
            brow_ratio = brow_dist / face_width

            # 2. Logic phán đoán
            # Lưu ý: Trong ảnh máy tính, trục Y càng lớn thì càng nằm thấp bên dưới
            
            # Nếu khóe miệng CAO HƠN môi trên -> Cười
            if mouth_corner_y < upper_lip_y: 
                return "Tích cực/Vui vẻ 😊"
            
            # [MỚI] Nếu khóe miệng THẤP HƠN trung tâm môi một khoảng -> Buồn
            # 0.02 là ngưỡng (threshold), bạn có thể chỉnh số này nếu thấy chưa nhạy
            elif mouth_corner_y > mouth_center_y + 0.0025:
                return "Buồn / Chán nản 😞"
            
            # Nếu lông mày quá gần nhau -> Stress
            elif brow_ratio < 0.16: 
                return "Căng thẳng/Stress 😖"
                
            else:
                return "Bình thường"
        except:
            return "Bình thường"
    def process_zone(self, frame_crop):
        if frame_crop is None or frame_crop.shape[0] == 0 or frame_crop.shape[1] == 0:
            return "NO DATA", (100, 100, 100)

        rgb_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
        rgb_crop.flags.writeable = False
        results = self.face_mesh.process(rgb_crop)
        rgb_crop.flags.writeable = True

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Tính toán các chỉ số
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            ear = (self.calculate_ear(landmarks, LEFT_EYE) + self.calculate_ear(landmarks, RIGHT_EYE)) / 2.0
            mar = self.calculate_mar(landmarks)
            pose = self.get_head_pose(landmarks)
            emotion = self.detect_emotion(landmarks)

            # --- LOGIC CHẨN ĐOÁN TÂM LÝ (QUAN TRỌNG) ---
            status = "Ổn định"
            color = (0, 255, 0) # Xanh lá

            # Ưu tiên 1: Các dấu hiệu sức khỏe thể chất (Mệt mỏi)
            if ear < 0.20:
                status = "Kiệt sức / Buồn ngủ 😴"
                color = (0, 0, 255) # Đỏ
            elif mar > 0.5:
                status = "Mệt mỏi / Thiếu oxy 🥱"
                color = (0, 165, 255) # Cam
            
            # Ưu tiên 2: Hành vi/Tư thế (Thu mình)
            elif pose == "Cúi đầu":
                 status = "Thu mình / Trầm tư 🙇"
                 color = (255, 0, 255) # Tím
            elif "Né tránh" in pose:
                status = f"Mất tập trung ({pose})"
                color = (0, 255, 255) # Vàng

            # Ưu tiên 3: Cảm xúc (Nếu tư thế và mắt bình thường)
            elif "Căng thẳng" in emotion:
                status = "Căng thẳng / Lo âu 😖"
                color = (128, 0, 128) # Tím đậm
            
            # [THÊM MỚI] Xử lý trạng thái Buồn
            elif "Buồn" in emotion:
                status = "Buồn / Chán nản 😞"
                color = (0, 100, 255) # Xanh dương đậm (Màu đặc trưng của nỗi buồn)

            elif "Tích cực" in emotion:
                status = "Tích cực / Hứng thú 😄"
                color = (0, 255, 127) # Xanh lơ

            return status, color
        else:
            return "Vắng / K.Thấy Mặt", (128, 128, 128)
# ======================================================
# 2. THREAD VIDEO (ĐÃ CẬP NHẬT TỐI ƯU & TRACKER)
# ======================================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_board_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._is_running = True
        self.is_monitoring = False
        
        self.ai = AIProcessor()
        
        # --- [THÊM MỚI] KHỞI TẠO BỘ THEO DÕI HÀNH VI ---
        # threshold_seconds=5: Phải duy trì trạng thái xấu 5 giây mới báo cáo
        self.tracker = BehaviorTracker(threshold_seconds=5) 
        
        self.video_width = 640
        self.video_height = 480
        self.zones = [] 
        self.current_drawing_zone = None
        self.frame_count = 0 
        self.last_statuses = {} 

        # Cấu hình tối ưu: 30 frame mới chạy AI 1 lần (tương đương 1 giây)
        self.SKIP_FRAMES = 30 

    def run(self):
        if sys.platform == 'win32':
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.camera_index)
            
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except:
            pass

        if not cap.isOpened():
            self.error_signal.emit("Không thể mở Camera!")
            return

        while self._is_running:
            try:
                ret, cv_img = cap.read()
                if not ret:
                    continue
                
                self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.frame_count += 1
                
                if self.is_monitoring:
                    final_results = []
                    
                    # --- [TỐI ƯU] Chỉ chạy AI mỗi 30 frame (1 giây/lần) ---
                    should_process_ai = (self.frame_count % self.SKIP_FRAMES == 0)

                    for zone in self.zones:
                        z_id = str(zone["id"]) # Đảm bảo ID là string để khớp với Tracker
                        
                        # 1. GIAI ĐOẠN XỬ LÝ AI (NẶNG - CHẠY ÍT)
                        if should_process_ai:
                            x, y, w, h = zone["rect"]
                            x = max(0, x); y = max(0, y)
                            w = min(w, self.video_width - x); h = min(h, self.video_height - y)
                            
                            if w > 0 and h > 0:
                                roi = cv_img[y:y+h, x:x+w]
                                status, color = self.ai.process_zone(roi)
                                
                                # Lưu vào Cache để vẽ liên tục
                                self.last_statuses[z_id] = {"status": status, "color": color}
                                
                                # --- [THÊM MỚI] GỌI TRACKER ĐỂ KIỂM TRA THỜI GIAN ---
                                report = self.tracker.update(z_id, status)
                                if report:
                                    # Gửi lên server ngay lập tức (Thêm self.current_class vào)
                                    send_to_server(report, self.current_class)      
                        
                        # 2. GIAI ĐOẠN VẼ (NHẸ - CHẠY LIÊN TỤC)
                        # Lấy dữ liệu từ cache ra vẽ
                        cached = self.last_statuses.get(z_id, {"status": "Đang tải...", "color": (200,200,200)})
                        
                        # Thêm vào danh sách để cập nhật bảng bên phải
                        final_results.append({"id": z_id, "status": cached["status"], "color": cached["color"]})
                        
                        # Vẽ khung hình chữ nhật trên video
                        cv2.rectangle(cv_img, (zone["rect"][0], zone["rect"][1]), 
                                      (zone["rect"][0]+zone["rect"][2], zone["rect"][1]+zone["rect"][3]), 
                                      cached["color"], 2)
                        # Vẽ ID nhỏ trên đầu khung
                        cv2.putText(cv_img, f"HS-{z_id}", (zone["rect"][0], zone["rect"][1]-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cached["color"], 1)
                    
                    self.update_board_signal.emit(final_results)

                else:
                    # Chế độ chờ (Vẽ khung vàng)
                    for zone in self.zones:
                         x, y, w, h = zone["rect"]
                         cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                         cv2.putText(cv_img, f"HS-{zone['id']}", (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if self.current_drawing_zone:
                        dx, dy, dw, dh = self.current_drawing_zone
                        cv2.rectangle(cv_img, (dx, dy), (dx+dw, dy+dh), (0, 165, 255), 2)

                # Chuyển đổi ảnh sang Qt
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image.copy())

            except Exception as e:
                print(f"Error in thread: {e}")
                traceback.print_exc()

        cap.release()

    def stop(self):
        self._is_running = False
        if self.is_monitoring:
            print("Đang tổng kết dữ liệu...")
            final_reports = self.tracker.finalize_all()
            for rep in final_reports:
                # TRUYỀN THÊM TÊN LỚP KHI TẮT
                send_to_server(rep, self.current_class) 
        self.wait()
# ======================================================
# 3. GIAO DIỆN NGƯỜI DÙNG (GIỮ NGUYÊN)
# ======================================================
class CameraMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ Thống Giám Sát Lớp Học - AI Smart Monitor")
        self.resize(1400, 850)
        self.thread = None
        self.is_drawing_mode = False
        self.start_point = None
        self.current_temp_zone = None 
        self.student_widgets = {} 
        self.is_monitoring_active = False 

        self.setup_ui()
        self.apply_styles()
        self.load_available_cameras()

    def setup_ui(self):
        self.setFont(QFont("Segoe UI", 10))
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.video_container = QWidget()
        v_layout = QVBoxLayout(self.video_container)
        v_layout.setContentsMargins(0,0,0,0)
        self.video_label = QLabel("Vui lòng chọn Camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setScaledContents(True) 
        self.video_label.setMouseTracking(True) 
        self.video_label.mousePressEvent = self.on_mouse_press
        self.video_label.mouseMoveEvent = self.on_mouse_move
        self.video_label.mouseReleaseEvent = self.on_mouse_release
        v_layout.addWidget(self.video_label)

        sidebar = QFrame()
        sidebar.setFixedWidth(450)
        sb_layout = QVBoxLayout(sidebar)
        
        self.grp_setup = QFrame()
        setup_layout = QVBoxLayout(self.grp_setup)
        self.combo_cam = QComboBox()
        self.combo_cam.currentIndexChanged.connect(self.start_camera_stream)
        setup_layout.addWidget(QLabel("Nguồn Camera:"))
        setup_layout.addWidget(self.combo_cam)
        setup_layout.addWidget(QLabel("🏫 Chọn Lớp Học:"))
        self.combo_class_select = QComboBox()
        # Danh sách các lớp (bạn có thể sửa tùy ý)
        self.combo_class_select.addItems(["Lớp 12A1", "Lớp 12A2", "Lớp 12A4", "Lớp 12A5"])
        self.combo_class_select.currentTextChanged.connect(self.update_class_name)
        setup_layout.addWidget(self.combo_class_select)
        self.btn_draw = QPushButton("🖌 THÊM VỊ TRÍ HỌC SINH")
        self.btn_draw.setCheckable(True)
        self.btn_draw.clicked.connect(self.toggle_drawing_mode)
        setup_layout.addWidget(self.btn_draw)
        
        self.btn_confirm_zone = QPushButton("✅ XÁC NHẬN VÙNG")
        self.btn_confirm_zone.clicked.connect(self.confirm_current_zone)
        self.btn_confirm_zone.setEnabled(False)
        setup_layout.addWidget(self.btn_confirm_zone)
        
        h_file = QHBoxLayout()
        btn_save = QPushButton("💾 Lưu Cấu Hình"); btn_save.clicked.connect(self.save_layout_to_file)
        btn_load = QPushButton("📂 Mở Cấu Hình"); btn_load.clicked.connect(self.load_layout_from_file)
        h_file.addWidget(btn_save); h_file.addWidget(btn_load)
        setup_layout.addLayout(h_file)
        
        sb_layout.addWidget(self.grp_setup)
        
        self.btn_start_monitor = QPushButton("▶ BẮT ĐẦU PHÂN TÍCH AI")
        self.btn_start_monitor.setMinimumHeight(60)
        self.btn_start_monitor.setStyleSheet("background-color: #2e7d32; font-size: 16px; font-weight: bold;")
        self.btn_start_monitor.clicked.connect(self.toggle_monitoring)
        sb_layout.addWidget(self.btn_start_monitor)
        
        sb_layout.addWidget(self.create_line())
        sb_layout.addWidget(QLabel("📊 TRẠNG THÁI HỌC SINH (Grid View)"))
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)
        sb_layout.addWidget(self.scroll_area)

        splitter.addWidget(self.video_container)
        splitter.addWidget(sidebar)
        splitter.setSizes([950, 450])
        main_layout.addWidget(splitter)

    def update_class_name(self, text):
        if self.thread:
            self.thread.current_class = text
            print(f"Đã đổi sang lớp: {text}")
    
    def handle_thread_error(self, err_msg):
        QMessageBox.critical(self, "Lỗi", err_msg)
        self.combo_cam.setCurrentIndex(-1)

    def start_camera_stream(self):
        idx = self.combo_cam.currentIndex()
        if idx < 0: return
        if self.thread: self.thread.stop()
        
        self.thread = VideoThread(camera_index=idx)
        # Cập nhật tên lớp ngay khi khởi tạo
        self.thread.current_class = self.combo_class_select.currentText() 
        
        self.thread.change_pixmap_signal.connect(lambda img: self.video_label.setPixmap(QPixmap.fromImage(img)))
        self.thread.update_board_signal.connect(self.update_student_panel)
        self.thread.error_signal.connect(self.handle_thread_error)
        self.thread.start()

    def toggle_monitoring(self):
        if not self.thread: return
        self.is_monitoring_active = not self.is_monitoring_active
        if self.is_monitoring_active:
            self.thread.is_monitoring = True
            self.btn_start_monitor.setText("⏹ DỪNG PHÂN TÍCH")
            self.btn_start_monitor.setStyleSheet("background-color: #c62828; font-size: 16px; font-weight: bold;")
            self.grp_setup.setEnabled(False)
            self.is_drawing_mode = False; self.btn_draw.setChecked(False)
        else:
            self.thread.is_monitoring = False
            self.btn_start_monitor.setText("▶ BẮT ĐẦU PHÂN TÍCH AI")
            self.btn_start_monitor.setStyleSheet("background-color: #2e7d32; font-size: 16px; font-weight: bold;")
            self.grp_setup.setEnabled(True)
            for w in self.student_widgets.values():
                w.lbl_stat.setText("Chờ..."); w.setStyleSheet("background-color: #424242; border-radius: 4px;")

    def update_student_panel(self, data_list):
        COLUMNS = 3
        # Xóa các widget của vùng đã bị xóa khỏi dữ liệu nhưng vẫn còn trên giao diện
        active_ids = [item['id'] for item in data_list]
        for sid in list(self.student_widgets.keys()):
            if sid not in active_ids:
                self.student_widgets[sid].setParent(None)
                del self.student_widgets[sid]

        for index, item in enumerate(data_list):
            sid = item['id']
            status_text = item['status']
            
            if sid not in self.student_widgets:
                card = QFrame()
                card.setFixedSize(130, 80) 
                l = QVBoxLayout(card); l.setContentsMargins(2,2,2,2); l.setSpacing(0)
                
                lbl_id = QLabel(f"HS {sid}"); lbl_id.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl_id.setStyleSheet("font-weight: bold; color: white;")
                
                lbl_stat = QLabel(status_text); lbl_stat.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl_stat.setWordWrap(True) 
                lbl_stat.setStyleSheet("font-size: 11px; color: white;")
                
                card.lbl_stat = lbl_stat
                l.addWidget(lbl_id); l.addWidget(lbl_stat)
                
                # --- [MỚI] CẤU HÌNH MENU CHUỘT PHẢI ---
                card.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                # Dùng lambda để truyền đúng ID của học sinh cần xóa
                card.customContextMenuRequested.connect(lambda pos, s=sid: self.show_context_menu(pos, s))
                # -------------------------------------

                self.grid_layout.addWidget(card, index // COLUMNS, index % COLUMNS)
                self.student_widgets[sid] = card
            
            # ... (Phần cập nhật màu sắc bên dưới giữ nguyên) ...
            w = self.student_widgets[sid]
            w.lbl_stat.setText(status_text)
            
            bg_color = "#2e7d32" 
            if "NGỦ" in status_text: bg_color = "#d32f2f" 
            elif "Vắng" in status_text: bg_color = "#424242" 
            elif "NGÁP" in status_text: bg_color = "#ef6c00" 
            elif "Mất tập trung" in status_text: bg_color = "#f9a825" 
            elif "Cúi" in status_text: bg_color = "#7b1fa2" 
            
            w.setStyleSheet(f"background-color: {bg_color}; border-radius: 6px;")

    def save_layout_to_file(self):
        if not self.thread or not self.thread.zones: return
        name, ok = QInputDialog.getText(self, "Lưu", "Nhập tên file (không dấu):")
        if ok and name:
            with open(f"{name}.json", 'w') as f: json.dump(self.thread.zones, f)
            QMessageBox.information(self, "Thành công", "Đã lưu bản đồ lớp học.")

    def load_layout_from_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Mở", "", "JSON (*.json)")
        if not fname or not self.thread: return
        try:
            with open(fname, 'r') as f: 
                self.thread.zones = json.load(f)
                for i in reversed(range(self.grid_layout.count())): 
                    self.grid_layout.itemAt(i).widget().setParent(None)
                self.student_widgets.clear()
                QMessageBox.information(self, "Thành công", f"Đã nạp {len(self.thread.zones)} vị trí học sinh.")
        except Exception as e: QMessageBox.warning(self, "Lỗi", str(e))

    def toggle_drawing_mode(self):
        self.is_drawing_mode = self.btn_draw.isChecked()
        self.btn_draw.setText("❌ HỦY VẼ" if self.is_drawing_mode else "🖌 THÊM VỊ TRÍ HỌC SINH")
        self.btn_draw.setStyleSheet("background-color: #d32f2f;" if self.is_drawing_mode else "")

    def confirm_current_zone(self):
        if self.current_temp_zone and self.thread:
            new_id = len(self.thread.zones) + 1
            self.thread.zones.append({"id": new_id, "rect": self.current_temp_zone})
            self.thread.current_drawing_zone = None
            self.current_temp_zone = None
            self.btn_confirm_zone.setEnabled(False)

    def get_real_coords(self, qpoint):
        if not self.thread: return 0, 0
        lbl_w, lbl_h = self.video_label.width(), self.video_label.height()
        vid_w, vid_h = self.thread.video_width, self.thread.video_height
        if lbl_w == 0 or lbl_h == 0: return 0, 0
        return int(qpoint.x() * vid_w / lbl_w), int(qpoint.y() * vid_h / lbl_h)

    def on_mouse_press(self, event):
        if self.is_drawing_mode and event.button() == Qt.MouseButton.LeftButton:
            self.start_point = self.get_real_coords(event.position())

    def on_mouse_move(self, event):
        if self.is_drawing_mode and self.start_point:
            cur = self.get_real_coords(event.position())
            self.thread.current_drawing_zone = (min(self.start_point[0], cur[0]), min(self.start_point[1], cur[1]), abs(self.start_point[0]-cur[0]), abs(self.start_point[1]-cur[1]))

    def on_mouse_release(self, event):
        if self.is_drawing_mode:
            self.current_temp_zone = self.thread.current_drawing_zone
            self.start_point = None
            self.current_temp_zone = self.thread.current_drawing_zone # Fix minor bug in original logic
            if self.current_temp_zone and self.current_temp_zone[2] > 10: self.btn_confirm_zone.setEnabled(True)

    def load_available_cameras(self):
        self.combo_cam.clear()
        for i in range(len(QMediaDevices.videoInputs())): self.combo_cam.addItem(f"Camera {i}")

    def create_line(self):
        l = QFrame(); l.setFrameShape(QFrame.Shape.HLine); l.setStyleSheet("color: #555;"); return l

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #fff; }
            QPushButton { padding: 8px; border-radius: 4px; background: #424242; color: white; border: 1px solid #555; }
            QPushButton:hover { background: #505050; }
            QScrollArea { border: none; background: transparent; }
            QWidget { background: transparent; }
            QComboBox { padding: 5px; background: #333; color: white; border: 1px solid #555; }
        """)

    def closeEvent(self, event):
        if self.thread: self.thread.stop()
        event.accept()
    
    # --- [MỚI] HÀM HIỆN MENU XÓA ---
    def show_context_menu(self, pos, student_id):
        menu = QMenu(self)
        delete_action = menu.addAction(f"❌ Xóa vị trí HS-{student_id}")
        
        # Lấy widget gửi tín hiệu để hiển thị menu đúng chỗ
        sender_widget = self.student_widgets.get(student_id)
        if sender_widget:
            # Chuyển đổi tọa độ để menu hiện ngay tại con chuột
            global_pos = sender_widget.mapToGlobal(pos)
            action = menu.exec(global_pos)
            
            if action == delete_action:
                self.delete_zone(student_id)

    # --- [MỚI] HÀM XỬ LÝ LOGIC XÓA ---
    def delete_zone(self, student_id):
        if not self.thread: return
        
        reply = QMessageBox.question(self, 'Xác nhận', 
                                   f"Bạn có chắc muốn xóa vùng theo dõi HS-{student_id}?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # 1. Xóa khỏi danh sách vùng trong VideoThread
            # Lọc lại danh sách, giữ lại những vùng KHÔNG phải là ID cần xóa
            self.thread.zones = [z for z in self.thread.zones if str(z['id']) != str(student_id)]
            
            # 2. Xóa widget trên giao diện
            if student_id in self.student_widgets:
                self.student_widgets[student_id].setParent(None)
                del self.student_widgets[student_id]
            
            # 3. Sắp xếp lại giao diện Grid
            # (Hàm update_student_panel sẽ tự lo việc vẽ lại các ô còn lại)
            print(f"Đã xóa vùng HS-{student_id}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraMonitorUI()
    window.show()
    sys.exit(app.exec())