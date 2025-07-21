import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image = None
        self.current_mode = 'outer'  # or 'inner'
        self.outer_mask = None
        self.inner_mask = None
        self.original_image = None
        self.scale_factor = 1.0  # to map display coords to original image coords

    def set_image(self, image):
        self.original_image = image.copy()
        self.image = image
        self.update_display()

        h, w = image.shape[:2]
        self.outer_mask = np.zeros((h, w), dtype=np.uint8)
        self.inner_mask = np.zeros((h, w), dtype=np.uint8)

    def update_display(self):
        if self.image is None:
            return
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)

        max_width = 1000
        max_height = 700

        w_ratio = max_width / w
        h_ratio = max_height / h
        self.scale_factor = min(w_ratio, h_ratio, 1.0)  # do not upscale

        new_width = int(w * self.scale_factor)
        new_height = int(h * self.scale_factor)

        pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())

    def follow_contour_segment(self, contour, start_idx, max_points=150):
        n = len(contour)
        forward_points = []
        backward_points = []

        for i in range(start_idx, min(start_idx + max_points, n)):
            forward_points.append(contour[i])

        for i in range(start_idx - 1, max(start_idx - max_points - 1, -1), -1):
            backward_points.append(contour[i])

        full_segment = backward_points[::-1] + forward_points
        return np.array(full_segment)

    def mousePressEvent(self, event):
        if self.image is None:
            return

        x_disp = event.pos().x()
        y_disp = event.pos().y()
        x = int(x_disp / self.scale_factor)
        y = int(y_disp / self.scale_factor)

        if x >= self.original_image.shape[1] or y >= self.original_image.shape[0]:
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        radius_threshold = 20
        nearby_contours = []
        for contour in contours:
            dists = np.linalg.norm(contour[:, 0, :] - np.array([x, y]), axis=1)
            if np.any(dists <= radius_threshold):
                nearby_contours.append(contour)

        if not nearby_contours:
            return

        if self.current_mode == 'outer':
            merged_points = np.vstack(nearby_contours)
            epsilon = 1.5
            merged_contour = cv2.approxPolyDP(merged_points, epsilon, False)
            cv2.drawContours(self.outer_mask, [merged_contour], -1, 255, 3)

        else:
            if self.inner_mask is not None and np.any(self.inner_mask):
                inner_mask_inv = cv2.bitwise_not(self.inner_mask)
                dist_transform = cv2.distanceTransform(inner_mask_inv, cv2.DIST_L2, 3)

                if dist_transform[y, x] <= 10:
                    best_contour = None
                    best_point_idx = None
                    min_dist = float('inf')

                    for contour in nearby_contours:
                        dists = np.linalg.norm(contour[:, 0, :] - np.array([x, y]), axis=1)
                        idx = np.argmin(dists)
                        dist_val = dists[idx]
                        if dist_val < min_dist:
                            min_dist = dist_val
                            best_contour = contour
                            best_point_idx = idx

                    if best_contour is not None:
                        segment = self.follow_contour_segment(best_contour, best_point_idx, max_points=200)
                        cv2.drawContours(self.inner_mask, [segment], -1, 255, 3)
                    else:
                        merged_points = np.vstack(nearby_contours)
                        epsilon = 1.5
                        merged_contour = cv2.approxPolyDP(merged_points, epsilon, False)
                        cv2.drawContours(self.inner_mask, [merged_contour], -1, 255, 3)
                else:
                    merged_points = np.vstack(nearby_contours)
                    epsilon = 1.5
                    merged_contour = cv2.approxPolyDP(merged_points, epsilon, False)
                    cv2.drawContours(self.inner_mask, [merged_contour], -1, 255, 3)
            else:
                merged_points = np.vstack(nearby_contours)
                epsilon = 1.5
                merged_contour = cv2.approxPolyDP(merged_points, epsilon, False)
                cv2.drawContours(self.inner_mask, [merged_contour], -1, 255, 3)

        overlay = self.original_image.copy()
        outer_contours, _ = cv2.findContours(self.outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if outer_contours:
            cv2.drawContours(overlay, outer_contours, -1, (0, 0, 255), 3)
        inner_contours, _ = cv2.findContours(self.inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if inner_contours:
            cv2.drawContours(overlay, inner_contours, -1, (255, 0, 0), 3)

        self.image = overlay
        self.update_display()

    def save_combined_annotation(self, output_path, base_name):
        if self.outer_mask is None or self.inner_mask is None:
            return

        h, w = self.outer_mask.shape
        annotated_img = np.zeros((h, w, 3), dtype=np.uint8)

        annotated_img[self.outer_mask > 0] = [0, 0, 255]  # red
        annotated_img[self.inner_mask > 0] = [255, 0, 0]  # blue

        cv2.imwrite(os.path.join(output_path, base_name + '_annotated.png'), annotated_img)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Pipe Wall Manual Annotator (Semi-Auto)")
        self.image_paths = []
        self.current_index = -1
        self.output_folder = ""

        self.image_label = ImageLabel()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)

        self.load_btn = QPushButton("Select Input Folder")
        self.load_btn.clicked.connect(self.select_input_folder)

        self.output_btn = QPushButton("Select Output Folder")
        self.output_btn.clicked.connect(self.select_output_folder)

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_image)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_image)

        self.save_btn = QPushButton("💾 Save Annotation")
        self.save_btn.clicked.connect(self.save_annotation)

        self.toggle_btn = QPushButton("Switch to Inner")
        self.toggle_btn.clicked.connect(self.toggle_mode)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.output_btn)
        top_layout.addWidget(self.prev_btn)
        top_layout.addWidget(self.next_btn)
        top_layout.addWidget(self.toggle_btn)
        top_layout.addWidget(self.save_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(scroll_area)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.image_paths = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]
            self.image_paths.sort()
            self.current_index = 0
            self.load_image()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder

    def load_image(self):
        if 0 <= self.current_index < len(self.image_paths):
            path = self.image_paths[self.current_index]
            img = cv2.imread(path)
            if img is not None:
                self.image_label.set_image(img)
                self.setWindowTitle(f"Annotating: {os.path.basename(path)}")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_image()

    def toggle_mode(self):
        if self.image_label.current_mode == 'outer':
            self.image_label.current_mode = 'inner'
            self.toggle_btn.setText("Switch to Outer")
        else:
            self.image_label.current_mode = 'outer'
            self.toggle_btn.setText("Switch to Inner")

    def save_annotation(self):
        if not self.output_folder:
            print("Please select an output folder before saving.")
            return
        base_name = os.path.splitext(os.path.basename(self.image_paths[self.current_index]))[0]
        self.image_label.save_combined_annotation(self.output_folder, base_name)
        print(f"Combined annotation saved for {base_name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
