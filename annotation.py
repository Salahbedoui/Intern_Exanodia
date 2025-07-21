import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QHBoxLayout, QScrollArea, QSlider, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.original_image = None
        self.proc_image = None
        self.scale_factor = 1.0
        self.proc_scale = 0.5 # Scale down for processing to speed up Canny, etc.

        self.current_mode = 'outer'
        self.outer_mask = None
        self.inner_mask = None
        self.cached_contours = []

        self.undo_stack_outer = []
        self.undo_stack_inner = []

        # Eraser
        self.eraser_radius = 12
        self.eraser_pos = None
        self.erase_target = 'both'  # or 'outer', 'inner'

        # Parameters for Canny and contour approximation
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.gaussian_kernel_size = 5 # For Gaussian blur
        self.approx_poly_epsilon = 3.0 # For cv2.approxPolyDP

    def set_image(self, image):
        self.original_image = image.copy()
        self.image = image.copy()
        h, w = image.shape[:2]
        self.outer_mask = np.zeros((h, w), dtype=np.uint8)
        self.inner_mask = np.zeros((h, w), dtype=np.uint8)

        self.undo_stack_outer.clear()
        self.undo_stack_inner.clear()

        self.proc_image = cv2.resize(self.original_image, (0, 0), fx=self.proc_scale, fy=self.proc_scale)
        gray = cv2.cvtColor(self.proc_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blurred_gray = cv2.GaussianBlur(gray, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_gray, self.canny_threshold1, self.canny_threshold2)

        # Optional: Apply morphological closing to connect broken edges
        # kernel = np.ones((3,3), np.uint8)
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        self.cached_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        self.update_display()

    def update_display(self):
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)

        max_width = 1000
        max_height = 700
        w_ratio = max_width / w
        h_ratio = max_height / h
        self.scale_factor = min(w_ratio, h_ratio, 1.0)

        new_width = int(w * self.scale_factor)
        new_height = int(h * self.scale_factor)
        pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())

    def follow_contour_segment(self, contour, start_idx, max_points=150):
        n = len(contour)
        forward_points = [contour[i] for i in range(start_idx, min(start_idx + max_points, n))]
        backward_points = [contour[i] for i in range(start_idx - 1, max(start_idx - max_points - 1, -1), -1)]
        return np.array(backward_points[::-1] + forward_points)

    def save_undo_state(self):
        self.undo_stack_outer.append(self.outer_mask.copy())
        self.undo_stack_inner.append(self.inner_mask.copy())
        if len(self.undo_stack_outer) > 20:
            self.undo_stack_outer.pop(0)
            self.undo_stack_inner.pop(0)

    def undo(self):
        if not self.undo_stack_outer or not self.undo_stack_inner:
            return
        if len(self.undo_stack_outer) > 1:
            self.undo_stack_outer.pop()
            self.undo_stack_inner.pop()
            self.outer_mask = self.undo_stack_outer[-1].copy()
            self.inner_mask = self.undo_stack_inner[-1].copy()
        else:
            self.outer_mask = np.zeros_like(self.outer_mask)
            self.inner_mask = np.zeros_like(self.inner_mask)
            self.undo_stack_outer.clear()
            self.undo_stack_inner.clear()
        self.redraw_masks()

    def redraw_masks(self):
        overlay = self.original_image.copy()
        if self.outer_mask is not None:
            contours, _ = cv2.findContours(self.outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        if self.inner_mask is not None:
            contours, _ = cv2.findContours(self.inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
        self.image = overlay
        self.update_display()

    def mousePressEvent(self, event):
        if self.original_image is None:
            return

        x_disp = event.pos().x()
        y_disp = event.pos().y()
        x = int(x_disp / self.scale_factor)
        y = int(y_disp / self.scale_factor)
        self.eraser_pos = (x_disp, y_disp)

        if x >= self.original_image.shape[1] or y >= self.original_image.shape[0]:
            return

        x_proc = int(x * self.proc_scale)
        y_proc = int(y * self.proc_scale)

        radius_threshold = 10
        nearby_contours = []
        for contour in self.cached_contours:
            dist_to_point = cv2.pointPolygonTest(contour, (x_proc, y_proc), True)
            if abs(dist_to_point) <= radius_threshold:
                nearby_contours.append(contour)

        if not nearby_contours:
            return

        # Scale contours back to original image size
        resized_contours = [np.array(cnt * (1 / self.proc_scale), dtype=np.int32) for cnt in nearby_contours]

        self.save_undo_state()

        if self.current_mode == 'eraser':
            for cnt in resized_contours:
                erase_mask = np.zeros_like(self.outer_mask)
                # Draw thicker line for easier intersection with eraser circle
                cv2.drawContours(erase_mask, [cnt], -1, 255, thickness=self.eraser_radius)
                erase_circle = np.zeros_like(self.outer_mask)
                cv2.circle(erase_circle, (x, y), self.eraser_radius, 255, -1)
                intersection = cv2.bitwise_and(erase_mask, erase_circle)

                if self.erase_target in ['outer', 'both']:
                    self.outer_mask = cv2.bitwise_and(self.outer_mask, cv2.bitwise_not(intersection))
                if self.erase_target in ['inner', 'both']:
                    self.inner_mask = cv2.bitwise_and(self.inner_mask, cv2.bitwise_not(intersection))
        else:
            if self.current_mode == 'outer':
                merged_points = np.vstack(resized_contours)
                # Apply approxPolyDP with configured epsilon
                merged_contour = cv2.approxPolyDP(merged_points, self.approx_poly_epsilon, False)
                cv2.drawContours(self.outer_mask, [merged_contour], -1, 255, 3)
            else:
                # Logic for inner mask drawing, apply approxPolyDP as well
                if np.any(self.inner_mask):
                    inv_inner = cv2.bitwise_not(self.inner_mask)
                    dist_transform = cv2.distanceTransform(inv_inner, cv2.DIST_L2, 3)
                    if dist_transform[y, x] <= 10:
                        best = min(
                            ((c, i, np.min(np.linalg.norm(c[:, 0, :] - np.array([x, y]), axis=1)))
                             for c in resized_contours for i in range(len(c))),
                            key=lambda x: x[2], default=(None, None, None)
                        )
                        if best[0] is not None:
                            segment = self.follow_contour_segment(best[0], best[1])
                            segment = cv2.approxPolyDP(segment, self.approx_poly_epsilon, False) # Apply smoothing
                            cv2.drawContours(self.inner_mask, [segment], -1, 255, 3)
                        else:
                            merged_points = np.vstack(resized_contours)
                            cv2.drawContours(self.inner_mask, [cv2.approxPolyDP(merged_points, self.approx_poly_epsilon, False)], -1, 255, 3)
                    else:
                        merged_points = np.vstack(resized_contours)
                        cv2.drawContours(self.inner_mask, [cv2.approxPolyDP(merged_points, self.approx_poly_epsilon, False)], -1, 255, 3)
                else:
                    merged_points = np.vstack(resized_contours)
                    cv2.drawContours(self.inner_mask, [cv2.approxPolyDP(merged_points, self.approx_poly_epsilon, False)], -1, 255, 3)

        self.redraw_masks()

    def mouseMoveEvent(self, event):
        if self.current_mode == 'eraser':
            self.eraser_pos = (event.pos().x(), event.pos().y())
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_mode == 'eraser' and self.eraser_pos:
            qp = QPainter(self)
            pen = QPen(Qt.cyan, 1, Qt.SolidLine)
            qp.setPen(pen)
            radius = int(self.eraser_radius * self.scale_factor)
            center = QPoint(int(self.eraser_pos[0]), int(self.eraser_pos[1]))
            qp.drawEllipse(center, radius, radius)


    def save_combined_annotation(self, output_path, base_name):
        if self.outer_mask is None or self.inner_mask is None:
            return
        h, w = self.outer_mask.shape
        annotated_img = np.zeros((h, w, 3), dtype=np.uint8)
        annotated_img[self.outer_mask > 0] = [0, 0, 255]
        annotated_img[self.inner_mask > 0] = [255, 0, 0]
        cv2.imwrite(os.path.join(output_path, base_name + '_annotated.png'), annotated_img)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Pipe Wall Annotator (Enhanced Eraser)")
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

        self.eraser_btn = QPushButton("Eraser OFF")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(self.toggle_eraser)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.image_label.undo)

        self.eraser_size = QSlider(Qt.Horizontal)
        self.eraser_size.setMinimum(5)
        self.eraser_size.setMaximum(50)
        self.eraser_size.setValue(12)
        self.eraser_size.setFixedWidth(100)
        self.eraser_size.valueChanged.connect(self.set_eraser_size)

        self.eraser_target = QComboBox()
        self.eraser_target.addItems(["both", "outer", "inner"])
        self.eraser_target.currentTextChanged.connect(self.set_eraser_target)

        # UI for Canny and approxPolyDP parameters (optional but recommended for tuning)
        self.canny1_slider = QSlider(Qt.Horizontal)
        self.canny1_slider.setMinimum(0)
        self.canny1_slider.setMaximum(255)
        self.canny1_slider.setValue(self.image_label.canny_threshold1)
        self.canny1_slider.setFixedWidth(100)
        self.canny1_slider.valueChanged.connect(self.set_canny1)
        self.canny1_label = QLabel(f"Canny Thresh1: {self.image_label.canny_threshold1}")

        self.canny2_slider = QSlider(Qt.Horizontal)
        self.canny2_slider.setMinimum(0)
        self.canny2_slider.setMaximum(255)
        self.canny2_slider.setValue(self.image_label.canny_threshold2)
        self.canny2_slider.setFixedWidth(100)
        self.canny2_slider.valueChanged.connect(self.set_canny2)
        self.canny2_label = QLabel(f"Canny Thresh2: {self.image_label.canny_threshold2}")

        self.gaussian_slider = QSlider(Qt.Horizontal)
        self.gaussian_slider.setMinimum(1)
        self.gaussian_slider.setMaximum(15) # Only odd numbers for kernel size
        self.gaussian_slider.setValue(self.image_label.gaussian_kernel_size)
        self.gaussian_slider.setSingleStep(2) # Ensure odd steps
        self.gaussian_slider.setFixedWidth(100)
        self.gaussian_slider.valueChanged.connect(self.set_gaussian_kernel)
        self.gaussian_label = QLabel(f"Gaussian Kernel: {self.image_label.gaussian_kernel_size}")

        self.approx_poly_slider = QSlider(Qt.Horizontal)
        self.approx_poly_slider.setMinimum(0)
        self.approx_poly_slider.setMaximum(200) # Representing 0.0 to 20.0 (scaled by 10)
        self.approx_poly_slider.setValue(int(self.image_label.approx_poly_epsilon * 10))
        self.approx_poly_slider.setFixedWidth(100)
        self.approx_poly_slider.valueChanged.connect(self.set_approx_poly_epsilon)
        self.approx_poly_label = QLabel(f"Approx Poly Epsilon: {self.image_label.approx_poly_epsilon:.1f}")


        top_layout = QHBoxLayout()
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.output_btn)
        top_layout.addWidget(self.prev_btn)
        top_layout.addWidget(self.next_btn)
        top_layout.addWidget(self.toggle_btn)
        top_layout.addWidget(self.eraser_btn)
        top_layout.addWidget(self.undo_btn)
        top_layout.addWidget(QLabel("Eraser Size:"))
        top_layout.addWidget(self.eraser_size)
        top_layout.addWidget(QLabel("Eraser Target:"))
        top_layout.addWidget(self.eraser_target)
        top_layout.addWidget(self.save_btn)

        # Add new sliders and labels to the layout
        params_layout = QHBoxLayout()
        params_layout.addWidget(self.canny1_label)
        params_layout.addWidget(self.canny1_slider)
        params_layout.addWidget(self.canny2_label)
        params_layout.addWidget(self.canny2_slider)
        params_layout.addWidget(self.gaussian_label)
        params_layout.addWidget(self.gaussian_slider)
        params_layout.addWidget(self.approx_poly_label)
        params_layout.addWidget(self.approx_poly_slider)


        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(params_layout) # Add parameters layout
        main_layout.addWidget(scroll_area)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def set_eraser_size(self, value):
        self.image_label.eraser_radius = value

    def set_eraser_target(self, text):
        self.image_label.erase_target = text

    def set_canny1(self, value):
        self.image_label.canny_threshold1 = value
        self.canny1_label.setText(f"Canny Thresh1: {value}")
        self.load_image() # Reload image to re-apply Canny

    def set_canny2(self, value):
        self.image_label.canny_threshold2 = value
        self.canny2_label.setText(f"Canny Thresh2: {value}")
        self.load_image() # Reload image to re-apply Canny

    def set_gaussian_kernel(self, value):
        # Ensure kernel size is odd
        if value % 2 == 0:
            value += 1
        self.image_label.gaussian_kernel_size = value
        self.gaussian_label.setText(f"Gaussian Kernel: {value}")
        self.load_image() # Reload image to re-apply Canny

    def set_approx_poly_epsilon(self, value):
        self.image_label.approx_poly_epsilon = value / 10.0 # Scale back to float
        self.approx_poly_label.setText(f"Approx Poly Epsilon: {self.image_label.approx_poly_epsilon:.1f}")
        self.image_label.redraw_masks() # Redraw just the masks with new epsilon

    def toggle_eraser(self):
        if self.eraser_btn.isChecked():
            self.image_label.current_mode = 'eraser'
            self.eraser_btn.setText("Eraser ON")
            self.toggle_btn.setEnabled(False)
        else:
            if self.toggle_btn.text().endswith("Inner"):
                self.image_label.current_mode = 'outer'
            else:
                self.image_label.current_mode = 'inner'
            self.eraser_btn.setText("Eraser OFF")
            self.toggle_btn.setEnabled(True)

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
            img = cv2.imread(self.image_paths[self.current_index])
            if img is not None:
                self.image_label.set_image(img)
                self.setWindowTitle(f"Annotating: {os.path.basename(self.image_paths[self.current_index])}")

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
        print(f"Annotation saved for {base_name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
