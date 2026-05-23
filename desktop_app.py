import sys
import os
import torch
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QProgressBar, QTabWidget, QMessageBox, QSlider,
    QGroupBox, QGridLayout, QSpacerItem, QSizePolicy, QStatusBar
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class InferenceThread(QThread):
    progress_updated = pyqtSignal(int)
    inference_finished = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, opt_data, sar_data, model, patch_size):
        super().__init__()
        self.opt_data = opt_data
        self.sar_data = sar_data
        self.model = model
        self.patch_size = patch_size
        
    def run(self):
        try:
            if self.opt_data is None or self.sar_data is None:
                self.error_occurred.emit("数据未加载")
                return
                
            if self.opt_data.shape[2] != self.sar_data.shape[2] or \
               self.opt_data.shape[3] != self.sar_data.shape[3]:
                self.error_occurred.emit("光学数据和SAR数据尺寸不匹配")
                return
                
            self.model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            doy_norm = np.random.rand(self.opt_data.shape[0])
            H, W = self.opt_data.shape[2], self.opt_data.shape[3]
            pred_map = np.zeros((H, W), dtype=np.int32)
            count_map = np.zeros((H, W), dtype=np.int32)
            
            stride = self.patch_size // 2
            rows = list(range(0, H - self.patch_size + 1, stride)) + [H - self.patch_size]
            cols = list(range(0, W - self.patch_size + 1, stride)) + [W - self.patch_size]
            
            total_steps = len(set(rows)) * len(set(cols))
            current_step = 0
            
            for r in set(rows):
                for c in set(cols):
                    opt_patch = self.opt_data[:, :, r:r+self.patch_size, c:c+self.patch_size]
                    sar_patch = self.sar_data[:, :, r:r+self.patch_size, c:c+self.patch_size]
                    
                    opt_t = torch.from_numpy(opt_patch).unsqueeze(0).float().to(device)
                    sar_t = torch.from_numpy(sar_patch).unsqueeze(0).float().to(device)
                    doy_t = torch.from_numpy(doy_norm).unsqueeze(0).float().to(device)
                    
                    dem_t = torch.zeros(1, 5, self.patch_size, self.patch_size).float().to(device)
                    
                    with torch.no_grad():
                        logits = self.model(opt_t, sar_t, dem_t, doy_t)
                        pred = logits.argmax(dim=1).cpu().numpy()[0]
                    
                    pred_map[r:r+self.patch_size, c:c+self.patch_size] += pred
                    count_map[r:r+self.patch_size, c:c+self.patch_size] += 1
                    
                    current_step += 1
                    self.progress_updated.emit(int(current_step / total_steps * 100))
            
            mask = count_map > 0
            pred_map[mask] = pred_map[mask] // count_map[mask]
            
            self.inference_finished.emit(pred_map)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌾 遥感影像作物分类系统")
        self.setGeometry(100, 100, 1200, 800)
        
        self.opt_data = None
        self.sar_data = None
        self.model = None
        self.pred_result = None
        self.patch_size = 32
        
        self.init_ui()
        self.auto_init_model()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.create_menu()
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::tab-bar { alignment: center; }")
        layout.addWidget(self.tabs)
        
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        
        self.tabs.addTab(self.tab1, "📥 数据输入")
        self.tabs.addTab(self.tab2, "🔄 模型推理")
        self.tabs.addTab(self.tab3, "📊 结果展示")
        
        self.init_tab1()
        self.init_tab2()
        self.init_tab3()
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请先加载数据")
    
    def create_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("文件")
        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)
        
        help_menu = menubar.addMenu("帮助")
        about_action = help_menu.addAction("关于")
        about_action.triggered.connect(self.show_about)
    
    def show_about(self):
        QMessageBox.about(self, "关于", "遥感影像作物分类系统\n\n版本: 1.0.0\n\n基于深度学习的多模态遥感影像农作物分类系统\n\n支持光学和SAR数据融合分类")
    
    def init_tab1(self):
        layout = QVBoxLayout(self.tab1)
        layout.setSpacing(10)
        
        group = QGroupBox("数据文件选择")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout.addWidget(group)
        group_layout = QGridLayout(group)
        group_layout.setSpacing(10)
        
        group_layout.addWidget(QLabel("光学时序数据:"), 0, 0)
        self.opt_path_label = QLabel("未选择")
        self.opt_path_label.setStyleSheet("color: #666;")
        group_layout.addWidget(self.opt_path_label, 0, 1)
        self.opt_btn = QPushButton("浏览...")
        self.opt_btn.setStyleSheet("padding: 5px 15px;")
        self.opt_btn.clicked.connect(self.select_opt_data)
        group_layout.addWidget(self.opt_btn, 0, 2)
        
        group_layout.addWidget(QLabel("SAR时序数据:"), 1, 0)
        self.sar_path_label = QLabel("未选择")
        self.sar_path_label.setStyleSheet("color: #666;")
        group_layout.addWidget(self.sar_path_label, 1, 1)
        self.sar_btn = QPushButton("浏览...")
        self.sar_btn.setStyleSheet("padding: 5px 15px;")
        self.sar_btn.clicked.connect(self.select_sar_data)
        group_layout.addWidget(self.sar_btn, 1, 2)
        
        self.data_info_group = QGroupBox("数据信息")
        layout.addWidget(self.data_info_group)
        info_layout = QVBoxLayout(self.data_info_group)
        self.data_info = QLabel("请选择数据文件")
        self.data_info.setStyleSheet("color: #888;")
        info_layout.addWidget(self.data_info)
        
        self.preview_group = QGroupBox("数据预览")
        layout.addWidget(self.preview_group)
        preview_layout = QHBoxLayout(self.preview_group)
        
        self.opt_figure = Figure(figsize=(4, 3))
        self.opt_canvas = FigureCanvas(self.opt_figure)
        self.opt_canvas.setStyleSheet("background-color: white;")
        preview_layout.addWidget(self.opt_canvas)
        
        self.sar_figure = Figure(figsize=(4, 3))
        self.sar_canvas = FigureCanvas(self.sar_figure)
        self.sar_canvas.setStyleSheet("background-color: white;")
        preview_layout.addWidget(self.sar_canvas)
    
    def init_tab2(self):
        layout = QVBoxLayout(self.tab2)
        layout.setSpacing(10)
        
        model_group = QGroupBox("模型配置")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout.addWidget(model_group)
        model_layout = QVBoxLayout(model_group)
        
        self.load_model_btn = QPushButton("📦 重新加载模型")
        self.load_model_btn.setStyleSheet("padding: 8px 20px; font-weight: bold;")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        
        self.model_status = QLabel("状态：模型未加载")
        self.model_status.setStyleSheet("color: #d9534f;")
        model_layout.addWidget(self.model_status)
        
        param_group = QGroupBox("推理参数")
        param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout.addWidget(param_group)
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(8)
        
        param_layout.addWidget(QLabel("Patch大小:"))
        self.patch_slider = QSlider(Qt.Horizontal)
        self.patch_slider.setRange(16, 64)
        self.patch_slider.setValue(32)
        self.patch_slider.setTickInterval(8)
        self.patch_slider.setTickPosition(QSlider.TicksBelow)
        param_layout.addWidget(self.patch_slider)
        self.patch_label = QLabel("当前值: 32")
        self.patch_slider.valueChanged.connect(self.update_patch_label)
        param_layout.addWidget(self.patch_label)
        
        self.infer_btn = QPushButton("🎯 开始推理")
        self.infer_btn.setStyleSheet("padding: 10px 30px; font-weight: bold; font-size: 14px;")
        self.infer_btn.clicked.connect(self.start_inference)
        self.infer_btn.setEnabled(False)
        layout.addWidget(self.infer_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar { height: 20px; }")
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("状态：等待")
        self.status_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.status_label)
    
    def init_tab3(self):
        layout = QVBoxLayout(self.tab3)
        layout.setSpacing(10)
        
        result_group = QGroupBox("分类结果")
        result_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout.addWidget(result_group)
        result_layout = QHBoxLayout(result_group)
        
        self.result_figure = Figure(figsize=(8, 6))
        self.result_canvas = FigureCanvas(self.result_figure)
        self.result_canvas.setStyleSheet("background-color: white;")
        result_layout.addWidget(self.result_canvas)
        
        stats_group = QGroupBox("统计信息")
        stats_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        result_layout.addWidget(stats_group)
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("请先进行推理")
        self.stats_label.setStyleSheet("color: #888;")
        stats_layout.addWidget(self.stats_label)
        
        self.pie_figure = Figure(figsize=(4, 3))
        self.pie_canvas = FigureCanvas(self.pie_figure)
        self.pie_canvas.setStyleSheet("background-color: white;")
        stats_layout.addWidget(self.pie_canvas)
        
        self.download_btn = QPushButton("💾 下载结果")
        self.download_btn.setStyleSheet("padding: 8px 20px;")
        self.download_btn.clicked.connect(self.download_result)
        self.download_btn.setEnabled(False)
        layout.addWidget(self.download_btn)
    
    def update_patch_label(self, value):
        self.patch_size = value
        self.patch_label.setText(f"当前值: {value}")
    
    def select_opt_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择光学数据", "", "NPY文件 (*.npy)"
        )
        if file_path:
            try:
                self.opt_data = np.load(file_path)
                self.validate_data_shape(self.opt_data, "光学")
                self.opt_path_label.setText(os.path.basename(file_path))
                self.opt_path_label.setStyleSheet("color: #3c763d;")
                self.update_data_info()
                self.show_opt_preview()
                self.status_bar.showMessage(f"已加载光学数据: {self.opt_data.shape}")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载光学数据失败: {str(e)}")
                self.opt_data = None
    
    def select_sar_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择SAR数据", "", "NPY文件 (*.npy)"
        )
        if file_path:
            try:
                self.sar_data = np.load(file_path)
                self.validate_data_shape(self.sar_data, "SAR")
                self.sar_path_label.setText(os.path.basename(file_path))
                self.sar_path_label.setStyleSheet("color: #3c763d;")
                self.update_data_info()
                self.show_sar_preview()
                self.status_bar.showMessage(f"已加载SAR数据: {self.sar_data.shape}")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载SAR数据失败: {str(e)}")
                self.sar_data = None
    
    def validate_data_shape(self, data, data_type):
        if len(data.shape) != 4:
            raise ValueError(f"{data_type}数据应为4维数组 (时间步长, 通道数, 高度, 宽度)")
        if data.shape[0] < 1:
            raise ValueError(f"{data_type}数据时间步长至少为1")
        if data.shape[1] < 1:
            raise ValueError(f"{data_type}数据通道数至少为1")
    
    def update_data_info(self):
        info = ""
        if self.opt_data is not None:
            info += f"<b>光学数据:</b><br>"
            info += f"  形状: {self.opt_data.shape}<br>"
            info += f"  时间步长: {self.opt_data.shape[0]}<br>"
            info += f"  通道数: {self.opt_data.shape[1]}<br>"
            info += f"  尺寸: {self.opt_data.shape[2]} x {self.opt_data.shape[3]}<br>"
        if self.sar_data is not None:
            info += f"<br><b>SAR数据:</b><br>"
            info += f"  形状: {self.sar_data.shape}<br>"
            info += f"  时间步长: {self.sar_data.shape[0]}<br>"
            info += f"  通道数: {self.sar_data.shape[1]}<br>"
            info += f"  尺寸: {self.sar_data.shape[2]} x {self.sar_data.shape[3]}<br>"
        
        self.data_info.setText(info)
        self.data_info.setStyleSheet("color: #333;")
        
        if self.opt_data is not None and self.sar_data is not None:
            if self.opt_data.shape[2] == self.sar_data.shape[2] and \
               self.opt_data.shape[3] == self.sar_data.shape[3]:
                self.infer_btn.setEnabled(True)
                self.status_bar.showMessage("数据准备完成，可以开始推理")
            else:
                QMessageBox.warning(self, "警告", "光学数据和SAR数据尺寸不匹配！")
                self.status_bar.showMessage("数据尺寸不匹配")
    
    def show_opt_preview(self):
        if self.opt_data is None:
            return
        
        ax = self.opt_figure.add_subplot(111)
        ax.clear()
        
        if self.opt_data.shape[1] >= 3:
            rgb = self.opt_data[0, [3, 2, 1], :, :].transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            ax.imshow(rgb)
        else:
            ax.imshow(self.opt_data[0, 0, :, :], cmap='gray')
        
        ax.set_title("光学影像预览")
        ax.axis('off')
        self.opt_canvas.draw()
    
    def show_sar_preview(self):
        if self.sar_data is None:
            return
        
        ax = self.sar_figure.add_subplot(111)
        ax.clear()
        
        if self.sar_data.shape[1] >= 1:
            ax.imshow(self.sar_data[0, 0, :, :], cmap='gray')
        
        ax.set_title("SAR影像预览")
        ax.axis('off')
        self.sar_canvas.draw()
    
    def auto_init_model(self):
        self.status_bar.showMessage("正在初始化模型...")
        QApplication.processEvents()
        self.load_model()
    
    def load_model(self, use_v5pro=False):
        try:
            from models.fusion_net_v5 import FusionCropNetV5
            from models.fusion_net_v5pro import FusionCropNetV5Pro

            self.model_status.setText("状态：正在加载模型...")
            QApplication.processEvents()

            if use_v5pro:
                self.model = FusionCropNetV5Pro(
                    opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                    feat_dim=512, backbone="resnet50", pretrained=False,
                    n_heads=16, win_size=4, n_layers=4
                )
            else:
                self.model = FusionCropNetV5(
                    opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                    feat_dim=512, backbone="resnet50", pretrained=False,
                    n_heads=16, win_size=4, n_layers=4
                )
            
            model_path = "checkpoints/best_phase2.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(checkpoint["model_state"], strict=False)
                self.model_status.setText("状态：✓ 模型加载成功 (预训练)")
                self.model_status.setStyleSheet("color: #3c763d;")
                self.status_bar.showMessage("模型加载成功（使用预训练权重）")
            else:
                self.model_status.setText("状态：✓ 模型加载成功 (随机初始化)")
                self.model_status.setStyleSheet("color: #f0ad4e;")
                self.status_bar.showMessage("模型加载成功（随机初始化，建议使用预训练权重）")
            
            if self.opt_data is not None and self.sar_data is not None:
                if self.opt_data.shape[2] == self.sar_data.shape[2] and \
                   self.opt_data.shape[3] == self.sar_data.shape[3]:
                    self.infer_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            self.model_status.setText("状态：✗ 模型加载失败")
            self.model_status.setStyleSheet("color: #d9534f;")
            self.status_bar.showMessage(f"模型加载失败: {str(e)}")
    
    def start_inference(self):
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        if not self.validate_data_before_inference():
            return
        
        self.status_label.setText("状态：推理中...")
        self.status_label.setStyleSheet("color: #337ab7;")
        self.progress_bar.setValue(0)
        self.infer_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        
        self.thread = InferenceThread(
            self.opt_data, self.sar_data, self.model, self.patch_size
        )
        self.thread.progress_updated.connect(self.progress_bar.setValue)
        self.thread.inference_finished.connect(self.on_inference_finished)
        self.thread.error_occurred.connect(self.on_error)
        self.thread.start()
    
    def validate_data_before_inference(self):
        if self.opt_data is None:
            QMessageBox.warning(self, "警告", "请先加载光学数据")
            return False
        if self.sar_data is None:
            QMessageBox.warning(self, "警告", "请先加载SAR数据")
            return False
        if self.opt_data.shape[0] != self.sar_data.shape[0]:
            QMessageBox.warning(self, "警告", "光学数据和SAR数据时间步长不一致")
            return False
        return True
    
    def on_inference_finished(self, pred_result):
        self.pred_result = pred_result
        self.status_label.setText("状态：✓ 推理完成")
        self.status_label.setStyleSheet("color: #3c763d;")
        self.infer_btn.setEnabled(True)
        self.download_btn.setEnabled(True)
        self.status_bar.showMessage("推理完成")
        self.tabs.setCurrentIndex(2)
        self.display_result()
    
    def on_error(self, error_msg):
        QMessageBox.critical(self, "错误", f"推理失败: {error_msg}")
        self.status_label.setText("状态：✗ 错误")
        self.status_label.setStyleSheet("color: #d9534f;")
        self.infer_btn.setEnabled(True)
        self.status_bar.showMessage(f"推理失败: {error_msg}")
    
    def display_result(self):
        if self.pred_result is None:
            return
        
        for ax in self.result_figure.axes:
            self.result_figure.delaxes(ax)
        
        ax = self.result_figure.add_subplot(111)
        
        crop_colors = {
            0: '#FFFFFF', 1: '#FFD700', 2: '#228B22',
            3: '#4682B4', 4: '#9ACD32', 5: '#FF8C00', 6: '#A9A9A9'
        }
        cmap = plt.cm.colors.ListedColormap(list(crop_colors.values()))
        
        im = ax.imshow(self.pred_result, cmap=cmap)
        
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5, 6])
        cbar.ax.set_yticklabels(["背景", "冬小麦", "夏玉米", "水稻", "大豆", "棉花", "其他"])
        
        ax.set_title("作物分类结果")
        ax.axis('off')
        self.result_canvas.draw()
        
        class_counts = np.bincount(self.pred_result.flatten(), minlength=7)
        class_names = ["背景", "冬小麦", "夏玉米", "水稻", "大豆", "棉花", "其他"]
        
        stats = "<b>统计信息：</b><br><br>"
        total_pixels = np.prod(self.pred_result.shape)
        crop_pixels = total_pixels - class_counts[0]
        
        for name, count in zip(class_names, class_counts):
            stats += f"{name}: {count:,} ({count/total_pixels*100:.1f}%)<br>"
        
        stats += f"<br><b>作物覆盖率:</b> {crop_pixels/total_pixels*100:.1f}%"
        self.stats_label.setText(stats)
        self.stats_label.setStyleSheet("color: #333;")
        
        for ax in self.pie_figure.axes:
            self.pie_figure.delaxes(ax)
        
        pie_ax = self.pie_figure.add_subplot(111)
        pie_ax.pie(class_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
        pie_ax.axis('equal')
        self.pie_canvas.draw()
    
    def download_result(self):
        if self.pred_result is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "classification_result.npy", "NPY文件 (*.npy)"
        )
        
        if file_path:
            try:
                np.save(file_path, self.pred_result)
                QMessageBox.information(self, "成功", "结果保存成功！")
                self.status_bar.showMessage(f"结果已保存: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"保存失败: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())