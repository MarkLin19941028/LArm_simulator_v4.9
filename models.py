import math
import numpy as np

# 嘗試導入常數，如果失敗則使用預設值 (避免單獨測試時報錯)
try:
    from constants import MAX_NOZZLE_SPEED_MMS
except ImportError:
    MAX_NOZZLE_SPEED_MMS = 250.0

class DispenseArm:
    def __init__(self, arm_id, pivot, home, length, p_start, p_end,
                 arm_artist=None, nozzle_artist=None):
        self.id = arm_id
        # 強制轉為 float numpy array，避免數據類型混亂
        self.pivot_pos = np.array(pivot, dtype=float)
        self.home_pos = np.array(home, dtype=float)
        self.arm_length = float(length)
        self.p_start = np.array(p_start, dtype=float)
        self.p_end = np.array(p_end, dtype=float)
        
        # 綁定繪圖物件 (Headless 模式下可能為 None)
        self.arm_line = arm_artist
        self.nozzle_head = nozzle_artist
        
        # --- 幾何計算區域 ---
        self.theta_start = math.atan2(self.p_start[1] - self.pivot_pos[1], self.p_start[0] - self.pivot_pos[0])
        self.theta_end = math.atan2(self.p_end[1] - self.pivot_pos[1], self.p_end[0] - self.pivot_pos[0])
        self.home_angle = math.atan2(self.home_pos[1] - self.pivot_pos[1], self.home_pos[0] - self.pivot_pos[0])
        
        # 計算 Center Position 相關參數
        center_angle = math.atan2(0 - self.pivot_pos[1], 0 - self.pivot_pos[0])
        self.center_pos_coords = self.angle_to_coords(center_angle)
        
        vec_center = self.center_pos_coords - self.p_start
        vec_path = self.p_end - self.p_start
        path_len_sq = np.dot(vec_path, vec_path)
        dot_product = np.dot(vec_center, vec_path)
        
        # 避免除以零
        ratio = np.clip(dot_product / path_len_sq if path_len_sq > 0 else 0, 0, 1)
        self.center_pos_percent = (ratio * 200) - 100

        # 計算最大速度百分比 (用於物理計算)
        angle_diff = self._get_angle_diff(self.theta_end, self.theta_start)
        arc_length = self.arm_length * abs(angle_diff)
        self.max_percent_speed = (MAX_NOZZLE_SPEED_MMS / arc_length) * 200 if arc_length > 0 else 0

    def _get_angle_diff(self, a1, a2):
        """計算兩個角度之間的最小差值 (-pi 到 pi)"""
        return (a1 - a2 + math.pi) % (2 * math.pi) - math.pi

    def angle_to_coords(self, angle):
        """輸入角度(弧度)，回傳 (x, y) 座標"""
        x = self.pivot_pos[0] + self.arm_length * math.cos(angle)
        y = self.pivot_pos[1] + self.arm_length * math.sin(angle)
        return np.array([x, y])

    def percent_to_angle(self, percent):
        """輸入路徑百分比 (-100~100)，回傳角度"""
        original_ratio = (percent + 100) / 200.0
        angle_diff = self._get_angle_diff(self.theta_end, self.theta_start)
        return self.theta_start + original_ratio * angle_diff

    def percent_to_coords(self, percent):
        """輸入路徑百分比，回傳 (x, y) 座標"""
        return self.angle_to_coords(self.percent_to_angle(percent))

    def coords_to_angle(self, coords):
        """輸入座標，回傳相對於轉軸的角度"""
        return math.atan2(coords[1] - self.pivot_pos[1], coords[0] - self.pivot_pos[0])

    def get_interpolated_coords(self, start_angle, end_angle, progress_ratio):
        """計算過渡期間的插值座標"""
        ratio = max(0.0, min(1.0, progress_ratio))
        angle_diff = self._get_angle_diff(end_angle, start_angle)
        current_angle = start_angle + ratio * angle_diff
        return self.angle_to_coords(current_angle)

    def update_artists(self, coords, color=None):
        """更新 GUI 上的圖形元件位置與顏色"""
        if self.nozzle_head is None or self.arm_line is None:
            return
            
        self.nozzle_head.center = (coords[0], coords[1])
        if color:
            self.nozzle_head.set_facecolor(color)
        
        self.arm_line.set_data([self.pivot_pos[0], coords[0]], [self.pivot_pos[1], coords[1]])
        
        self.nozzle_head.set_visible(True)
        self.arm_line.set_visible(True)

    def go_home(self):
        """讓手臂回到 Home 點並變灰"""
        self.update_artists(self.home_pos, color='gray')

    def get_artists(self):
        """回傳所有的繪圖物件列表，供 Blitting 動畫使用"""
        artists = []
        if self.arm_line: artists.append(self.arm_line)
        if self.nozzle_head: artists.append(self.nozzle_head)
        return artists