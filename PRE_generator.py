import numpy as np
import math
import os
import matplotlib.pyplot as plt
from models import DispenseArm
from simulation_engine import SimulationEngine
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    PRE_ALPHA, PRE_BETA, PRE_GRID_SIZE,
    PRE_Q_REF, PRE_GAMMA_BASE
)

class PREGenerator:
    def __init__(self, app_instance):
        self.app = app_instance
        self.alpha = PRE_ALPHA
        self.beta = PRE_BETA
        self.impact_radius = PRE_GRID_SIZE
        self.q_ref = PRE_Q_REF
        self.gamma_base = PRE_GAMMA_BASE

    def generate(self, recipe, filepath, progress_widgets=None):
        """
        Cleaning Dose 模擬邏輯：
        1. 使用結合流量補償與再附著衰減的混合型一階動力學模型
        2. k = (alpha * omega^1.5 * r) + (beta * C_q)
        3. Dose_eff = sum( [ k * exp(-gamma_eff * r) * dt ] )
        4. 空間分佈：擴散至 PRE_GRID_SIZE 範圍
        """
        # 1. 初始化 Headless Arms
        headless_arms = {i: DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], None, None) 
                         for i, geo in ARM_GEOMETRIES.items()}

        # 獲取物理參數
        water_params = self.app._get_water_params()
        water_params_dict = {i: {
            'viscosity': water_params['viscosity'],
            'surface_tension': water_params['surface_tension'],
            'evaporation_rate': water_params['evaporation_rate']
        } for i in [1, 2, 3]}

        # 2. 實例化引擎
        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True)
        
        # 3. 準備 Dose 矩陣 (1.0mm per pixel)
        grid_size = 300
        dose_matrix = np.zeros((grid_size, grid_size))
        
        # 4. 設定模擬步長
        report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
        dt = 1.0 / report_fps
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0

        # 5. 執行模擬
        while True:
            snapshot = engine.update(dt)
            sim_clock += dt
            
            # 更新 UI 進度
            if progress_widgets:
                try:
                    p_bar = progress_widgets['bar']
                    p_label = progress_widgets['label']
                    p_bar['value'] = min(sim_clock, total_duration)
                    p_label.config(text=f"Dose Simulation: {sim_clock:.1f}s / {total_duration:.1f}s")
                    progress_widgets['window'].update_idletasks()
                except: pass

            # 流量與補償係數計算
            current_proc = recipe['processes'][snapshot['process_idx']]
            q_actual = current_proc.get('flow_rate', self.q_ref)
            flow_ratio = q_actual / self.q_ref
            c_q = math.sqrt(flow_ratio) # 衝擊補償
            g_q = 1.0 / math.sqrt(flow_ratio) if flow_ratio > 0 else 1.0 # 再附著修正
            gamma_eff = self.gamma_base * g_q

            # 獲取當前旋轉參數
            current_rpm = snapshot['rpm']
            omega = (current_rpm / 60.0) * 2 * math.pi
            
            # 獲取座標轉換矩陣 (晶圓座標系)
            rad_wafer = math.radians(snapshot['wafer_angle'])
            cos_t, sin_t = np.cos(-rad_wafer), np.sin(-rad_wafer)
            rot_back = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

            # 遍歷粒子進行 Dose 計算與空間擴散
            for particles in engine.particle_systems.values():
                for p in particles:
                    if p['state'] == 'on_wafer':
                        # 1. 座標轉換與半徑計算
                        center = np.dot(rot_back, p['pos'][:2])
                        r_val = np.linalg.norm(center)
                        
                        # 2. 瞬時強度計算 (Instantaneous Intensity, k)
                        shear_part = self.alpha * (abs(omega) ** 1.5) * r_val
                        impact_part = self.beta * c_q
                        k_raw = shear_part + impact_part
                        
                        # 3. 有效劑量因子 (Redeposition decay)
                        eta = math.exp(-gamma_eff * r_val)
                        
                        # 4. 該步最終權重
                        dose_contribution = k_raw * eta * dt
                        
                        # 5. 空間擴散累加
                        self._apply_dose_contribution(
                            dose_matrix, center[0], center[1], 
                            dose_contribution, self.impact_radius, grid_size
                        )

            if snapshot.get('is_finished') or sim_clock > (total_duration + 10.0):
                break

        # 6. 輸出結果
        self._export_results(dose_matrix, filepath)
        return True

    def _apply_dose_contribution(self, matrix, x, y, contribution, radius, grid_size):
        """
        將清洗 Dose 貢獻擴散到指定半徑內的像素
        使用距離線性加權：(radius - dist) / radius
        """
        # 座標轉換：從 (-150, 150) 轉為 (0, 300)
        idx_x = x + 150
        idx_y = y + 150
        
        # 取得影響範圍
        r_pixel = int(math.ceil(radius))
        min_i = max(0, int(math.floor(idx_x - r_pixel)))
        max_i = min(grid_size - 1, int(math.ceil(idx_x + r_pixel)))
        min_j = max(0, int(math.floor(idx_y - r_pixel)))
        max_j = min(grid_size - 1, int(math.ceil(idx_y + r_pixel)))

        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                dist_sq = (i - idx_x)**2 + (j - idx_y)**2
                if dist_sq <= radius**2:
                    dist = math.sqrt(dist_sq)
                    spatial_weight = (radius - dist) / radius
                    matrix[i, j] += contribution * spatial_weight

    def _export_results(self, matrix, filepath):
        base_path, _ = os.path.splitext(filepath)
        png_path = filepath
        real_base = base_path.replace("_Cleaning_Dose", "")
        csv_path = f"{real_base}_Cleaning_Dose_RawData.csv"
        radial_png_path = f"{real_base}_Cleaning_Dose_Radial_Distribution.png"

        data = matrix.T

        # 1. 繪製並儲存 PNG
        plt.figure(figsize=(11, 9), dpi=120)
        im = plt.imshow(
            data,
            origin='lower',
            extent=[-150, 150, -150, 150],
            cmap='viridis',
            interpolation='bilinear'
        )
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Accumulated Effective Cleaning Dose (A.U.)')

        wafer_circle = plt.Circle((0, 0), 150, color='red', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_artist(wafer_circle)

        plt.title("Wafer Cleaning Dose Distribution (Redeposition Model)", fontsize=14, pad=15)
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")

        if data.size > 0:
            h_max = np.max(data)
            h_mean = np.mean(data[data > 0]) if np.any(data > 0) else 0.0
            h_std = np.std(data)
        else:
            h_max = h_mean = h_std = 0.0

        stats_text = (
            f"Max Dose:     {h_max:.4f}\n"
            f"Mean Dose(>0):{h_mean:.4f}\n"
            f"Std Dev:      {h_std:.4f}\n"
            f"Alpha:        {self.alpha:.6f}\n"
            f"Beta:         {self.beta:.4f}\n"
            f"Gamma Base:   {self.gamma_base:.4f}\n"
            f"Impact Radius:{self.impact_radius}mm"
        )
        plt.text(-145, -145, stats_text, color='white', fontsize=10,
                family='monospace', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 2. 儲存 CSV
        try:
            header_str = (f"Cleaning Dose Data (Redeposition Model), Q_ref: {self.q_ref}mL/min, "
                         f"Gamma_base: {self.gamma_base}, Impact Radius: {self.impact_radius}mm")
            np.savetxt(csv_path, data, delimiter=",", fmt='%.6f', header=header_str)
        except Exception as e:
            print(f"Failed to write CSV: {e}")

        # 3. 輸出徑向分佈圖 (Radial Distribution)
        self._export_radial_distribution(matrix, radial_png_path)

    def _export_radial_distribution(self, matrix, filepath):
        grid_size = matrix.shape[0]
        center = grid_size / 2.0
        y, x = np.indices(matrix.shape)
        r = np.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
        r_rounded = r.astype(int)
        max_r = 150
        radial_sum = np.zeros(max_r + 1)
        radial_count = np.zeros(max_r + 1)
        mask = r_rounded <= max_r
        np.add.at(radial_sum, r_rounded[mask], matrix[mask])
        np.add.at(radial_count, r_rounded[mask], 1)
        radial_avg = np.divide(radial_sum, radial_count, out=np.zeros_like(radial_sum), where=radial_count > 0)
        
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(np.arange(len(radial_avg)), radial_avg, color='blue', linewidth=2, label='Average Dose')
        plt.fill_between(np.arange(len(radial_avg)), radial_avg, alpha=0.2, color='blue')
        plt.title("Radial Cleaning Dose Distribution (Redeposition Model)", fontsize=14, pad=15)
        plt.xlabel("Radius (mm)", fontsize=12)
        plt.ylabel("Average Cleaning Dose (A.U.)", fontsize=12)
        plt.xlim(0, max_r)
        plt.xticks(np.arange(0, max_r + 1, 10))
        plt.ylim(0, np.max(radial_avg) * 1.1 if np.max(radial_avg) > 0 else 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
