import numpy as np
import math
import os
import matplotlib.pyplot as plt
from models import DispenseArm
from simulation_engine import SimulationEngine
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    STATE_RUNNING_PROCESS, STATE_MOVING_FROM_CENTER_TO_START,
    GRID_SIZE, CHAMBER_SIZE, ETCHING_TAU,
    ETCHING_IMPINGEMENT_TIME, ETCHING_IMPINGEMENT_BONUS,
    ETCHING_GEO_SMOOTHING, ETCHING_SATURATION_THRESHOLD
)

class EtchingAmountGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def generate(self, recipe, filepath, config=None, progress_widgets=None):
        """
        核心蝕刻量模擬邏輯：
        1. 採用老化模型加權：contribution = exp(-(t - birth_time) / tau)
        2. 空間分佈：擴散至 NOZZLE_RADIUS_MM 範圍
        3. 輸出 PNG 與 CSV
        """
        # 合併配置
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()

        # 提取蝕刻參數
        etch_tau = config.get('ETCHING_TAU', ETCHING_TAU)
        grid_radius = config.get('GRID_SIZE', GRID_SIZE)
        imp_time = config.get('ETCHING_IMPINGEMENT_TIME', ETCHING_IMPINGEMENT_TIME)
        imp_bonus = config.get('ETCHING_IMPINGEMENT_BONUS', ETCHING_IMPINGEMENT_BONUS)
        geo_smoothing = config.get('ETCHING_GEO_SMOOTHING', ETCHING_GEO_SMOOTHING)
        sat_threshold = config.get('ETCHING_SATURATION_THRESHOLD', ETCHING_SATURATION_THRESHOLD)

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
        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True, config=config)
        
        # 3. 準備蝕刻矩陣 (1.0mm per pixel)
        grid_size = 300
        etch_matrix = np.zeros((grid_size, grid_size))
        
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
                    p_label.config(text=f"Etching Amount Simulation: {sim_clock:.1f}s / {total_duration:.1f}s")
                    progress_widgets['window'].update_idletasks()
                except: pass

            # [新增] 每一時步初始化暫存矩陣，用於像素級飽和計算
            temp_step_matrix = np.zeros((grid_size, grid_size))

            # 核心累加邏輯
            rad_wafer = math.radians(snapshot['wafer_angle'])
            cos_t, sin_t = np.cos(-rad_wafer), np.sin(-rad_wafer)
            rot_back = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

            for particles in engine.particle_systems.values():
                for p in particles:
                    if p['state'] == 'on_wafer':
                        # 1. 計算粒子在晶圓座標系下的位置
                        center = np.dot(rot_back, p['pos'][:2])
                        
                        # 2. 老化模型加權貢獻 (含噴嘴衝擊加權)
                        current_time = engine.simulation_time_elapsed
                        age = max(0, current_time - p['birth_time'])
                        tow = p.get('time_on_wafer', 0.0)
                        
                        # 基本老化模型
                        base_contribution = math.exp(-age / etch_tau) * dt
                        
                        # [修正] 噴嘴衝擊加權：使用 time_on_wafer 判定
                        if tow < imp_time:
                            base_contribution *= imp_bonus
                            
                        # 3. 空間加權貢獻 (GRID_SIZE)
                        # 將該粒子的貢獻先累加到 temp_step_matrix 中
                        self._apply_etched_contribution(
                            temp_step_matrix, center[0], center[1], 
                            base_contribution, grid_radius, grid_size,
                            geo_smoothing=geo_smoothing
                        )

            # [新增] 整體強度飽和：對整個暫存矩陣執行一次 tanh 運算 (符合物理邏輯：針對像素點的總接收量飽和)
            if sat_threshold > 0:
                temp_step_matrix = sat_threshold * np.tanh(
                    temp_step_matrix / sat_threshold
                )

            # [新增] 將飽和後的單步貢獻累加進主矩陣
            etch_matrix += temp_step_matrix

            if snapshot.get('is_finished') or sim_clock > (total_duration + 10.0):
                break

        # 6. 輸出結果
        self._export_results(etch_matrix, filepath, config=config)
        return True

    def _apply_etched_contribution(self, matrix, x, y, contribution, radius, grid_size, geo_smoothing=7.0):
        """
        將蝕刻貢獻擴散到指定半徑內的像素
        使用圓形分佈、距離加權，並引入幾何稀釋效應 (Geometric Normalization)
        """
        # 座標轉換：從 (-150, 150) 轉為 (0, 300)
        idx_x = x + 150
        idx_y = y + 150
        
        # 取得範圍
        r_pixel = int(math.ceil(radius))
        min_i = max(0, int(math.floor(idx_x - r_pixel)))
        max_i = min(grid_size - 1, int(math.ceil(idx_x + r_pixel)))
        min_j = max(0, int(math.floor(idx_y - r_pixel)))
        max_j = min(grid_size - 1, int(math.ceil(idx_y + r_pixel)))

        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                # 1. 空間權重：計算該像素中心到粒子中心的距離
                dist_sq = (i - idx_x)**2 + (j - idx_y)**2
                if dist_sq <= radius**2:
                    dist = math.sqrt(dist_sq)
                    spatial_weight = (radius - dist) / radius
                    
                    # 2. 幾何稀釋 (Geometric Normalization)：
                    # 計算該點距離晶圓中心的半徑 r_wafer，模擬藥液隨離心力鋪開後的稀釋。
                    # 座標系補正：(i, j) 從 (0, 300) 轉回 (-150, 150)
                    x_wafer = i - 150
                    y_wafer = j - 150
                    r_wafer = math.sqrt(x_wafer**2 + y_wafer**2)
                    
                    # 引用幾何平滑常數，防止 r=0 時數值爆炸
                    geo_factor = (r_wafer + geo_smoothing) / 150
                    
                    # 3. 累加原始強度 (不在此處飽和，改由外部統一處理)
                    matrix[i, j] += contribution * spatial_weight * geo_factor

    def _export_results(self, matrix, filepath, config=None):
        base_path, _ = os.path.splitext(filepath)
        png_path = filepath
        
        # 獲取真正的使用者自訂檔名 (去除剛才加上的標籤)
        real_base = base_path.replace("_Etching_Amount", "")
        csv_path = f"{real_base}_Etching_RawData.csv"
        radial_png_path = f"{real_base}_Etching_Radial_Distribution.png"

        # 轉置以符合座標系 (Matplotlib imshow origin='lower')
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
        cbar.set_label('Simulated Etching Amount (A.U.)')

        wafer_circle = plt.Circle((0, 0), 150, color='red', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_artist(wafer_circle)

        plt.title("Wafer Etching Amount Distribution (Aging Model)", fontsize=14, pad=15)
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")

        # 統計數據
        if data.size > 0:
            h_max = np.max(data)
            h_median = np.median(data[data > 0]) if np.any(data > 0) else 0.0
            h_std = np.std(data)
        else:
            h_max = h_median = h_std = 0.0

        etch_tau = config.get('ETCHING_TAU', ETCHING_TAU) if config else ETCHING_TAU
        grid_radius = config.get('GRID_SIZE', GRID_SIZE) if config else GRID_SIZE

        stats_text = (
            f"Max Amount:   {h_max:.4f}\n"
            f"Median(>0):   {h_median:.4f}\n"
            f"Std Dev:      {h_std:.4f}\n"
            f"Aging Tau:    {etch_tau}s\n"
            f"Grid Size:    {grid_radius}mm"
        )
        plt.text(-145, -145, stats_text, color='white', fontsize=10,
                family='monospace', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 2. 儲存 CSV
        try:
            np.savetxt(csv_path, data, delimiter=",", fmt='%.6f', 
                       header="Etching Amount Data (Aging Model), Resolution: 1.0mm/pixel, Range: -150 to 150 mm")
        except Exception as e:
            print(f"Failed to write CSV: {e}")

        # 3. 輸出徑向分佈圖 (Radial Distribution)
        self._export_radial_distribution(matrix, radial_png_path)

    def _export_radial_distribution(self, matrix, filepath):
        """
        計算並輸出隨半徑變化的蝕刻量分佈圖 (Radial Distribution)
        """
        grid_size = matrix.shape[0]
        center = grid_size / 2.0
        
        # 建立坐標網格
        y, x = np.indices(matrix.shape)
        r = np.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
        
        # 將半徑以 1mm 為單位分組 (0 to 150)
        r_rounded = r.astype(int)
        max_r = int(WAFER_RADIUS)
        
        radial_sum = np.zeros(max_r + 1)
        radial_count = np.zeros(max_r + 1)
        
        # 向量化累加 (僅考慮晶圓範圍內)
        mask = r_rounded <= max_r
        np.add.at(radial_sum, r_rounded[mask], matrix[mask])
        np.add.at(radial_count, r_rounded[mask], 1)
        
        # 計算平均值 (避免除以零)
        radial_avg = np.divide(radial_sum, radial_count, out=np.zeros_like(radial_sum), where=radial_count > 0)
        
        # 繪圖
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(np.arange(len(radial_avg)), radial_avg, color='blue', linewidth=2, label='Average EA')
        
        # 填色優化
        plt.fill_between(np.arange(len(radial_avg)), radial_avg, alpha=0.2, color='blue')
        
        plt.title("Radial Etching Amount Distribution", fontsize=14, pad=15)
        plt.xlabel("Radius (mm)", fontsize=12)
        plt.ylabel("Average Etching Amount (A.U.)", fontsize=12)
        plt.xlim(0, max_r)
        plt.xticks(np.arange(0, max_r + 1, 10))
        plt.ylim(0, np.max(radial_avg) * 1.1 if np.max(radial_avg) > 0 else 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
