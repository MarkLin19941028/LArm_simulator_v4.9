import numpy as np
import math
import os
import matplotlib.pyplot as plt
from numba import njit  # [新增] 引入 Numba

from models import DispenseArm
from simulation_engine import SimulationEngine
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    GRID_SIZE, ETCHING_TAU,
    ETCHING_IMPINGEMENT_TIME, ETCHING_IMPINGEMENT_BONUS,
    ETCHING_GEO_SMOOTHING, ETCHING_SATURATION_THRESHOLD
)

# --- [新增] Numba 加速核心函數 (放在 Class 外面) ---
@njit(fastmath=True, cache=True)
def _numba_apply_etch_kernel(matrix, center_x, center_y, contribution, radius, grid_size, geo_smoothing):
    """
    Numba 加速版的蝕刻累加器。
    完全對應原本 _apply_etched_contribution 的邏輯，但速度快 50-100 倍。
    """
    # 座標轉換：從 (-150, 150) 轉為 (0, 300)
    idx_x = center_x + 150.0
    idx_y = center_y + 150.0
    
    # 計算邊界 (避免超出矩陣)
    r_pixel = int(math.ceil(radius))
    
    # Numba 中使用 max/min 確保索引安全
    min_i = max(0, int(math.floor(idx_x - r_pixel)))
    max_i = min(grid_size - 1, int(math.ceil(idx_x + r_pixel)))
    min_j = max(0, int(math.floor(idx_y - r_pixel)))
    max_j = min(grid_size - 1, int(math.ceil(idx_y + r_pixel)))

    radius_sq = radius * radius
    
    # 雙層迴圈 (在 Numba 中這裡會被展開並向量化)
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            dist_sq = (i - idx_x)**2 + (j - idx_y)**2
            
            if dist_sq <= radius_sq:
                dist = math.sqrt(dist_sq)
                # 1. 空間權重
                spatial_weight = (radius - dist) / radius
                # spatial_weight = 1
                
                # 2. 幾何稀釋 (Geometric Normalization)
                # 轉回晶圓中心座標 (-150, 150) 來計算 r_wafer
                x_wafer = i - 150.0
                y_wafer = j - 150.0
                r_wafer = math.sqrt(x_wafer**2 + y_wafer**2)
                
                geo_factor = (r_wafer + geo_smoothing) / 150.0
                # geo_factor = (r_wafer * r_wafer + geo_smoothing**2) / (150.0 * 150.0)
                # geo_factor = 1
                
                # 3. 累加
                matrix[i, j] += contribution * spatial_weight * geo_factor

class EtchingAmountGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def generate(self, recipe, filepath, config=None, progress_widgets=None):
        """
        核心蝕刻量模擬邏輯 (Numba 加速版)
        """
        # 合併配置
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()

        # 提取參數
        etch_tau = config.get('ETCHING_TAU', ETCHING_TAU)
        grid_radius = config.get('GRID_SIZE', GRID_SIZE)
        imp_time = config.get('ETCHING_IMPINGEMENT_TIME', ETCHING_IMPINGEMENT_TIME)
        imp_bonus = config.get('ETCHING_IMPINGEMENT_BONUS', ETCHING_IMPINGEMENT_BONUS)
        geo_smoothing = config.get('ETCHING_GEO_SMOOTHING', ETCHING_GEO_SMOOTHING)
        sat_threshold = config.get('ETCHING_SATURATION_THRESHOLD', ETCHING_SATURATION_THRESHOLD)

        # 1. 初始化 Headless Arms
        headless_arms = {i: DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], None, None) 
                         for i, geo in ARM_GEOMETRIES.items()}

        water_params = self.app._get_water_params()
        water_params_dict = {i: {
            'viscosity': water_params['viscosity'],
            'surface_tension': water_params['surface_tension'],
            'evaporation_rate': water_params['evaporation_rate']
        } for i in [1, 2, 3]}

        # 2. 實例化引擎
        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True, config=config)
        
        # 3. 準備蝕刻矩陣
        grid_size = 300
        etch_matrix = np.zeros((grid_size, grid_size), dtype=np.float64) # 明確指定型態
        
        report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
        dt = 1.0 / report_fps
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0

        # 5. 執行模擬
        while True:
            snapshot = engine.update(dt) 
            sim_clock += dt
            
            if progress_widgets:
                try:
                    p_bar = progress_widgets['bar']
                    p_label = progress_widgets['label']
                    p_bar['value'] = min(sim_clock, total_duration)
                    p_label.config(text=f"Etching Amount (Accelerated): {sim_clock:.1f}s / {total_duration:.1f}s")
                    progress_widgets['window'].update_idletasks()
                except: pass

            # 初始化單步暫存矩陣
            temp_step_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)

            # 暫時測試用參數
            vel_coeff = 1
            imp_decay_tau = 0.05

            # 優化：直接從引擎的 NumPy 陣列提取 (現在引擎直接提供旋轉座標系下的座標)
            on_wafer_mask = engine.particles_state == 2 # P_ON_WAFER
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                current_time = engine.simulation_time_elapsed
                
                for i in indices:
                    # 1. 取得相對座標
                    rel_x, rel_y = engine.particles_pos[i, 0], engine.particles_pos[i, 1]
                    
                    # 2. 計算流體與晶圓的「相對滑動速度」 (重要！)
                    # 粒子相對於晶圓的滑動速度向量 (vx, vy)
                    vx, vy = engine.particles_vel[i, 0], engine.particles_vel[i, 1]
                    rel_speed = math.sqrt(vx**2 + vy**2)
                    
                    # [建議1實作] 速度依賴加成：模擬邊界層削薄 (Boundary Layer Thinning)
                    # 使用平方根關係式，模擬流體力學常見的 Re^0.5 關係
                    # velocity_factor = 1.0 + vel_coeff * math.sqrt(rel_speed)
                    velocity_factor = 1.0 + vel_coeff * rel_speed
                    
                    # 3. [建議2實作] 平滑衝擊加權 (Exponential Decay Impingement)
                    tow = engine.particles_time_on_wafer[i]
                    # 使用連續指數衰減取代原本的 if/else 硬切斷
                    # imp_bonus 是倍率加成，例如 3 代表增加 200% 的能力
                    smooth_imp_factor = 1.0 + (imp_bonus - 1.0) * math.exp(-tow / imp_decay_tau)
                    
                    # 4. 老化模型 (化學消耗)
                    age = max(0.0, current_time - engine.particles_birth_time[i])
                    chemical_potential = math.exp(-age / etch_tau)
                    
                    # 最終整合貢獻度
                    base_contribution = chemical_potential * 1 * smooth_imp_factor * dt
                        
                    # 5. 呼叫 Numba 核心累加
                    _numba_apply_etch_kernel(
                        temp_step_matrix, 
                        rel_x, rel_y, 
                        base_contribution, 
                        grid_radius, 
                        grid_size,
                        geo_smoothing
                    )

            # 飽和度計算
            if sat_threshold > 0:
                np.tanh(temp_step_matrix / sat_threshold, out=temp_step_matrix)
                temp_step_matrix *= sat_threshold

            etch_matrix += temp_step_matrix

            if snapshot.get('is_finished') or sim_clock > (total_duration + 10.0):
                break

        self._export_results(etch_matrix, filepath, config=config)
        return True

    def _export_results(self, matrix, filepath, config=None):
        base_path, _ = os.path.splitext(filepath)
        png_path = filepath
        real_base = base_path.replace("_Etching_Amount", "")
        csv_path = f"{real_base}_Etching_RawData.csv"
        radial_png_path = f"{real_base}_Etching_Radial_Distribution.png"
        
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
        if data.size > 0 and np.any(data > 0):
            valid_data = data[data > 0]
            h_max = np.max(data)
            h_min = np.min(valid_data)
            h_mean = np.mean(valid_data)
            h_uni = (h_max - h_min) / (2 * h_mean) * 100 if h_mean > 0 else 0.0
        else:
            h_max = h_min = h_mean = h_uni = 0.0

        # 獲取 Physics & System 參數
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()
            
        # 整理所有參數資訊
        params_lines = []
        from simulation_config_def import PARAMETER_DEFINITIONS
        for category, params in PARAMETER_DEFINITIONS.items():
            for key, info in params.items():
                label = info[0]
                val = config.get(key, info[1])
                params_lines.append(f"{label}: {val}")
        params_text = "\n".join(params_lines)

        stats_text = (
            f"Max: {h_max:.4f}\n"
            f"Min(>0): {h_min:.4f}\n"
            f"Uniformity: {h_uni:.2f}%\n"
            f"------------------\n"
            f"{params_text}"
        )
        plt.text(-145, -145, stats_text, color='white', fontsize=8,
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
        grid_size = matrix.shape[0]
        center = grid_size / 2.0
        y, x = np.indices(matrix.shape)
        r = np.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
        r_rounded = r.astype(int)
        max_r = int(WAFER_RADIUS)
        radial_sum = np.zeros(max_r + 1)
        radial_count = np.zeros(max_r + 1)
        mask = r_rounded <= max_r
        np.add.at(radial_sum, r_rounded[mask], matrix[mask])
        np.add.at(radial_count, r_rounded[mask], 1)
        radial_avg = np.divide(radial_sum, radial_count, out=np.zeros_like(radial_sum), where=radial_count > 0)
        
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(np.arange(len(radial_avg)), radial_avg, color='blue', linewidth=2, label='Average EA')
        plt.fill_between(np.arange(len(radial_avg)), radial_avg, alpha=0.2, color='blue')
        plt.title("Radial Etching Amount Distribution", fontsize=14, pad=15)
        plt.xlabel("Radius (mm)", fontsize=12)
        plt.ylabel("Average Etching Amount (A.U.)", fontsize=12)
        plt.xlim(0, max_r)
        plt.xticks(np.arange(0, max_r + 1, 10))
        plt.ylim(0, np.max(radial_avg) * 1.1 if np.max(radial_avg) > 0 else 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 統計資訊 (針對徑向分布資料)
        if radial_avg.size > 0 and np.any(radial_avg > 0):
            valid_r = radial_avg[radial_avg > 0]
            r_max = np.max(radial_avg)
            r_min = np.min(valid_r)
            r_mean = np.mean(valid_r)
            r_uni = (r_max - r_min) / (2 * r_mean) * 100 if r_mean > 0 else 0.0
            
            stats_text = (
                f"Max: {r_max:.4f}\n"
                f"Min(>0): {r_min:.4f}\n"
                f"Uniformity: {r_uni:.2f}%"
            )
            plt.text(0.02, 0.05, stats_text, transform=plt.gca().transAxes,
                    color='blue', fontsize=10, family='monospace', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
