import numpy as np
import math
import os
import matplotlib.pyplot as plt
from numba import njit # [新增]

from models import DispenseArm
from simulation_engine import SimulationEngine
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    PRE_ALPHA, PRE_BETA, PRE_GRID_SIZE,
    PRE_Q_REF, PRE_GAMMA_BASE
)

# --- [新增] Numba 加速核心函數 ---
@njit(fastmath=True, cache=True)
def _numba_apply_pre_kernel(matrix, center_x, center_y, contribution, radius, grid_size):
    """
    Numba 加速版的 PRE Dose 累加器。
    """
    idx_x = center_x + 150.0
    idx_y = center_y + 150.0
    
    r_pixel = int(math.ceil(radius))
    
    min_i = max(0, int(math.floor(idx_x - r_pixel)))
    max_i = min(grid_size - 1, int(math.ceil(idx_x + r_pixel)))
    min_j = max(0, int(math.floor(idx_y - r_pixel)))
    max_j = min(grid_size - 1, int(math.ceil(idx_y + r_pixel)))

    radius_sq = radius * radius
    
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            dist_sq = (i - idx_x)**2 + (j - idx_y)**2
            if dist_sq <= radius_sq:
                dist = math.sqrt(dist_sq)
                # 線性空間權重
                spatial_weight = (radius - dist) / radius
                
                # 累加
                matrix[i, j] += contribution * spatial_weight

class PREGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def generate(self, recipe, filepath, config=None, progress_widgets=None):
        """
        Cleaning Dose 模擬邏輯 (Numba 加速版)
        """
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()

        pre_alpha = config.get('PRE_ALPHA', PRE_ALPHA)
        pre_beta = config.get('PRE_BETA', PRE_BETA)
        pre_grid_radius = config.get('PRE_GRID_SIZE', PRE_GRID_SIZE)
        pre_q_ref = config.get('PRE_Q_REF', PRE_Q_REF)
        pre_gamma_base = config.get('PRE_GAMMA_BASE', PRE_GAMMA_BASE)

        headless_arms = {i: DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], None, None) 
                         for i, geo in ARM_GEOMETRIES.items()}

        water_params = self.app._get_water_params()
        water_params_dict = {i: {
            'viscosity': water_params['viscosity'],
            'surface_tension': water_params['surface_tension'],
            'evaporation_rate': water_params['evaporation_rate']
        } for i in [1, 2, 3]}

        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True, config=config)
        
        grid_size = 300
        dose_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
        dt = 1.0 / report_fps
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0

        while True:
            snapshot = engine.update(dt) 
            sim_clock += dt
            
            if progress_widgets:
                try:
                    p_bar = progress_widgets['bar']
                    p_label = progress_widgets['label']
                    p_bar['value'] = min(sim_clock, total_duration)
                    p_label.config(text=f"Dose Simulation (Accelerated): {sim_clock:.1f}s / {total_duration:.1f}s")
                    progress_widgets['window'].update_idletasks()
                except: pass

            current_proc = recipe['processes'][snapshot['process_idx']]
            q_actual = current_proc.get('flow_rate', pre_q_ref)
            flow_ratio = q_actual / pre_q_ref
            c_q = math.sqrt(flow_ratio) 
            g_q = 1.0 / math.sqrt(flow_ratio) if flow_ratio > 0 else 1.0 
            gamma_eff = pre_gamma_base * g_q

            current_rpm = snapshot['rpm']
            omega = (current_rpm / 60.0) * 2 * math.pi
            
            rad_wafer = math.radians(snapshot['wafer_angle'])
            cos_t, sin_t = np.cos(-rad_wafer), np.sin(-rad_wafer)

            # 優化：直接從引擎的 NumPy 陣列提取
            on_wafer_mask = engine.particles_state == 2 # P_ON_WAFER
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                for i in indices:
                    # 1. 座標轉換
                    px, py = engine.particles_pos[i, 0], engine.particles_pos[i, 1]
                    center_x = px * cos_t - py * sin_t
                    center_y = px * sin_t + py * cos_t
                    
                    r_val = math.sqrt(center_x**2 + center_y**2)
                    
                    # 2. 瞬時強度計算
                    shear_part = pre_alpha * (abs(omega) ** 1.5) * r_val
                    impact_part = pre_beta * c_q
                    k_raw = shear_part + impact_part
                    
                    # 3. 有效劑量因子
                    eta = math.exp(-gamma_eff * r_val)
                    dose_contribution = k_raw * eta * dt
                    
                    # 4. [修改] 呼叫 Numba 核心
                    _numba_apply_pre_kernel(
                        dose_matrix, 
                        center_x, center_y, 
                        dose_contribution, 
                        pre_grid_radius, 
                        grid_size
                    )

            if snapshot.get('is_finished') or sim_clock > (total_duration + 10.0):
                break

        self._export_results(dose_matrix, filepath, config=config)
        return True
    
    def _export_results(self, matrix, filepath, config=None):
        base_path, _ = os.path.splitext(filepath)
        png_path = filepath
        real_base = base_path.replace("_Cleaning_Dose", "")
        csv_path = f"{real_base}_Cleaning_Dose_RawData.csv"
        radial_png_path = f"{real_base}_Cleaning_Dose_Radial_Distribution.png"
        
        data = matrix.T

        # 提取參數用於顯示
        alpha_val = config.get('PRE_ALPHA', PRE_ALPHA) if config else PRE_ALPHA
        beta_val = config.get('PRE_BETA', PRE_BETA) if config else PRE_BETA
        gamma_base_val = config.get('PRE_GAMMA_BASE', PRE_GAMMA_BASE) if config else PRE_GAMMA_BASE
        impact_rad_val = config.get('PRE_GRID_SIZE', PRE_GRID_SIZE) if config else PRE_GRID_SIZE
        q_ref_val = config.get('PRE_Q_REF', PRE_Q_REF) if config else PRE_Q_REF

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
            f"Alpha:        {alpha_val:.6f}\n"
            f"Beta:         {beta_val:.4f}\n"
            f"Gamma Base:   {gamma_base_val:.4f}\n"
            f"Impact Radius:{impact_rad_val}mm"
        )
        plt.text(-145, -145, stats_text, color='white', fontsize=10,
                family='monospace', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 2. 儲存 CSV
        try:
            header_str = (f"Cleaning Dose Data (Redeposition Model), Q_ref: {q_ref_val}mL/min, "
                         f"Gamma_base: {gamma_base_val}, Impact Radius: {impact_rad_val}mm")
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
