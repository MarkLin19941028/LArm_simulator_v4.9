import numpy as np
import math
import os
import matplotlib.pyplot as plt
from numba import njit, prange

from simulation_engine import SimulationEngine
from models import DispenseArm
from constants import (
    ARM_GEOMETRIES, WAFER_RADIUS, REPORT_FPS, 
    GRID_SIZE, CHARGING_BASE_SPIN_DECAY,
    VACUUM_PERMITTIVITY, WATER_RELATIVE_PERMITTIVITY, DEFAULT_CONDUCTIVITY
)

# ==========================================
# Numba Kernel 1: 電荷生成 (Source Term)
# ==========================================
@njit(fastmath=True, cache=True)
def _numba_deposit_charge(charge_matrix, film_matrix, 
                          pos_x, pos_y, vel_x, vel_y, 
                          radius, grid_size, dt, 
                          charging_efficiency):
    """
    物理機制: 流動帶電 (Streaming Current)
    邏輯: 
    1. 當液體粒子接觸晶圓，根據其「相對速度」產生電荷分離。
    2. 速度越快 (摩擦越大) -> 產生電荷越多。
    """
    # 座標轉換 (-150mm~150mm -> 0~300 pixel)
    center_offset = 150.0
    idx_x = pos_x + center_offset
    idx_y = pos_y + center_offset
    
    r_pixel = int(math.ceil(radius))
    min_i = max(0, int(math.floor(idx_x - r_pixel)))
    max_i = min(grid_size - 1, int(math.ceil(idx_x + r_pixel)))
    min_j = max(0, int(math.floor(idx_y - r_pixel)))
    max_j = min(grid_size - 1, int(math.ceil(idx_y + r_pixel)))
    
    radius_sq = radius * radius
    
    # [關鍵物理計算 1]: 流動電流生成率
    # I_gen ∝ v (速度) * Area (接觸面積) * Efficiency (材料/Zeta電位係數)
    speed = math.sqrt(vel_x**2 + vel_y**2)
    
    # 這裡假設單顆粒子帶來的電荷量
    q_gen = charging_efficiency * speed * dt

    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            dist_sq = (i - idx_x)**2 + (j - idx_y)**2
            if dist_sq <= radius_sq:
                # 簡單的高斯分佈或是均勻分佈權重
                # 只有當該處有液膜存在時，電荷才能附著
                # (這裡同時維護一個簡易的 film_matrix 以計算電容)
                if film_matrix[i, j] > 0:
                    charge_matrix[i, j] += q_gen

# ==========================================
# Numba Kernel 2: 電荷演化 (Relaxation & Transport)
# ==========================================
@njit(fastmath=True, parallel=True, cache=True)
def _numba_evolve_charge(charge_matrix, film_matrix, dt, 
                         conductivity, relative_permittivity, 
                         wafer_radius, spin_decay_rate):
    """
    物理機制: 介電鬆弛 (Dielectric Relaxation) 與 物理傳輸
    邏輯:
    1. 電荷會穿過液膜流向晶圓基板 (接地)。
    2. 電荷會隨著液體被甩出晶圓邊緣。
    """
    rows, cols = charge_matrix.shape
    epsilon = relative_permittivity * VACUUM_PERMITTIVITY
    
    # [關鍵物理計算 2]: 介電鬆弛時間 (Relaxation Time)
    # tau = epsilon / sigma
    # 導電率(sigma)越低，tau 越大，電荷消散越慢 (累積越多)
    sigma = max(conductivity, 1e-12) # 避免除以零
    tau = epsilon / sigma
    
    # 衰減因子 (Exponential Decay)
    relax_factor = math.exp(-dt / tau)
    
    center_offset = 150.0

    for i in prange(rows):
        for j in range(cols):
            q = charge_matrix[i, j]
            h = film_matrix[i, j]
            
            if q != 0:
                # 1. 介電鬆弛 (電荷流向 Substrate)
                # 只有當有液膜連接到地面時才會發生 (簡化模型)
                if h > 1e-6:
                    q *= relax_factor
                
                # 2. 物理甩乾 (Spin-off)
                # 電荷附著在液體上，液體被甩走，電荷也跟著走
                dx = i - center_offset
                dy = j - center_offset
                r = math.sqrt(dx*dx + dy*dy)
                
                # 簡單模擬液膜變薄帶走電荷
                if r <= wafer_radius:
                    # 邊緣甩得快
                    local_decay = spin_decay_rate * (1.0 + r/wafer_radius)
                    q *= (1.0 - local_decay * dt)
                else:
                    q = 0.0 # 離開晶圓
                
                charge_matrix[i, j] = q
            
            # 同步更新簡易膜厚 (為了計算電位用)
            if h > 0:
                dx = i - center_offset
                dy = j - center_offset
                r = math.sqrt(dx*dx + dy*dy)
                if r <= wafer_radius:
                     local_decay = spin_decay_rate * (1.0 + r/wafer_radius)
                     film_matrix[i, j] *= (1.0 - local_decay * dt)
                else:
                     film_matrix[i, j] = 0.0

class ChargingGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def generate(self, recipe, filepath, config=None, progress_widgets=None, play_speed_multiplier=1.0):
        """
        執行電荷累積模擬
        """
        # 1. 讀取設定
        if config is None:
            from simulation_config_def import get_default_config
            config = get_default_config()
        
        # 關鍵參數讀取
        cond = config.get('FLUID_CONDUCTIVITY', DEFAULT_CONDUCTIVITY)
        perm = config.get('FLUID_RELATIVE_PERMITTIVITY', WATER_RELATIVE_PERMITTIVITY)
        eff_factor = config.get('CHARGING_EFFICIENCY', 1e-10) # 經驗係數
        base_spin_decay = config.get('CHARGING_BASE_SPIN_DECAY', CHARGING_BASE_SPIN_DECAY)
        
        # 2. 初始化模擬引擎
        # 為了獨立運作，我們需要自己的 SimulationEngine 來跑粒子軌跡
        headless_arms = {i: DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], None, None) 
                         for i, geo in ARM_GEOMETRIES.items()}
        water_params = self.app._get_water_params() # 沿用主程式的水參數
        
        # 為了相容性，簡單包裝
        wp_dict = {1: water_params, 2: water_params, 3: water_params}
        
        engine = SimulationEngine(recipe, headless_arms, wp_dict, headless=True, config=config)
        
        # 3. 初始化網格
        grid_size = 300
        # charge_matrix: 儲存累積電荷量 Q (Coulombs)
        charge_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
        # film_matrix: 儲存液膜厚度 (mm)，用於計算電容與判定導通
        film_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)

        # 影片同步設定
        VIDEO_FPS = 30.0
        record_interval = (1.0 / VIDEO_FPS) * play_speed_multiplier
        next_record_time = 0.0
        video_buffer = []

        report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
        dt = 1.0 / report_fps
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0
        
        import time
        last_ui_update_time = time.time()
        print(f"Starting Charging Simulation (Cond={cond:.2e} S/m)...")

        # 4. 主迴圈
        while True:
            # 更新粒子物理
            snapshot = engine.update(dt)
            sim_clock += dt
            
            # 影片快照
            if sim_clock >= next_record_time:
                video_buffer.append({
                    'charge': charge_matrix.copy(),
                    'film': film_matrix.copy(),
                    'time': sim_clock
                })
                next_record_time += record_interval

            # --- A. 簡易液膜生成 (為了支撐電荷計算) ---
            on_wafer_mask = engine.particles_state == 2 # P_ON_WAFER
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                for idx in indices:
                    pos = engine.particles_pos[idx]
                    self._simple_deposit_film(film_matrix, pos[0], pos[1], 2.0, 0.005) # 2mm半徑, 0.005厚度增量
            
            # --- B. 電荷生成 (Source) ---
            if np.any(on_wafer_mask):
                indices = np.where(on_wafer_mask)[0]
                for idx in indices:
                    pos = engine.particles_pos[idx]
                    vel = engine.particles_vel[idx]
                    _numba_deposit_charge(
                        charge_matrix, film_matrix,
                        pos[0], pos[1], vel[0], vel[1],
                        2.0, grid_size, dt,
                        eff_factor
                    )

            # --- C. 電荷演化 (Sink) ---
            rpm = snapshot.get('rpm', 0)
            # 使用從 config 讀取的 CHARGING_BASE_SPIN_DECAY
            current_spin_decay = base_spin_decay * (1.0 + abs(rpm)/500.0)
            
            _numba_evolve_charge(
                charge_matrix, film_matrix, dt,
                cond, perm, WAFER_RADIUS, current_spin_decay
            )

            # 更新進度條
            if progress_widgets:
                if time.time() - last_ui_update_time >= 0.5:
                    try:
                        p_bar = progress_widgets['bar']
                        p_label = progress_widgets['label']
                        p_bar['value'] = min(sim_clock, total_duration)
                        percent = (min(sim_clock, total_duration) / total_duration) * 100
                        p_label.config(text=f"Charging: {sim_clock:.1f}s / {total_duration:.1f}s ({percent:.0f}%)")
                        progress_widgets['window'].update_idletasks()
                        last_ui_update_time = time.time()
                    except: pass

            if snapshot.get('is_finished') or sim_clock > total_duration + 2.0:
                break
        
        # 5. 結果輸出
        self._export_results(charge_matrix, film_matrix, filepath, perm, config, video_buffer, VIDEO_FPS)
        return True

    @staticmethod
    @njit(fastmath=True)
    def _simple_deposit_film(matrix, x, y, r, val):
        cx, cy = x + 150.0, y + 150.0
        ri = int(r)
        for i in range(int(cx-ri), int(cx+ri+1)):
            for j in range(int(cy-ri), int(cy+ri+1)):
                if 0 <= i < 300 and 0 <= j < 300:
                    if (i-cx)**2 + (j-cy)**2 <= r*r:
                        matrix[i, j] += val

    def _export_results(self, charge_Q, film_H, filepath, rel_perm, config, video_buffer, fps):
        base_path, _ = os.path.splitext(filepath)
        # 檔名處理
        real_base = filepath.replace("_Charging.png", "")
        radial_png_path = f"{real_base}_Charging_Radial_Distribution.png"
        video_path = f"{real_base}_Charging_Simulation.mp4"

        # 計算電位矩陣
        potential_map = self._calculate_potential(charge_Q, film_H, rel_perm)

        # 1. 輸出 Heatmap PNG
        self._export_potential_map(potential_map, filepath, config)

        # 2. 輸出 Radial Distribution
        self._export_radial_distribution(potential_map, radial_png_path)

        # 3. 輸出影片
        self._export_charging_video(video_buffer, video_path, rel_perm, fps)

    def _calculate_potential(self, charge_Q, film_H, rel_perm):
        epsilon = rel_perm * VACUUM_PERMITTIVITY
        area = 1e-6 # 1mm^2
        potential_map = np.zeros_like(charge_Q)
        mask = film_H > 1e-6
        if np.any(mask):
            potential_map[mask] = (charge_Q[mask] * (film_H[mask] * 1e-3)) / (epsilon * area)
        return potential_map

    def _export_potential_map(self, potential_map, filepath, current_config):
        v_max = np.max(potential_map)
        v_min = np.min(potential_map)
        abs_max = max(abs(v_max), abs(v_min))
        if abs_max == 0: abs_max = 1.0
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(potential_map.T, 
                        origin='lower', 
                        cmap='seismic_r', 
                        extent=[-150, 150, -150, 150],
                        vmin=-abs_max, vmax=abs_max)
        
        cbar = plt.colorbar(im)
        cbar.set_label('Surface Potential (Volts)')
        plt.title('Simulated Wafer Surface Potential')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')

        # 參數資訊
        params_lines = []
        from simulation_config_def import PARAMETER_DEFINITIONS
        for category, params in PARAMETER_DEFINITIONS.items():
            for key, info in params.items():
                label = info[0]
                val = current_config.get(key, info[1])
                params_lines.append(f"{label}: {val}")
        params_text = "\n".join(params_lines)
        stats_text = (
            f"Max: {v_max:.4f}V\n"
            f"Min: {v_min:.4f}V\n"
            f"Range: {abs(v_max-v_min):.4f}V\n"
            f"------------------\n"
            f"{params_text}"
        )
        plt.text(-145, -145, stats_text, color='white', fontsize=7,
                family='monospace', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()

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
        plt.plot(np.arange(len(radial_avg)), radial_avg, color='red', linewidth=2, label='Avg Potential')
        plt.fill_between(np.arange(len(radial_avg)), radial_avg, alpha=0.2, color='red')
        plt.title("Radial Surface Potential Distribution", fontsize=14, pad=15)
        plt.xlabel("Radius (mm)", fontsize=12)
        plt.ylabel("Potential (Volts)", fontsize=12)
        plt.xlim(0, max_r)
        plt.grid(True, linestyle='--', alpha=0.7)

        v_max = np.max(radial_avg)
        v_min = np.min(radial_avg)
        stats_text = f"Max: {v_max:.4f}V\nMin: {v_min:.4f}V"
        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
                color='red', fontsize=10, family='monospace', fontweight='bold',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()

    def _export_charging_video(self, video_buffer, output_path, rel_perm, fps):
        import cv2
        if not video_buffer: return

        view_size = 400
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (view_size, view_size))

        mask = np.zeros((view_size, view_size), dtype=np.uint8)
        cv2.circle(mask, (view_size//2, view_size//2), view_size//2, 255, -1)

        # 預計算最終最大值作為歸一化基準
        final_potential = self._calculate_potential(video_buffer[-1]['charge'], video_buffer[-1]['film'], rel_perm)
        v_max_final = np.max(final_potential)
        v_min_final = np.min(final_potential)
        abs_max_global = max(abs(v_max_final), abs(v_min_final), 0.1)

        print(f"Exporting Charging Video...")
        for frame_data in video_buffer:
            p_map = self._calculate_potential(frame_data['charge'], frame_data['film'], rel_perm)
            
            # 歸一化到 0-255，且 0V 剛好在中間 (127)
            # (val - (-abs_max)) / (2 * abs_max) * 255
            norm_map = ((p_map.T + abs_max_global) / (2 * abs_max_global) * 255)
            norm_map = np.clip(norm_map, 0, 255).astype(np.uint8)
            
            # 使用 seismic_r 對應的 OpenCV 色階 (此處手動模擬或使用 COLORMAP_JET)
            color_view = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            color_view = cv2.resize(color_view, (view_size, view_size), interpolation=cv2.INTER_LINEAR)
            color_view = cv2.bitwise_and(color_view, color_view, mask=mask)
            
            # 加上時間文字
            cv2.putText(color_view, f"Time: {frame_data['time']:.1f}s", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(color_view)

        out.release()
