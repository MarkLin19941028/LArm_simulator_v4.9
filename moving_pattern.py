import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import math
import cv2
import os
import time

from constants import *
from models import DispenseArm
from simulation_engine import SimulationEngine

class MovingPatternGenerator:
    def __init__(self, app_instance):
        self.app = app_instance

    def generate(self, recipe, filepath_img, filepath_vid, config=None, progress_widgets=None, play_speed_multiplier=1.0):
        try:
            self._run_headless_pattern_generation(recipe, filepath_img, filepath_vid, progress_widgets, play_speed_multiplier, config)
            return True
        except Exception as e:
            print(f"Error in MovingPatternGenerator: {e}")
            raise e

    def _run_headless_pattern_generation(self, recipe, filepath_img, filepath_vid, progress_widgets=None, play_speed_multiplier=1.0, config=None):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(7, 4.5), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-350, 350)
        ax.set_ylim(-225, 225)
        ax.set_facecolor('#111111')
        ax.add_patch(plt.Circle((0, 0), WAFER_RADIUS, facecolor='#333333', edgecolor='cyan', lw=1.5, zorder=1))

        # 準備影片輸出 (VideoWriter)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(filepath_vid, fourcc, 30.0, (w, h))

        # 準備計算覆蓋面積的遮罩 (1mm = 1 pixel)
        grid_size = int(WAFER_RADIUS * 2) + 2
        offset = grid_size // 2
        coverage_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # 建立晶圓圓形遮罩，用來計算總有效面積
        wafer_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        cv2.circle(wafer_mask, (offset, offset), int(WAFER_RADIUS), 1, -1)
        total_wafer_pixels = np.sum(wafer_mask)

        # Matplotlib 動態畫線的暫存容器
        arm_lines = {1: [], 2: [], 3: []}
        arm_colors = {1: 'lime', 2: 'magenta', 3: 'yellow'}

        # 建立 config 並指定模式
        pattern_config = config if config else self.app.get_current_config() # 獲取目前的物理參數
        max_speed = pattern_config.get('MAX_NOZZLE_SPEED_MMS', 250.0)

        headless_arms = {}
        for i in range(1, 4):
            geo = ARM_GEOMETRIES[i]
            headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo.get('p_start'), geo.get('p_end'), None, None, max_nozzle_speed_mms=max_speed)

        pattern_config['SIMULATION_MODE'] = 'pattern_only' # 強制覆蓋為純軌跡模式

        engine = SimulationEngine(recipe, headless_arms, {}, headless=True, config=pattern_config)

        arm_trajectories = {1: [], 2: [], 3: []}
        
        # 決定高解析度的取樣率 (參考 export_simulation_report 的邏輯)
        max_rpm = 0
        for proc in recipe['processes']:
            spin = proc['spin_params']
            current_max = spin['rpm'] if spin['mode'] == 'Simple' else max(spin['start_rpm'], spin['end_rpm'])
            if current_max > max_rpm: max_rpm = current_max
        sim_fps = max(800, int(max_rpm * 4))
        
        # 影片輸出為 30 FPS
        video_fps = 30.0
        sim_dt = 1.0 / sim_fps
        video_dt = (1.0 / video_fps) * play_speed_multiplier

        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        if total_duration <= 0: total_duration = 1.0

        last_ui_update_time = time.time()
        last_video_frame_time = -video_dt # 保證第一張立刻拍

        last_active_id = None
        last_was_spraying = False

        # 這裡的邏輯改回原始程式碼的標準單一噴嘴追蹤，並結合覆蓋面積遮罩計算
        last_pts = {1: None, 2: None, 3: None}

        while True:
            snapshot = engine.update(sim_dt)
            if progress_widgets:
                # FPS = 每 0.5 秒更新一次 UI
                if time.time() - last_ui_update_time >= 0.5:
                    try:
                        p_bar, p_label = progress_widgets['bar'], progress_widgets['label']
                        p_bar['value'] = min(snapshot['time'], total_duration)
                        percent = (min(snapshot['time'], total_duration) / total_duration) * 100
                        p_label.config(text=f"Processing Pattern: {snapshot['time']:.1f}s / {total_duration:.1f}s ({percent:.0f}%)")
                        progress_widgets['window'].update_idletasks()
                        last_ui_update_time = time.time()
                    except: pass

            curr_arm_id = snapshot['active_arm_id']
            curr_spraying = snapshot['is_spraying']

            # 獲取當前製程流量，只有流量 > 0 才記錄
            current_proc_idx = snapshot['process_idx']
            current_flow = recipe['processes'][current_proc_idx].get('flow_rate', 0)

            curr_pts = {1: None, 2: None, 3: None}
            if curr_arm_id != 0 and curr_spraying and current_flow > 0:
                if not last_was_spraying or curr_arm_id != last_active_id:
                    arm_trajectories[curr_arm_id].append([])
                    arm_lines[curr_arm_id].append(ax.plot([], [], color=arm_colors[curr_arm_id], linewidth=NOZZLE_RADIUS_MM * 2, solid_capstyle='round', alpha=0.6, zorder=10)[0])
                
                abs_pos = snapshot['nozzle_pos'][:2]
                rad_wafer = math.radians(snapshot['wafer_angle'])
                cos_a, sin_a = math.cos(-rad_wafer), math.sin(-rad_wafer)
                inv_rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                nozzle_pos_rotated = np.dot(inv_rot_matrix, abs_pos)
                
                arm_trajectories[curr_arm_id][-1].append(nozzle_pos_rotated)
                curr_pts[curr_arm_id] = nozzle_pos_rotated

                # 畫到 coverage mask 上
                if last_pts[curr_arm_id] is not None:
                    p1 = (int(last_pts[curr_arm_id][0]) + offset, int(last_pts[curr_arm_id][1]) + offset)
                    p2 = (int(nozzle_pos_rotated[0]) + offset, int(nozzle_pos_rotated[1]) + offset)
                    cv2.line(coverage_mask, p1, p2, 1, int(NOZZLE_RADIUS_MM * 2))

            last_was_spraying = (curr_spraying and current_flow > 0)
            last_active_id = curr_arm_id
            last_pts = curr_pts.copy()

            # 只在時間間隔到達 video_dt 時更新 UI 畫布並寫入影片 frame
            if snapshot['time'] - last_video_frame_time >= video_dt:
                for arm_id, segments in arm_trajectories.items():
                    if len(segments) > 0 and len(segments[-1]) > 0:
                        coords = np.array(segments[-1])
                        arm_lines[arm_id][-1].set_data(coords[:, 0], coords[:, 1])

                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
                out_vid.write(frame_bgr)
                last_video_frame_time += video_dt

            if snapshot.get('is_finished') or snapshot['time'] > (total_duration + 30.0): break

        # 如果最後沒有寫入剛好結束的 frame，補一張
        for arm_id, segments in arm_trajectories.items():
            if len(segments) > 0 and len(segments[-1]) > 0:
                coords = np.array(segments[-1])
                arm_lines[arm_id][-1].set_data(coords[:, 0], coords[:, 1])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        out_vid.write(frame_bgr)

        out_vid.release()

        has_any_trajectory = False
        
        # 建立 Legend 用的 Handles
        from matplotlib.lines import Line2D
        legend_elements = []

        for arm_id, segments in arm_trajectories.items():
            drawn_this_arm = False
            for segment in segments:
                if len(segment) > 0:
                    has_any_trajectory = True
                    drawn_this_arm = True
            
            if drawn_this_arm:
                label = f"Nozzle {arm_id}"
                legend_elements.append(Line2D([0], [0], color=arm_colors[arm_id], lw=4, label=label))

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', facecolor='#222222', edgecolor='gray', labelcolor='white', fontsize=9)

        # 計算覆蓋面積
        valid_coverage = cv2.bitwise_and(coverage_mask, wafer_mask)
        covered_pixels = np.sum(valid_coverage)
        if total_wafer_pixels > 0:
            coverage_percentage = (covered_pixels / total_wafer_pixels) * 100.0
        else:
            coverage_percentage = 0.0

        if has_any_trajectory:
            # 標示覆蓋率
            ax.text(0.02, 0.02, f"Coverage Area: {coverage_percentage:.2f}%", 
                    transform=ax.transAxes, color='white', fontsize=12,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

            # 取得最後一個有效的噴嘴位置作標記
            final_pos = snapshot['nozzle_pos']
            if isinstance(final_pos, list):
                for p in final_pos:
                    ax.plot(p[0], p[1], 'o', color='white', markersize=4, zorder=15)
            else:
                ax.plot(final_pos[0], final_pos[1], 'o', color='white', markersize=4, zorder=15)

        fig.savefig(filepath_img, bbox_inches='tight', dpi=100)
        plt.close(fig)