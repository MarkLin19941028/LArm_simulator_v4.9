import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib.animation as animation
import matplotlib.transforms
import time
import numpy as np
import math
import csv
import cv2
import os

from utils import (
    calculate_water_velocity,
    calculate_water_counts_by_radius,
)

from models import DispenseArm
from simulation_engine import SimulationEngine
from constants import * # 導入所有常數
from recipe_manager import RecipeManager
from video_generator import VideoGenerator
# from HeatmapVideoGenerator import HeatmapVideoGenerator

class WaterColumn:
    def __init__(self, ax, flow_rate_ml_per_min):
        self.ax = ax
        self.flow_rate = flow_rate_ml_per_min
        
        if self.ax:
            # 優化下落水滴：增加柔和感
            self.artist, = self.ax.plot([], [], 'o', color=(0.6, 0.8, 1.0, 0.4), markersize=WATER_DROP_SIZE, zorder=10)
            
            # 優化晶圓水膜：採用極高的透明度 (0.15) 讓重疊部分自然融合
            self.on_wafer_artist, = self.ax.plot([], [], '.', color=(0.6, 0.8, 1.0, 0.15), markersize=WATER_ON_WAFER_SIZE, zorder=3)
        else:
            self.artist = None
            self.on_wafer_artist = None

    def draw(self, falling_xy, on_wafer_xy):
        """僅負責將引擎算好的座標畫出來"""
        self.clear()
        
        if self.artist:
            if falling_xy:
                self.artist.set_data([p[0] for p in falling_xy], [p[1] for p in falling_xy])
                self.artist.set_visible(True)
            else:
                self.artist.set_data([], [])
        
        if self.on_wafer_artist:
            if on_wafer_xy:
                self.on_wafer_artist.set_data([p[0] for p in on_wafer_xy], [p[1] for p in on_wafer_xy])
                self.on_wafer_artist.set_visible(True)
            else:
                self.on_wafer_artist.set_data([], [])

    def clear(self):
        if self.artist: self.artist.set_data([], [])
        if self.on_wafer_artist: self.on_wafer_artist.set_data([], [])

    def reset(self):
        """清理畫面上的殘留水滴"""
        if self.artist:
            self.artist.set_data([], [])
        if self.on_wafer_artist:
            self.on_wafer_artist.set_data([], [])

class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recipe Editor")
        # --- UI Font Style and Size ---
        style = ttk.Style(self.root)

        # Windows: "Microsoft JhengHei UI", "Segoe UI"
        # macOS: "PingFang TC", "Helvetica Neue"
        default_font = ("Microsoft JhengHei UI", 10)
        
        # 為所有 ttk 元件設定預設字體
        # '.' 代表套用到所有 ttk 元件的基礎樣式
        style.configure('.', font=default_font)

        # 您也可以針對特定元件微調
        # 例如，讓 LabelFrame 的標題字體加粗
        labelframe_font = ("Microsoft JhengHei UI", 11, "bold")
        style.configure('TLabelframe.Label', font=labelframe_font)
        # --- 字體設定 END ---

        self.root.geometry("800x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.ani = None
        self.sim_window = None
        self.canvas = None
        self.processes_data = []
        self.arms = {}
        self.water_columns = {}
        self.arm_flow_rate_vars = {}
        self.recipe = []
        self.animation_state = STATE_RUNNING_PROCESS
        self.active_arm_id = 0
        self.current_process_index = 0
        self.time_offset_for_current_process = 0.0
        self.cumulative_physics_time = 0.0
        self.last_nozzle_pos = np.array([0.0, 0.0])
        self.transition_start_angle = 0.0
        self.transition_end_angle = 0.0
        self.transition_start_time = 0.0
        self.transition_start_rpm = 0.0
        self.transition_end_rpm = 0.0
        self.current_step_segment_str = ""
        self.display_time_accumulator = 0.0
        self.wafer_angle = 0.0
        self.is_paused = False
        self.play_speed_multiplier = 1.0
        self.simulation_time_elapsed = 0.0
        self.display_water_var = tk.BooleanVar(value=True)
        # --- 動態渲染緩存 ---
        self.water_fade_history = {} # 用於實現尾跡效果 (可選)
        
        # --- 初始化管理器 (必須在 create_editor_widgets 之前) ---
        self.recipe_manager = RecipeManager(self)
        self.video_generator = VideoGenerator(self)

        # --- MODIFICATION START: Remove diffusion_var ---
        self.water_setting_mode_var = tk.StringVar(value="Auto")
        self.viscosity_var = tk.StringVar(value='1.0')  # Unit: mPa·s (like water)
        self.surface_tension_var = tk.StringVar(value='72.8')  # Unit: mN/m (water at 20°C)
        self.evaporation_rate_var = tk.StringVar(value='0.0')  # Unit: arbitrary, proportion per second
        # --- MODIFICATION END ---

        self.create_editor_widgets()
        self._create_dummy_artists()

    # Water physics description
    def _validate_value_with_warning(self, string_var, min_val, max_val, entry_name):
        if not self.root.winfo_exists():
            return
        try:
            val = float(string_var.get())
            if not (min_val <= val <= max_val):
                # If the value is out of range, trigger a ValueError to enter the except block below
                raise ValueError("Value out of range")
        except (ValueError, TypeError):
            # --- MODIFICATION START: Custom warning messages ---

            title = f"Invalid Value for '{entry_name}'"
            message = ""

            range_info = f"Please enter a value between {min_val} and {max_val}."

            if entry_name == "Surface Tension":
                title = "Invalid Surface Tension Value"
                message = (
                    f"{range_info}\n\n"
                    "[Physical Principle]\n"
                    "Surface tension is the cohesive force between liquid molecules, causing the liquid to contract to the minimum possible surface area (like forming a droplet).\n\n"
                    "[Simulation Parameter Explanation]\n"
                    "In the simulation, a higher value causes the liquid to clump together and spread less easily; a lower value allows it to spread more readily.\n\n"
                    "[Reference Value]\n"
                    "The surface tension of pure water at 20°C is approximately 72.8 mN/m."
                )
            elif entry_name == "Viscosity":
                title = "Invalid Viscosity Value"
                message = (
                    f"{range_info}\n\n"
                    "[Physical Principle]\n"
                    "Viscosity is the internal friction of a fluid, representing its resistance to flow. It can be thought of as the fluid's \"thickness.\"\n\n"
                    "[Simulation Parameter Explanation]\n"
                    "In the simulation, a higher value results in slower spreading under rotation; a lower value improves fluidity and causes faster spreading.\n\n"
                    "[Reference Value]\n"
                    "The viscosity of pure water at 20°C is approximately 1.0 mPa·s."
                )
            elif entry_name == "Evaporation Rate":
                title = "Invalid Evaporation Rate Value"
                message = (
                    f"{range_info}\n\n"
                    "[Physical Principle]\n"
                    "The evaporation rate represents the speed at which a liquid turns into a gas and disappears from a surface.\n\n"
                    "[Simulation Parameter Explanation]\n"
                    "This parameter controls the visual effect of a particle disappearing. 0 means no evaporation, while 10.0 represents extremely fast evaporation.\n\n"
                    "[Reference Value]\n"
                    "In this simulation, the default value for water is 0.0, representing an idealized, non-volatile scenario."
                )
            else:
                # Provide a generic fallback message for any other parameters
                message = range_info

            messagebox.showwarning(title, message)
            string_var.set(str(min_val))  # Reset the invalid value to the minimum value
            # --- MODIFICATION END ---

    def _on_water_setting_mode_change(self, *args):
        """Shows or hides the manual water parameter entry fields based on the selected mode."""
        mode = self.water_setting_mode_var.get()
        if mode == "Manual":
            self.manual_water_settings_frame.grid(row=2, column=0, columnspan=5, sticky=tk.W, pady=5)
        else:  # Auto mode
            self.manual_water_settings_frame.grid_remove()

    def create_editor_widgets(self):
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        main_canvas = tk.Canvas(container)
        v_scrollbar = ttk.Scrollbar(container, orient="vertical", command=main_canvas.yview)
        h_scrollbar = ttk.Scrollbar(container, orient="horizontal", command=main_canvas.xview)
        main_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        main_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.scrollable_frame = ttk.Frame(main_canvas, padding="10")
        main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        content_frame = ttk.Frame(self.scrollable_frame)
        content_frame.pack(fill="both", expand=True)
        io_frame = ttk.LabelFrame(content_frame, text="Export / Import Recipe", padding="10")
        io_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        # 連結到外部recipe_manager
        ttk.Button(io_frame, text="Import Recipe", command=self.recipe_manager.import_recipe).pack(side="left", padx=5)
        ttk.Button(io_frame, text="Export Recipe", command=self.recipe_manager.export_recipe).pack(side="left", padx=5)

        report_frame = ttk.LabelFrame(content_frame, text="Reporting", padding="10")
        report_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(report_frame, text="Simulation Report", command=self.export_simulation_report).pack(side="left", padx=5)
        ttk.Button(report_frame, text="Accumulation Heatmap", command=self.export_accumulation_heatmap).pack(side="left", padx=5)
        # 連結到外部Video_generator
        ttk.Button(report_frame, text="Generate Video", command=self.export_simulation_video).pack(side="left", padx=5)

        ttk.Button(report_frame, text="Moving Pattern", command=self.export_nozzle_pattern).pack(side="left", padx=15)
        global_frame = ttk.LabelFrame(content_frame, text="Global Parameters", padding="10")
        global_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(global_frame, text="Wafer Spin Direction:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.spin_dir = tk.StringVar(value='ccw')
        ttk.Radiobutton(global_frame, text="Counter-Clockwise (ccw)", variable=self.spin_dir, value='ccw').grid(row=0, column=1, columnspan=2, sticky=tk.W)
        ttk.Radiobutton(global_frame, text="Clockwise (cw)", variable=self.spin_dir, value='cw').grid(row=0, column=3, columnspan=2, sticky=tk.W)

        # Water Setting Mode
        ttk.Label(global_frame, text="Water Setting Mode:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(global_frame, text="Auto (Default)", variable=self.water_setting_mode_var, value="Auto", command=self._on_water_setting_mode_change).grid(row=1, column=1, columnspan=2, sticky=tk.W)
        ttk.Radiobutton(global_frame, text="Manual", variable=self.water_setting_mode_var, value="Manual", command=self._on_water_setting_mode_change).grid(row=1, column=3, columnspan=2, sticky=tk.W)

        # Frame for Manual Water Settings (initially hidden)
        self.manual_water_settings_frame = ttk.Frame(global_frame)

        # Surface Tension
        ttk.Label(self.manual_water_settings_frame, text="Surface Tension (mN/m):").grid(row=0, column=0, sticky=tk.W, padx=(0, 2))
        st_entry = ttk.Entry(self.manual_water_settings_frame, textvariable=self.surface_tension_var, width=8)
        st_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        # 允許使用者設定的上下限值
        st_entry.bind('<FocusOut>', lambda event: self._validate_value_with_warning(self.surface_tension_var, 0.000001, 500.0, "Surface Tension"))

        # Viscosity
        ttk.Label(self.manual_water_settings_frame, text="Viscosity (mPa·s):").grid(row=0, column=2, sticky=tk.W, padx=(0, 2))
        viscosity_entry = ttk.Entry(self.manual_water_settings_frame, textvariable=self.viscosity_var, width=8)
        viscosity_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 10))
        # 允許使用者設定的上下限值
        viscosity_entry.bind('<FocusOut>', lambda event: self._validate_value_with_warning(self.viscosity_var, 0.000001, 500.0, "Viscosity"))

        # Solvent Evaporation Rate
        ttk.Label(self.manual_water_settings_frame, text="Evaporation Rate:").grid(row=0, column=4, sticky=tk.W, padx=(0, 2))
        er_entry = ttk.Entry(self.manual_water_settings_frame, textvariable=self.evaporation_rate_var, width=8)
        er_entry.grid(row=0, column=5, sticky=tk.W, padx=(0, 10))
        # 允許使用者設定的上下限值
        er_entry.bind('<FocusOut>', lambda event: self._validate_value_with_warning(self.evaporation_rate_var, 0.0, 10.0, "Evaporation Rate"))

        # Number of Processes
        ttk.Label(global_frame, text="Number of Processes:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.num_processes = tk.IntVar(value=1)
        ttk.OptionMenu(global_frame, self.num_processes, 1, *range(1, 51), command=self.recreate_process_widgets).grid(row=3, column=1, sticky=tk.W)

        self.processes_container = ttk.Frame(content_frame)
        self.processes_container.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        control_frame = ttk.Frame(container)
        control_frame.grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=5)
        ttk.Button(control_frame, text="Start / Update Simulation", command=self.start_or_update_simulation).pack()

        self.recreate_process_widgets()

        self._on_water_setting_mode_change() # Call this to set the initial visibility

    def _create_callback(self, func, *args):
        return lambda new_value: func(*args)

    def _on_spin_mode_change(self, process_index):
        proc_data = self.processes_data[process_index]
        container = proc_data['spin_widgets_frame']
        for widget in container.winfo_children(): widget.destroy()
        mode = proc_data['spin_mode_var'].get()
        if mode == "Simple":
            ttk.Label(container, text="Spin Speed (RPM):").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(container, textvariable=proc_data['simple_rpm_var'], width=10).grid(row=0, column=1, sticky=tk.W)
        elif mode == "Speed Ramp":
            ttk.Label(container, text="Start RPM:").grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(container, textvariable=proc_data['start_rpm_var'], width=10).grid(row=0, column=1, sticky=tk.W, padx=(0, 5))
            ttk.Label(container, text="End RPM:").grid(row=0, column=2, sticky=tk.W)
            ttk.Entry(container, textvariable=proc_data['end_rpm_var'], width=10).grid(row=0, column=3, sticky=tk.W)

    def _on_arm_change(self, process_index):
        proc_data = self.processes_data[process_index]
        arm_str = proc_data['arm_var'].get()
        new_state = 'disabled' if arm_str == 'None' else 'normal'
        proc_data['sfc_checkbox'].config(state=new_state)
        if 'flow_rate_spinbox' in proc_data:
            proc_data['flow_rate_spinbox'].config(state=new_state)
        for child in proc_data['steps_container'].winfo_children():
            child.config(state=new_state)

    def recreate_process_widgets(self, *args, imported_data=None):
        for widget in self.processes_container.winfo_children(): widget.destroy()
        self.processes_data = []
        num_processes = len(imported_data) if imported_data else self.num_processes.get()
        for i in range(num_processes):
            process_labelframe = ttk.LabelFrame(self.processes_container, text=f"Process Recipe {i+1}", padding="10")
            process_labelframe.grid(row=0, column=i, padx=10, pady=5, sticky="ns")
            proc_params_frame = ttk.Frame(process_labelframe)
            proc_params_frame.grid(row=0, column=0, sticky=tk.W, columnspan=2)
            ttk.Label(proc_params_frame, text="Dispense Arm:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
            arm_var = tk.StringVar(value="Arm 1")
            arm_menu = ttk.OptionMenu(proc_params_frame, arm_var, "Arm 1", "None", "Arm 1", "Arm 2", "Arm 3", command=self._create_callback(self._on_arm_change, i))
            arm_menu.grid(row=0, column=1, sticky=tk.W)
            
            # Flow Rate for this process
            ttk.Label(proc_params_frame, text="Flow (mL/min):").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
            flow_rate_var = tk.StringVar(value='500')
            flow_rate_spinbox = ttk.Spinbox(
                proc_params_frame, from_=10, to=3000, increment=10,
                textvariable=flow_rate_var, width=8
            )
            flow_rate_spinbox.grid(row=0, column=3, sticky=tk.W)
            flow_rate_spinbox.bind('<FocusOut>', lambda event, v=flow_rate_var, idx=i: self._validate_value_with_warning(
                v, 10.0, 3000.0, f"Process {idx+1} Flow Rate"
            ))

            ttk.Label(proc_params_frame, text="Total Process Time (s):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
            duration_var = tk.StringVar(value='10')
            ttk.Entry(proc_params_frame, textvariable=duration_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
            ttk.Label(proc_params_frame, text="Spin Mode:").grid(row=2, column=0, sticky=tk.W, pady=2)
            spin_mode_var = tk.StringVar(value="Simple")
            spin_menu = ttk.OptionMenu(proc_params_frame, spin_mode_var, "Simple", "Simple", "Speed Ramp", command=self._create_callback(self._on_spin_mode_change, i))
            spin_menu.grid(row=2, column=1, sticky=tk.W)
            spin_widgets_frame = ttk.Frame(proc_params_frame)
            spin_widgets_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=2)
            ttk.Label(proc_params_frame, text="Number of Steps:").grid(row=1, column=2, sticky=tk.W, padx=(10, 5), pady=(5, 0))
            initial_steps = imported_data[i]['steps'] if imported_data and 'steps' in imported_data[i] else 2
            num_steps_var = tk.IntVar(value=initial_steps)
            steps_menu = ttk.OptionMenu(proc_params_frame, num_steps_var, initial_steps, *range(2, 21), command=self._create_callback(self.recreate_step_entries, i))
            steps_menu.grid(row=1, column=3, sticky=tk.W)
            start_from_center_var = tk.BooleanVar(value=False)
            sfc_checkbox = ttk.Checkbutton(proc_params_frame, text="Start from center", variable=start_from_center_var)
            sfc_checkbox.grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=10)
            steps_container = ttk.LabelFrame(process_labelframe, text="Step Parameters", padding="10")
            steps_container.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
            proc_data_dict = {
                'arm_var': arm_var, 
                'flow_rate_var': flow_rate_var,
                'flow_rate_spinbox': flow_rate_spinbox,
                'duration_var': duration_var, 
                'spin_mode_var': spin_mode_var, 
                'simple_rpm_var': tk.StringVar(value='200'), 
                'start_rpm_var': tk.StringVar(value='0'), 
                'end_rpm_var': tk.StringVar(value='200'), 
                'spin_widgets_frame': spin_widgets_frame, 
                'num_steps_var': num_steps_var, 
                'start_from_center_var': start_from_center_var, 
                'sfc_checkbox': sfc_checkbox, 
                'steps_container': steps_container, 
                'step_entries': []
            }
            self.processes_data.append(proc_data_dict)
            self._on_spin_mode_change(i)
            self.recreate_step_entries(i)

    def recreate_step_entries(self, process_index):
        proc_data = self.processes_data[process_index]
        container = proc_data['steps_container']
        for widget in container.winfo_children(): widget.destroy()
        proc_data['step_entries'].clear()
        num_steps = proc_data['num_steps_var'].get()
        ttk.Label(container, text="Step").grid(row=0, column=0, padx=5)
        ttk.Label(container, text="Target Position (%)").grid(row=0, column=1, padx=5)
        ttk.Label(container, text="Target Speed (%)").grid(row=0, column=2, padx=5)
        for j in range(num_steps):
            ttk.Label(container, text=f"{j+1}").grid(row=j + 1, column=0)
            pos_var, speed_var = tk.StringVar(), tk.StringVar()
            if not hasattr(self, 'is_importing') or not self.is_importing:
                if j == 0:
                    pos_var.set('-100')
                    speed_var.set('100')
                elif j == 1:
                    pos_var.set('100')
                    speed_var.set('100')
            ttk.Entry(container, textvariable=pos_var, width=10).grid(row=j + 1, column=1, padx=2, pady=2)
            ttk.Entry(container, textvariable=speed_var, width=10).grid(row=j + 1, column=2, pady=2)
            proc_data['step_entries'].append({'pos': pos_var, 'speed': speed_var})
        self._on_arm_change(process_index)

    def _get_water_params(self):
        """Returns a dictionary of water parameters based on the selected UI mode."""
        if self.water_setting_mode_var.get() == "Auto":
            return {
                'viscosity': 1.0,
                'surface_tension': 72.8,
                'evaporation_rate': 0.0,
            }
        else: # Manual
            try:
                return {
                    'viscosity': float(self.viscosity_var.get()),
                    'surface_tension': float(self.surface_tension_var.get()),
                    'evaporation_rate': float(self.evaporation_rate_var.get()),
                }
            except (ValueError, TypeError):
                messagebox.showerror("Invalid Water Parameter", "One of the manual water parameters is not a valid number. Using defaults.")
                return {
                    'viscosity': 1.0,
                    'surface_tension': 72.8,
                    'evaporation_rate': 0.0
                }

    def export_simulation_video(self):
        # 1. 解析 Recipe
        parsed_recipe = self.parse_and_prepare_recipe()
        if not parsed_recipe:
            # 可以考慮在此處加入一個 messagebox 提示使用者解析失敗
            # messagebox.showwarning("Warning", "Failed to parse or prepare recipe.")
            return

        # 2. 選擇檔案路徑
        filepath = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Video Files", "*.mp4"), ("All Files", "*.*")],
            title="Export Simulation Video As..."
        )
        if not filepath:
            # messagebox.showinfo("Info", "Video export cancelled.")
            return

        # 3. 建立進度視窗
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Generating Video")
        progress_window.geometry("400x120")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.resizable(False, False)
        ttk.Label(progress_window, text="Generating simulation video, please wait...", padding=10).pack()

        # 計算總時長用於進度條 (這部分似乎是為機械動作準備的)
        total_duration = sum(p['total_duration'] for p in parsed_recipe['processes'])
        if total_duration == 0:
            total_duration = 1

        progress_label = ttk.Label(progress_window, text=f"Processing Time: 0.0s / {total_duration:.1f}s (0%)", padding=(0, 5))
        progress_label.pack()
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=350, mode="determinate", maximum=total_duration)
        progress_bar.pack(pady=10)
        progress_widgets = {'window': progress_window, 'bar': progress_bar, 'label': progress_label}

        try:
            # 讀取當前播放倍率 (如果視窗未開啟則預設 1.0)
            try:
                current_multiplier = float(self.speed_var.get().replace('x', ''))
            except (AttributeError, ValueError):
                current_multiplier = 1.0

            # 直接執行 Simulator 動畫影片生成
            self.video_generator._run_headless_video_generation(
                parsed_recipe, filepath, progress_widgets, play_speed_multiplier=current_multiplier
            )

            messagebox.showinfo("Success", f"Simulation video exported successfully to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during video generation: {e}")
        finally:
            # 確保進度視窗在任何情況下都被關閉
            if progress_window.winfo_exists():
                progress_window.destroy()

    def export_simulation_report(self):
        """
        導出報表邏輯優化：
        1. 實作 Atomic Lock 確保不會重複觸發。
        2. 確保在計算過程中禁用 UI 相關回調。
        """
        if getattr(self, '_report_export_lock', False):
            print("Report export already in progress. Ignoring duplicate request.")
            return
        self._report_export_lock = True
        
        try:
            parsed_recipe = self.parse_and_prepare_recipe()
            if not parsed_recipe:
                self._report_export_lock = False
                return

            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Export Simulation Report As..."
            )
            if not filepath:
                self._report_export_lock = False
                return

            # --- 自動計算建議的 REPORT_FPS ---
            max_rpm = 0
            for proc in parsed_recipe['processes']:
                spin = proc['spin_params']
                if spin['mode'] == 'Simple':
                    current_max = spin['rpm']
                else:
                    current_max = max(spin['start_rpm'], spin['end_rpm'])
                if current_max > max_rpm: max_rpm = current_max
            
            # 提高採樣率限制，確保 1000rpm 以上仍然精確
            suggested_fps = max(800, int(max_rpm * 4))
            parsed_recipe['dynamic_report_fps'] = suggested_fps

            # 3. 建立進度視窗
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Generating Report")
            progress_window.geometry("400x120")
            progress_window.transient(self.root)
            progress_window.grab_set()
            progress_window.resizable(False, False)
            ttk.Label(progress_window, text="Generating simulation report, please wait...", padding=10).pack()

            total_duration = sum(p['total_duration'] for p in parsed_recipe['processes'])
            if total_duration <= 0: total_duration = 1.0
            
            progress_label = ttk.Label(progress_window, text=f"Processing Time: 0.0s / {total_duration:.1f}s (0%)", padding=(0, 5))
            progress_label.pack()
            progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=350, mode="determinate", maximum=total_duration)
            progress_bar.pack(pady=10)
            progress_widgets = {'window': progress_window, 'bar': progress_bar, 'label': progress_label}
            
            # --- 核心模擬導出 ---
            report_data, particle_data, _ = self._run_headless_simulation(parsed_recipe, progress_widgets)
            
            # 關閉進度視窗
            if progress_window.winfo_exists():
                progress_window.destroy()
            
            # --- 檔案寫入 ---
            if report_data:
                try:
                    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                        if report_data:
                            writer = csv.DictWriter(csvfile, fieldnames=report_data[0].keys())
                            writer.writeheader()
                            writer.writerows(report_data)
                    messagebox.showinfo("Success", f"Simulation report exported successfully to:\n{filepath}")
                except Exception as e:
                    messagebox.showerror("Export Error", f"Failed to write simulation report to file: {e}")
                    
            if particle_data:
                base_path, _ = os.path.splitext(filepath)
                particle_filepath = f"{base_path}_Particle_Calculation.csv"
                
                processed_particle_data = []
                for p in particle_data:
                    if p['time_on_wafer'] > 0:
                        avg_velocity = (p['path_length'] / p['time_on_wafer']) if p['time_on_wafer'] > 0 else 0
                        processed_particle_data.append({
                            'Particle ID': p['id'],
                            'Residence Time (s)': f"{p['time_on_wafer']:.4f}",
                            'Path Length (mm)': f"{p['path_length']:.4f}",
                            'Average Velocity (mm/s)': f"{avg_velocity:.4f}"
                        })
                
                try:
                    with open(particle_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                        if processed_particle_data:
                            headers = ['Particle ID', 'Residence Time (s)', 'Path Length (mm)', 'Average Velocity (mm/s)']
                            writer = csv.DictWriter(csvfile, fieldnames=headers)
                            writer.writeheader()
                            writer.writerows(processed_particle_data)
                    messagebox.showinfo("Success", f"Particle calculation report also exported successfully to:\n{particle_filepath}")
                except Exception as e:
                    messagebox.showerror("Export Error", f"Failed to write particle report to file: {e}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed during report generation phase: {e}")
            if 'progress_window' in locals() and progress_window.winfo_exists():
                progress_window.destroy()
        finally:
            # 確保在結束前重置標記
            self._report_export_lock = False

    def export_accumulation_heatmap(self):
        """
        獨立導出熱力圖功能：
        1. 執行模擬獲得 heatmap_matrix。
        2. 輸出 PNG 圖片。
        3. 輸出原始數據 CSV 檔案。
        """
        if getattr(self, '_heatmap_export_lock', False):
            return
        self._heatmap_export_lock = True
        
        try:
            parsed_recipe = self.parse_and_prepare_recipe()
            if not parsed_recipe:
                self._heatmap_export_lock = False
                return

            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")],
                title="Export Accumulation Heatmap As..."
            )
            if not filepath:
                self._heatmap_export_lock = False
                return

            # --- 自動計算建議的 REPORT_FPS ---
            max_rpm = 0
            for proc in parsed_recipe['processes']:
                spin = proc['spin_params']
                if spin['mode'] == 'Simple':
                    current_max = spin['rpm']
                else:
                    current_max = max(spin['start_rpm'], spin['end_rpm'])
                if current_max > max_rpm: max_rpm = current_max
            
            suggested_fps = max(800, int(max_rpm * 4))
            parsed_recipe['dynamic_report_fps'] = suggested_fps

            # 建立進度視窗
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Generating Heatmap")
            progress_window.geometry("400x120")
            progress_window.transient(self.root)
            progress_window.grab_set()
            progress_window.resizable(False, False)
            ttk.Label(progress_window, text="Generating accumulation heatmap, please wait...", padding=10).pack()

            total_duration = sum(p['total_duration'] for p in parsed_recipe['processes'])
            if total_duration <= 0: total_duration = 1.0
            
            progress_label = ttk.Label(progress_window, text=f"Processing: 0.0s / {total_duration:.1f}s (0%)", padding=(0, 5))
            progress_label.pack()
            progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=350, mode="determinate", maximum=total_duration)
            progress_bar.pack(pady=10)
            progress_widgets = {'window': progress_window, 'bar': progress_bar, 'label': progress_label}
            
            # --- 核心模擬導出 ---
            _, _, heatmap_matrix = self._run_headless_simulation(parsed_recipe, progress_widgets)
            
            # 關閉進度視窗
            if progress_window.winfo_exists():
                progress_window.destroy()

            if heatmap_matrix is not None:
                base_path, _ = os.path.splitext(filepath)
                heatmap_png_path = filepath
                heatmap_csv_path = f"{base_path}_RawData.csv"
                
                # 1. 輸出 PNG
                # 數據轉置以符合 (X, Y) 座標對齊
                data = heatmap_matrix.T
        
                # 強制檢查維度
                grid_dim = int(np.sqrt(data.size))
                if data.ndim == 1:
                    data = data.reshape((grid_dim, grid_dim))
                    
                # 計算統計數據
                if data.size > 0:
                    h_max = np.max(data)
                    h_median = np.median(data[data > 0]) if np.any(data > 0) else 0.0
                    h_std = np.std(data)
                else:
                    h_max = h_median = h_std = 0.0

                # 開始繪圖
                plt.figure(figsize=(11, 9), dpi=120)
                im = plt.imshow(
                    data,
                    origin='lower',
                    extent=[-150, 150, -150, 150],
                    cmap='magma',
                    interpolation='nearest'
                )

                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                cbar.set_label('Accumulated Residence Time (Seconds)')

                wafer_circle = plt.Circle((0, 0), 150, color='white', fill=False, linestyle='--', alpha=0.5)
                plt.gca().add_artist(wafer_circle)

                plt.title("Wafer Water Accumulation Heatmap (Quantitative)", fontsize=14, pad=15)
                plt.xlabel("X Position (mm)")
                plt.ylabel("Y Position (mm)")
                
                stats_text = (
                    f"Max Time:    {h_max:.4f} s\n"
                    f"Median(>0):  {h_median:.4f} s\n"
                    f"Std Dev:     {h_std:.4f}\n"
                    f"Resolution:  1.0 mm/pixel"
                )
                plt.text(-145, -145, stats_text, color='white', fontsize=10,
                        family='monospace', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

                plt.tight_layout()
                plt.savefig(heatmap_png_path, bbox_inches='tight', dpi=300)
                plt.close()

                # 2. 輸出 CSV (原始數據)
                try:
                    np.savetxt(heatmap_csv_path, data, delimiter=",", fmt='%.6f', 
                               header="Raw Accumulation Data (Seconds), Resolution: 1.0mm/pixel, Range: -150 to 150 mm")
                except Exception as e:
                    messagebox.showerror("Export Error", f"Failed to write heatmap raw data to CSV: {e}")
                
                messagebox.showinfo("Success", f"Heatmap PNG and Raw Data CSV exported successfully:\n{heatmap_png_path}\n{heatmap_csv_path}")

        except Exception as e:
            messagebox.showerror("Heatmap Error", f"Failed during heatmap generation phase: {e}")
            if 'progress_window' in locals() and progress_window.winfo_exists():
                progress_window.destroy()
        finally:
            self._heatmap_export_lock = False

    def export_nozzle_pattern(self):
        """
        處理 UI 流程，並呼叫核心繪製邏輯。
        """
        parsed_recipe = self.parse_and_prepare_recipe()
        if not parsed_recipe:
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            title="Export Moving Pattern Image As..."
        )
        if not filepath:
            return

        # 建立進度視窗
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Generating Pattern")
        progress_window.geometry("400x120")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.resizable(False, False)
        ttk.Label(progress_window, text="Generating moving pattern image, please wait...", padding=10).pack()

        total_duration = sum(p['total_duration'] for p in parsed_recipe['processes'])
        if total_duration <= 0: total_duration = 1.0

        progress_label = ttk.Label(progress_window, text=f"Processing Time: 0.0s / {total_duration:.1f}s (0%)", padding=(0, 5))
        progress_label.pack()
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=350, mode="determinate", maximum=total_duration)
        progress_bar.pack(pady=10)
        progress_widgets = {'window': progress_window, 'bar': progress_bar, 'label': progress_label}

        try:
            self._run_headless_pattern_generation(parsed_recipe, filepath, progress_widgets)
            messagebox.showinfo("Success", f"Moving Pattern image exported successfully to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate pattern: {e}")
        finally:
            if progress_window.winfo_exists():
                progress_window.destroy()

    def _run_headless_pattern_generation(self, recipe, filepath, progress_widgets=None):
        """
        運行精簡無頭模擬，追蹤噴嘴軌跡（考慮晶圓旋轉），並繪製成圖。
        採用 SimulationEngine 以確保物理運算準確。
        """
        # 1. 初始化 Matplotlib 繪圖環境
        fig = Figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-CHAMBER_SIZE / 2, CHAMBER_SIZE / 2)
        ax.set_ylim(-CHAMBER_SIZE / 2, CHAMBER_SIZE / 2)
        ax.set_facecolor('#111111')
        ax.add_patch(plt.Circle((0, 0), WAFER_RADIUS, facecolor='#333333', edgecolor='cyan', lw=1.5, zorder=1))

        # 2. 初始化引擎
        headless_arms = {}
        for i in range(1, 4):
            geo = ARM_GEOMETRIES[i]
            headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], None, None)

        engine = SimulationEngine(recipe, headless_arms, {}, headless=True)

        # 軌跡數據列表 (按手臂與線段區分)
        arm_trajectories = {1: [], 2: [], 3: []}
        dt = 1.0 / REPORT_FPS
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        if total_duration <= 0: total_duration = 1.0

        last_active_id = None
        last_was_spraying = False

        # 3. 運行狀態機迴圈
        last_snapshot = None
        while True:
            snapshot = engine.update(dt)
            last_snapshot = snapshot

            # 更新 UI 進度
            if progress_widgets:
                try:
                    p_bar, p_label = progress_widgets['bar'], progress_widgets['label']
                    p_bar['value'] = min(snapshot['time'], total_duration)
                    percent = (min(snapshot['time'], total_duration) / total_duration) * 100
                    p_label.config(text=f"Processing Pattern: {snapshot['time']:.1f}s / {total_duration:.1f}s ({percent:.0f}%)")
                    progress_widgets['window'].update_idletasks()
                except: pass

            curr_arm_id = snapshot['active_arm_id']
            curr_spraying = snapshot['is_spraying']

            # 只有當正在噴灑時記錄軌跡 (確保不記錄非製程移動)
            if curr_arm_id != 0 and curr_spraying:
                # 檢查是否需要開始新線段 (手臂更換或噴灑中斷後重新開始)
                if not last_was_spraying or curr_arm_id != last_active_id:
                    arm_trajectories[curr_arm_id].append([])

                # 1. 取得絕對座標
                abs_pos = snapshot['nozzle_pos'][:2]

                # 2. 計算反向旋轉矩陣，將噴嘴轉化為晶圓相對座標
                rad_wafer = math.radians(snapshot['wafer_angle'])
                cos_a, sin_a = math.cos(-rad_wafer), math.sin(-rad_wafer)
                inv_rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

                # 3. 轉換並儲存至當前手臂的當前線段
                nozzle_pos_rotated = np.dot(inv_rot_matrix, abs_pos)
                arm_trajectories[curr_arm_id][-1].append(nozzle_pos_rotated)

            last_was_spraying = curr_spraying
            last_active_id = curr_arm_id

            # 結束判定
            if snapshot.get('is_finished'):
                break

            # 安全鎖
            if snapshot['time'] > (total_duration + 30.0): break

        # 4. 繪製圖形
        arm_colors = {1: 'lime', 2: 'magenta', 3: 'yellow'}
        has_any_trajectory = False
        for arm_id, segments in arm_trajectories.items():
            for segment in segments:
                if segment:
                    has_any_trajectory = True
                    coords = np.array(segment)
                    # 繪製各手臂專屬顏色的軌跡線
                    ax.plot(coords[:, 0], coords[:, 1],
                            color=arm_colors[arm_id],
                            linewidth=NOZZLE_RADIUS_MM * 2,
                            solid_capstyle='round',
                            alpha=0.6,
                            zorder=10)

        if has_any_trajectory:
            # 繪製噴嘴中心 (最後位置)
            if last_snapshot is not None:
                final_pos = last_snapshot['nozzle_pos']
                ax.plot(final_pos[0], final_pos[1], 'o', color='yellow', markersize=4, zorder=15)

        # 5. 儲存圖片
        fig.savefig(filepath, bbox_inches='tight', dpi=100)
        plt.close(fig)

    def _run_headless_simulation(self, recipe, progress_widgets=None):
        """
        純數據模擬優化版：
        1. 徹底解決 ID 偏移與數量遺失。
        2. 改採「主動紀錄」模式紀錄所有粒子生命週期。
        """
        # 1. 初始化 Headless Arms
        headless_arms = {i: DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], None, None) 
                         for i, geo in ARM_GEOMETRIES.items()}

        # 獲取物理參數
        global_water_params = self._get_water_params()
        water_params_dict = {i: {
            'viscosity': global_water_params['viscosity'],
            'surface_tension': global_water_params['surface_tension'],
            'evaporation_rate': global_water_params['evaporation_rate']
        } for i in [1, 2, 3]}

        # 3. 實例化引擎 (傳入 headless=True 防止自動循環)
        engine = SimulationEngine(recipe, headless_arms, water_params_dict, headless=True)
        engine.next_particle_id = 0 # 強制重置 ID
       
        # 4. 準備數據容器
        report_data = []
        # 使用字典追蹤所有出現過的粒子，徹底解決遺失問題
        particle_registry = {} 
        
        # 初始化熱力圖
        grid_size = 300
        heatmap_accum = np.zeros((grid_size, grid_size))
       
        # 5. 設定模擬參數
        report_fps = recipe.get('dynamic_report_fps', REPORT_FPS)
        dt = 1.0 / report_fps
        report_log_interval = 0.01 # 報表記錄間隔 (s)
        time_since_last_log = 0.0
       
        total_duration = sum(p['total_duration'] for p in recipe['processes'])
        sim_clock = 0.0

        while True:
            # A. 呼叫引擎更新
            snapshot = engine.update(dt)
            sim_clock += dt
            time_since_last_log += dt
           
            # B. 主動註冊並更新所有粒子的最新狀態
            for arm_id, particles in engine.particle_systems.items():
                for p in particles:
                    pid = p['id']
                    if pid not in particle_registry:
                        # 第一次看見該粒子，建立完整紀錄
                        particle_registry[pid] = {
                            'id': pid,
                            'time_on_wafer': 0.0,
                            'path_length': 0.0
                        }
                    # 只有粒子在晶圓上時才累加統計量
                    if p['state'] == 'on_wafer':
                        particle_registry[pid]['time_on_wafer'] = p['time_on_wafer']
                        particle_registry[pid]['path_length'] = p['path_length']

            # C. 熱力圖累積 (計量導向：累積水滴滯留時間)
            rad_wafer = math.radians(snapshot['wafer_angle'])
            cos_t, sin_t = np.cos(-rad_wafer), np.sin(-rad_wafer)
            rot_back = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

            active_coords = []
            for particles in engine.particle_systems.values():
                for p_item in particles:
                    if p_item['state'] == 'on_wafer':
                        # 取得粒子在晶圓座標系下的位置
                        center = np.dot(rot_back, p_item['pos'][:2])
                        active_coords.append(center)

            if active_coords:
                # 每一點代表該位置在 dt 期間內有一個粒子存在，累積滯留時間 dt
                hist, _, _ = np.histogram2d(
                    np.array(active_coords)[:,0], 
                    np.array(active_coords)[:,1], 
                    bins=grid_size, 
                    range=[[-150, 150], [-150, 150]]
                )
                heatmap_accum += hist * dt

            # D. 紀錄報表 (每 0.01s 或 到達模擬結束點)
            is_finished = snapshot.get('is_finished', False)
            if time_since_last_log >= report_log_interval or is_finished:
                time_since_last_log = 0.0
                if progress_widgets:
                    try:
                        p_bar = progress_widgets['bar']
                        p_label = progress_widgets['label']
                        current_val = min(sim_clock, p_bar['maximum'])
                        p_bar['value'] = current_val
                        p_label.config(text=f"Processing Report: {snapshot['time']:.1f}s / (Simulating...)")
                        progress_widgets['window'].update_idletasks()
                    except: pass

                nozzle_pos = snapshot['nozzle_pos']
                all_on_wafer_coords = []
                for particles in engine.particle_systems.values():
                    all_on_wafer_coords.extend([p['pos'][:2] for p in particles if p['state'] == 'on_wafer'])

                radial_counts = calculate_water_counts_by_radius(all_on_wafer_coords, WAFER_RADIUS, REPORT_INTERVAL_MM)
                nozzle_r = np.linalg.norm(nozzle_pos) if snapshot['active_arm_id'] != 0 else 0.0

                row_data = {
                    'Time Elapsed': f"{snapshot['time']:.2f}",
                    'Process Recipe Number': snapshot['process_idx'] + 1,
                    'Dispense Arm Number': snapshot['active_arm_id'] if snapshot['active_arm_id'] != 0 else 'N/A',
                    'State': snapshot['state'],
                    'Process Time': "Running" if snapshot['state'] in [STATE_RUNNING_PROCESS, STATE_MOVING_FROM_CENTER_TO_START] else "N/A",
                    'Spin speed': f"{snapshot['rpm']:.2f}",
                    'Nozzle X-Position': f"{nozzle_pos[0]:.3f}" if snapshot['active_arm_id'] != 0 else 'N/A',
                    'Nozzle Y-Position': f"{nozzle_pos[1]:.3f}" if snapshot['active_arm_id'] != 0 else 'N/A',
                    'Nozzle Radius': f"{nozzle_r:.3f}" if snapshot['active_arm_id'] != 0 else 'N/A',
                }
                row_data.update(radial_counts)
                report_data.append(row_data)

            # 檢查結束時機 (每影格檢查，但記錄則遵循 0.01s 間隔，確保最後一幀必存)
            if is_finished:
                # 強制補上最後一幀的數據 (僅當 report_data 不為空且時間點不同時)
                if not report_data or report_data[-1]['Time Elapsed'] != f"{snapshot['time']:.2f}":
                    report_data.append(row_data)
                break
                
            # 安全鎖
            if snapshot['time'] > (total_duration + 30.0): break

        # 轉換註冊表為報表需要的列表格式
        final_particles_list = list(particle_registry.values())

        # ======= Simulation Summary for Terminal =======
        print("\n" + "="*60)
        print(" [Simulation Summary - Headless Mode] ")
        print("-" * 60)
       
        # 1. Particle Statistics
        total_captured = len(final_particles_list)
        print(f" ● Total Particles Captured       : {total_captured:,} pts")

        # 2. Process & Time Parameters
        total_recipe_time = sum(p['total_duration'] for p in recipe['processes'])
        print(f" ● Simulated Duration             : {sim_clock:.2f} s / {total_recipe_time:.2f} s")
        print(f" ● Time Step (dt) / Frame Rate    : {dt:.6e} s / {report_fps} FPS")

        # 3. Heatmap Metrics
        max_heat = np.max(heatmap_accum)
        median_heat = np.median(heatmap_accum)
        range_heat = max_heat - np.min(heatmap_accum)
        std_heat = np.std(heatmap_accum)
        print(f" ● Heatmap Intensity              : Max={max_heat:.4f}, Median{median_heat:.4f}, Range={range_heat:.4f}, Std={std_heat:.4f}")

        # 4. Data Logging
        print(f" ● Report Log Entries             : {len(report_data):,} lines")
       
        # 5. Status Update
        print("-" * 60)
        print(f" Status: Calculation completed")
        print("="*60 + "\n")

        return report_data, final_particles_list, heatmap_accum

    def parse_and_prepare_recipe(self):
        try:
            recipe = {'processes': []}
            recipe['spin_dir'] = self.spin_dir.get()
               
            # 1. 讀取製程參數
            for i, proc_data in enumerate(self.processes_data):
                # 手臂 ID
                arm_str = proc_data['arm_var'].get()
                arm_id = 0 if arm_str == "None" else int(arm_str.split(" ")[1])

                # Flow Rate
                flow_rate = float(proc_data['flow_rate_var'].get())
               
                # 時間與模式
                duration = float(proc_data['duration_var'].get())
                if duration <= 0: raise ValueError(f"Process {i+1} total time must be > 0.")
               
                start_from_center = proc_data['start_from_center_var'].get()
                spin_mode = proc_data['spin_mode_var'].get()
               
                # 轉速參數打包
                spin_params = {'mode': spin_mode}
                if spin_mode == 'Simple':
                    spin_params['rpm'] = float(proc_data['simple_rpm_var'].get())
                else:
                    spin_params['start_rpm'] = float(proc_data['start_rpm_var'].get())
                    spin_params['end_rpm'] = float(proc_data['end_rpm_var'].get())
               
                # 3. 讀取步驟 (只讀取原始輸入，不進行物理計算)
                steps = []
                if arm_id != 0:
                    last_pos = -float('inf')
                    for j, entry in enumerate(proc_data['step_entries']):
                        pos = float(entry['pos'].get())
                        speed = float(entry['speed'].get())
                       
                        if not (-120 <= pos <= 120 and 0 <= speed <= 100):
                            raise ValueError("Parameter out of range.")
                        if pos < last_pos:
                            raise ValueError(f"Process {i+1}: Steps must be increasing.")
                        last_pos = pos
                       
                        # 注意：這裡不需要算 vel_percent_s，Engine 會自己算
                        steps.append({'pos': pos, 'speed': speed})
               
                recipe['processes'].append({
                    'arm_id': arm_id,
                    'flow_rate': flow_rate,
                    'total_duration': duration,
                    'spin_params': spin_params,
                    'start_from_center': start_from_center,
                    'steps': steps
                })
           
            return recipe

        except Exception as e:
            messagebox.showerror("Input Error", f"Error during parsing: {e}")
            return None

    def start_or_update_simulation(self):
        # 0. 防止重複點擊觸發
        if hasattr(self, '_is_starting_sim') and self._is_starting_sim:
            return
        self._is_starting_sim = True
        
        # 1. 解析並準備 Recipe 數據
        parsed_recipe = self.parse_and_prepare_recipe()
        if not parsed_recipe:
            self._is_starting_sim = False
            return
        self.recipe = parsed_recipe

        # 2. 徹底清理舊動畫與引擎
        if hasattr(self, 'ani') and self.ani:
            try:
                if self.ani.event_source:
                    self.ani.event_source.stop()
            except:
                pass
            finally:
                self.ani = None

        # 3. 獲取物理參數並打包
        global_water_params = self._get_water_params()
        water_params_dict = {arm_id: {
            'viscosity': global_water_params['viscosity'],
            'surface_tension': global_water_params['surface_tension'],
            'evaporation_rate': global_water_params['evaporation_rate']
        } for arm_id in [1, 2, 3]}

        # 4. 建立 (或重啟) 模擬視窗
        self.display_water_var.set(True)
        if not self.sim_window or not self.sim_window.winfo_exists():
            self.create_simulator_window()

        # 5. 建立 SimulationEngine
        self.engine = SimulationEngine(self.recipe, self.arms, water_params_dict)

        # 6. 【動態步長系統初始化】
        self.is_paused = False
        self.speed_var.set("1x")
        self.pause_button.config(text="Pause")
        
        # 根據 constants.py 設定基礎步長，確保模擬速度與 FPS 掛鉤
        self.fixed_dt = 1.0 / FPS 
        # 移除舊有的慢速累積器，改用動態 dt 實現絲滑動畫
       
        # 7. 啟動動畫
        self.run_animation()
        self._is_starting_sim = False

    def prepare_water_params_from_ui(self):
        """
        輔助函式：如果您有專門的 UI 變數，可以在這裡統一打包
        """
        params_dict = {}
        for arm_id in self.arms.keys():
            # 範例：從您的 Tkinter 變數中讀取
            params_dict[arm_id] = {
                'viscosity': float(self.viscosity_entries[arm_id].get()),
                'surface_tension': float(self.tension_entries[arm_id].get()),
                'flow_rate': float(self.flow_entries[arm_id].get())
            }
        return params_dict

    def _create_dummy_artists(self):
        if self.arms: return
        fig = Figure()
        ax = fig.add_subplot(111)
        self.arms = {}
        for i in range(1, 4):
            arm_line, = ax.plot([], [])
            nozzle_head = plt.Circle((0, 0), 10)
            geo = ARM_GEOMETRIES[i]
            self.arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], arm_line, nozzle_head)

    def _on_simulator_close(self):
        """Handle simulator window closure safely."""
        self.ani_running = False
        
        # 立即停止動畫事件源
        try:
            if hasattr(self, 'ani') and self.ani:
                if self.ani.event_source:
                    self.ani.event_source.stop()
                # 移除對動畫對象的引用，幫助 GC
                self.ani = None
        except Exception:
            pass
            
        # 使用 root.after 給予 100ms 讓 Matplotlib 執行緒安全退出，避免正在進行的繪圖操作引發崩潰
        if self.sim_window:
            self.root.after(100, self._safe_destroy_sim_window)

    def _safe_destroy_sim_window(self):
        if self.sim_window:
            try:
                self.sim_window.destroy()
            except:
                pass
            self.sim_window = None
            self.ani = None

    def create_simulator_window(self):
        self.sim_window = tk.Toplevel(self.root)
        self.sim_window.title("Simulator")
        self.sim_window.geometry("800x700")
        self.sim_window.resizable(False, False)
        self.sim_window.protocol("WM_DELETE_WINDOW", self._on_simulator_close)
        sim_control_frame = ttk.Frame(self.sim_window, padding=5)
        sim_control_frame.pack(side="top", fill="x")
        self.pause_button = ttk.Button(sim_control_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=5)
        self.water_toggle_button = ttk.Button(sim_control_frame, text="Hide Water", command=self.toggle_water_display)
        self.water_toggle_button.pack(side="left", padx=5)
        
        # --- 速度控制區：改用按鈕遞增方式，徹底避開 OptionMenu 點擊阻塞 ---
        ttk.Label(sim_control_frame, text="Speed:").pack(side="left", padx=(10, 2))
        self.speed_var = tk.StringVar(value="1x")
        self.speed_label = ttk.Label(sim_control_frame, textvariable=self.speed_var, width=5, foreground="blue", font=("Arial", 10, "bold"))
        self.speed_label.pack(side="left")
        
        ttk.Button(sim_control_frame, text="<<", width=3, command=lambda: self._adjust_speed(-1)).pack(side="left", padx=2)
        ttk.Button(sim_control_frame, text=">>", width=3, command=lambda: self._adjust_speed(1)).pack(side="left", padx=2)
        
        self.speed_options = ["0.1x", "0.25x", "0.5x", "1x", "1.25x", "1.5x", "2x", "5x", "10x", "20x"]
        self.speed_idx = 3 # 預設 1x
        # -------------------------------------------------------------
        
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.sim_window)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.init_plot()

    def toggle_water_display(self):
        current_state = self.display_water_var.get()
        
        self.display_water_var.set(not current_state)
        if self.display_water_var.get():
            self.water_toggle_button.config(text="Hide Water")
        else:
            self.water_toggle_button.config(text="Show Water")

    def toggle_pause(self):
        if not self.ani: return
        if self.is_paused:
            self.ani.resume()
            self.pause_button.config(text="Pause")
        else:
            self.ani.pause()
            self.pause_button.config(text="Play")
        self.is_paused = not self.is_paused

    def _adjust_speed(self, delta):
        """點擊按鈕調整速度，非阻塞且安全"""
        self.speed_idx = max(0, min(len(self.speed_options) - 1, self.speed_idx + delta))
        new_speed = self.speed_options[self.speed_idx]
        self.speed_var.set(new_speed)
        # 此處不進行任何重計算，僅修改變數，由 update_anim 順便讀取

    def change_play_speed(self, *args):
        pass

    def init_plot(self):
        # 1. 清除舊畫布與設定背景
        self.ax.clear()
        self.ax.set_aspect('equal', 'box')
        self.ax.set_facecolor('black')
        
        # 使用 Constants 設定邊界
        self.ax.set_xlim(-CHAMBER_SIZE / 2, CHAMBER_SIZE / 2)
        self.ax.set_ylim(-CHAMBER_SIZE / 2, CHAMBER_SIZE / 2)

        # 2. 繪製靜態背景 (Chamber, Wafer, Mask)
        # Chamber 外框
        self.ax.add_patch(plt.Rectangle(
            (-CHAMBER_SIZE / 2, -CHAMBER_SIZE / 2), CHAMBER_SIZE, CHAMBER_SIZE,
            facecolor='none', edgecolor='gray', lw=2
        ))
       
        # Wafer 本體
        self.ax.add_patch(plt.Circle(
            (0, 0), WAFER_RADIUS,
            facecolor='#222222', edgecolor='cyan', lw=1.5, zorder=1
        ))
       
        # Wafer 中心點標記
        self.ax.add_patch(plt.Circle((0, 0), 3, color='cyan', zorder=2))
       
        # Notch (缺口) - 初始化為原點，稍後由 update_anim 更新位置
        self.notch_patch = plt.Polygon(
            [[0, 0], [0, 0], [0, 0]],
            closed=True, facecolor='black', edgecolor='cyan', lw=1.5, zorder=2
        )
        self.ax.add_patch(self.notch_patch)
       
        # 圓形遮罩 (Mask) - 遮住晶圓外部區域
        mask_inner_radius = WAFER_RADIUS + 10
        mask_outer_radius = CHAMBER_SIZE / 2
        circular_mask = patches.Wedge(
            center=(0, 0),
            r=mask_outer_radius,
            theta1=0,
            theta2=360,
            width=mask_outer_radius - mask_inner_radius,
            facecolor='black',
            zorder=11
        )
        self.ax.add_patch(circular_mask)

        # 3. 【關鍵修正】初始化手臂與繪圖物件
        self.arms = {}
        arm_colors = {1: 'lime', 2: 'magenta', 3: 'yellow'}
       
        for i in range(1, 4):
            # A. 先建立 Matplotlib 的繪圖物件 (View)
            # 手臂連桿 (Line2D)
            arm_line, = self.ax.plot([], [], color='gray', lw=4, visible=False, zorder=12)
            # 噴嘴頭 (Circle)
            nozzle_head = plt.Circle((0, 0), 10, facecolor=arm_colors[i], visible=False, zorder=13)
            self.ax.add_patch(nozzle_head)
           
            # B. 建立 DispenseArm 模型並綁定繪圖物件 (Model + View)
            geo = ARM_GEOMETRIES[i]
            self.arms[i] = DispenseArm(
                arm_id=i,
                pivot=geo['pivot'],
                home=geo['home'],
                length=geo['length'],
                p_start=geo['p_start'],
                p_end=geo['p_end'],
                # 這裡傳入剛建立的 artist
                arm_artist=arm_line,
                nozzle_artist=nozzle_head
            )

        # 4. 初始化水柱 (Water Columns)
        self.water_columns = {}
        
        for i in range(1, 4):
            # 這裡建立 WaterColumn (預設流量，實際模擬時會依據 Process 調整)
            self.water_columns[i] = WaterColumn(self.ax, 500.0)

        # 5. 初始化狀態文字
        font_properties = {
            'family': 'serif',
            'color':  'white',
            'verticalalignment': 'top',
            'size': 11
        } 
        self.status_text = self.ax.text(
            0.02, 0.98, '',
            transform=self.ax.transAxes,
            fontdict=font_properties,
            zorder=20
        )
       
        # 回傳空列表 (因為這是初始化，blitting 會由 init_anim 處理)
        return []

    def run_animation(self):
        """啟動動畫循環"""
        # 1. 確保 Figure 存在
        if not hasattr(self, 'fig') or self.fig is None:
            print("Error: self.fig is not defined. Ensure create_simulator_window is called.")
            return

        # 2. 清理舊動畫
        if hasattr(self, 'ani') and self.ani:
            try:
                if self.ani.event_source:
                    self.ani.event_source.stop()
                # 斷開與舊畫布的 resize 連結
                if hasattr(self, 'canvas') and self.canvas:
                    self.canvas.mpl_disconnect(self.ani._resize_id)
            except Exception as e:
                print(f"Animation Cleanup Warning: {e}")

        # 3. 初始化時間與狀態
        self.last_frame_time = time.time()
        self.ani_running = True
        self.is_paused = False

        # 4. 建立新的動畫實例
        from matplotlib.animation import FuncAnimation
        # 動畫更新間隔根據 constants.FPS 動態調整 (例如 60fps = 16ms)
        anim_interval = int(1000 / FPS)
        
        self.ani = FuncAnimation(
            self.fig,
            self.update_anim,
            init_func=self.init_anim,
            interval=anim_interval,
            blit=False,
            cache_frame_data=False
        )
       
        # 5. 刷新畫布
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    def update_anim(self, frame):
        """每一幀的渲染邏輯 - 採用【動態步長優化系統】"""
        if not self.ani_running or not hasattr(self, 'engine') or self.engine is None:
            return []
            
        if self.is_paused:
            return self.get_current_artists()

        # 1. 獲取當前倍率
        try:
            multiplier = float(self.speed_var.get().replace('x', ''))
        except:
            multiplier = 1.0

        # 2. 動態調整步長與步數 (動態 FPS 核心邏輯)
        # 目的：在慢速播放時增加模擬精細度(提高模擬FPS)，高速時維持效能平衡
        
        if multiplier <= 1.0:
            # 【慢速/標準模式】：每一幀都更新，但縮小 dt。
            # 優點：即便在 0.1x 下畫面依然有 60fps (或 FPS 設定值) 的更新率，極其絲滑。
            # CPU 負擔：與 1x 播放時完全相同。
            steps_to_run = 1
            dynamic_dt = self.fixed_dt * multiplier
        else:
            # 【高速模式】：為了保持物理模擬的穩定性，限制單步 dt 不過大，改用多步更新。
            # 例如 5x 播放時，每幀執行 5 步，每步維持 fixed_dt 長度。
            steps_to_run = int(multiplier)
            dynamic_dt = self.fixed_dt * (multiplier / steps_to_run)

        # 3. 執行物理步進
        snapshot = None
        for _ in range(steps_to_run):
            snapshot = self.engine.update(dynamic_dt)

        if snapshot is None:
            return self.get_current_artists()

        # 4. 更新視覺元件

        # 3. 更新手臂視覺元件
        active_id = snapshot.get('active_arm_id')
        nozzle_pos = snapshot.get('nozzle_pos', np.array([0.0, 0.0]))
        # 【關鍵修正】從引擎讀取是否正在噴水
        is_spraying = snapshot.get('is_spraying', False)

        for arm_id, arm in self.arms.items():
            if arm_id == active_id:
                # 只有當引擎說 "is_spraying" 為 True 時，才顯示黃色
                color = 'yellow' if is_spraying else 'gray'
               
                arm.update_artists(nozzle_pos, color=color)
               
                # 確保可見性 (假設您的 Arm 類別有 line 和 nozzle 屬性)
                if hasattr(arm, 'line'): arm.line.set_visible(True)
                if hasattr(arm, 'nozzle'): arm.nozzle.set_visible(True)
            else:
                # 非作用中手臂回原點
                arm.go_home()
                # 視需求決定是否隱藏
                # if hasattr(arm, 'line'): arm.line.set_visible(False)

        # 4. 更新水滴視覺元件 (從引擎 snapshot 讀取)
        water_render_data = snapshot.get('water_render', {})
        if self.display_water_var.get():
            for arm_id, data in water_render_data.items():
                if arm_id in self.water_columns:
                    # 調用 WaterColumn 內的 draw 方法
                    self.water_columns[arm_id].draw(data.get('falling', []), data.get('on_wafer', []))
        else:
            for arm_id in self.water_columns:
                self.water_columns[arm_id].clear()

        # 5. 更新晶圓旋轉 (修正之前的 water_plote 拼字錯誤)
        if hasattr(self, 'wafer_plot'):
            # 使用 matplotlib.transforms 進行座標變換
            import matplotlib.transforms as mtransforms
            trans = mtransforms.Affine2D().rotate_deg(snapshot['wafer_angle']) + self.ax.transData
            self.wafer_plot.set_transform(trans)

        # 引擎回傳的是計算好的多邊形頂點座標 (3x2 矩陣)
        notch_coords = snapshot.get('notch_coords')
        if notch_coords is not None and hasattr(self, 'notch_patch'):
            self.notch_patch.set_xy(notch_coords)
       
        # 6. 更新左上角狀態文字
        status_text = (f"Time: {snapshot['time']:.2f}s\n"
                       f"Process: {snapshot['process_idx'] + 1}\n"
                       f"State: {snapshot['state']}\n"
                       f"Step: {snapshot['step_str']}\n"
                       f"Process Time: {snapshot['process_time_str']}\n"
                       f"RPM: {snapshot['rpm']:.0f}")
        self.status_text.set_text(status_text)

        # 7. 回傳所有 Artist 進行重繪
        # 修正：加上 self.
        return self.get_current_artists()

    def init_anim(self):
        """動畫初始化：清空所有視覺元件的數據"""
        # 1. 重置手臂
        for arm in self.arms.values():
            arm.update_artists(arm.home_pos)
           
        # 2. 重置水柱
        for wc in self.water_columns.values():
            wc.reset()
           
        # 3. 重置晶圓旋轉
        if hasattr(self, 'wafer_plot'):
            self.wafer_plot.set_transform(self.ax.transData)
           
        # 4. 重置文字
        self.status_text.set_text("Initializing...")
       
        # 回傳初始的 Artist 列表
        return self.get_current_artists()

    def get_current_artists(self):
        """蒐集當前畫面上所有需要重繪的 Artist 物件"""
        artists = []
       
        # A. 基礎元件
        if hasattr(self, 'status_text'):
            artists.append(self.status_text)
        if hasattr(self, 'wafer_plot'):
            artists.append(self.wafer_plot)
           
        # B. 手臂元件 (獲取每隻手臂的 line 和 nozzle)
        for arm in self.arms.values():
            # 假設您的 DispenseArm.get_artists() 回傳 [self.line, self.nozzle]
            if hasattr(arm, 'get_artists'):
                artists.extend(arm.get_artists())
            else:
                # 備案：如果沒有 get_artists 方法，直接手動加入
                if hasattr(arm, 'line'): artists.append(arm.line)
                if hasattr(arm, 'nozzle'): artists.append(arm.nozzle)
               
        # C. 水柱元件 (獲取 falling 和 on_wafer 兩層)
        for wc in self.water_columns.values():
            if hasattr(wc, 'artist'):
                artists.append(wc.artist)
            if hasattr(wc, 'on_wafer_artist'):
                artists.append(wc.on_wafer_artist)
               
        return artists

    def get_rpm_at_time(self, process, time_in_proc=0):
        if self.animation_state in [STATE_RUNNING_PROCESS, STATE_MOVING_FROM_CENTER_TO_START, STATE_PAUSE_AT_CENTER]:
            spin_params = process['spin_params']
            if spin_params['mode'] == 'Simple': return spin_params['rpm']
            else:
                total_dur = process['total_duration']
                ratio = min(1.0, max(0.0, time_in_proc / total_dur if total_dur > 0 else 1.0))
                return spin_params['start_rpm'] * (1 - ratio) + spin_params['end_rpm'] * ratio
        else:
            time_in_state = self.simulation_time_elapsed - self.transition_start_time
            active_arm = self.arms.get(self.active_arm_id)
            if not active_arm:
                prev_proc_idx = (self.current_process_index - 1 + len(self.recipe['processes'])) % len(self.recipe['processes'])
                prev_proc = self.recipe['processes'][prev_proc_idx]
                if prev_proc['arm_id'] != 0:
                    active_arm = self.arms[prev_proc['arm_id']]
                else:
                    return self.transition_start_rpm
            angle_diff = active_arm._get_angle_diff(self.transition_end_angle, self.transition_start_angle)
            arc_dist = abs(angle_diff) * active_arm.arm_length
            target_speed = MAX_NOZZLE_SPEED_MMS * 0.75
            transition_duration = arc_dist / target_speed if target_speed > 0 else float('inf')
            ratio = min(1.0, max(0.0, time_in_state / transition_duration if transition_duration > 0 else 1.0))
            return self.transition_start_rpm * (1 - ratio) + self.transition_end_rpm * ratio

    def on_closing(self):
        # 先停止動畫，防止關閉過程中繼續觸發渲染
        try:
            if hasattr(self, 'ani') and self.ani and self.ani.event_source:
                self.ani.event_source.stop()
        except Exception:
            pass
        self.ani = None
        self.ani_running = False
        
        # 關閉模擬視窗 (如果存在)
        if self.sim_window and self.sim_window.winfo_exists():
            try:
                self.sim_window.destroy()
            except Exception:
                pass
            self.sim_window = None

        # 銷毀主視窗
        self.root.destroy()


