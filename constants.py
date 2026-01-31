import numpy as np

# --- Constants ---
WAFER_DIAMETER = 300
WAFER_RADIUS = WAFER_DIAMETER / 2
MAX_NOZZLE_SPEED_MMS = 250
CHAMBER_SIZE = 450
FPS = 30
REPORT_FPS = 800  # 目前已根據 recipe 動態調整，這個只是預設最低數值
NOTCH_DEPTH = 15
NOTCH_HALF_WIDTH = 7.5

# Simulation Report 參數
REPORT_INTERVAL_MM = 2

# --- Heatmap Physical & Visualization Constants ---
# [物理飽和限制] 定義網格每秒能吸收的最大有效貢獻時間。
# 數值 1.2 代表 nozzle 正下方飽和時，每秒貢獻 1.2 秒的有效清洗劑量。
# EFFICIENCY_CAP = 1.0  

# [Heatmap Quality & Physics Emulation]
# HEATMAP_SMOOTHING_SIGMA = 1.5
# HEATMAP_SAMPLE_COUNT = 3
# HEATMAP_SPREAD_SCALE = 80

# --- Simulation Density Parameters ---
PARTICLE_MAX_COUNT = 3000         # 調降最大粒子數，顯著減輕 CPU 繪圖負擔
PARTICLE_SPAWN_MULTIPLIER = 1.0   # 粒子生成速率的乘數

# --- 視覺連貫性優化參數 ---
WATER_RENDER_INTERPOLATION_LIMIT = 12 # 適度降低插值點上限
WATER_JITTER_AMOUNT = 3.0            # 隨機微擾動幅度 (mm)
WATER_DROP_SIZE = 10                  # 噴嘴下落水滴尺寸
WATER_ON_WAFER_SIZE = 10              # 晶圓表面水滴尺寸

# --- Physics Constants ---
GRAVITY_MMS2 = 9800  # Gravity in mm/s^2

# --- Simulation Geometry & Speed ---
NOZZLE_RADIUS_MM = 2.0            # 噴嘴半徑 (mm)
NOZZLE_Z_HEIGHT = 15.0            # 噴嘴到晶圓的初始垂直距離 (mm)
TRANSITION_ARM_SPEED_RATIO = 0.8  # Arm 轉換狀態下的速度比例 (用於乘以 MAX_NOZZLE_SPEED_MMS)

# --- Etching Amount Simulation Constants ---
ETCHING_TAU = 0.3                 # 老化模型衰減常數 (s)
GRID_SIZE = 10.0                  # 蝕刻影響半徑 (mm)
ETCHING_IMPINGEMENT_TIME = 0.01   # 判定為衝擊區的在晶圓時間門檻 (s)
ETCHING_IMPINGEMENT_BONUS = 2.0   # 衝擊區的強度加成倍數
ETCHING_GEO_SMOOTHING = 7.0      # 幾何稀釋平滑常數
ETCHING_SATURATION_THRESHOLD = 0.002 # 每一步長單個像素點的最大蝕刻貢獻飽和值 (配合像素級飽和邏輯)

# --- Timing & Pause ---
ARM_CHANGE_PAUSE_TIME = 1.0       # Arm 切換之間的停頓時間 (s)
CENTER_PAUSE_TIME = 0.8           # Arm 抵達晶圓中心後停頓的時間 (s)

# State Machine Constants
STATE_RUNNING_PROCESS = "RUNNING_PROCESS"
STATE_ARM_MOVE_FROM_HOME = "ARM_MOVE_FROM_HOME"
STATE_ARM_MOVE_TO_HOME = "ARM_MOVE_TO_HOME"
STATE_ARM_CHANGE_PAUSE = "ARM_CHANGE_PAUSE"
STATE_PAUSE_AT_CENTER = "PAUSE_AT_CENTER"
STATE_MOVING_TO_CENTER_ARC = "MOVING_TO_CENTER_ARC"
STATE_MOVING_FROM_CENTER_TO_START = "MOVING_FROM_CENTER_TO_START"

# --- Arm Geometric Definitions ---

ARM_GEOMETRIES = {
    1: {
        "pivot": np.array([-225.0, -225.0]), 
        "length": 318.198,
        "home": np.array([-200.0, 94.02]),
        "p_start": np.array([-126.241, 79.231]), 
        "p_end": np.array([79.231, -126.241])
    },
    2: {
        "pivot": np.array([225.0, -225.0]), 
        "length": 318.198,
        "home": np.array([200.0, 94.02]),
        "p_start": np.array([126.241, 79.231]), 
        "p_end": np.array([-79.231, -126.241])
    },
    3: {
        "pivot": np.array([225.0, -225.0]), 
        "length": 318.198,
        "home": np.array([-94.02, -200.0]),
        "p_start": np.array([-79.231, -126.241]), 
        "p_end": np.array([126.241, 79.231])
    }
}
