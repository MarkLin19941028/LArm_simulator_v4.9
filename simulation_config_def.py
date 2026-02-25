# 定義參數結構：Key 為參數名稱，Value 為 UI 設定
# 格式: (Label顯示名稱, 預設值, 變數類型, (最小值, 最大值), 提示訊息)
# 變數類型: 'float', 'int'

PARAMETER_DEFINITIONS = {
    "General": {
        "TRANSITION_ARM_SPEED_RATIO": ("Trans. Speed Ratio", 0.8, 'float', (0.1, 2.0), "Arm 不噴灑時移動的速度(相對最快速度）"),
        "ARM_CHANGE_PAUSE_TIME":      ("Arm Change Pause (s)", 1.0, 'float', (0.0, 10.0), "Arm 切換之間的停頓時間 (s)"),
        "CENTER_PAUSE_TIME":          ("Center Pause (s)", 0.8, 'float', (0.0, 10.0), "Arm 抵達晶圓中心後停頓的時間 (s)"),
        "REPORT_INTERVAL_MM":         ("Report Interval (mm)", 2.0, 'float', (0.1, 50.0), "Simulation Report 徑向間隔 (mm)"),
        "REPORT_LOG_INTERVAL":        ("Report Log Interval (s)", 0.01, 'float', (0.001, 5.0), "Simulation Report 時間記錄間隔 (s)"),
    },
    "Etching Amount": {
        "GRID_SIZE":                  ("Grid Size (radius)", 5.0, 'float', (0.0000001, 150.0), "單個粒子的影響半徑 (mm)。影響渲染的解析度與路徑平滑度。"),
        "ETCHING_TAU":                ("Etching Tau", 0.3, 'float', (0.0000001, 50.0), "化學老化常數 (s)。模擬藥液活性隨時間衰減的速度。"),
        "ETCHING_SATURATION_THICKNESS":("Saturation Thickness", 0.002, 'float', (0.00000001, 10.0), "反應飽和與膜厚。模擬化學反應在表面完全潤濕後的飽和上限。"),
        "ETCHING_BASE_SPIN_DECAY":    ("Base Spin Decay", 2.0, 'float', (0.00000001, 10.0), "基礎甩乾速率。模擬液體因旋轉與蒸發離開表面的速度。"),
        "ETCHING_IMPINGEMENT_BONUS":  ("Impingement Bonus", 1.2, 'float', (1.0, 50.0), "衝擊加成倍數。噴嘴正下方新鮮藥液撞擊帶來的蝕刻增益。"),
        "ETCHING_GEO_SMOOTHING":      ("Geo Smoothing", 0.1, 'float', (0.0, 150.0), "幾何平滑係數。配合平方項校正公式，用於微調中心點的數值。"),
        "ETCHING_SATURATION_THRESHOLD":("Sat. Threshold", 0.0, 'float', (0.0, 10.0), "最終蝕刻量飽和閥值。用於 np.tanh 限制極端值的數學處理。"),
    },
    "Particle Removal": {
        "PRE_ALPHA":                  ("Alpha (Shear)", 0.001, 'float', (0.0, 1.0), "剪切項係數"),
        "PRE_BETA":                   ("Beta (Impact)", 0.5, 'float', (0.0, 10.0), "衝擊項保底係數"),
        "PRE_GRID_SIZE":              ("PRE Grid Size (mm)", 5.0, 'float', (1.0, 30.0), "清洗影響半徑 (mm)"),
        "PRE_Q_REF":                  ("Q Ref (mL/min)", 1000.0, 'float', (100.0, 5000.0), "參考流量 (mL/min)"),
        "PRE_GAMMA_BASE":             ("Gamma Base", 0.001, 'float', (0.0, 1.0), "基礎再附著係數 (1/mm)"),
    },
    "Charging Simulation": {
        "FLUID_CONDUCTIVITY":         ("Conductivity (S/m)", 5.0e-09, 'float', (1.0e-16, 10.0), "藥液導電率 (S/m)。DIW 約 5e-6，化學液約 1.0。"),
        "FLUID_RELATIVE_PERMITTIVITY":("Rel. Permittivity", 80.0, 'float', (1.0, 100.0), "相對介電常數。水約為 80。"),
        "CHARGING_EFFICIENCY":        ("Charging Efficiency", -5.0e-5, 'float', (-1.0, 1.0), "電荷產生效率經驗係數。"),
        "CHARGING_BASE_SPIN_DECAY":    ("Base Spin Decay", 2.0, 'float', (0.00000001, 10.0), "基礎甩乾速率。模擬液體因旋轉與蒸發離開表面的速度。"),
    },
    "Advanced Physics Parameters": {
        "PHYSICS_PRESSURE_PUSH_STRENGTH": ("Pressure Push", 5.0, 'float', (0.0, 5000.0), "中心區域推力強度 (解決中心堆積)"),
        "PHYSICS_PRESSURE_CORE_RADIUS":   ("Core Radius (mm)", 80.0, 'float', (1.0, 150.0), "中心推力影響半徑 (mm)"),
        "PHYSICS_ST_RESIST_BASE":         ("ST Resist Base", 0.3, 'float', (0.0, 10.0), "表面張力基礎阻力係數"),
        "PHYSICS_WEBER_COEFF":            ("Weber Coeff", 0.01, 'float', (0.0, 0.1), "韋伯數係數 (速度對張力的削弱)"),
        "PHYSICS_VISCOSITY_DAMPING":      ("Visc. Damping", 2.0, 'float', (0.0, 10.0), "基礎阻尼係數 (控制整體流動)"),
        "PHYSICS_FILM_THINNING_FACTOR":   ("Thinning Factor", 1.0, 'float', (0.0, 10.0), "膜厚變薄阻力係數 (Emslie 模型)"),
        "PHYSICS_DRYING_VISC_MULT":       ("Drying Visc Mult", 5.0, 'float', (1.0, 20.0), "乾燥時的黏度倍率"),
        "PHYSICS_RPM_EVAP_COEFF":         ("RPM Evap Coeff", 0.005, 'float', (0.0, 0.1), "轉速依賴蒸發係數"),
        "PHYSICS_SPRAY_SPREAD_BASE":      ("Spray Spread", 600.0, 'float', (100.0, 2000.0), "噴嘴擴散基數 (霧化程度)"),
        "PHYSICS_JET_SPEED_FACTOR":       ("Jet Speed Factor", 0.05, 'float', (0.0, 1.0), "噴嘴垂直初速係數"),
    }
}

def get_default_config():
    """回傳一個扁平化的預設配置字典"""
    config = {}
    for section in PARAMETER_DEFINITIONS.values():
        for key, val in section.items():
            config[key] = val[1]
    return config
