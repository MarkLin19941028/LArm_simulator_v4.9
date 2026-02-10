# 定義參數結構：Key 為參數名稱，Value 為 UI 設定
# 格式: (Label顯示名稱, 預設值, 變數類型, (最小值, 最大值), 提示訊息)
# 變數類型: 'float', 'int'

PARAMETER_DEFINITIONS = {
    "Timing & Speed": {
        "TRANSITION_ARM_SPEED_RATIO": ("Trans. Speed Ratio", 0.8, 'float', (0.1, 2.0), "Arm 不噴灑時移動的速度(相對最快速度）"),
        "ARM_CHANGE_PAUSE_TIME":      ("Arm Change Pause (s)", 1.0, 'float', (0.0, 10.0), "Arm 切換之間的停頓時間 (s)"),
        "CENTER_PAUSE_TIME":          ("Center Pause (s)", 0.8, 'float', (0.0, 10.0), "Arm 抵達晶圓中心後停頓的時間 (s)"),
    },
    "Etching Amount": {
        "ETCHING_TAU":                ("Decay Tau (s)", 0.3, 'float', (0.01, 100.0), "老化模型衰減常數 (s)"),
        "GRID_SIZE":                  ("Grid Size (mm)", 5.0, 'float', (1.0, 30.0), "蝕刻影響半徑 (mm)"),
        "ETCHING_IMPINGEMENT_TIME":   ("Impingement Time (s)", 0.01, 'float', (0.0, 5.0), "判定為衝擊區的在晶圓時間門檻 (s)"),
        "ETCHING_IMPINGEMENT_BONUS":  ("Impingement Bonus", 1.2, 'float', (1.0, 10.0), "衝擊區的強度加成倍數"),
        "ETCHING_GEO_SMOOTHING":      ("Geo Smoothing", 7.0, 'float', (0.1, 50.0), "幾何釋平滑常數"),
        "ETCHING_SATURATION_THRESHOLD":("Sat. Threshold", 0.002, 'float', (0.0001, 1.0), "最大蝕刻貢獻飽和值"),
    },
    "Particle Removal": {
        "PRE_ALPHA":                  ("Alpha (Shear)", 0.001, 'float', (0.0, 1.0), "剪切項係數"),
        "PRE_BETA":                   ("Beta (Impact)", 0.5, 'float', (0.0, 10.0), "衝擊項保底係數"),
        "PRE_GRID_SIZE":              ("PRE Grid Size (mm)", 5.0, 'float', (1.0, 30.0), "清洗影響半徑 (mm)"),
        "PRE_Q_REF":                  ("Q Ref (mL/min)", 1000.0, 'float', (100.0, 5000.0), "參考流量 (mL/min)"),
        "PRE_GAMMA_BASE":             ("Gamma Base", 0.001, 'float', (0.0, 1.0), "基礎再附著係數 (1/mm)"),
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
