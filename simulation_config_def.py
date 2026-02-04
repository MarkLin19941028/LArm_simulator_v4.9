# 定義參數結構：Key 為參數名稱，Value 為 UI 設定
# 格式: (Label顯示名稱, 預設值, 變數類型, (最小值, 最大值), 提示訊息)
# 變數類型: 'float', 'int'

PARAMETER_DEFINITIONS = {
    "Timing & Speed": {
        "TRANSITION_ARM_SPEED_RATIO": ("Trans. Speed Ratio", 0.8, 'float', (0.1, 2.0), "Arm 不噴灑移動的速度(相對最快移動速度）"),
        "ARM_CHANGE_PAUSE_TIME":      ("Arm Change Pause (s)", 1.0, 'float', (0.0, 10.0), "Arm 切換之間的停頓時間 (s)"),
        "CENTER_PAUSE_TIME":          ("Center Pause (s)", 0.8, 'float', (0.0, 10.0), "Arm 抵達晶圓中心後停頓的時間 (s)"),
    },
    "Etching Simulation": {
        "ETCHING_TAU":                ("Decay Tau (s)", 0.3, 'float', (0.01, 100.0), "老化模型衰減常數 (s)"),
        "GRID_SIZE":                  ("Grid Size (mm)", 10.0, 'float', (1.0, 30.0), "蝕刻影響半徑 (mm)"),
        "ETCHING_IMPINGEMENT_TIME":   ("Impingement Time (s)", 0.01, 'float', (0.0, 5.0), "判定為衝擊區的在晶圓時間門檻 (s)"),
        "ETCHING_IMPINGEMENT_BONUS":  ("Impingement Bonus", 2.0, 'float', (1.0, 10.0), "衝擊區的強度加成倍數"),
        "ETCHING_GEO_SMOOTHING":      ("Geo Smoothing", 7.0, 'float', (0.1, 50.0), "幾何釋平滑常數"),
        "ETCHING_SATURATION_THRESHOLD":("Sat. Threshold", 0.002, 'float', (0.0001, 1.0), "每一步長單個像素點的最大蝕刻貢獻飽和值 (配合像素級飽和邏輯)"),
    },
    "PRE Simulation": {
        "PRE_ALPHA":                  ("Alpha (Shear)", 0.001, 'float', (0.0, 1.0), "剪切項係數"),
        "PRE_BETA":                   ("Beta (Impact)", 0.5, 'float', (0.0, 10.0), "衝擊項保底係數"),
        "PRE_GRID_SIZE":              ("PRE Grid Size (mm)", 10.0, 'float', (1.0, 30.0), "清洗影響半徑 (mm)"),
        "PRE_Q_REF":                  ("Q Ref (mL/min)", 1000.0, 'float', (100.0, 5000.0), "參考流量 (mL/min)"),
        "PRE_GAMMA_BASE":             ("Gamma Base", 0.001, 'float', (0.0, 1.0), "基礎再附著係數 (1/mm)"),
    }
}

def get_default_config():
    """回傳一個扁平化的預設配置字典"""
    config = {}
    for section in PARAMETER_DEFINITIONS.values():
        for key, val in section.items():
            config[key] = val[1]
    return config
