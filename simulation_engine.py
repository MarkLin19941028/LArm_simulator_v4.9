import numpy as np
import math
import random
from numba import njit, prange
from constants import *

# Particle States
P_INACTIVE = 0
P_FALLING = 1
P_ON_WAFER = 2

@njit(fastmath=True, cache=True)
def _physics_kernel(states, pos, vel, last_pos, life, time_on_wafer, path_length, arm_ids,
                    dt, omega, cos_t, sin_t, 
                    viscosities, evap_rates, surface_tensions,
                    gravity, wafer_radius):
    """
    Numba 加速的物理步進核心。
    使用平坦陣列 (SoA) 以獲得最佳快取效率。
    """
    n = states.shape[0]
    for i in range(n):
        state = states[i]
        if state == P_INACTIVE:
            continue
            
        arm_id = arm_ids[i]
        visc = viscosities[arm_id]
        evap = evap_rates[arm_id]
        st_val = surface_tensions[arm_id]

        # 1. 蒸發邏輯
        if evap > 0:
            life[i] -= evap * dt
            if life[i] <= 0:
                states[i] = P_INACTIVE
                continue

        # 紀錄上一位置
        last_pos[i, 0] = pos[i, 0]
        last_pos[i, 1] = pos[i, 1]

        if state == P_FALLING:
            # 重力加速與位置更新
            vel[i, 2] -= gravity * dt
            pos[i, 0] += vel[i, 0] * dt
            pos[i, 1] += vel[i, 1] * dt
            pos[i, 2] += vel[i, 2] * dt
            
            # 碰撞檢查 (落在晶圓上)
            if pos[i, 2] <= 0:
                dist_sq = pos[i, 0]**2 + pos[i, 1]**2
                if dist_sq <= wafer_radius**2:
                    states[i] = P_ON_WAFER
                    pos[i, 2] = 0.0
                    vel[i, 0] *= 0.1
                    vel[i, 1] *= 0.1
                    vel[i, 2] = 0.0
                else:
                    # 掉出晶圓外
                    states[i] = P_INACTIVE
                    continue
        
        elif state == P_ON_WAFER:
            time_on_wafer[i] += dt
            
            x, y = pos[i, 0], pos[i, 1]
            dist_sq = x*x + y*y
            dist = math.sqrt(dist_sq)
            
            if dist > 1e-4:
                # 1. 基礎離心加速度
                inv_dist = 1.0 / dist
                nx, ny = x * inv_dist, y * inv_dist
                centrifugal_acc = omega * omega * dist
                
                # 2. 表面張力阻力 (徑向向心)
                st_resistance = st_val * 0.3
                
                total_acc_mag = centrifugal_acc - st_resistance
                
                # 更新速度
                vel[i, 0] += nx * total_acc_mag * dt
                vel[i, 1] += ny * total_acc_mag * dt
                
                # 黏度阻尼
                damping = (1.0 - 0.05 * visc * dt)
                vel[i, 0] *= damping
                vel[i, 1] *= damping
                
            # 更新位置
            old_x, old_y = pos[i, 0], pos[i, 1]
            pos[i, 0] += vel[i, 0] * dt
            pos[i, 1] += vel[i, 1] * dt
            
            # 更新路徑長度
            move_dist = math.sqrt((pos[i, 0] - old_x)**2 + (pos[i, 1] - old_y)**2)
            path_length[i] += move_dist
            
            # 旋轉補償 (晶圓旋轉)
            rx, ry = pos[i, 0], pos[i, 1]
            pos[i, 0] = rx * cos_t - ry * sin_t
            pos[i, 1] = rx * sin_t + ry * cos_t

        # 移除邏輯
        dist_sq_final = pos[i, 0]**2 + pos[i, 1]**2
        if dist_sq_final > (wafer_radius + 20)**2 or pos[i, 2] < -10:
            states[i] = P_INACTIVE

# >>>>+++ REPLACE


class SimulationEngine:
    def __init__(self, recipe, arms_dict, water_params_dict, headless=False, config=None):
        self.recipe = recipe
        self.arms = arms_dict
        self.water_params = water_params_dict
        self.headless = headless
        self.config = config if config else {}

        self.simulation_mode = self.config.get('SIMULATION_MODE', 'full')
        self.transition_arm_speed_ratio = self.config.get('TRANSITION_ARM_SPEED_RATIO', TRANSITION_ARM_SPEED_RATIO)
        self.arm_change_pause_time = self.config.get('ARM_CHANGE_PAUSE_TIME', ARM_CHANGE_PAUSE_TIME)
        self.center_pause_time = self.config.get('CENTER_PAUSE_TIME', CENTER_PAUSE_TIME)
        
        # --- Numba 陣列初始化 ---
        self.max_particles = PARTICLE_MAX_COUNT
        self.particles_state = np.zeros(self.max_particles, dtype=np.int32)
        self.particles_pos = np.zeros((self.max_particles, 3), dtype=np.float64)
        self.particles_vel = np.zeros((self.max_particles, 3), dtype=np.float64)
        self.particles_last_pos = np.zeros((self.max_particles, 2), dtype=np.float64)
        self.particles_life = np.zeros(self.max_particles, dtype=np.float64)
        self.particles_time_on_wafer = np.zeros(self.max_particles, dtype=np.float64)
        self.particles_path_length = np.zeros(self.max_particles, dtype=np.float64)
        self.particles_birth_time = np.zeros(self.max_particles, dtype=np.float64)
        self.particles_arm_id = np.zeros(self.max_particles, dtype=np.int32)
        self.particles_id = np.zeros(self.max_particles, dtype=np.int32)
        
        self.next_particle_id = 0
        self._spawn_accumulator = {arm_id: 0.0 for arm_id in arms_dict.keys()}
        
        # 預先處理參數陣列以利 Numba 存取 (index 1-3)
        self.viscosities = np.ones(10, dtype=np.float64)
        self.evap_rates = np.zeros(10, dtype=np.float64)
        self.surface_tensions = np.full(10, 72.8, dtype=np.float64)
        
        for arm_id, p in water_params_dict.items():
            if arm_id < 10:
                self.viscosities[arm_id] = p.get('viscosity', 1.0)
                self.evap_rates[arm_id] = p.get('evaporation_rate', 0.0)
                self.surface_tensions[arm_id] = p.get('surface_tension', 72.8)

        self._pre_calculate_physics()
        
        self.current_notch_coords = np.array([[WAFER_RADIUS, 0],
                                              [WAFER_RADIUS-NOTCH_DEPTH, NOTCH_HALF_WIDTH],
                                              [WAFER_RADIUS-NOTCH_DEPTH, -NOTCH_HALF_WIDTH]])
        self.reset()

    @property
    def particle_systems(self):
        """
        相容性 Property：將 NumPy 陣列包裝成舊版的字典列表格式。
        供 AccuHeatmapGenerator 等組件暫時使用。
        """
        systems = {arm_id: [] for arm_id in self.arms.keys()}
        for i in range(self.max_particles):
            state_val = self.particles_state[i]
            if state_val == P_INACTIVE:
                continue
            
            arm_id = self.particles_arm_id[i]
            p_dict = {
                'id': self.particles_id[i],
                'state': 'falling' if state_val == P_FALLING else 'on_wafer',
                'life': self.particles_life[i],
                'birth_time': self.particles_birth_time[i],
                'time_on_wafer': self.particles_time_on_wafer[i],
                'path_length': self.particles_path_length[i],
                'pos': self.particles_pos[i].copy(),
                'last_pos': self.particles_last_pos[i].copy()
            }
            if arm_id in systems:
                systems[arm_id].append(p_dict)
        return systems

    def reset(self):
        self.simulation_time_elapsed = 0.0
        self.current_process_index = 0
        self.time_offset_for_current_process = 0.0
        self.cumulative_physics_time = 0.0
        self.wafer_angle = 0.0
        self.is_looping_back = False
        self.current_step_label = "Init"
        self.current_rpm_value = 0.0
        
        self.particles_state.fill(P_INACTIVE)
        self.next_particle_id = 0

        first_proc = self.recipe['processes'][0]
        self.active_arm_id = first_proc['arm_id']
        
        if not self.active_arm_id or self.active_arm_id == 0:
            self.animation_state = STATE_RUNNING_PROCESS
            self.last_nozzle_pos = np.array([0.0, 0.0])
        else:
            arm = self.arms[self.active_arm_id]
            self.animation_state = STATE_ARM_MOVE_FROM_HOME
            self.last_nozzle_pos = arm.home_pos.copy()
            self.transition_start_time = 0.0
            self.transition_start_angle = arm.home_angle
            
            if first_proc.get('start_from_center'):
                self.transition_end_angle = arm.coords_to_angle(arm.center_pos_coords)
            else:
                first_step_pos = first_proc['steps'][0]['pos'] if first_proc.get('steps') else 0
                self.transition_end_angle = arm.percent_to_angle(first_step_pos)

    def update(self, dt):
        self.prev_nozzle_pos = self.last_nozzle_pos.copy()
        
        if self.current_process_index < len(self.recipe['processes']):
            current_process = self.recipe['processes'][self.current_process_index]
        else:
            current_process = {'total_duration': 0, 'steps': []}

        if self.animation_state == STATE_RUNNING_PROCESS:
            wall_time_in_proc = self.simulation_time_elapsed - self.time_offset_for_current_process
        else:
            wall_time_in_proc = 0.0

        if self.animation_state == STATE_RUNNING_PROCESS:
            self.current_rpm_value = self._get_rpm_at_time(current_process, wall_time_in_proc)
        
        current_rpm = self.current_rpm_value
        spin_dir = self.recipe.get('spin_dir', 'cw')

        if self.simulation_mode == 'full':
            SUB_STEPS = min(10, 5 + int(current_rpm / 300)) 
            sub_dt = dt / SUB_STEPS
            
            omega = (current_rpm / 60.0) * 2 * math.pi * (-1 if spin_dir == 'cw' else 1)
            d_theta = omega * sub_dt
            cos_t, sin_t = math.cos(d_theta), math.sin(d_theta)

            for i in range(SUB_STEPS):
                # 1. 分步生成粒子
                if (self.active_arm_id and self.active_arm_id != 0) and (self.animation_state == STATE_RUNNING_PROCESS):
                    frac = (i + 1) / SUB_STEPS
                    interp_pos = self.prev_nozzle_pos + (self.last_nozzle_pos - self.prev_nozzle_pos) * frac
                    self._spawn_particles(self.active_arm_id, sub_dt, interp_pos)

                # 2. 物理步進 (Numba 加速)
                _physics_kernel(
                    self.particles_state, self.particles_pos, self.particles_vel, self.particles_last_pos,
                    self.particles_life, self.particles_time_on_wafer, self.particles_path_length, self.particles_arm_id,
                    sub_dt, omega, cos_t, sin_t,
                    self.viscosities, self.evap_rates, self.surface_tensions,
                    GRAVITY_MMS2, WAFER_RADIUS
                )

        if self.animation_state == STATE_RUNNING_PROCESS:
            if wall_time_in_proc >= current_process.get('total_duration', 0):
                self._handle_process_transition(current_process)
            elif self.active_arm_id and self.active_arm_id != 0:
                arm = self.arms.get(self.active_arm_id)
                self._calculate_physics_movement(current_process, arm, dt)
        
        elif self.animation_state == STATE_ARM_CHANGE_PAUSE:
            if self.simulation_time_elapsed - self.transition_start_time >= self.arm_change_pause_time:
                self._prepare_next_arm_move()
        elif self.animation_state == STATE_PAUSE_AT_CENTER:
            if self.simulation_time_elapsed - self.transition_start_time >= self.center_pause_time:
                self._prepare_move_center_to_start(current_process)
        else:
            self._handle_arm_transition(current_process)

        direction_mult = -1 if spin_dir == 'cw' else 1
        self.wafer_angle += (current_rpm / 60.0 * 360.0 * dt) * direction_mult
        
        rad = math.radians(self.wafer_angle)
        rot_matrix = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        base_notch = np.array([[WAFER_RADIUS, 0], [WAFER_RADIUS-NOTCH_DEPTH, NOTCH_HALF_WIDTH], [WAFER_RADIUS-NOTCH_DEPTH, -NOTCH_HALF_WIDTH]])
        self.current_notch_coords = np.dot(base_notch, rot_matrix.T)

        if self.simulation_mode == 'full':
            render_data = {arm_id: self._get_render_paths(arm_id, dt, current_rpm, spin_dir) for arm_id in self.arms.keys()}
        else:
            render_data = {}

        is_finished = False
        if self.current_process_index >= len(self.recipe['processes']) - 1:
            if self.is_looping_back:
                arm = self.arms.get(self.active_arm_id)
                if arm:
                    angle_diff = arm._get_angle_diff(self.transition_end_angle, self.transition_start_angle)
                    dist_arc = abs(angle_diff) * arm.arm_length
                    dur = max(0.05, dist_arc / (MAX_NOZZLE_SPEED_MMS * self.transition_arm_speed_ratio))
                    if (self.simulation_time_elapsed - self.transition_start_time) >= dur:
                        is_finished = True
                else:
                    is_finished = True
            elif wall_time_in_proc >= current_process.get('total_duration', 0):
                if self.active_arm_id == 0:
                    is_finished = True

        if not self.headless:
            self.simulation_time_elapsed += dt
        else:
            if not is_finished:
                self.simulation_time_elapsed += dt

        return {
            'time': self.simulation_time_elapsed,
            'state': self.animation_state,
            'active_arm_id': self.active_arm_id,
            'nozzle_pos': self.last_nozzle_pos,
            'wafer_angle': self.wafer_angle,
            'notch_coords': self.current_notch_coords,
            'rpm': current_rpm,
            'process_idx': self.current_process_index,
            'process_time_str': f"{max(0, wall_time_in_proc):.2f}s / {current_process.get('total_duration', 0):.2f}s",
            'step_str': self.current_step_label,
            'water_render': render_data,
            'is_spraying': (self.animation_state == STATE_RUNNING_PROCESS),
            'removed_particles': [],
            'is_finished': is_finished
        }

    def _calculate_physics_movement(self, process, arm, dt):
        self.cumulative_physics_time += dt
        cycle = process.get('physics_cycle_time', 0)
        segs = process.get('physics_segments', [])
        
        if cycle > 0 and segs:
            t_in = self.cumulative_physics_time % cycle
            t_acc = 0.0
            found = False
            for s in segs:
                if t_acc + s['t'] >= t_in:
                    dt_in_seg = t_in - t_acc
                    pos_pct = s['pi'] + s['vi'] * dt_in_seg + 0.5 * s['a'] * (dt_in_seg**2)
                    self.last_nozzle_pos = arm.percent_to_coords(pos_pct)
                    self.current_step_label = s['label']
                    found = True
                    break
                t_acc += s['t']
            
            if not found and segs:
                last = segs[-1]
                ds = last['t']
                self.last_nozzle_pos = arm.percent_to_coords(last['pi'] + last['vi']*ds + 0.5*last['a']*ds**2)

    def _spawn_particles(self, arm_id, dt, custom_pos=None):
        current_process = self.recipe['processes'][self.current_process_index]
        flow = current_process.get('flow_rate', 500.0)
        params = self.water_params.get(arm_id, {})
        st_val = params.get('surface_tension', 72.8)
        spread_base = 150.0 / (st_val + 5.0) 
        
        expected_particles = (flow * 0.5) * dt
        self._spawn_accumulator[arm_id] += expected_particles
        
        count = int(self._spawn_accumulator[arm_id])
        self._spawn_accumulator[arm_id] -= count 
        
        if count <= 0: return
        
        # 決定噴嘴基準位置
        nozzle_pos = custom_pos if custom_pos is not None else self.last_nozzle_pos
        
        # 尋找空位
        inactive_indices = np.where(self.particles_state == P_INACTIVE)[0]
        if len(inactive_indices) == 0: return
        
        spawn_count = min(count, len(inactive_indices))
        target_indices = inactive_indices[:spawn_count]
        
        # 批次初始化
        for idx in target_indices:
            self.particles_state[idx] = P_FALLING
            self.particles_life[idx] = 1.0
            self.particles_birth_time[idx] = self.simulation_time_elapsed
            self.particles_time_on_wafer[idx] = 0.0
            self.particles_path_length[idx] = 0.0
            self.particles_arm_id[idx] = arm_id
            self.particles_id[idx] = self.next_particle_id
            
            off = (np.random.rand(2) - 0.5) * spread_base
            self.particles_pos[idx] = [nozzle_pos[0]+off[0], nozzle_pos[1]+off[1], 15.0]
            self.particles_vel[idx] = [0.0, 0.0, -flow * 0.05]
            self.particles_last_pos[idx] = self.particles_pos[idx, :2]
            
            self.next_particle_id += 1

    def _get_render_paths(self, arm_id, dt, rpm, spin_dir):
        """
        產生渲染路徑。
        目前保持 Python 實作以處理動態點數，但底層存取 NumPy 陣列。
        """
        f_xy = []
        o_xy = []
        
        omega = (rpm / 60.0) * 2 * math.pi * (-1 if spin_dir == 'cw' else 1)
        tdt = omega * dt
        interp_steps = max(2, int(abs(math.degrees(tdt)) / 0.3)) 
        interp_steps = min(interp_steps, WATER_RENDER_INTERPOLATION_LIMIT)
        
        # 過濾該 arm 的活躍粒子
        mask = (self.particles_arm_id == arm_id) & (self.particles_state != P_INACTIVE)
        indices = np.where(mask)[0]
        
        for i in indices:
            state = self.particles_state[i]
            pos = self.particles_pos[i]
            last_p = self.particles_last_pos[i]
            
            if state == P_FALLING:
                ps, pe = last_p, pos[:2]
                dist = math.sqrt((pe[0]-ps[0])**2 + (pe[1]-ps[1])**2)
                f_steps = max(1, int(dist / 0.5))
                for j in range(1, f_steps + 1):
                    frac = j / f_steps
                    f_xy.append((ps[0] + (pe[0]-ps[0])*frac, ps[1] + (pe[1]-ps[1])*frac))
                    
            elif state == P_ON_WAFER:
                ps, pe = last_p, pos[:2]
                rs, re = math.sqrt(ps[0]**2 + ps[1]**2), math.sqrt(pe[0]**2 + pe[1]**2)
                ts = math.atan2(ps[1], ps[0])
                
                for j in range(1, interp_steps + 1):
                    frac = j / interp_steps
                    ri = rs + (re - rs) * frac
                    ti = ts + tdt * frac
                    
                    j_amt = WATER_JITTER_AMOUNT * (ri / WAFER_RADIUS)
                    jx = (random.random() - 0.5) * j_amt
                    jy = (random.random() - 0.5) * j_amt
                    
                    o_xy.append((ri * math.cos(ti) + jx, ri * math.sin(ti) + jy))
                
        return {
            'falling': np.array(f_xy) if f_xy else np.empty((0, 2)),
            'on_wafer': np.array(o_xy) if o_xy else np.empty((0, 2))
        }

    def _handle_process_transition(self, current_process_obj):
        arm = self.arms.get(self.active_arm_id)
        current_angle = arm.coords_to_angle(self.last_nozzle_pos) if arm else 0.0
        curr_time = self.simulation_time_elapsed

        next_idx = self.current_process_index + 1
        if next_idx >= len(self.recipe['processes']):
            if self.active_arm_id and self.active_arm_id != 0:
                self.animation_state = STATE_ARM_MOVE_TO_HOME
                self.transition_start_time = curr_time
                self.transition_start_angle = current_angle
                self.transition_end_angle = arm.home_angle
                self.is_looping_back = True
            else:
                if not self.headless:
                    self._reset_to_start()
            return

        prev_p = current_process_obj
        next_p = self.recipe['processes'][next_idx]
        self.current_process_index = next_idx

        if prev_p['arm_id'] == next_p['arm_id'] and prev_p['arm_id'] != 0:
            self.transition_start_time = curr_time
            self.transition_start_angle = current_angle
            if next_p.get('start_from_center'):
                self.animation_state = STATE_MOVING_TO_CENTER_ARC
                self.transition_end_angle = arm.coords_to_angle(arm.center_pos_coords)
            else:
                self.animation_state = STATE_ARM_MOVE_FROM_HOME
                target_pos = next_p['steps'][0]['pos'] if next_p.get('steps') else 0
                self.transition_end_angle = arm.percent_to_angle(target_pos)
        else:
            if prev_p['arm_id'] != 0 and arm:
                self.animation_state = STATE_ARM_MOVE_TO_HOME
                self.transition_start_time = curr_time
                self.transition_start_angle = current_angle
                self.transition_end_angle = arm.home_angle
            else:
                self._prepare_next_arm_move()

    def _handle_arm_transition(self, current_process):
        arm = self.arms.get(self.active_arm_id)
        if not arm: return
        
        angle_diff = arm._get_angle_diff(self.transition_end_angle, self.transition_start_angle)
        dist_arc = abs(angle_diff) * arm.arm_length
        dur = max(0.05, dist_arc / (MAX_NOZZLE_SPEED_MMS * self.transition_arm_speed_ratio))
        
        t = self.simulation_time_elapsed - self.transition_start_time
        
        if t >= dur:
            self.last_nozzle_pos = arm.angle_to_coords(self.transition_end_angle)
            if self.animation_state == STATE_ARM_MOVE_TO_HOME:
                if self.is_looping_back:
                    if not self.headless:
                        self._reset_to_start()
                else:
                    if self.recipe['processes'][self.current_process_index]['arm_id'] == 0:
                        self.animation_state, self.active_arm_id, self.time_offset_for_current_process = STATE_RUNNING_PROCESS, 0, self.simulation_time_elapsed
                    else:
                        self.animation_state, self.transition_start_time = STATE_ARM_CHANGE_PAUSE, self.simulation_time_elapsed
            elif self.animation_state == STATE_ARM_MOVE_FROM_HOME:
                if current_process.get('start_from_center'):
                    self.animation_state, self.transition_start_time = STATE_MOVING_TO_CENTER_ARC, self.simulation_time_elapsed
                    self.transition_start_angle, self.transition_end_angle = arm.coords_to_angle(self.last_nozzle_pos), arm.coords_to_angle(arm.center_pos_coords)
                else:
                    self.animation_state, self.time_offset_for_current_process, self.cumulative_physics_time = STATE_RUNNING_PROCESS, self.simulation_time_elapsed, 0.0
            elif self.animation_state == STATE_MOVING_TO_CENTER_ARC:
                self.animation_state, self.transition_start_time = STATE_PAUSE_AT_CENTER, self.simulation_time_elapsed
            elif self.animation_state == STATE_MOVING_FROM_CENTER_TO_START:
                self.animation_state, self.time_offset_for_current_process, self.cumulative_physics_time = STATE_RUNNING_PROCESS, self.simulation_time_elapsed, current_process.get('sfc_start_time', 0.0)
        else:
            frac = t / dur
            self.last_nozzle_pos = arm.get_interpolated_coords(self.transition_start_angle, self.transition_end_angle, frac)

    def _prepare_next_arm_move(self):
        next_p = self.recipe['processes'][self.current_process_index]
        self.active_arm_id = next_p['arm_id']
        arm = self.arms.get(self.active_arm_id)
        if arm:
            self.animation_state, self.transition_start_time = STATE_ARM_MOVE_FROM_HOME, self.simulation_time_elapsed
            self.transition_start_angle, self.last_nozzle_pos = arm.home_angle, arm.home_pos.copy()
            if next_p.get('start_from_center'):
                self.transition_end_angle = arm.coords_to_angle(arm.center_pos_coords)
            else:
                first_pos = next_p['steps'][0]['pos'] if next_p.get('steps') else 0
                self.transition_end_angle = arm.percent_to_angle(first_pos)
        else:
            self.animation_state, self.time_offset_for_current_process = STATE_RUNNING_PROCESS, self.simulation_time_elapsed

    def _prepare_move_center_to_start(self, current_process):
        self.animation_state, self.transition_start_time = STATE_MOVING_FROM_CENTER_TO_START, self.simulation_time_elapsed
        arm = self.arms[self.active_arm_id]
        self.transition_start_angle = arm.coords_to_angle(arm.center_pos_coords)
        target_idx = current_process.get('sfc_target_idx', 0)
        if current_process.get('steps'):
            target_pos = current_process['steps'][target_idx]['pos']
            self.transition_end_angle = arm.percent_to_angle(target_pos)
        else:
            self.transition_end_angle = self.transition_start_angle

    def _pre_calculate_physics(self):
        for process in self.recipe['processes']:
            arm_id = process.get('arm_id', 0)
            if not arm_id or arm_id == 0: continue
            arm = self.arms[arm_id]
            steps = process.get('steps', [])
            if len(steps) < 2: continue
            
            def create_segments(step_list, is_forward=True):
                segs = []
                for j in range(len(step_list) - 1):
                    p_i, p_f = float(step_list[j]['pos']), float(step_list[j+1]['pos'])
                    v_i_mag = (float(step_list[j].get('speed', 0)) / 100.0) * arm.max_percent_speed
                    v_f_mag = (float(step_list[j+1].get('speed', 0)) / 100.0) * arm.max_percent_speed
                    
                    dist = p_f - p_i
                    if abs(dist) < 1e-6:
                        continue
                    
                    direction = 1.0 if dist > 0 else -1.0
                    v_i = v_i_mag * direction
                    v_f = v_f_mag * direction
                    
                    v_avg_mag = (v_i_mag + v_f_mag) / 2.0
                    if v_avg_mag < 0.1:
                        v_avg_mag = 0.1
                    
                    t_d = abs(dist) / v_avg_mag
                    accel = (v_f - v_i) / t_d if t_d > 0 else 0
                    
                    label_from = j + 1 if is_forward else len(step_list) - j
                    label_to = j + 2 if is_forward else len(step_list) - j - 1
                    
                    segs.append({
                        'pi': p_i,
                        'vi': v_i,
                        'a': accel,
                        't': max(t_d, 0.01),
                        'label': f"Step {label_from}->{label_to}"
                    })
                return segs

            f_segs = create_segments(steps, is_forward=True)
            b_segs = create_segments(steps[::-1], is_forward=False)
            
            all_s = f_segs + b_segs
            process['physics_segments'], process['physics_cycle_time'] = all_s, sum(s['t'] for s in all_s)
            
            sfc_t, sfc_idx, min_d = 0.0, 0, float('inf')
            if process.get('start_from_center'):
                for k, step in enumerate(steps):
                    d = abs(step['pos'] - arm.center_pos_percent)
                    if d < min_d: min_d, sfc_idx = d, k
                
                sfc_t = 0.0
                for m in range(min(sfc_idx, len(f_segs))):
                    sfc_t += f_segs[m]['t']
                    
            process['sfc_start_time'], process['sfc_target_idx'] = sfc_t, sfc_idx

    def _get_rpm_at_time(self, process, time_in_proc):
        spin, total_d = process.get('spin_params', {}), process.get('total_duration', 1.0)
        if spin.get('mode', 'Simple') == 'Simple': return float(spin.get('rpm', 0))
        sr, er = float(spin.get('start_rpm', 0)), float(spin.get('end_rpm', 0))
        return sr + (er - sr) * max(0.0, min(1.0, time_in_proc / total_d)) if total_d > 0 else sr

    def _reset_to_start(self): self.reset()
