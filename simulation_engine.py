import numpy as np
import math
import random
from constants import *

class SimulationEngine:
    def __init__(self, recipe, arms_dict, water_params_dict, headless=False):
        self.recipe = recipe
        self.arms = arms_dict
        self.water_params = water_params_dict
        self.headless = headless # 新增 headless 標記
        
        self.particle_systems = {arm_id: [] for arm_id in arms_dict.keys()}
        self.next_particle_id = 0
        # 精確生成累加器
        self._spawn_accumulator = {arm_id: 0.0 for arm_id in arms_dict.keys()}
        
        self._pre_calculate_physics()
        
        self.current_notch_coords = np.array([[WAFER_RADIUS, 0],
                                              [WAFER_RADIUS-NOTCH_DEPTH, NOTCH_HALF_WIDTH],
                                              [WAFER_RADIUS-NOTCH_DEPTH, -NOTCH_HALF_WIDTH]])
        self.reset()

    def reset(self):
        self.simulation_time_elapsed = 0.0
        self.current_process_index = 0
        self.time_offset_for_current_process = 0.0
        self.cumulative_physics_time = 0.0
        self.wafer_angle = 0.0
        self.is_looping_back = False
        self.current_step_label = "Init"
        self.current_rpm_value = 0.0
        
        for arm_id in self.particle_systems:
            self.particle_systems[arm_id] = []
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
        # 紀錄上一影格噴嘴位置，用於生成插值
        self.prev_nozzle_pos = self.last_nozzle_pos.copy()
        
        # 安全取得當前製程
        if self.current_process_index < len(self.recipe['processes']):
            current_process = self.recipe['processes'][self.current_process_index]
        else:
            current_process = {'total_duration': 0, 'steps': []}

        # Process Time 計時
        if self.animation_state == STATE_RUNNING_PROCESS:
            wall_time_in_proc = self.simulation_time_elapsed - self.time_offset_for_current_process
        else:
            wall_time_in_proc = 0.0

        # RPM 計算
        if self.animation_state == STATE_RUNNING_PROCESS:
            self.current_rpm_value = self._get_rpm_at_time(current_process, wall_time_in_proc)
        
        current_rpm = self.current_rpm_value
        spin_dir = self.recipe.get('spin_dir', 'cw')

        # 物理模擬 (效能優化：降低子步數至上限 10 步，大幅提升順暢度)
        removed_this_frame = []
        SUB_STEPS = min(10, 5 + int(current_rpm / 300)) 
        sub_dt = dt / SUB_STEPS
        
        for i in range(SUB_STEPS):
            # 1. 分步生成粒子 (確保流體連貫性)
            if (self.active_arm_id and self.active_arm_id != 0) and (self.animation_state == STATE_RUNNING_PROCESS):
                # 計算子步內的噴嘴位置插值
                frac = (i + 1) / SUB_STEPS
                interp_pos = self.prev_nozzle_pos + (self.last_nozzle_pos - self.prev_nozzle_pos) * frac
                self._spawn_particles(self.active_arm_id, sub_dt, interp_pos)

            # 2. 物理步進
            removed = self._physics_step(sub_dt, current_rpm, spin_dir, current_process, wall_time_in_proc)
            removed_this_frame.extend(removed)

        # 狀態機
        if self.animation_state == STATE_RUNNING_PROCESS:
            if wall_time_in_proc >= current_process.get('total_duration', 0):
                self._handle_process_transition(current_process)
            elif self.active_arm_id and self.active_arm_id != 0:
                arm = self.arms.get(self.active_arm_id)
                self._calculate_physics_movement(current_process, arm, dt)
        
        elif self.animation_state == STATE_ARM_CHANGE_PAUSE:
            if self.simulation_time_elapsed - self.transition_start_time >= ARM_CHANGE_PAUSE_TIME:
                self._prepare_next_arm_move()
                
        elif self.animation_state == STATE_PAUSE_AT_CENTER:
            if self.simulation_time_elapsed - self.transition_start_time >= CENTER_PAUSE_TIME:
                self._prepare_move_center_to_start(current_process)
                
        else:
            self._handle_arm_transition(current_process)

        # 晶圓旋轉
        direction_mult = -1 if spin_dir == 'cw' else 1
        self.wafer_angle += (current_rpm / 60.0 * 360.0 * dt) * direction_mult
        
        # Notch 計算
        rad = math.radians(self.wafer_angle)
        rot_matrix = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        base_notch = np.array([[WAFER_RADIUS, 0], [WAFER_RADIUS-NOTCH_DEPTH, NOTCH_HALF_WIDTH], [WAFER_RADIUS-NOTCH_DEPTH, -NOTCH_HALF_WIDTH]])
        self.current_notch_coords = np.dot(base_notch, rot_matrix.T)

        render_data = {arm_id: self._get_render_paths(arm_id, dt, current_rpm, spin_dir) for arm_id in self.arms.keys()}

        # 【極致精簡結束判定】
        # 判斷是否所有製程都已結束 (包含最後一個製程結束後的手臂收回動作)
        is_finished = False
        if self.current_process_index >= len(self.recipe['processes']) - 1:
            if self.is_looping_back:
                # 已經在最後的歸位階段
                arm = self.arms.get(self.active_arm_id)
                if arm:
                    angle_diff = arm._get_angle_diff(self.transition_end_angle, self.transition_start_angle)
                    dist_arc = abs(angle_diff) * arm.arm_length
                    dur = max(0.05, dist_arc / (MAX_NOZZLE_SPEED_MMS * 0.8))
                    if (self.simulation_time_elapsed - self.transition_start_time) >= dur:
                        is_finished = True
                else:
                    is_finished = True
            elif wall_time_in_proc >= current_process.get('total_duration', 0):
                # 剛完成最後一個製程的 duration (或是無手臂製程)
                if self.active_arm_id == 0:
                    is_finished = True
                # 若 active_arm_id != 0，則等待下一幀進入 is_looping_back 狀態

        if not self.headless:
            self.simulation_time_elapsed += dt
        else:
            # 在導出模式下，如果還沒結束，時間才繼續跑
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
            'removed_particles': [], # 報表所需的空列表預留，避免報錯
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

    def _physics_step(self, dt, rpm, spin_dir, current_process, wall_time):
        removed_particles = []
        omega = (rpm / 60.0) * 2 * math.pi * (-1 if spin_dir == 'cw' else 1)
        d_theta = omega * dt
        cos_t, sin_t = math.cos(d_theta), math.sin(d_theta)
        
        for arm_id, particles in self.particle_systems.items():
            new_particles = []
            params = self.water_params.get(arm_id, {})
            visc = params.get('viscosity', 1.0)
            evap = params.get('evaporation_rate', 0.0)
            st_val = params.get('surface_tension', 72.8)
            
            for p in particles:
                # 蒸發邏輯 (簡單生命值機制)
                if evap > 0:
                    p['life'] = p.get('life', 1.0) - evap * dt
                    if p['life'] <= 0:
                        continue

                p['last_pos'] = p['pos'][:2].copy()
                if p['state'] == 'falling':
                    p['vel'][2] -= GRAVITY_MMS2 * dt
                    p['pos'] += p['vel'] * dt
                    if p['pos'][2] <= 0:
                        if np.linalg.norm(p['pos'][:2]) <= WAFER_RADIUS:
                            p['state'], p['pos'][2], p['vel'][:2] = 'on_wafer', 0, p['vel'][:2]*0.1
                        else: continue
                elif p['state'] == 'on_wafer':
                    # 粒子路徑追蹤邏輯 (報表所需)
                    p['time_on_wafer'] = p.get('time_on_wafer', 0.0) + dt
                    
                    dist = np.linalg.norm(p['pos'][:2])
                    if dist > 1e-4:
                        # 1. 基礎離心加速度
                        centrifugal_acc = (p['pos'][:2]/dist) * (omega**2 * dist)
                        
                        # 2. 表面張力產生一個徑向向心的力
                        st_resistance = -(p['pos'][:2]/dist) * (st_val * 0.3)
                        
                        # 合力加速度
                        total_acc = centrifugal_acc + st_resistance
                        
                        # 更新速度
                        old_vel = p['vel'][:2].copy()
                        p['vel'][:2] += total_acc * dt
                        
                        # 黏度阻尼 (影響流速)
                        p['vel'][:2] *= (1.0 - 0.05 * visc * dt)
                        
                    # 更新位置
                    old_xy = p['pos'][:2].copy()
                    p['pos'][:2] += p['vel'][:2] * dt
                    
                    # 更新路徑長度 (報表所需)
                    move_dist = np.linalg.norm(p['pos'][:2] - old_xy)
                    p['path_length'] = p.get('path_length', 0.0) + move_dist
                    
                    # 旋轉補償
                    x, y = p['pos'][0], p['pos'][1]
                    p['pos'][0], p['pos'][1] = x*cos_t - y*sin_t, x*sin_t + y*cos_t

                # 強化移除邏輯：一旦超出晶圓邊界一定距離，或掉落太深，立刻移除
                is_out_of_bounds = np.linalg.norm(p['pos'][:2]) > (WAFER_RADIUS + 20)
                is_too_deep = p['pos'][2] < -10
                
                if not (is_out_of_bounds or is_too_deep):
                    new_particles.append(p)
                else:
                    removed_particles.append(p)
            self.particle_systems[arm_id] = new_particles
        return removed_particles

    def _spawn_particles(self, arm_id, dt, custom_pos=None):
        """
        優化版粒子生成：
        1. 採用體積累加制，確保生成量精確。
        2. 支援噴嘴位置插值 (custom_pos)，消除低轉速下的成串感。
        """
        current_process = self.recipe['processes'][self.current_process_index]
        flow = current_process.get('flow_rate', 500.0)
        params = self.water_params.get(arm_id, {})
        st_val = params.get('surface_tension', 72.8)
        spread_base = 150.0 / (st_val + 5.0) 
        
        expected_particles = (flow * 0.5) * dt
        self._spawn_accumulator[arm_id] += expected_particles
        
        count = int(self._spawn_accumulator[arm_id])
        self._spawn_accumulator[arm_id] -= count 
        
        # 決定噴嘴基準位置
        nozzle_pos = custom_pos if custom_pos is not None else self.last_nozzle_pos

        for _ in range(count):
            off = (np.random.rand(2) - 0.5) * spread_base
            
            self.particle_systems[arm_id].append({
                'id': self.next_particle_id, 
                'state': 'falling',
                'life': 1.0,
                'birth_time': self.simulation_time_elapsed,
                'time_on_wafer': 0.0,
                'path_length': 0.0,
                'pos': np.array([nozzle_pos[0]+off[0], nozzle_pos[1]+off[1], 15.0]),
                'vel': np.array([0.0, 0.0, -flow * 0.05])
            })
            self.next_particle_id += 1

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
                # 在 headless 模式下，禁止重頭模擬
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
        dur = max(0.05, dist_arc / (MAX_NOZZLE_SPEED_MMS * 0.8))
        
        t = self.simulation_time_elapsed - self.transition_start_time
        
        if t >= dur:
            self.last_nozzle_pos = arm.angle_to_coords(self.transition_end_angle)
            if self.animation_state == STATE_ARM_MOVE_TO_HOME:
                if self.is_looping_back:
                    # 在 headless 模式下，即便標記為回原點完成，也不要觸發 reset_to_start
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
        """
        預先計算手臂移動路徑，實現平滑的線性加速/減速。
        """
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
                    
                    # 計算平均速度以求得時間 (假設線性加速度)
                    v_avg_mag = (v_i_mag + v_f_mag) / 2.0
                    # 避免除以零或極慢速導致卡死
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
            # 乒乓往返：從最後一步回到第一步
            b_segs = create_segments(steps[::-1], is_forward=False)
            
            all_s = f_segs + b_segs
            process['physics_segments'], process['physics_cycle_time'] = all_s, sum(s['t'] for s in all_s)
            
            sfc_t, sfc_idx, min_d = 0.0, 0, float('inf')
            if process.get('start_from_center'):
                for k, step in enumerate(steps):
                    d = abs(step['pos'] - arm.center_pos_percent)
                    if d < min_d: min_d, sfc_idx = d, k
                
                # 計算從第一個 f_seg 開始累計到目標 step 的時間
                # 注意：這裡假設 start_from_center 是從正向路徑切入
                sfc_t = 0.0
                for m in range(min(sfc_idx, len(f_segs))):
                    sfc_t += f_segs[m]['t']
                    
            process['sfc_start_time'], process['sfc_target_idx'] = sfc_t, sfc_idx

    def _get_render_paths(self, arm_id, dt, rpm, spin_dir):
        """
        大幅強化渲染連貫性：極致解析度版
        """
        f_xy = []
        for p in self.particle_systems[arm_id]:
            if p['state'] == 'falling':
                ps, pe = p.get('last_pos', p['pos'][:2]), p['pos'][:2]
                dist = np.linalg.norm(pe - ps)
                f_steps = max(1, int(dist / 0.5)) # 提高解析度
                for i in range(1, f_steps + 1):
                    f_xy.append(tuple(ps + (pe - ps) * (i / f_steps)))

        o_xy = []
        omega = (rpm / 60.0) * 2 * math.pi * (-1 if spin_dir == 'cw' else 1)
        tdt = omega * dt
        
        # 效能平衡：每 0.3 度一個採樣點
        interp_steps = max(2, int(abs(math.degrees(tdt)) / 0.3)) 
        interp_steps = min(interp_steps, WATER_RENDER_INTERPOLATION_LIMIT)

        for p in self.particle_systems[arm_id]:
            if p['state'] != 'on_wafer': continue
            
            ps, pe = p.get('last_pos', p['pos'][:2]), p['pos'][:2]
            rs, re = np.linalg.norm(ps), np.linalg.norm(pe)
            ts = math.atan2(ps[1], ps[0])
            
            for i in range(1, interp_steps + 1):
                frac = i / interp_steps
                ri = rs + (re - rs) * frac
                ti = ts + tdt * frac
                
                # 徑向隨機度，打破同心圓
                j_amt = WATER_JITTER_AMOUNT * (ri / WAFER_RADIUS) # 越邊緣擾動越大
                jx = (random.random() - 0.5) * j_amt
                jy = (random.random() - 0.5) * j_amt
                
                o_xy.append((ri * math.cos(ti) + jx, ri * math.sin(ti) + jy))
                
        return {'falling': f_xy, 'on_wafer': o_xy}

    def _get_rpm_at_time(self, process, time_in_proc):
        spin, total_d = process.get('spin_params', {}), process.get('total_duration', 1.0)
        if spin.get('mode', 'Simple') == 'Simple': return float(spin.get('rpm', 0))
        sr, er = float(spin.get('start_rpm', 0)), float(spin.get('end_rpm', 0))
        return sr + (er - sr) * max(0.0, min(1.0, time_in_proc / total_d)) if total_d > 0 else sr

    def _reset_to_start(self): self.reset()
