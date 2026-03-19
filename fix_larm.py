import os
import glob
import re

def fix_dispense_arm_calls(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # We need to replace TREOS specific DispenseArm logic with LArm specific one.
    # Pattern to look for DispenseArm creations using ARM_GEOMETRIES

    # In video_generator.py
    if "video_generator.py" in file_path:
        content = re.sub(
            r"if i == 2:\s*side_arm_line, = ax\.plot\(\[\], \[\], color='gray', lw=4, zorder=12\)\s*side_nozzle_head = plt\.Circle\(\(0, 0\), 10, facecolor='yellow', zorder=13\)\s*ax\.add_patch\(side_nozzle_head\)\s*arms\[i\] = DispenseArm\(i, geo\['pivot'\], geo\['home'\], geo\['length'\], \s*arm_line, nozzle_head, \s*side_arm_length=geo\.get\('side_arm_length'\), \s*side_arm_angle_offset=geo\.get\('side_arm_angle_offset'\),\s*side_arm_branch_dist=geo\.get\('side_arm_branch_dist'\),\s*side_arm_artist=side_arm_line, side_nozzle_artist=side_nozzle_head\)\s*else:\s*arms\[i\] = DispenseArm\(i, geo\['pivot'\], geo\['home'\], geo\['length'\], arm_line, nozzle_head\)",
            "arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], arm_line, nozzle_head)",
            content,
            flags=re.MULTILINE
        )

    # In app.py
    if "app.py" in file_path:
        content = re.sub(
            r"if i == 2:\s*self\.arms\[i\] = DispenseArm\(i, geo\['pivot'\], geo\['home'\], geo\['length'\],\s*arm_line, nozzle_head,\s*side_arm_length=geo\.get\('side_arm_length'\),\s*side_arm_angle_offset=geo\.get\('side_arm_angle_offset'\),\s*side_arm_branch_dist=geo\.get\('side_arm_branch_dist'\),\s*side_arm_artist=side_arm_line, side_nozzle_artist=side_nozzle_head\)\s*else:\s*self\.arms\[i\] = DispenseArm\(i, geo\['pivot'\], geo\['home'\], geo\['length'\], arm_line, nozzle_head\)",
            "self.arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], arm_line, nozzle_head)",
            content,
            flags=re.MULTILINE
        )
        # Also clean up side_arm_line and side_nozzle_head creations if any
        content = re.sub(
            r"if i == 2:\s*side_arm_line, = self\.ax\.plot\(\[\], \[\], color='gray', lw=4, zorder=12\)\s*side_nozzle_head = plt\.Circle\(\(0, 0\), 10, facecolor='yellow', zorder=13\)\s*self\.ax\.add_patch\(side_nozzle_head\)\s*self\.arms\[i\] =",
            "self.arms[i] =",
            content,
            flags=re.MULTILINE
        )
        content = re.sub(
            r"side_arm_line, = self\.ax\.plot\(\[\], \[\], color='gray', lw=4, zorder=12\)\s*side_nozzle_head = plt\.Circle\(\(0, 0\), 10, facecolor='yellow', zorder=13\)\s*self\.ax\.add_patch\(side_nozzle_head\)",
            "",
            content,
            flags=re.MULTILINE
        )


    # In the generators (etchingamount, PRE, charging)
    for gen in ["etchingamount_generator.py", "PRE_generator.py", "charging_generator.py", "accu_heatmap_generator.py"]:
        if gen in file_path:
            content = re.sub(
                r"if i == 2:\s*headless_arms\[i\] = DispenseArm\(i, geo\['pivot'\], geo\['home'\], geo\['length'\], None, None,\s*side_arm_length=geo\.get\('side_arm_length'\), \s*side_arm_angle_offset=geo\.get\('side_arm_angle_offset'\),\s*side_arm_branch_dist=geo\.get\('side_arm_branch_dist'\)\)\s*else:\s*headless_arms\[i\] = DispenseArm\(i, geo\['pivot'\], geo\['home'\], geo\['length'\], None, None\)",
                "headless_arms[i] = DispenseArm(i, geo['pivot'], geo['home'], geo['length'], geo['p_start'], geo['p_end'], None, None)",
                content,
                flags=re.MULTILINE
            )
            # Fix actual_flow logic in generators:
            content = re.sub(
                r"actual_flow = current_proc\.get\('flow_rate_2' if p_arm_id == 3 else 'flow_rate', 500\.0\)",
                "actual_flow = current_proc.get('flow_rate', 500.0)",
                content,
                flags=re.MULTILINE
            )

    # In simulation_engine.py
    if "simulation_engine.py" in file_path:
        # Revert the multiple sources logic back to simple logic for LArm
        # actually, LArm has 3 arms, TREOS has 2.
        # In TREOS simulation_engine:
        # sources = []
        # if arm_id == 2:
        #   if isinstance(nozzle_end, list) and len(nozzle_end) == 2: ...
        # This whole block is wrong for LArm.
        content = re.sub(
            r"if arm_id == 2:\s*# 處理雙噴嘴.*?else:\s*sources\.append\(\{'id': arm_id, 'flow': current_proc.*?\}\)",
            "sources.append({'id': arm_id, 'flow': current_process.get('flow_rate', 500.0), 'start_pos': nozzle_start, 'end_pos': nozzle_end})",
            content,
            flags=re.DOTALL
        )
        content = re.sub(
            r"current_flows = \{1: 0\.0, 2: 0\.0, 3: 0\.0\}\s*if self\.animation_state == STATE_RUNNING_PROCESS:\s*if self\.active_arm_id == 1:\s*current_flows\[1\] = current_process\.get\('flow_rate', 0\.0\)\s*elif self\.active_arm_id == 2:\s*current_flows\[2\] = current_process\.get\('flow_rate', 0\.0\)\s*current_flows\[3\] = current_process\.get\('flow_rate_2', 0\.0\)",
            "current_flows = {1: 0.0, 2: 0.0, 3: 0.0}\n        if self.animation_state == STATE_RUNNING_PROCESS:\n            current_flows[self.active_arm_id] = current_process.get('flow_rate', 0.0)",
            content,
            flags=re.MULTILINE
        )


    # In app.py - flow_rate_2 ui updates
    if "app.py" in file_path:
        content = re.sub(
            r"gui_proc\['flow_rate_var_2'\] = tk\.StringVar\(\)\s*gui_proc\['flow_rate_var_2'\]\.set\('1500'\)\s*ttk\.Label\(frame_arm2, text=\"Nozzle 3 \(Side\) Flow \(cc/min\):\"\)\.grid.*?ttk\.Entry\(frame_arm2, textvariable=gui_proc\['flow_rate_var_2'\], width=10\)\.grid.*?\n",
            "",
            content,
            flags=re.DOTALL
        )
        content = re.sub(
            r"if 'flow_rate_2' in gui_proc:\s*proc_data\['flow_rate_2'\] = float\(gui_proc\['flow_rate_var_2'\]\.get\(\)\)",
            "",
            content,
            flags=re.MULTILINE
        )
        content = re.sub(
            r"if 'flow_rate_2' in proc_data:\s*gui_proc\['flow_rate_var_2'\]\.set\(proc_data\['flow_rate_2'\]\)\s*else:\s*gui_proc\['flow_rate_var_2'\]\.set\('1500'\) # 預設值",
            "",
            content,
            flags=re.MULTILINE
        )


    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

for py_file in glob.glob("/tmp/LArm_upgrade/*.py"):
    fix_dispense_arm_calls(py_file)
    
print("Fix script applied.")
