import os
import time
import mujoco
import mujoco.viewer
import numpy as np


# モデルの読み込みとデータの初期化
# XMLファイルを指定 '~' を絶対パスに変換する
xml_path = os.path.expanduser("~/mujoco/model/humanoid/humanoid.xml")
# モデルの読み込みとデータの初期化
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. ラジオ体操「深呼吸」の正確な動作を再現するポリシー関数
# まっすぐ上に腕を伸ばす深呼吸ポリシー関数
def radio_taiso_straight_arms_policy(model, data):
    action = np.zeros(model.nu)
    kp = 50.0  # 姿勢を保持・制御する力の強さ（比例ゲイン*）
    
    # 1サイクルの時間（秒）：上げるのに2秒、下ろすのに2秒 ＝ 計4秒
    cycle_time = 4.0
    total_cycles = 4
    max_duration = cycle_time * total_cycles
    
    current_time = data.time
    
    if current_time >= max_duration:
        # 4回完了した後は「気をつけ」の姿勢を維持する
        phase_up_down = 0.0
    else:
        # 1サイクルの中での経過時間を計算
        time_in_cycle = current_time % cycle_time
        
        if time_in_cycle < 2.0:
            # 前から上に上げるフェーズ（0秒 〜 2秒）： 0.0から1.0へ増加
            phase_up_down = time_in_cycle / 2.0
        else:
            # 横から下に下ろすフェーズ（2秒 〜 4秒）： 1.0から0.0へ減少
            phase_up_down = (4.0 - time_in_cycle) / 2.0
            
    for i in range(model.nu):
        trnid = model.actuator_trnid[i, 0]
        if trnid < 0:
            continue
            
        jnt_id = trnid
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        
        # 基本の「気をつけ（直立）」の角度を目標にする
        target_pos = model.qpos0[model.jnt_qposadr[jnt_id]]
        
        # 肩の関節を利用し、腕をまっすぐ上まで引き上げるため、加算値を「2.2」に大きくする
        if jnt_name and 'shoulder' in jnt_name:
            target_pos += 2.2 * phase_up_down
            
        current_pos = data.qpos[model.jnt_qposadr[jnt_id]]
        ctrl_val = -kp * (current_pos - target_pos)
        
        ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
        action[i] = np.clip(ctrl_val, ctrl_min, ctrl_max)
        
    return action

# 3. シミュレーションと可視化の実行ループ
with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetData(model, data)
    
    while viewer.is_running():
        step_start = time.time()
        
        # ポリシーから制御入力を決定
        action = radio_taiso_straight_arms_policy(model, data)
        data.ctrl[:] = action
        
        # 物理シミュレーションを進める
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # 実時間に合わせるための待機処理
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
