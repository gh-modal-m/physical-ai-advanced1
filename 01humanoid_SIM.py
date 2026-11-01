import os
import time
import mujoco
import mujoco.viewer
import numpy as np

# 1. モデルの読み込みとデータの初期化
# XMLファイルを指定 '~' を絶対パスに変換する
xml_path = os.path.expanduser("~/mujoco/model/humanoid/humanoid.xml")
# モデルの読み込みとデータの初期化
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. ヒューマノイドを立たせるためのポリシー（簡易的なP制御）
def humanoid_policy(model, data):
    # 直立姿勢を維持するため、初期状態（qpos0）を目標角度とし、
    # 現在の角度とのズレに比例した力を各関節のモーター（アクチュエータ）に与えます。
    
    action = np.zeros(model.nu)
    kp = 50.0  # 姿勢を戻そうとする力の強さ（比例ゲイン*）
    
    for i in range(model.nu):
        # 各アクチュエータが対応する関節IDを取得
        trnid = model.actuator_trnid[i, 0]
        if trnid < 0:
            continue
            
        # 関節の現在の位置と、目標位置（初期位置）の差を計算
        current_pos = data.qpos[model.jnt_qposadr[trnid]]
        target_pos = model.qpos0[model.jnt_qposadr[trnid]]
        
        # ズレを埋めるための制御トルクを計算
        ctrl_val = -kp * (current_pos - target_pos)
        
        # モーターの出力制限（可動範囲）内に収める
        ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
        action[i] = np.clip(ctrl_val, ctrl_min, ctrl_max)
        
    return action

# 3. シミュレーションと可視化の実行ループ
with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetData(model, data)
    
    while viewer.is_running():
        step_start = time.time()
        
        # ポリシーから人型ロボットへの行動（制御入力）を決定
        action = humanoid_policy(model, data)
        
        # 制御入力をアクチュエータに適用
        data.ctrl[:] = action
        
        # 物理シミュレーションを1ステップ進める
        mujoco.mj_step(model, data)
        
        # 画面の更新
        viewer.sync()
        
        # 実時間に同期するための待機処理
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
