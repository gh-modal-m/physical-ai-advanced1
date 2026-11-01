import gymnasium as gym
import numpy as np

class RadioTaisoDeepBreathingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 物理モデルの情報を取得
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        
        # 深呼吸の周期設定（4秒で1往復）
        self.period = 4.0
        self.omega = (2 * np.pi) / self.period
        
        # 各アクチュエータ（モーター）のインデックスを取得
        self.r_shoulder_pitch_id = self._get_actuator_id('shoulder_1_right')
        self.r_shoulder_roll_id  = self._get_actuator_id('shoulder_2_right')
        self.l_shoulder_pitch_id = self._get_actuator_id('shoulder_1_left')
        self.l_shoulder_roll_id  = self._get_actuator_id('shoulder_2_left')
        
        # 初期状態のルート位置（高さなど）を保持しておく変数
        self.initial_root_z = 1.4  

    def _get_actuator_id(self, name):
        import mujoco
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        except ValueError:
            return None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # リセット時にもルートのの高さを安定させる
        if self.data.qpos.shape[0] >= 3:
            self.data.qpos[2] = self.initial_root_z
        return observation, info

    def step(self, action):
        # 1. 転倒対策：毎フレーム、ルート座標（位置と姿勢）を固定して重力で崩れ落ちるのを防ぐ
        if self.data.qpos.shape[0] >= 7:
            # 位置(x, y, z)の z を固定し、速度もゼロにする
            self.data.qpos[0] = 0.0  # X
            self.data.qpos[1] = 0.0  # Y
            self.data.qpos[2] = self.initial_root_z  # Z (高さ)
            
            # ルートの速度も安定させるためにゼロクリア
            if self.data.qvel.shape[0] >= 6:
                self.data.qvel[0:6] = 0.0

        # 2. 現在のシミュレーション時間から「深呼吸の理想の関節角度」を計算
        t = self.data.time
        target_pitch = -0.6 * np.sin(self.omega * t) - 0.2
        target_roll  = 0.5 * (1.0 - np.cos(self.omega * t))
        
        # 3. アクチュエータ（モーター）の強制制御
        # 腕の動きをきれいに見せるため、ターゲットの角度を直接モーターの制御値（ctrl）に反映させる
        if self.r_shoulder_pitch_id is not None and self.r_shoulder_pitch_id < len(self.data.ctrl):
            self.data.ctrl[self.r_shoulder_pitch_id] = target_pitch
        if self.l_shoulder_pitch_id is not None and self.l_shoulder_pitch_id < len(self.data.ctrl):
            self.data.ctrl[self.l_shoulder_pitch_id] = target_pitch
        if self.r_shoulder_roll_id is not None and self.r_shoulder_roll_id < len(self.data.ctrl):
            self.data.ctrl[self.r_shoulder_roll_id]  = -target_roll
        if self.l_shoulder_roll_id is not None and self.l_shoulder_roll_id < len(self.data.ctrl):
            self.data.ctrl[self.l_shoulder_roll_id]  = target_roll

        # 4. 通常の環境ステップを進める
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        
        # トラッキングの評価（ペナルティ計算）
        tracking_penalty = 0.0
        valid_motors = 0
        for motor_id, target in [
            (self.r_shoulder_pitch_id, target_pitch),
            (self.l_shoulder_pitch_id, target_pitch),
            (self.r_shoulder_roll_id, -target_roll),
            (self.l_shoulder_roll_id, target_roll)
        ]:
            if motor_id is not None and motor_id < len(action):
                tracking_penalty += (action[motor_id] - target) ** 2
                valid_motors += 1
        
        if valid_motors > 0:
            tracking_penalty /= valid_motors

        imitation_reward = np.exp(-2.0 * tracking_penalty)
        custom_reward = 0.3 * original_reward + 0.7 * imitation_reward
        
        return observation, custom_reward, terminated, truncated, info

# --- 環境の生成とテスト実行の例 ---
if __name__ == "__main__":
    # Humanoid-v5 環境を呼び出し（描画モードをhumanに）
    base_env = gym.make("Humanoid-v5", render_mode="human")
    env = RadioTaisoDeepBreathingWrapper(base_env)
    
    obs, info = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
            
    env.close()


