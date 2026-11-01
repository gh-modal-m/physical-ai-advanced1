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
        # Humanoid-v5の標準的な肩のモーター名（環境によって異なる場合はエラーを防ぐためNoneを許容）
        self.r_shoulder_pitch_id = self._get_actuator_id('shoulder_1_right')
        self.r_shoulder_roll_id  = self._get_actuator_id('shoulder_2_right')
        self.l_shoulder_pitch_id = self._get_actuator_id('shoulder_1_left')
        self.l_shoulder_roll_id  = self._get_actuator_id('shoulder_2_left')

    def _get_actuator_id(self, name):
        import mujoco
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        except ValueError:
            return None

    def step(self, action):
        # 1. 通常の環境ステップを進める
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        
        # 2. 現在のシミュレーション時間から「深呼吸の理想の関節角度」を計算
        t = self.data.time
        
        # 理想の制御値
        target_pitch = -0.6 * np.sin(self.omega * t) - 0.2
        target_roll  = 0.5 * (1.0 - np.cos(self.omega * t))
        
        # 3. 報酬の再計算
        tracking_penalty = 0.0
        valid_motors = 0
        
        # 各肩のモーターの入力がターゲットにどれだけ近いかを評価（安全に存在するものだけ計算）
        for motor_id, target in [
            (self.r_shoulder_pitch_id, target_pitch),
            (self.l_shoulder_pitch_id, target_pitch),
            (self.r_shoulder_roll_id, -target_roll),
            (self.l_shoulder_roll_id, target_roll)
        ]:
            if motor_id is not None and motor_id < len(action):
                tracking_penalty += (action[motor_id] - target) ** 2
                valid_motors += 1
        
        # モーターが見つからない場合のゼロ除算を防ぐ
        if valid_motors > 0:
            tracking_penalty /= valid_motors

        # 4. 新しい報酬関数の設計
        imitation_reward = np.exp(-2.0 * tracking_penalty)  # ぴったり一致で1.0に近づく
        
        # 直立を維持する報酬と深呼吸の軌道をなぞる報酬をブレンド
        custom_reward = 0.3 * original_reward + 0.7 * imitation_reward
        
        return observation, custom_reward, terminated, truncated, info

# --- 環境の生成とテスト実行の例 ---
if __name__ == "__main__":
    # Humanoid-v5 環境を呼び出し（描画モードをhumanに）
    base_env = gym.make("Humanoid-v5", render_mode="human")
    # 学習時は render_mode を指定しない（または None にする）
    #base_env = gym.make("Humanoid-v5", render_mode=None)

    env = RadioTaisoDeepBreathingWrapper(base_env)
    
    obs, info = env.reset()
    
    for _ in range(1000):
        # テストとしてのランダム行動
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
            
    env.close()
