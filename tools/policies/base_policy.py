from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    """
    所有推理策略的抽象基类。

    obs 格式（来自仿真 ZMQ obs topic，与 act_policy.ACTPolicy 一致）:
        obs['puppet']['arm_left_position_raw']['data']            (7,)  左臂关节角
        obs['puppet']['arm_right_position_raw']['data']           (7,)  右臂关节角
        obs['puppet']['end_effector_left_position_raw']['data']   (6,)  左手指（实测，sim 不可靠）
        obs['puppet']['end_effector_right_position_raw']['data']  (6,)  右手指（同上）
        obs['camera_observations']['color_images']['camera_head'] (H,W,3) uint8 RGB, 原生 1280×720

    infer 返回: np.ndarray, shape=[action_dim]，顺序为
        [left_arm(7), left_hand(6), right_arm(7), right_hand(6)]
        或
        [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
        取决于具体机器人配置
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self):
        """加载模型权重，返回 policy 对象。"""
        ...

    @abstractmethod
    def reset(self):
        """每个 episode 开始时重置内部状态（chunk buffer、action queue 等）。"""
        ...

    @abstractmethod
    def infer(self, obs: dict) -> np.ndarray:
        """输入标准化 obs dict，输出单步 action array。"""
        ...
