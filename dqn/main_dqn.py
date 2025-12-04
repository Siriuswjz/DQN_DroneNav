import gym
import time
import yaml
import torch
import torch.nn as nn
from torchvision import models

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VGG16FeaturesExtractor(BaseFeaturesExtractor):
    """
    使用VGG16作为特征提取器
    """
    def __init__(self, observation_space, features_dim=512, use_pretrained=True):
        super().__init__(observation_space, features_dim)
        
        # 加载VGG16
        vgg16 = models.vgg16(pretrained=use_pretrained)
        
        # 只使用VGG16的特征提取部分（卷积层）
        self.features = vgg16.features
        
        # 冻结预训练权重（可选）
        if use_pretrained:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # 自适应池化，将任意尺寸输出变为固定尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # observations shape: (batch, channels, height, width)
        x = self.features(observations)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Get train environment configs
with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# Create a DummyVecEnv
env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:airsim-env-v0", 
        ip_address="127.0.0.1", 
        image_shape=(50,50,3),
        env_config=env_config["TrainEnv"]
    )
)])

# Wrap env as VecTransposeImage (Channel last to channel first)
env = VecTransposeImage(env)

# Policy kwargs with custom feature extractor
policy_kwargs = dict(
    features_extractor_class=VGG16FeaturesExtractor,
    features_extractor_kwargs=dict(
        features_dim=512,
        use_pretrained=True  
    ),
)

# Initialize DQN with VGG16
model = DQN(
    'CnnPolicy',
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device="cuda",
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=10000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    tensorboard_log="./tb_logs/",
)

# Evaluation callback
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=4,
    best_model_save_path="./dqn",
    log_path="./dqn",
    eval_freq=500,
)

callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

log_name = "dqn_vgg16_run_" + str(time.time())

print("开始DQN训练（使用VGG16特征提取器）...")
model.learn(
    total_timesteps=280000,
    tb_log_name=log_name,
    **kwargs
)

# Save policy weights
model.save("dqn/dqn_vgg16_navigation_policy")
print("训练完成！模型已保存。")
