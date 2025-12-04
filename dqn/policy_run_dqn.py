import gym
import yaml
import time
import torch
import torch.nn as nn
from torchvision import models

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VGG16FeaturesExtractor(BaseFeaturesExtractor):
    """
    使用VGG16作为特征提取器（需要和训练时一致）
    """
    def __init__(self, observation_space, features_dim=512, use_pretrained=True):
        super().__init__(observation_space, features_dim)
        
        vgg16 = models.vgg16(pretrained=use_pretrained)
        self.features = vgg16.features
        
        if use_pretrained:
            for param in self.features.parameters():
                param.requires_grad = False
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
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
        "scripts:test-env-v0", 
        ip_address="127.0.0.1", 
        image_shape=(50,50,3),
        env_config=env_config["TrainEnv"]
    )
)])

# Wrap env as VecTransposeImage (Channel last to channel first)
env = VecTransposeImage(env)

# Load the trained model
print("加载DQN模型...")
model = DQN.load(env=env, path="dqn/best_model")

# Run the trained policy
print("开始测试...")
obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, dones, info = env.step(action)
    time.sleep(0.05)  # 添加延迟，让动作看起来更慢
