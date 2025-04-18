import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ==================== Game Settings ==================== #
SCREEN_WIDTH = 400  # Increased window size
SCREEN_HEIGHT = 600
FPS = 2 # Slower frame rate

COLORS = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0)
}
# Player settings (larger size)
PLAYER_WIDTH = 10
PLAYER_HEIGHT = 20
PLAYER_SPEED = 1
PLAYER_SCALE_FACTOR = 0.3  # Reduced from 1.8 to make the player smaller

# Enemy settings (balanced for avoidability)
ENEMY_WIDTH = 70
ENEMY_WIDTH = 70
ENEMY_HEIGHT = 70
ENEMY_SPEED = 1  # Slightly reduced speed
MIN_ENEMY_SPAWN_DISTANCE = 150  # Prevent clustered spawning

# Collectible settings
ITEM_WIDTH = 60
ITEM_HEIGHT = 60
ITEM_SPEED = 1

# Spawn intervals (adjusted for fairness)
SPAWN_INTERVAL_MIN = 50
SPAWN_INTERVAL_MAX = 70
ITEM_SPAWN_INTERVAL_MIN = 50
ITEM_SPAWN_INTERVAL_MAX = 200


class DodgeGameAI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Enhanced Dodge Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 40, bold=True)

        # Load and scale game assets
        self._load_assets()
        self.reset()

    def _load_assets(self):
        """Load and scale game assets with proper error handling"""
        try:
            # Background
            self.background = pygame.transform.scale(
                pygame.image.load('background.jpeg').convert(),
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )

            # Player with larger scaling
            self.player_img = pygame.transform.scale_by(
                pygame.image.load('antyou.png').convert_alpha(),
                PLAYER_SCALE_FACTOR
            )

            # Enemy
            self.enemy_img = pygame.transform.scale(
                pygame.image.load('fish.png').convert_alpha(),
                (ENEMY_WIDTH, ENEMY_HEIGHT)
            )

            # Banana
            self.banana_img = pygame.transform.scale(
                pygame.image.load('banana.png').convert_alpha(),
                (ITEM_WIDTH, ITEM_HEIGHT)
            )

            # Initialize the mixer module
            pygame.mixer.init()

            # Load background music
            pygame.mixer.music.load('music.mp3')
            pygame.mixer.music.set_volume(0.5)  # Adjust volume as needed
            pygame.mixer.music.play(-1)  # The -1 means the music will loop indefinitely

            # Load banana sound effect
            self.banana_sound = pygame.mixer.Sound('banana.wav')
            self.banana_sound.set_volume(0.7)  # Adjust volume as needed

        except Exception as e:
            print(f"Error loading assets: {e}")

        except pygame.error as e:
            print(f"Asset loading error: {e}")
            sys.exit()

    def check_collisions(self):
        # Check for collision between player and banana
        if self.player_rect.colliderect(self.banana_rect):
            # Play the banana sound effect
            self.banana_sound.play()

            # Handle the banana collection logic
            self.score += 1
            self.spawn_new_banana()

    def reset(self):
        """Reset game state with balanced initial values"""
        self.score = 1
        self.game_over = False
        self.player = pygame.Rect(
            SCREEN_WIDTH // 2 - PLAYER_WIDTH // 2,
            SCREEN_HEIGHT - PLAYER_HEIGHT - 50,
            PLAYER_WIDTH,
            PLAYER_HEIGHT
        )
        self.enemies = []
        self.items = []
        self.spawn_timer = random.randint(SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX)
        self.item_spawn_timer = random.randint(ITEM_SPAWN_INTERVAL_MIN, ITEM_SPAWN_INTERVAL_MAX)
        return self._get_state()

    def _get_state(self):
        """Improved state representation with danger assessment"""
        state = [
            self.player.x / SCREEN_WIDTH,  # Normalized player position
            0.0,  # Nearest enemy X
            0.0,  # Nearest enemy Y
            0.0  # Danger level (distance to nearest enemy)
        ]

        if self.enemies:
            # Find most immediate threat (lowest enemy)
            nearest_enemy = min(self.enemies, key=lambda e: SCREEN_HEIGHT - e.y)
            state[1] = nearest_enemy.x / SCREEN_WIDTH
            state[2] = nearest_enemy.y / SCREEN_HEIGHT
            state[3] = (self.player.x - nearest_enemy.x) / SCREEN_WIDTH  # Horizontal danger

        return np.array(state, dtype=np.float32)

    def _valid_spawn_position(self, new_x):
        """Ensure enemies don't spawn too close to each other"""
        return all(
            abs(new_x - e.x) > MIN_ENEMY_SPAWN_DISTANCE
            for e in self.enemies
        )

    def play_step(self, action):
        """Main game loop step with balanced mechanics"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Process action (now with momentum-based movement)
        action_idx = np.argmax(action)
        if action_idx == 1:
            self.player.x -= PLAYER_SPEED + 2  # Slight left boost
        elif action_idx == 2:
            self.player.x += PLAYER_SPEED + 2  # Slight right boost

        # Keep player in bounds
        self.player.x = max(0, min(SCREEN_WIDTH - PLAYER_WIDTH, self.player.x))

        # Enemy spawning with safety checks
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            attempts = 0
            while attempts < 5:  # Prevent infinite loop
                new_x = random.randint(0, SCREEN_WIDTH - ENEMY_WIDTH)
                if self._valid_spawn_position(new_x):
                    self.enemies.append(pygame.Rect(
                        new_x, -ENEMY_HEIGHT, ENEMY_WIDTH, ENEMY_HEIGHT
                    ))
                    self.spawn_timer = random.randint(SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX)
                    break
                attempts += 1

        # Item spawning
        self.item_spawn_timer -= 1
        if self.item_spawn_timer <= 0:
            self.items.append(pygame.Rect(
                random.randint(0, SCREEN_WIDTH - ITEM_WIDTH),
                -ITEM_HEIGHT,
                ITEM_WIDTH,
                ITEM_HEIGHT
            ))
            self.item_spawn_timer = random.randint(ITEM_SPAWN_INTERVAL_MIN, ITEM_SPAWN_INTERVAL_MAX)

        # Update positions
        for obj in [*self.enemies, *self.items]:
            obj.y += ENEMY_SPEED if obj in self.enemies else ITEM_SPEED

        # Cleanup off-screen objects
        self.enemies = [e for e in self.enemies if e.y < SCREEN_HEIGHT]
        self.items = [i for i in self.items if i.y < SCREEN_HEIGHT]

        # Collision detection
        reward = 1
        self.score += 1

        # Enemy collision
        for enemy in self.enemies:
            if self.player.colliderect(enemy):
                self.game_over = True
                reward = -100
                break

        # Banana collection
        for item in self.items[:]:
            if self.player.colliderect(item):
                self.score += 20
                self.items.remove(item)
                reward += 5
                self.banana_sound.play()

        self._update_display()
        return self._get_state(), reward, self.game_over, self.score

    def _update_display(self):
        """Enhanced visual rendering"""
        self.screen.blit(self.background, (0, 0))

        # Draw player centered on hitbox
        player_pos = (
            self.player.x - (self.player_img.get_width() - PLAYER_WIDTH) // 2,
            self.player.y - (self.player_img.get_height() - PLAYER_HEIGHT) // 2
        )
        self.screen.blit(self.player_img, player_pos)

        # Draw game objects
        for enemy in self.enemies:
            self.screen.blit(self.enemy_img, enemy)
        for item in self.items:
            self.screen.blit(self.banana_img, item)

        # Score display
        score_text = self.font.render(f"Score: {self.score:,}", True, COLORS['white'])
        self.screen.blit(score_text, (20, 20))
        pygame.display.flip()


class ImprovedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # Note: The dropout layer isn't connected to the network in forward().
        # If you intend to use dropout, integrate it into the net sequentially or call it explicitly.

    def forward(self, x):
        return self.net(x)

class BalancedAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.policy_net = ImprovedDQN(state_dim, action_dim).to(device)
        self.target_net = ImprovedDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.0005)
        self.memory = deque(maxlen=20000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.02

    def select_action(self, state, train_mode=True):
        if train_mode and random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).argmax().item()

    def store_experience(self, *args):
        self.memory.append(args)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors with proper typing
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions)
        # Compute next Q values
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        # Compute target Q values
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = nn.SmoothL1Loss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def play_agent(episodes=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DodgeGameAI()
    agent = BalancedAgent(state_dim=4, action_dim=3, device=device)

    # Load the trained model
    agent.policy_net.load_state_dict(torch.load("balanced_dqn.pth", map_location=device))
    agent.policy_net.eval()  # Set the model to evaluation mode

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action_idx = agent.select_action(state, train_mode=False)  # Disable epsilon-greedy in evaluation
            action = np.zeros(3)
            action[action_idx] = 1

            next_state, reward, done, score = env.play_step(action)
            state = next_state
            total_reward += reward

        # Print the result of the episode
        print(f"Episode {episode + 1:4d} | Score: {score:6d} | Total Reward: {total_reward:.1f}")

if __name__ == "__main__":
    play_agent()
