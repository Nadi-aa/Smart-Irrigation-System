import pygame
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from env import SmartIrrigationEnv
from stable_baselines3 import DQN

def moisture_color(moisture):
    if moisture < 30:
        return (194, 178, 128)  # sandy brown
    elif moisture < 70:
        return (85, 107, 47)    # greenish
    else:
        return (0, 128, 128)    # deep teal

def load_image(name, size):
    img = pygame.image.load(os.path.join("assets", name)).convert_alpha()
    return pygame.transform.scale(img, size)

def run_visual_simulation(env, model, steps=200):
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Smart Irrigation RL Simulator")
    font = pygame.font.SysFont("Arial", 24, bold=True)

    # Load background images
    backgrounds = {
        "day": load_image("sunny.png", (width, height)),
        "cloudy": load_image("cloudy.png", (width, height)),
        "rainy": load_image("rainy.png", (width, height)),
        "night": load_image("night.png", (width, height)),
    }

    # Load weather icons
    weather_icons = [
        load_image("sun_icon.png", (50, 50)),
        load_image("cloud_icon.png", (50, 50)),
        load_image("rain_icon.png", (50, 50))
    ]

    # Load plant icons
    plant_icons = [
        load_image("plant1.png", (50, 50)),
        load_image("plant2.png", (50, 50)),
        load_image("plant3.png", (50, 50))
    ]

    obs = env.reset()
    clock = pygame.time.Clock()
    running = True

    # Data logs
    moisture_history = []
    action_history = []
    reward_history = []
    plant_history = []
    time_history = []

    for _ in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(int(action))

        # Unpack observation
        soil, weather, time_of_day, plant_type = obs

        # Log data
        moisture_history.append(soil)
        action_history.append(int(action))
        reward_history.append(reward)
        plant_history.append(int(plant_type))
        time_history.append(time_of_day)

        # Determine background
        if time_of_day < 6 or time_of_day > 18:
            bg_img = backgrounds["night"]
        else:
            if weather == 0:
                bg_img = backgrounds["day"]
            elif weather == 1:
                bg_img = backgrounds["cloudy"]
            else:
                bg_img = backgrounds["rainy"]

        screen.blit(bg_img, (0, 0))

        # Draw plant field
        field_rect = pygame.Rect(250, 150, 300, 300)
        pygame.draw.rect(screen, moisture_color(soil), field_rect, border_radius=12)

        # Weather icon
        screen.blit(weather_icons[int(weather)], (width - 70, 20))

        # Plant icon
        plant_index = int(plant_type) % len(plant_icons)
        screen.blit(plant_icons[plant_index], (width - 420, 260))

        # Info texts
        text_color = (255, 255, 255) if (time_of_day < 6 or time_of_day > 18) else (0, 0, 0)
        time_text = font.render(f"Time: {int(time_of_day):02d}:00", True, text_color)
        moisture_text = font.render(f"Soil Moisture: {soil:.1f}%", True, text_color)
        reward_text = font.render(f"Reward: {reward:.2f}", True, text_color)
        plant_text = font.render(f"Plant Type: {int(plant_type)}", True, text_color)
        actions = ["None", "Low", "Medium", "High"]
        action_text = font.render(f"Action: {actions[int(action)]}", True, text_color)

        screen.blit(time_text, (50, 20))
        screen.blit(moisture_text, (50, 60))
        screen.blit(action_text, (50, 100))
        screen.blit(reward_text, (50, 140))
        screen.blit(plant_text, (50, 180))

        pygame.display.flip()
        time.sleep(0.2)
        clock.tick(10)

        if done or not running:
            break

    pygame.quit()

    # --- Enhanced Plotting ---
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 10))
    plt.suptitle("Smart Irrigation Simulation Overview", fontsize=16, weight='bold', color='navy')

    # Moisture
    plt.subplot(4, 1, 1)
    plt.plot(moisture_history, color='royalblue', linewidth=2, label="Soil Moisture (%)")
    plt.ylabel("Moisture")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Actions
    plt.subplot(4, 1, 2)
    plt.step(range(len(action_history)), action_history, where='post', color='limegreen', linewidth=2, label="Irrigation Action")
    plt.ylabel("Action")
    plt.yticks([0, 1, 2, 3], ["None", "Low", "Medium", "High"])
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Rewards
    plt.subplot(4, 1, 3)
    plt.plot(reward_history, color='purple', linewidth=2, label="Reward")
    plt.ylabel("Reward")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plant Types
    plt.subplot(4, 1, 4)
    plt.plot(plant_history, color='saddlebrown', linewidth=2, label="Plant Type")
    plt.ylabel("Plant Type")
    plt.xlabel("Step")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    env = SmartIrrigationEnv()
    model = DQN.load("models/irrigation_dqn")
    run_visual_simulation(env, model)