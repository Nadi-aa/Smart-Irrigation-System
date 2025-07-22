# 🌱 Smart Irrigation System using Reinforcement Learning (PPO)

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Gym](https://img.shields.io/badge/OpenAI%20Gym-Custom%20Env-orange)
![PPO](https://img.shields.io/badge/RL%20Algorithm-PPO-purple)

An intelligent multi-zone irrigation control system powered by Reinforcement Learning. This project uses a custom OpenAI Gym environment and trains a PPO (Proximal Policy Optimization) agent to optimize water usage while maintaining ideal soil conditions for different plant types under varying weather conditions.

---

## 🚀 Features

- ✅ Multi-grid (3-zone) plant irrigation simulation
- ✅ Weather-aware, plant-specific irrigation control
- ✅ Custom reward function for moisture balance and water efficiency
- ✅ PPO-based reinforcement learning agent using Stable-Baselines3
- ✅ Interactive Pygame-based simulation GUI
- ✅ Training and simulation visualizations using Matplotlib & Seaborn

---

## 🧠 Problem Statement

> Traditional irrigation systems lack adaptability. This project introduces an AI-driven solution that adjusts irrigation levels dynamically based on:
- Soil moisture
- Plant type requirements
- Weather conditions
- Time of day

The goal: 🌿 healthy plants + 💧 minimum water waste.

---

## 🏗️ Project Structure

smart-irrigation-rl/
├── assets/ # GUI icons & backgrounds
├── models/ # Trained PPO models
├── env.py # Custom OpenAI Gym environment
├── train_agent.py # PPO training script
├── run_gui.py # Pygame-based visual simulator
├── requirements.txt # Dependencies
└── README.md

## 🧪 Training the PPO Agent

- Trains a PPO agent using Stable-Baselines3  
- Logs rewards per episode  
- Saves the model to models/irrigation_ppo_multigrid  
- Plots training rewards after learning

---

## 🕹️ Run the Interactive Simulation

- Loads the trained PPO model
- Visualizes 3 irrigation grids using Pygame
- Weather, soil color, plant icons, and actions update in real-time
- Ends with performance plots: moisture levels, actions, rewards

⚠️ Make sure your assets/ folder contains:
- sunny.png, cloudy.png, rainy.png, night.png  
- sun_icon.png, cloud_icon.png, rain_icon.png  
- plant1.png, plant2.png, plant3.png

---

## 🌦️ Environment Overview

Observation Space:  
[soil_0, weather_0, plant_0, soil_1, weather_1, plant_1, soil_2, weather_2, plant_2, time_of_day]

Action Space:  
MultiDiscrete([4, 4, 4])  # 0=None, 1=Low, 2=Medium, 3=High (per grid)

Reward Function:  
- Penalizes deviation from optimal moisture (plant-specific)
- Small penalty for higher irrigation levels
- Weather & plant types influence soil evaporation

---

## 📸 Screenshots


<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/2802c6cd-d732-4834-a0fe-f05450b2c114" />

<img width="994" height="710" alt="image" src="https://github.com/user-attachments/assets/b7ade540-90cf-4e34-bf91-3660dcba5015" />



## 👨‍💻 Author

Made by NADI-AA + 🤖 
