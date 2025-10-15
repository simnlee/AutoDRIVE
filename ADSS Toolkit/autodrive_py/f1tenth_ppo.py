"""
f1tenth_ppo.py
--------------
Train F1TENTH racing agent using PPO (Stable Baselines 3).

Usage:
    # Train a new model
    python f1tenth_ppo.py --timesteps 1000000

    # Continue training from checkpoint
    python f1tenth_ppo.py --load models/f1tenth_ppo_500000_steps.zip --timesteps 1000000
"""

import os
import argparse
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from autodrive_gym import AutoDRIVEGymEnv


def make_env(waypoint_csv: str = "track_waypoints_trimmed.csv",
             log_dir: Optional[str] = None):
    """Create and wrap the AutoDRIVE environment."""
    def _init():
        env = AutoDRIVEGymEnv(waypoint_csv_path=waypoint_csv)
        if log_dir:
            env = Monitor(env, log_dir)
        return env
    return _init


def train_ppo(
    waypoint_csv: str = "track_waypoints_trimmed.csv",
    total_timesteps: int = 1_000_000,
    save_freq: int = 50_000,
    model_dir: str = "models",
    log_dir: str = "logs",
    tensorboard_log: str = "tensorboard_logs",
    load_path: Optional[str] = None,
):
    """
    Train PPO agent for F1TENTH racing.

    Args:
        waypoint_csv: Path to track waypoints CSV
        total_timesteps: Total training timesteps
        save_freq: Save checkpoint every N steps
        model_dir: Directory to save models
        log_dir: Directory for Monitor logs
        tensorboard_log: Directory for TensorBoard logs
        load_path: Path to load existing model (optional)
    """
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)

    # Create environment
    print("Creating training environment...")
    env = DummyVecEnv([make_env(waypoint_csv, log_dir)])

    # PPO hyperparameters
    ppo_kwargs = {
        "policy": "MultiInputPolicy",
        "env": env,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # Exploration bonus
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "tensorboard_log": tensorboard_log,
    }

    # Load existing model or create new one
    if load_path and os.path.exists(load_path):
        print(f"Loading model from {load_path}...")
        model = PPO.load(load_path, env=env, **ppo_kwargs)
    else:
        print("Creating new PPO model...")
        model = PPO(**ppo_kwargs)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix="f1tenth_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Train
    print(f"\nTraining PPO for {total_timesteps:,} timesteps...")
    print(f"Models will be saved to: {model_dir}")
    print(f"TensorBoard logs: {tensorboard_log}")
    print(f"Monitor logs: {log_dir}")
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {tensorboard_log}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model
    final_model_path = os.path.join(model_dir, "f1tenth_ppo_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}.zip")

    # Cleanup
    env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train F1TENTH PPO agent")

    # Common arguments
    parser.add_argument("--waypoints", type=str, default="track_waypoints_trimmed.csv",
                        help="Path to track waypoints CSV")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to load existing model")

    # Training arguments
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for Monitor logs")
    parser.add_argument("--tensorboard-log", type=str, default="tensorboard_logs",
                        help="Directory for TensorBoard logs")

    args = parser.parse_args()

    # Train
    train_ppo(
        waypoint_csv=args.waypoints,
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        tensorboard_log=args.tensorboard_log,
        load_path=args.load,
    )


if __name__ == "__main__":
    main()
