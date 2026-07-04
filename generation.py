# All necessary imports
import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")   # This error was appearing due to some optional image processing library that is not relevant for our code, so I safely ignore it.
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
# Adding src directly to the working directory
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# Imports from the Pape (2025) repository
from environments.full_yahtzee_env import Action, Observation, Phase
from utilities.scoring_helper import ScoreCategory, NUMBER_OF_CATEGORIES
from yahtzee_agent.features import create_features
from yahtzee_agent.model import (
    YahtzeeAgent,
    convert_rolling_action_to_hold_mask,
    phi,
    select_action,
)
from yahtzee_agent.trainer import Algorithm, YahtzeeAgentTrainer

CATEGORY_LABELS = ScoreCategory.LABELS
NORMAL_SCORE_MAX = np.array([5, 10, 15, 20, 25, 30, 30, 30, 25, 30, 40, 50, 30])

class ExpertDataGenerator:
    #Generate expert trajectories from trained RL models for my experiments
    def __init__(self, checkpoint_path: str):
        # Load expert model from checkpoint.
        print(f"Model loaded from the file: {checkpoint_path}...")
        torch.serialization.add_safe_globals([Algorithm])
        
        # Load checkpoint (basically a chosen model from Pape (2025) repo)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        hparams = checkpoint["hyper_parameters"]
        
        # Recreating features to correctly fit the model architecture
        features = create_features(hparams["phi_features"].split(","))
        self.phi_features = features
        self.phi_feature_names = [f.name for f in features]
        
        # Create model with same architecture
        self.model = YahtzeeAgent(
            hidden_size=hparams["hidden_size"],
            num_hidden=hparams["num_hidden"],
            dropout_rate=hparams["dropout_rate"],
            activation_function=hparams["activation_function"],
            features=features,
            rolling_action_representation=hparams["rolling_action_representation"],
            he_kaiming_initialization=False,
            use_layer_norm=hparams.get("use_layer_norm", True),
        )
        
        # Loading weights from a specified checkpoint
        policy_net_state_dict = {
            k.replace("policy_net.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("policy_net.")
        }
        self.model.load_state_dict(policy_net_state_dict)
        self.model.eval()
        
        # Standard practice of getting the device as working with torch and tensors
        self.device = next(self.model.parameters()).device
        self.model = self.model.to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Phi features ({len(features)}): {self.phi_feature_names}")

    def generate_trajectories(self, num_games: int = 100) -> list[dict[str, Any]]:
        # Generate state-action trajectories from expert policy.
        env = gym.make("FullYahtzee-v1") # same as in the source code
        trajectories = []
        print(f"\nGenerating {num_games} expert trajectories:")
        total_rewards_list = []
        duplicate_category_count = 0 # some sanity checks to ensure that each category isnt taken multiple times

        with torch.no_grad():
            for game_idx in tqdm(range(num_games), desc="Games"):
                obs, _ = env.reset()
                episode_data = {
                    "game_id": game_idx,
                    "steps": [],
                    "total_reward": 0.0,
                    "num_steps": 0,
                    "categories_used": [],
                    "categories_unused": [],
                }

                # Track which categories have been used (as a sanity check to see if model is not reusing some categories)
                used_categories = set()

                while True:
                    # Store complete step information
                    step_data = self._create_step_data(
                        obs=obs,
                        model=self.model,
                        device=self.device,
                        used_categories=used_categories,
                    )

                    # Get action from expert
                    with torch.no_grad():
                        state_tensor = phi(obs, self.model.features, self.device).unsqueeze(0)
                        rolling_probs, scoring_probs, _, _ = self.model.forward(state_tensor)
                        rolling_action, scoring_action_tensor = select_action(
                            rolling_probs,
                            scoring_probs,
                            self.model.rolling_action_representation,
                        )

                    # Convert actions to environment format
                    hold_mask = convert_rolling_action_to_hold_mask(
                        rolling_action, self.model.rolling_action_representation
                    )
                    scoring_category = int(scoring_action_tensor.item())
                    
                    step_data["action"] = {
                        "rolling_action_tensor": rolling_action.cpu().numpy().tolist(),
                        "scoring_action_tensor": int(scoring_action_tensor.item()),
                        "hold_mask": hold_mask.tolist(),
                        "score_category": scoring_category,
                        "score_category_name": CATEGORY_LABELS[scoring_category],
                    }
                    
                    if obs["phase"] == Phase.SCORING:
                        if scoring_category in used_categories:
                            duplicate_category_count += 1
                            print(f"\n⚠️  WARNING: Category {CATEGORY_LABELS[scoring_category]} used twice in game {game_idx}!") # Another extra safety measure to prevent model using same categories more than once
                        used_categories.add(scoring_category)
                    
                    action_dict = {
                        "hold_mask": hold_mask,
                        "score_category": scoring_category,
                    }
                    obs, reward, done, truncated, _ = env.step(action_dict)
                    
                    step_data["reward"] = float(reward)
                    episode_data["steps"].append(step_data)
                    episode_data["total_reward"] += float(reward)

                    if done or truncated:
                        break

                episode_data["num_steps"] = len(episode_data["steps"])
                episode_data["categories_used"] = sorted(list(used_categories))
                episode_data["categories_unused"] = sorted(
                    [i for i in range(13) if i not in used_categories]
                )
                
                trajectories.append(episode_data)
                total_rewards_list.append(episode_data["total_reward"])

        env.close()
        print(f"\nGenerated {len(trajectories)} trajectories")
        print(f"  Average reward: {np.mean(total_rewards_list):.2f}")
        print(f"  Std dev reward: {np.std(total_rewards_list):.2f}")
        print(f"  Min reward: {np.min(total_rewards_list):.2f}")
        print(f"  Max reward: {np.max(total_rewards_list):.2f}")
        
        if duplicate_category_count > 0:
            print(f"\n⚠️  Found {duplicate_category_count} duplicate category uses (should be 0!)")
        else:
            print(f"\n✓ No duplicate categories found. All games valid.")

        return trajectories

    def _create_step_data(
        self,
        obs: Observation,
        model: YahtzeeAgent,
        device: torch.device,
        used_categories: set,
    ) -> dict[str, Any]:
        # Create detailed step information including all phi features.
        phi_features_dict = {}
        for feature in self.phi_features:
            computed = feature.compute(obs)
            phi_features_dict[feature.name] = computed.tolist()

        # Category usage info
        category_usage = {}
        for i, label in enumerate(CATEGORY_LABELS):
            is_available = obs["score_sheet_available_mask"][i] == 1
            category_usage[label] = {
                "index": i,
                "available": bool(is_available),
                "used": i in used_categories,
                "current_score": int(obs["score_sheet"][i]),
            }

        return {
            "observation": {
                "dice": obs["dice"].tolist(),
                "rolls_used": int(obs["rolls_used"]),
                "phase": int(obs["phase"]),
                "phase_name": "ROLLING" if obs["phase"] == Phase.ROLLING else "SCORING",
                "score_sheet": obs["score_sheet"].tolist(),
                "score_sheet_available_mask": obs["score_sheet_available_mask"].tolist(),
            },
            "phi_features": phi_features_dict,
            "category_status": category_usage,
            "action": None,  
            "reward": None,  
        }

    def save_trajectories(
        self,
        trajectories: list[dict[str, Any]],
        output_dir: str = "expert_data_pkl",
    ) -> None:
        Path(output_dir).mkdir(exist_ok=True)

        # Original trajectory storage
        pickle_path = os.path.join(output_dir, "trajectories.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"✓ Saved raw trajectories to {pickle_path}")

        # Generate dataset variations for three experiments
        self._save_experiment_1_rolling_phase(trajectories, output_dir)
        self._save_experiment_2_scoring_phase(trajectories, output_dir)

        # Statistics and metadata (was not used ultimately)
        stats = self._compute_statistics(trajectories)
        stats_path = os.path.join(output_dir, "statistics.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics to {stats_path}")

        # Feature metadata
        feature_metadata = self._create_feature_metadata()
        metadata_path = os.path.join(output_dir, "feature_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(feature_metadata, f, indent=2)
        print(f"✓ Saved feature metadata to {metadata_path}")

    def _save_experiment_1_rolling_phase(
        self,
        trajectories: list[dict[str, Any]],
        output_dir: str,
    ) -> None:
        """
        Experiment 1: Only for determining which dices to roll or not.
        "Given a state -> which dice should I hold/reroll?"
        
        Features: Current dice, rolls used, available categories
        Target: hold_mask (binary for each die)
        """
        exp1_dir = os.path.join(output_dir, "experiment_1_rolling_phase")
        Path(exp1_dir).mkdir(exist_ok=True)

        dataset_samples = []
        
        for trajectory in trajectories:
            for step_idx, step in enumerate(trajectory["steps"]):
                # Only collect from rolling phase
                if step["observation"]["phase"] == Phase.ROLLING and step["action"] is not None:
                    sample = {
                        "game_id": trajectory["game_id"],
                        "step_idx": step_idx,
                        # Features
                        "dice": np.array(step["observation"]["dice"]),
                        "rolls_used": step["observation"]["rolls_used"],
                        "available_categories": np.array(step["observation"]["score_sheet_available_mask"]),
                        "score_sheet": np.array(step["observation"]["score_sheet"]),
                        "phi_features": step["phi_features"],
                        # Target
                        "hold_mask": np.array(step["action"]["hold_mask"], dtype=np.int32),
                    }
                    dataset_samples.append(sample)

        # Save as pickle 
        dataset_path = os.path.join(exp1_dir, "rolling_phase_dataset.pkl")
        with open(dataset_path, "wb") as f:
            pickle.dump(dataset_samples, f)
        print(f"✓ Experiment 1 - Rolling Phase: {len(dataset_samples)} samples -> {dataset_path}")

        # Save statistics (not used ultimately)
        exp1_stats = {
            "total_samples": len(dataset_samples),
            "games_represented": len(trajectories),
            "avg_samples_per_game": len(dataset_samples) / len(trajectories),
        }
        stats_path = os.path.join(exp1_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(exp1_stats, f, indent=2)

    def _save_experiment_2_scoring_phase(
        self,
        trajectories: list[dict[str, Any]],
        output_dir: str,
    ) -> None:
        """
        Experiment 2: Only for choosing a category based on the stage of the game.
        "Given a dice and stage -> which category should I choose?"
        
        Features: Current dice, rolls used (showing stage), available categories, score_sheet
        Target: score_category (13 categories, 0-12)
        """
        exp2_dir = os.path.join(output_dir, "experiment_2_scoring_phase")
        Path(exp2_dir).mkdir(exist_ok=True)

        dataset_samples = []
        
        for trajectory in trajectories:
            for step_idx, step in enumerate(trajectory["steps"]):
                # Only collect from scoring phase
                if step["observation"]["phase"] == Phase.SCORING and step["action"] is not None:
                    sample = {
                        "game_id": trajectory["game_id"],
                        "step_idx": step_idx,
                        "turn_number": len([s for s in trajectory["steps"][:step_idx+1] 
                                           if s["observation"]["phase"] == Phase.SCORING]),
                        # Features
                        "dice": np.array(step["observation"]["dice"]),
                        "rolls_used": step["observation"]["rolls_used"],
                        "available_categories": np.array(step["observation"]["score_sheet_available_mask"]),
                        "score_sheet": np.array(step["observation"]["score_sheet"]),
                        "phi_features": step["phi_features"],
                        # Target
                        "score_category": step["action"]["score_category"],
                        "score_category_name": step["action"]["score_category_name"],
                    }
                    dataset_samples.append(sample)

        # Save as pickle
        dataset_path = os.path.join(exp2_dir, "scoring_phase_dataset.pkl")
        with open(dataset_path, "wb") as f:
            pickle.dump(dataset_samples, f)
        print(f"✓ Experiment 2 - Scoring Phase: {len(dataset_samples)} samples -> {dataset_path}")

        # Save statistics (again, ultimately not used)
        category_counts = {}
        for sample in dataset_samples:
            cat = sample["score_category"]
            if cat not in category_counts:
                category_counts[cat] = 0
            category_counts[cat] += 1

        exp2_stats = {
            "total_samples": len(dataset_samples),
            "games_represented": len(trajectories),
            "avg_samples_per_game": len(dataset_samples) / len(trajectories),
            "category_distribution": {
                CATEGORY_LABELS[k]: v for k, v in sorted(category_counts.items())
            },
        }
        stats_path = os.path.join(exp2_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(exp2_stats, f, indent=2)

    
    def _create_feature_metadata(self) -> dict[str, Any]:
        # Create metadata about features for all experiments.
        return {
            "phi_feature_names": self.phi_feature_names,
            "phi_feature_count": len(self.phi_feature_names),
            "target_categories": CATEGORY_LABELS,
            "num_categories": len(CATEGORY_LABELS),
            "experiments": {
                "experiment_1_rolling_phase": {
                    "description": "Rolling phase only: dice state -> hold mask",
                    "features": ["dice", "rolls_used", "available_categories", "score_sheet", "phi_features"],
                    "target": "hold_mask (5 binary values per die)",
                    "num_classes": 2,
                    "task_type": "multi-output binary classification",
                },
                "experiment_2_scoring_phase": {
                    "description": "Scoring phase only: game state -> category choice",
                    "features": ["dice", "rolls_used", "available_categories", "score_sheet", "phi_features"],
                    "target": "score_category (0-12)",
                    "num_classes": 13,
                    "task_type": "multi-class classification",
                },
            },
        }

    @staticmethod
    def _compute_statistics(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        # Compute statistics about trajectories including Yahtzee bonus tracking. 
        rewards = [t["total_reward"] for t in trajectories]
        step_counts = [len(t["steps"]) for t in trajectories]
        
        # Per-category reward analysis
        category_rewards = {i: [] for i in range(13)}
        category_zero_count = {i: 0 for i in range(13)}
        category_round_scored = {i: [] for i in range(13)}  # Track which round each category was scored
        category_yahtzee_bonus = {i: 0 for i in range(13)}  # Count Yahtzee bonuses per category
        
        total_yahtzee_bonuses = 0
        yahtzee_bonus_rounds = []  # Track at which round bonuses occur
        
        for traj in trajectories:
            # Track scoring order for each game
            scoring_turn = 0
            
            for step_idx, step in enumerate(traj["steps"]):
                if step["action"] is not None and step["observation"]["phase_name"] == "SCORING":
                    cat = step["action"]["score_category"]
                    reward = step["reward"]
                    
                    category_rewards[cat].append(reward)
                    category_round_scored[cat].append(scoring_turn)
                    
                    if reward == 0:
                        category_zero_count[cat] += 1
                    
                    # Detect Yahtzee bonus
                    if cat == ScoreCategory.YAHTZEE:
                        # Yahtzee base is 50, bonus is 100, so if we see >= 100, it's with bonus
                        if reward >= 100:  # Yahtzee with bonus
                            category_yahtzee_bonus[cat] += 1
                            total_yahtzee_bonuses += 1
                            yahtzee_bonus_rounds.append(scoring_turn)
                    
                    scoring_turn += 1
        
        all_scoring_actions = []
        for traj in trajectories:
            for step in traj["steps"]:
                if step["action"] is not None:
                    all_scoring_actions.append(step["action"]["score_category"])

        category_usage_count = {i: 0 for i in range(13)}
        for traj in trajectories:
            for cat in traj["categories_used"]:
                category_usage_count[cat] += 1

        return {
            "num_games": len(trajectories),
            "total_steps": sum(step_counts),
            "reward_statistics": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "median": float(np.median(rewards)),
                "q25": float(np.percentile(rewards, 25)),
                "q75": float(np.percentile(rewards, 75)),
            },
            "steps_per_game": {
                "mean": float(np.mean(step_counts)),
                "std": float(np.std(step_counts)),
                "min": int(np.min(step_counts)),
                "max": int(np.max(step_counts)),
            },
            "category_usage": {
                CATEGORY_LABELS[i]: {
                    "index": i,
                    "times_used": category_usage_count[i],
                    "percentage": (category_usage_count[i] / len(trajectories)) * 100,
                }
                for i in range(13)
            },
            "category_reward_analysis": {
                CATEGORY_LABELS[i]: {
                    "times_used": len(category_rewards[i]),
                    "mean_reward": float(np.mean(category_rewards[i])) if category_rewards[i] else 0.0,
                    "std_reward": float(np.std(category_rewards[i])) if category_rewards[i] else 0.0,
                    "min_reward": float(np.min(category_rewards[i])) if category_rewards[i] else 0.0,
                    "max_reward": float(np.max(category_rewards[i])) if category_rewards[i] else 0.0,
                    "median_reward": float(np.median(category_rewards[i])) if category_rewards[i] else 0.0,
                    "zero_count": int(category_zero_count[i]),
                    "zero_percentage": float((category_zero_count[i] / len(category_rewards[i]) * 100) if category_rewards[i] else 0.0),
                }
                for i in range(13)
            },
            "yahtzee_bonus_analysis": {
                "total_bonuses_applied": int(total_yahtzee_bonuses),
                "bonus_rate": float((total_yahtzee_bonuses / len(trajectories)) * 100),  # % of games with bonus
                "avg_bonuses_per_game": float(total_yahtzee_bonuses / len(trajectories)),
                "yahtzee_bonus_round_stats": {
                    "rounds_with_bonuses": yahtzee_bonus_rounds,
                    "average_round_for_bonus": float(np.mean(yahtzee_bonus_rounds)) if yahtzee_bonus_rounds else 0.0,
                    "std_round_for_bonus": float(np.std(yahtzee_bonus_rounds)) if yahtzee_bonus_rounds else 0.0,
                    "earliest_bonus_round": int(min(yahtzee_bonus_rounds)) if yahtzee_bonus_rounds else None,
                    "latest_bonus_round": int(max(yahtzee_bonus_rounds)) if yahtzee_bonus_rounds else None,
                }
            },
            # Round-based category analysis
            "category_round_analysis": {
                CATEGORY_LABELS[i]: {
                    "times_used": len(category_round_scored[i]),
                    "avg_round_scored": float(np.mean(category_round_scored[i])) if category_round_scored[i] else 0.0,
                    "std_round_scored": float(np.std(category_round_scored[i])) if category_round_scored[i] else 0.0,
                    "min_round": int(min(category_round_scored[i])) if category_round_scored[i] else None,
                    "max_round": int(max(category_round_scored[i])) if category_round_scored[i] else None,
                }
                for i in range(13)
            }
        }

def main():
    # Main function to generate and save expert trajectories.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the trained checkpoint
    checkpoint_path = "C:\\Users\\gryfi\\Desktop\\Thesis (1)\\case-studies-final-project\\checkpoints\\a2c_1m.ckpt"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nSearching for available checkpoints...")
        models_dir = os.path.join(script_dir, "models")
        if os.path.exists(models_dir):
            found_any = False
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(".ckpt"):
                        print(f"  Found: {os.path.join(root, file)}")
                        found_any = True
            if not found_any:
                print("  No checkpoints found!")
        else:
            print(f"Models directory not found at {models_dir}")
        return

    # Initialize generator
    generator = ExpertDataGenerator(checkpoint_path)

    # Generate trajectories
    trajectories = generator.generate_trajectories(num_games=10000)

    # Save in all formats
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "expert_data_pkl")
    generator.save_trajectories(trajectories, output_dir=output_dir)

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\n📁 Directory structure:")
    print("  expert_data_pkl/")
    print("  ├── trajectories.pkl                     # Raw trajectories (all data)")
    print("  ├── feature_metadata.json                # Feature descriptions")
    print("  ├── statistics.json                      # Overall statistics")
    print("  │")
    print("  ├── experiment_1_rolling_phase/          # EXPERIMENT 1: Rolling phase only")
    print("  │   ├── rolling_phase_dataset.pkl")
    print("  │   │                                    # Target: hold_mask (5 binary)")
    print("  │   └── stats.json")
    print("  │")
    print("  ├── experiment_2_scoring_phase/          # EXPERIMENT 2: Scoring phase only")
    print("  │   ├── scoring_phase_dataset.pkl")
    print("  │   │                                    # Target: score_category (0-12)")
    print("  │   └── stats.json")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()