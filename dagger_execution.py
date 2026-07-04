# DAGGER TRAINING EXPERIMENT FOR YAHTZEE DECISION TREES (FIXED + DUAL FALLBACK RUN)
# ---------------------------------------------------------------------------
# Key fixes:
#   * All missing helper functions added: is_yahtzee, _turn_idx_from_avail, _subtree_for_turn, _is_three_state
#   * Added missing constants: CATEGORY_NAMES, CAT_NAME_TO_IDX
#   * Fixed function names: _fallback_cat -> fallback_cat
#   * Fixed prepare_features to return consistent 1D arrays (removed reshape in build_feature_vector)
#   * Evaluation now uses identical logic to working simulation
#   * Seeded, paired evaluation for reproducibility
#   * EXPERT MODE: beta=1.0 uses pure expert (no tree, no fallbacks)
#   * PLOTS: BC baseline REMOVED entirely from plots and data passed to plotting
#   * RUNS TWICE: once with FALLBACK_MODE='greedy', once with FALLBACK_MODE='random'
#     -> produces dagger_results_greedy.png and dagger_results_random.png
#   * Global SEED fixes numpy/torch RNGs so repeated runs are reproducible
# ---------------------------------------------------------------------------

# CONFIGURATION
SRC_PATH        = r'C:\Users\Szymon\Desktop\Thesis_use_this\case-studies-final-project'
CHECKPOINT_PATH = r'C:\Users\Szymon\Desktop\Thesis_use_this\case-studies-final-project\checkpoints\a2c_1m.ckpt'

# DAgger hyperparameters
DAGGER_ITERATIONS   = 10     # how many DAgger rounds
GAMES_PER_ITERATION = 1000     # games collected each round
EVAL_GAMES          = 10000    # games used to evaluate after each round
ROLL_DEPTH          = 20     # thesis rolling tree depth
SCORE_DEPTH         = 15     # thesis scoring tree depth
SEED                = 42
FULL_RUN            = True

# Run both fallback modes in sequence; each gets its own image file.
FALLBACK_MODES_TO_RUN = ['greedy', 'random']

# Reference lines for the plots
EXPERT_MEAN_REF  = 241.36   # expert mean (10k-game run)
EXPERT_BONUS_REF = 25.3     # expert upper-bonus rate (%)

# All necessary imports
import os
import sys
import warnings
import random
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
from gymnasium.vector import SyncVectorEnv
import gymnasium as gym


def set_global_seed(seed):
    # for reproducibility, set all relevant seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_global_seed(SEED)


def create_env():
    return gym.make("FullYahtzee-v1")

# Add source to path
src = os.path.join(SRC_PATH, 'src')
if src not in sys.path:
    sys.path.insert(0, src)

from environments.full_yahtzee_env import Phase
from utilities.scoring_helper import ScoreCategory
from yahtzee_agent.features import create_features
from yahtzee_agent.model import (
    YahtzeeAgent,
    convert_rolling_action_to_hold_mask,
    phi,
    select_action,
)
from yahtzee_agent.trainer import Algorithm

print("Imports OK")

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY NAMES AND MAPPING
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_NAMES = [
    'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
    '3-Kind', '4-Kind', 'Full House', 'S-Straight', 'L-Straight', 'Yahtzee', 'Chance'
]
CAT_NAME_TO_IDX = {n: i for i, n in enumerate(CATEGORY_NAMES)}

SIM_CATEGORIES = [
    'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
    '3-Kind', '4-Kind', 'Full House', 'S-Straight', 'L-Straight', 'Yahtzee', 'Chance']

SIM_TO_PAPE = {
    'Ones': 'Ones', 'Twos': 'Twos', 'Threes': 'Threes', 'Fours': 'Fours',
    'Fives': 'Fives', 'Sixes': 'Sixes', '3-Kind': 'Three of a Kind',
    '4-Kind': 'Four of a Kind', 'Full House': 'Full House',
    'S-Straight': 'Small Straight', 'L-Straight': 'Large Straight',
    'Yahtzee': 'Yahtzee', 'Chance': 'Chance'}
PAPE_TO_SIM = {v: k for k, v in SIM_TO_PAPE.items()}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def is_yahtzee(dice_row):
    """Check if all dice are the same."""
    return len(set(int(d) for d in dice_row)) == 1

def _is_three_state(m):
    """Check if model has three-state structure."""
    return hasattr(m, 'early') and hasattr(m, 'mid') and hasattr(m, 'end')

def _subtree_for_turn(model, turn_idx):
    """Select appropriate subtree based on turn index."""
    if _is_three_state(model):
        if   turn_idx < 5:  return model.early
        elif turn_idx < 10: return model.mid
        else:               return model.end
    return model

def _turn_idx_from_avail(avail_row):
    """0-based turn index = number of categories already filled (0..12)."""
    return 13 - int(round(float(np.sum(avail_row))))

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING + FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def dice_decision_to_class(row):
    binary_str = ''.join(
        [str(int(row[col])) for col in
         ["hold_die_1", "hold_die_2", "hold_die_3", "hold_die_4", "hold_die_5"]])
    return int(binary_str, 2)

# Experiment 1 (rolling)
df_exp1 = pd.read_csv(os.path.join(
    SRC_PATH, 'experiment_1_data', 'experiment_1_rolling_phase_full_data.csv'))
print(f"Loaded Experiment 1 data with shape: {df_exp1.shape}")

df = df_exp1.copy()
dice_cols = ['dice_1', 'dice_2', 'dice_3', 'dice_4', 'dice_5']
for face in range(1, 7):
    df[f'count_{face}'] = df[dice_cols].apply(lambda r: int(sum(r == face)), axis=1)
df['dice_sum'] = df[dice_cols].sum(axis=1)

def hand_patterns(row):
    counts = row.value_counts().values
    return pd.Series({
        'has_pair':       int(2 in counts),
        'has_three':      int(3 in counts),
        'has_four':       int(4 in counts),
        'has_yahtzee':    int(5 in counts),
        'has_full_house': int(set(counts) == {2, 3}),
        'has_two_pair':   int(list(counts).count(2) == 2),
    })

df = pd.concat([df, df[dice_cols].apply(hand_patterns, axis=1)], axis=1)
df_exp1_extended = df.copy()

feature_rich = (
    ["dice_1", "dice_2", "dice_3", "dice_4", "dice_5", "rolls_used"] +
    [f"cat_available_{i}" for i in range(13)] +
    [f"score_sheet_{i}" for i in range(13)] +
    [f"count_{i}" for i in range(1, 7)] +
    ["dice_sum", "has_pair", "has_three", "has_four",
     "has_two_pair", "has_full_house", "has_yahtzee"]
)
y_rich = df_exp1.apply(dice_decision_to_class, axis=1)
X_rich = df_exp1_extended[feature_rich].copy()
X_train_rich, X_test_rich, y_train_rich, y_test_rich = train_test_split(
    X_rich, y_rich, test_size=0.2, random_state=SEED)

# Seed BC rolling tree
complex_model_rich_exp1 = DecisionTreeClassifier(max_depth=ROLL_DEPTH, random_state=SEED)
complex_model_rich_exp1.fit(X_train_rich, y_train_rich)
print(f"Seed rolling tree trained (depth {ROLL_DEPTH}): {len(X_train_rich):,} samples")

# Experiment 2 (scoring)
df_exp2 = pd.read_csv(os.path.join(
    SRC_PATH, 'experiment_2_data', 'experiment_2_scoring_phase_full_data.csv'))
print(f"Loaded Experiment 2 data with shape: {df_exp2.shape}")

df = df_exp2.copy()
for face in range(1, 7):
    df[f'count_{face}'] = df[dice_cols].apply(lambda r: int(sum(r == face)), axis=1)
df['dice_sum'] = df[dice_cols].sum(axis=1)
df = pd.concat([df, df[dice_cols].apply(hand_patterns, axis=1)], axis=1)
df_exp2_extended = df.copy()

feature_rich_exp2 = (
    ["dice_1", "dice_2", "dice_3", "dice_4", "dice_5", "rolls_used"] +
    [f"cat_available_{i}" for i in range(13)] +
    [f"score_sheet_{i}" for i in range(13)] +
    [f"count_{i}" for i in range(1, 7)] +
    ["dice_sum", "has_pair", "has_three", "has_four",
     "has_two_pair", "has_full_house", "has_yahtzee"]
)
y_rich_exp2 = df_exp2['score_category_name'].copy()
X_rich_exp2 = df_exp2_extended[feature_rich_exp2].copy()
X_train_rich_exp2, X_test_rich_exp2, y_train_rich_exp2, y_test_rich_exp2 = train_test_split(
    X_rich_exp2, y_rich_exp2, test_size=0.2, random_state=SEED)

# Seed BC scoring tree
complex_model_rich_exp2 = DecisionTreeClassifier(max_depth=SCORE_DEPTH, random_state=SEED)
complex_model_rich_exp2.fit(X_train_rich_exp2, y_train_rich_exp2)
print(f"Seed scoring tree trained (depth {SCORE_DEPTH}): {len(X_train_rich_exp2):,} samples")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERT MODEL
# ─────────────────────────────────────────────────────────────────────────────
torch.serialization.add_safe_globals([Algorithm])
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
hparams    = checkpoint['hyper_parameters']
features   = create_features(hparams['phi_features'].split(','))

model = YahtzeeAgent(
    hidden_size                   = hparams['hidden_size'],
    num_hidden                    = hparams['num_hidden'],
    dropout_rate                  = hparams['dropout_rate'],
    activation_function           = hparams['activation_function'],
    features                      = features,
    rolling_action_representation = hparams['rolling_action_representation'],
    he_kaiming_initialization     = False,
    use_layer_norm                = hparams.get('use_layer_norm', True),
)
state_dict = {k.replace('policy_net.', ''): v
              for k, v in checkpoint['state_dict'].items()
              if k.startswith('policy_net.')}
model.load_state_dict(state_dict)
model.eval()

DEVICE          = next(model.parameters()).device
CATEGORY_LABELS = ScoreCategory.LABELS
print(f"Expert model ready on {DEVICE}")

# EXPERT OBSERVATION + QUERY HELPERS
def make_obs(dice, rolls_used, phase, available_categories, score_sheet=None):
    if score_sheet is None:
        score_sheet = [0] * 13
    phase_val = Phase.ROLLING if phase.upper() == 'ROLLING' else Phase.SCORING
    return {
        'dice':                       np.array(dice, dtype=np.int64),
        'rolls_used':                 np.int64(rolls_used),
        'phase':                      phase_val,
        'score_sheet':                np.array(score_sheet, dtype=np.int64),
        'score_sheet_available_mask': np.array(available_categories, dtype=np.int64),
    }

def query_expert(obs, greedy=True):
    with torch.no_grad():
        state_tensor = phi(obs, model.features, DEVICE).unsqueeze(0)
        rolling_probs, scoring_probs, _, _ = model.forward(state_tensor)
        if greedy:
            rolling_action        = rolling_probs.argmax(dim=-1)
            scoring_action_tensor = scoring_probs.argmax(dim=-1)
        else:
            rolling_action, scoring_action_tensor = select_action(
                rolling_probs, scoring_probs, model.rolling_action_representation)
        hold_mask = convert_rolling_action_to_hold_mask(
            rolling_action, model.rolling_action_representation)
        scoring_category = int(scoring_action_tensor.item())
    return {
        'hold_mask':      [int(h) for h in hold_mask.tolist()],
        'category_index': scoring_category,
        'category_name':  CATEGORY_LABELS[scoring_category],
    }

# FEATURE BUILDERS (CONSISTENT WITH WORKING SIMULATION)

def prepare_features(dice_row, score_sheet_row, avail_row, feature_cols, rolls_used=2):
    """Build feature vector (returns 1D array)."""
    fd = {}
    for i, d in enumerate(dice_row, 1):
        fd[f'dice_{i}'] = int(d)
    fd['rolls_used'] = int(rolls_used)
    for i in range(13):
        fd[f'cat_available_{i}'] = int(avail_row[i])
        fd[f'score_sheet_{i}']   = int(score_sheet_row[i]) if score_sheet_row[i] is not None else 0
    for face in range(1, 7):
        fd[f'count_{face}'] = sum(1 for d in dice_row if int(d) == face)
    fd['dice_sum'] = int(sum(int(d) for d in dice_row))
    counts = [fd[f'count_{f}'] for f in range(1, 7)]
    fd['has_pair']       = int(2 in counts)
    fd['has_three']      = int(3 in counts)
    fd['has_four']       = int(4 in counts)
    fd['has_yahtzee']    = int(max(counts) == 5)
    fd['has_full_house'] = int(sorted(counts, reverse=True)[:2] == [3, 2])
    fd['has_two_pair']   = int(sum(1 for c in counts if c == 2) == 2)
    return np.array([fd.get(col, 0) for col in feature_cols])


# SCORING + FALLBACK

def calc_score(dice, c):
    """Calculate score for a given category."""
    dice   = [int(d) for d in dice]
    counts = [dice.count(f) for f in range(1, 7)]
    s = sum(dice); mx = max(counts); top2 = sorted(counts, reverse=True)[:2]
    if c < 6:    return counts[c] * (c + 1)
    elif c == 6: return s if mx >= 3 else 0
    elif c == 7: return s if mx >= 4 else 0
    elif c == 8: return 25 if top2 == [3, 2] else 0
    elif c == 9:
        faces = set(dice)
        return 30 if ({1, 2, 3, 4} <= faces or {2, 3, 4, 5} <= faces or {3, 4, 5, 6} <= faces) else 0
    elif c == 10: return 40 if set(dice) in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}) else 0
    elif c == 11: return 50 if mx == 5 else 0
    elif c == 12: return s
    return 0

def greedy_available_cat(dice, available_cats, rng):
    """Choose best available category by greedy score."""
    avail = [c for c in range(13) if available_cats[c]]
    if not avail:
        return None
    scored = [(calc_score(dice, c), c) for c in avail]
    best   = max(s for s, _ in scored)
    return int(rng.choice([c for s, c in scored if s == best]))

def random_available_cat(available_cats, rng):
    """Choose random available category."""
    avail = [c for c in range(13) if available_cats[c]]
    return int(rng.choice(avail)) if avail else None

def fallback_cat(dice, available_cats, rng, mode='greedy'):
    """Fallback strategy when tree prediction is unavailable."""
    if mode == 'random':
        return random_available_cat(available_cats, rng)
    return greedy_available_cat(dice, available_cats, rng)

def _hold_vector_to_class(hold_vector):
    """Convert hold mask to class label."""
    return int(''.join(str(int(h)) for h in hold_vector), 2)

def _normalise_scoring_label(label):
    """Normalize scoring label to integer."""
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, str):
        if label in SIM_CATEGORIES:
            return SIM_CATEGORIES.index(label)
        sim_name = PAPE_TO_SIM.get(label)
        if sim_name in SIM_CATEGORIES:
            return SIM_CATEGORIES.index(sim_name)
    raise ValueError(f"Cannot convert scoring label to int: {label!r}")

def _normalise_rolling_label(label):
    """Normalize rolling label to integer."""
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, str):
        bits = [int(b) for b in label.strip('[]').replace(' ', '').split(',')]
        return int(''.join(str(b) for b in bits), 2)
    raise ValueError(f"Cannot convert rolling label to int: {label!r}")

def normalise_labels(y_array, fn):
    """Normalize all labels using a function."""
    return np.array([fn(v) for v in y_array], dtype=np.int64)


# EXPERT ACTIONS FOR COLLECTION

def _expert_rolling_action(dice, rolls_used, available_cats, score_sheet):
    """Get expert's rolling action."""
    obs = make_obs([int(d) for d in dice], int(rolls_used), 'ROLLING',
                   [int(a) for a in available_cats],
                   [int(s) if s is not None else 0 for s in score_sheet])
    return query_expert(obs, greedy=True)['hold_mask']

def _expert_scoring_action(dice, available_cats, score_sheet):
    """Get expert's scoring action."""
    obs = make_obs([int(d) for d in dice], 2, 'SCORING',
                   [int(a) for a in available_cats],
                   [int(s) if s is not None else 0 for s in score_sheet])
    pape_name = query_expert(obs, greedy=True)['category_name']
    sim_name  = PAPE_TO_SIM.get(pape_name)
    return SIM_CATEGORIES.index(sim_name) if sim_name in SIM_CATEGORIES else None


# EXPERT EVALUATION (PURE EXPERT MODE, NO FALLBACKS)

def run_expert_simulations(batch_size=100, seed=42):
    """Run batch simulation using pure expert (no fallbacks)."""
    envs = SyncVectorEnv([create_env for _ in range(batch_size)])
    observations = envs.reset(seed=seed)[0]
    total_rewards        = np.zeros(batch_size, dtype=np.float64)
    cumulative_rewards   = []
    yahtzee_bonus_counts = np.zeros(batch_size, dtype=np.int32)

    with torch.no_grad():
        for step in range(39):
            phases     = observations["phase"]
            dice_batch = observations["dice"]
            ss_batch   = observations["score_sheet"]
            av_batch   = observations["score_sheet_available_mask"]
            ru_batch   = observations["rolls_used"]

            # ── Rolling (Expert) ──────────────────────────────────────────────
            hold_masks = []
            for i in range(batch_size):
                if phases[i] == Phase.ROLLING:
                    ru_i = int(np.array(ru_batch[i]).reshape(-1)[0])
                    hm = _expert_rolling_action(dice_batch[i], ru_i, av_batch[i], ss_batch[i])
                    hm = np.array(hm, dtype=bool)
                else:
                    hm = np.zeros(5, dtype=bool)
                hold_masks.append(hm)

            # ── Scoring (Expert) ──────────────────────────────────────────────
            score_cats = []
            for i in range(batch_size):
                if phases[i] == Phase.SCORING:
                    cat = _expert_scoring_action(dice_batch[i], av_batch[i], ss_batch[i])
                    if cat is None:
                        cat = 0
                else:
                    cat = 0
                score_cats.append(cat)

            actions = {"hold_mask": np.array(hold_masks), "score_category": np.array(score_cats)}
            observations, rewards = envs.step(actions)[:2]
            total_rewards += rewards
            cumulative_rewards.append(float(np.mean(total_rewards)))

    envs.close()
    fs = observations["score_sheet"]
    upper = np.sum(fs[:, 0:6], axis=1)
    lower = np.sum(fs[:, 6:13], axis=1)
    return (total_rewards, upper, lower, cumulative_rewards, fs,
            int(np.sum(upper >= 63)), int(np.sum(yahtzee_bonus_counts > 0)))


# EVALUATION (IDENTICAL LOGIC TO WORKING SIMULATION)

def run_hybrid_simulations(model, scoring_tree, feature_cols_scoring,
                           batch_size=100, seed=42, mode='dt_roll_dt_score',
                           rolling_tree=None, fallback_mode='greedy'):
    """Run batch simulation with given trees."""
    envs = SyncVectorEnv([create_env for _ in range(batch_size)])
    observations = envs.reset(seed=seed)[0]
    total_rewards        = np.zeros(batch_size, dtype=np.float64)
    cumulative_rewards   = []
    yahtzee_bonus_counts = np.zeros(batch_size, dtype=np.int32)
    fallback_counts      = np.zeros(batch_size, dtype=np.int32)
    ones_fallback_counts = np.zeros(batch_size, dtype=np.int32)
    rng = np.random.default_rng(seed)
    n_features_scoring = len(feature_cols_scoring) if feature_cols_scoring else 0

    with torch.no_grad():
        for step in range(39):
            phases     = observations["phase"]
            dice_batch = observations["dice"]
            ss_batch   = observations["score_sheet"]
            av_batch   = observations["score_sheet_available_mask"]
            ru_batch   = observations["rolls_used"]

            # ── Rolling ───────────────────────────────────────────────────────
            hold_masks = []
            for i in range(batch_size):
                if phases[i] == Phase.ROLLING:
                    try:
                        turn_idx = _turn_idx_from_avail(av_batch[i])
                        rmodel   = rolling_tree if rolling_tree is not None else complex_model_rich_exp1
                        rtree    = _subtree_for_turn(rmodel, turn_idx)
                        ru_i     = int(np.array(ru_batch[i]).reshape(-1)[0])
                        feat = prepare_features(dice_batch[i], ss_batch[i], av_batch[i],
                                                feature_rich, rolls_used=ru_i)
                        cls  = rtree.predict(feat.reshape(1, -1))[0]
                        hm   = np.array([int(b) for b in format(int(cls), '05b')], dtype=bool)
                    except Exception:
                        hm = rng.integers(0, 2, size=5).astype(bool)
                else:
                    hm = np.zeros(5, dtype=bool)
                hold_masks.append(hm)

            # ── Scoring ───────────────────────────────────────────────────────
            score_cats = []
            for i in range(batch_size):
                if phases[i] == Phase.SCORING:
                    try:
                        turn_idx = _turn_idx_from_avail(av_batch[i])
                        smodel   = scoring_tree if scoring_tree is not None else complex_model_rich_exp2
                        stree    = _subtree_for_turn(smodel, turn_idx)
                        feat = prepare_features(dice_batch[i], ss_batch[i], av_batch[i],
                                                feature_cols_scoring)
                        if feat.shape[0] == n_features_scoring:
                            pred    = stree.predict(feat.reshape(1, -1))[0]
                            dt_pred = int(pred) if not isinstance(pred, str) else CAT_NAME_TO_IDX.get(pred, 0)
                            if av_batch[i][dt_pred]:
                                cat = dt_pred
                            else:
                                fallback_counts[i] += 1
                                if dt_pred == 0:
                                    ones_fallback_counts[i] += 1
                                cat = fallback_cat(dice_batch[i], av_batch[i], rng, fallback_mode)
                        else:
                            fallback_counts[i] += 1
                            cat = fallback_cat(dice_batch[i], av_batch[i], rng, fallback_mode)
                    except Exception:
                        fallback_counts[i] += 1
                        cat = fallback_cat(dice_batch[i], av_batch[i], rng, fallback_mode)

                    if is_yahtzee(dice_batch[i]) and not av_batch[i][11]:
                        yahtzee_bonus_counts[i] += 1
                else:
                    cat = 0
                score_cats.append(cat)

            actions = {"hold_mask": np.array(hold_masks), "score_category": np.array(score_cats)}
            observations, rewards = envs.step(actions)[:2]
            total_rewards += rewards
            cumulative_rewards.append(float(np.mean(total_rewards)))

    envs.close()
    fs = observations["score_sheet"]
    upper = np.sum(fs[:, 0:6], axis=1)
    lower = np.sum(fs[:, 6:13], axis=1)
    return (total_rewards, upper, lower, cumulative_rewards, fs,
            int(np.sum(upper >= 63)), int(np.sum(yahtzee_bonus_counts > 0)),
            fallback_counts, ones_fallback_counts)


# DATA COLLECTION (DAGGER ITERATION)

def run_dagger_iteration(rolling_tree, scoring_tree, fcr, fcs,
                         n_games=1000, beta=0.5, fallback_mode='greedy', rng=None):
    """Collect data from one DAgger iteration with mixed expert/learner rollouts."""
    if rng is None:
        rng = np.random.default_rng()
    Xr, yr, Xs, ys = [], [], [], []
    roll_agree = roll_total = score_agree = score_total = 0

    for _ in range(n_games):
        score_sheet = [None] * 13
        available   = [True] * 13

        for _turn in range(13):
            dice = [int(rng.integers(1, 7)) for _ in range(5)]

            # Rolling phase
            for ru in range(2):
                feat = prepare_features(dice, score_sheet, available, fcr, rolls_used=ru)
                expert_hold  = _expert_rolling_action(dice, ru, available, score_sheet)
                expert_class = _hold_vector_to_class(expert_hold)
                Xr.append(feat)
                yr.append(expert_class)

                if beta >= 0.9999:
                    # Pure expert mode (beta=1.0)
                    hold = expert_hold
                else:
                    try:
                        tree_class = int(rolling_tree.predict(feat.reshape(1, -1))[0])
                    except Exception:
                        tree_class = -1
                    roll_total += 1
                    roll_agree += int(tree_class == expert_class)

                    # Follow tree with probability (1-beta), expert with probability beta
                    if rng.random() < beta or tree_class < 0:
                        hold = expert_hold
                    else:
                        hold = [int(b) for b in format(tree_class % 32, '05b')]
                
                for i in range(5):
                    if not hold[i]:
                        dice[i] = int(rng.integers(1, 7))

            # Scoring phase
            feat_s = prepare_features(dice, score_sheet, available, fcs, rolls_used=2)
            expert_cat = _expert_scoring_action(dice, available, score_sheet)
            if expert_cat is None or not available[expert_cat]:
                expert_cat = greedy_available_cat(dice, available, rng)
            Xs.append(feat_s)
            ys.append(int(expert_cat))

            if beta >= 0.9999:
                # Pure expert mode (beta=1.0)
                cat = expert_cat
            else:
                try:
                    p   = scoring_tree.predict(feat_s.reshape(1, -1))[0]
                    raw = int(p) if isinstance(p, (int, np.integer)) else _normalise_scoring_label(p)
                except Exception:
                    raw = -1
                score_total += 1
                score_agree += int(raw == expert_cat)

                # Follow tree with probability (1-beta), expert with probability beta
                if rng.random() < beta or raw < 0 or raw >= 13:
                    cat = expert_cat
                else:
                    cat = raw if available[raw] else fallback_cat(dice, available, rng, fallback_mode)
            
            if cat is None:
                break
            score_sheet[cat] = calc_score(dice, cat)
            available[cat]   = False

    roll_fid = roll_agree / max(roll_total, 1) if roll_total > 0 else None
    score_fid = score_agree / max(score_total, 1) if score_total > 0 else None
    return (np.array(Xr), np.array(yr), np.array(Xs), np.array(ys), roll_fid, score_fid)

# EVALUATION FUNCTION
def evaluate_trees(rolling_tree, scoring_tree,
                   feature_cols_rolling, feature_cols_scoring,
                   n_eval_games=500, fallback_mode='greedy', seed=42):
    """Evaluate trees on fresh games."""
    rewards, upper, lower, _, _, upper_bonus_hits, _ , \
    fallback_counts, ones_fallback_counts = run_hybrid_simulations(
        model=None,
        scoring_tree=scoring_tree,
        feature_cols_scoring=feature_cols_scoring,
        batch_size=n_eval_games,
        seed=seed,
        mode='dt_roll_dt_score',
        rolling_tree=rolling_tree,
        fallback_mode=fallback_mode
    )

    scores = np.array(rewards)
    total_fb = int(np.sum(fallback_counts))
    total_ones = int(np.sum(ones_fallback_counts))

    return {
        'mean_score': float(np.mean(scores)),
        'median_score': float(np.median(scores)),
        'std_score': float(np.std(scores)),
        'bonus_rate': 100.0 * upper_bonus_hits / n_eval_games,
        'mean_upper': float(np.mean(upper)),
        'fallbacks_per_game': total_fb / n_eval_games,
        'ones_share': (100.0 * total_ones / total_fb if total_fb > 0 else 0.0),
        'scores': scores.tolist()
    }

def evaluate_expert(n_eval_games=500, seed=42):
    """Evaluate pure expert on fresh games."""
    rewards, upper, lower, _, fs, upper_bonus_hits, _ = run_expert_simulations(
        batch_size=n_eval_games,
        seed=seed
    )

    scores = np.array(rewards)

    return {
        'mean_score': float(np.mean(scores)),
        'median_score': float(np.median(scores)),
        'std_score': float(np.std(scores)),
        'bonus_rate': 100.0 * upper_bonus_hits / n_eval_games,
        'mean_upper': float(np.mean(upper)),
        'fallbacks_per_game': 0.0,
        'ones_share': 0.0,
        'scores': scores.tolist()
    }


# FULL DAGGER LOOP

def run_full_dagger(initial_rolling_tree, initial_scoring_tree,
                    feature_cols_rolling, feature_cols_scoring,
                    seed_X_roll, seed_y_roll, seed_X_score, seed_y_score,
                    n_iterations=DAGGER_ITERATIONS,
                    games_per_iteration=GAMES_PER_ITERATION,
                    eval_games=EVAL_GAMES,
                    roll_depth=ROLL_DEPTH, score_depth=SCORE_DEPTH,
                    fallback_mode='greedy', seed=SEED):
    """Run full DAgger algorithm for a single fallback_mode.

    NOTE: every call is reseeded internally (collect_rng + eval seed) so that
    running this twice with different fallback_mode values, in the same
    process or separately, always reproduces identical numbers for a given
    fallback_mode.
    """
    # Re-fix global seeds at the start of every run so the two fallback-mode
    # runs are fully independent and reproducible regardless of call order.
    set_global_seed(seed)

    agg_X_roll  = np.asarray(seed_X_roll,  dtype=float)
    agg_y_roll  = normalise_labels(seed_y_roll,  _normalise_rolling_label)
    agg_X_score = np.asarray(seed_X_score, dtype=float)
    agg_y_score = normalise_labels(seed_y_score, _normalise_scoring_label)

    current_rolling = initial_rolling_tree
    current_scoring = initial_scoring_tree
    betas = np.linspace(1.0, 0.0, n_iterations)
    collect_rng = np.random.default_rng(seed)

    history = {k: [] for k in
               ['iteration', 'beta', 'dataset_size_roll', 'dataset_size_score',
                'mean_score', 'median_score', 'std_score', 'bonus_rate',
                'mean_upper', 'fallbacks_per_game', 'ones_share',
                'roll_fidelity', 'score_fidelity']}

    print("\n" + "=" * 65)
    print(f"BASELINE EVALUATION (BC trees)  [fallback_mode={fallback_mode}]")
    print("=" * 65)
    baseline = evaluate_trees(current_rolling, current_scoring,
                              feature_cols_rolling, feature_cols_scoring,
                              n_eval_games=eval_games, fallback_mode=fallback_mode, seed=seed)
    print(f"  Mean {baseline['mean_score']:.2f} | Median {baseline['median_score']:.2f} "
          f"| Bonus {baseline['bonus_rate']:.1f}% | Fallbacks/game {baseline['fallbacks_per_game']:.2f}")
    print("  (Baseline computed for reference only - excluded from saved history/plots)")

    # NOTE: the BC baseline is intentionally NOT appended to `history`.
    # history therefore only ever contains DAgger iterations 1..n_iterations.

    for it in range(n_iterations):
        beta = float(betas[it])
        if beta >= 0.9999:
            print("\n" + "=" * 65)
            print(f"DAGGER ITERATION {it+1}/{n_iterations}   beta=1.00 (PURE EXPERT, collecting {games_per_iteration} games)")
            print("=" * 65)
        else:
            print("\n" + "=" * 65)
            print(f"DAGGER ITERATION {it+1}/{n_iterations}   beta={beta:.2f}  (collecting {games_per_iteration} games)")
            print("=" * 65)

        nXr, nyr, nXs, nys, roll_fid, score_fid = run_dagger_iteration(
            current_rolling, current_scoring,
            feature_cols_rolling, feature_cols_scoring,
            n_games=games_per_iteration, beta=beta,
            fallback_mode=fallback_mode, rng=collect_rng)

        agg_X_roll  = np.vstack([agg_X_roll,  nXr])
        agg_y_roll  = np.concatenate([agg_y_roll,  nyr.astype(np.int64)])
        agg_X_score = np.vstack([agg_X_score, nXs])
        agg_y_score = np.concatenate([agg_y_score, nys.astype(np.int64)])

        current_rolling = DecisionTreeClassifier(max_depth=roll_depth, random_state=seed)
        current_rolling.fit(agg_X_roll, agg_y_roll)
        current_scoring = DecisionTreeClassifier(max_depth=score_depth, random_state=seed)
        current_scoring.fit(agg_X_score, agg_y_score)

        if beta >= 0.9999:
            result = evaluate_expert(n_eval_games=eval_games, seed=seed)
            print(f"  (Expert mode - no fidelity, no fallbacks) | "
                  f"Mean {result['mean_score']:.2f} (baseline {baseline['mean_score']:.2f}) | "
                  f"Bonus {result['bonus_rate']:.1f}%")
        else:
            result = evaluate_trees(current_rolling, current_scoring,
                                    feature_cols_rolling, feature_cols_scoring,
                                    n_eval_games=eval_games, fallback_mode=fallback_mode, seed=seed)
            print(f"  fidelity roll={roll_fid:.3f} score={score_fid:.3f} | "
                  f"Mean {result['mean_score']:.2f} (baseline {baseline['mean_score']:.2f}) | "
                  f"Bonus {result['bonus_rate']:.1f}% | Fallbacks/game {result['fallbacks_per_game']:.2f} "
                  f"(Ones {result['ones_share']:.1f}%)")

        for k, v in [('iteration', it + 1), ('beta', beta),
                     ('dataset_size_roll', len(agg_X_roll)), ('dataset_size_score', len(agg_X_score)),
                     ('mean_score', result['mean_score']), ('median_score', result['median_score']),
                     ('std_score', result['std_score']), ('bonus_rate', result['bonus_rate']),
                     ('mean_upper', result['mean_upper']),
                     ('fallbacks_per_game', result['fallbacks_per_game']),
                     ('ones_share', result['ones_share']),
                     ('roll_fidelity', roll_fid), ('score_fidelity', score_fid)]:
            history[k].append(v)

    print("\n" + "=" * 65)
    print(f"DAGGER COMPLETE  [fallback_mode={fallback_mode}]")
    print("=" * 65)
    print(f"  BC baseline mean (reference only) : {baseline['mean_score']:.2f}")
    print(f"  Final DAgger mean                 : {history['mean_score'][-1]:.2f}")
    print(f"  Improvement vs BC                 : {history['mean_score'][-1] - baseline['mean_score']:+.2f} points")
    return current_rolling, current_scoring, history, baseline


# PLOTTING (BC baseline fully excluded - no marker, no reference scatter point)

def plot_dagger_results(history, save_path, fallback_mode,
                        expert_mean=EXPERT_MEAN_REF, expert_bonus=EXPERT_BONUS_REF):
    """Plot DAgger training progress for iterations 1..N only.
    The BC baseline (iteration 0) is not part of `history` at all, so it
    cannot appear in any panel."""
    iters = np.array(history['iteration'])
    m = np.array(history['mean_score'])
    s = np.array(history['std_score'])
    med = np.array(history['median_score'])
    bonus = np.array(history['bonus_rate'])
    fb_per_game = np.array(history['fallbacks_per_game'])
    ones_share = np.array(history['ones_share'])
    it_f = [i for i, v in zip(iters, history['roll_fidelity']) if v is not None]
    rf   = [v for v in history['roll_fidelity'] if v is not None]
    sf   = [v for v in history['score_fidelity'] if v is not None]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'DAgger Training Progress (Expert at β=1.0, {fallback_mode.capitalize()} Fallback)',
                 fontsize=14, fontweight='bold')

    # Score
    ax = axes[0, 0]
    ax.plot(iters, m, marker='o', color='#4C72B0', label='DAgger mean', linewidth=2)
    ax.plot(iters, med, marker='s', ls='--', color='#DD8452', label='DAgger median', linewidth=2)
    ax.fill_between(iters, m - s, m + s, alpha=0.15, color='#4C72B0')
    ax.axhline(expert_mean, color='#2D6A4F', ls=':', lw=1.8, label=f'Expert ({expert_mean:.0f})')
    ax.set_xlabel('DAgger iteration', fontsize=11)
    ax.set_ylabel('Full-game score', fontsize=11)
    ax.set_title('Score vs DAgger iteration', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(min(iters) - 0.5, max(iters) + 0.5)

    # Upper bonus
    ax = axes[0, 1]
    ax.plot(iters, bonus, marker='o', color='#DD8452', linewidth=2)
    ax.axhline(expert_bonus, color='#2D6A4F', ls=':', lw=1.8, label=f'Expert ({expert_bonus:.1f}%)')
    ax.set_xlabel('DAgger iteration', fontsize=11)
    ax.set_ylabel('Upper-bonus rate (%)', fontsize=11)
    ax.set_title('Upper-section bonus vs iteration', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(min(iters) - 0.5, max(iters) + 0.5)

    # Local fidelity
    ax = axes[1, 0]
    if it_f and rf:
        ax.plot(it_f, np.array(rf) * 100, marker='o', color='#4C72B0', label='Rolling', linewidth=2)
        ax.plot(it_f, np.array(sf) * 100, marker='s', color='#DD8452', label='Scoring', linewidth=2)
    ax.set_xlabel('DAgger iteration', fontsize=11)
    ax.set_ylabel('Expert agreement (%)', fontsize=11)
    ax.set_title('Local fidelity on learner-visited states', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Fallback rate
    ax = axes[1, 1]
    ax.plot(iters, fb_per_game, marker='o', color='#8172B3', linewidth=2,
            label='Fallbacks/game')
    ax.set_xlabel('DAgger iteration', fontsize=11)
    ax.set_ylabel('Fallbacks per game', color='#8172B3', fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(iters, ones_share, marker='s', ls='--', color='#937860', linewidth=2,
             label='Ones share (%)')
    ax2.set_ylabel('Ones share of fallbacks (%)', color='#937860', fontsize=11)
    ax.set_title('Scoring fallbacks vs iteration', fontsize=12)
    ax.grid(alpha=0.3)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=9, loc='center right')
    ax.set_xlim(min(iters) - 0.5, max(iters) + 0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close(fig)
    return fig

def print_results_table(history, fallback_mode, baseline=None):
    """Print summary table. Baseline is printed separately for reference,
    but is NOT part of `history` and is not included in the table rows."""
    print("\n" + "=" * 92)
    print(f"DAGGER RESULTS SUMMARY  [fallback_mode={fallback_mode}]")
    print("=" * 92)
    if baseline is not None:
        print(f"(Reference only, not in plots/history) BC baseline: "
              f"mean={baseline['mean_score']:.2f} bonus={baseline['bonus_rate']:.1f}% "
              f"fb/gm={baseline['fallbacks_per_game']:.2f}")
        print("-" * 92)
    print(f"{'Iter':>4} {'Beta':>5} {'Mean':>7} {'Median':>7} {'Std':>6} "
          f"{'Bonus%':>7} {'FB/gm':>6} {'Ones%':>6} {'RollFid':>8} {'ScoreFid':>9}")
    print("-" * 92)
    for i in range(len(history['iteration'])):
        rf = history['roll_fidelity'][i]
        sf = history['score_fidelity'][i]
        rf_s = f"{rf:.3f}" if rf is not None else "  BC "
        sf_s = f"{sf:.3f}" if sf is not None else "  BC "
        print(f"{history['iteration'][i]:>4} {history['beta'][i]:>5.2f} "
              f"{history['mean_score'][i]:>7.2f} {history['median_score'][i]:>7.2f} "
              f"{history['std_score'][i]:>6.2f} {history['bonus_rate'][i]:>7.1f} "
              f"{history['fallbacks_per_game'][i]:>6.2f} {history['ones_share'][i]:>6.1f} "
              f"{rf_s:>8} {sf_s:>9}")
    print("=" * 92)


# MAIN — runs DAgger once per fallback mode in FALLBACK_MODES_TO_RUN

if __name__ == '__main__' or True:
    if not FULL_RUN:
        print("\nFULL_RUN=False - quick smoke test\n")
        _n_iter, _n_games, _n_eval = 2, 50, 100
    else:
        _n_iter, _n_games, _n_eval = DAGGER_ITERATIONS, GAMES_PER_ITERATION, EVAL_GAMES

    all_results = {}

    for fb_mode in FALLBACK_MODES_TO_RUN:
        print("\n\n" + "#" * 79)
        print(f"#  RUNNING DAGGER WITH FALLBACK_MODE = '{fb_mode}'  (seed={SEED})")
        print("#" * 79)

        dagger_rolling_tree, dagger_scoring_tree, dagger_history, baseline_result = run_full_dagger(
            initial_rolling_tree = complex_model_rich_exp1,
            initial_scoring_tree = complex_model_rich_exp2,
            feature_cols_rolling = feature_rich,
            feature_cols_scoring = feature_rich_exp2,
            seed_X_roll  = X_train_rich,       seed_y_roll  = y_train_rich,
            seed_X_score = X_train_rich_exp2,  seed_y_score = y_train_rich_exp2,
            n_iterations = _n_iter, games_per_iteration = _n_games, eval_games = _n_eval,
            fallback_mode = fb_mode, seed = SEED,
        )

        print_results_table(dagger_history, fallback_mode=fb_mode, baseline=baseline_result)

        out_png = f'dagger_results_{fb_mode}.png'
        plot_dagger_results(dagger_history, save_path=out_png, fallback_mode=fb_mode)

        print(f"\nFinal evaluation (10000 games, {fb_mode} fallback) for thesis table")
        final_eval = evaluate_trees(dagger_rolling_tree, dagger_scoring_tree,
                                    feature_rich, feature_rich_exp2,
                                    n_eval_games=10000, fallback_mode=fb_mode, seed=SEED)
        print(f"  Final DAgger : mean {final_eval['mean_score']:.2f} | "
              f"median {final_eval['median_score']:.2f} | bonus {final_eval['bonus_rate']:.1f}% | "
              f"fallbacks/game {final_eval['fallbacks_per_game']:.2f} (Ones {final_eval['ones_share']:.1f}%)")
        print(f"  BC baseline (reference only) : mean {baseline_result['mean_score']:.2f} | "
              f"bonus {baseline_result['bonus_rate']:.1f}%")
        print(f"  Expert (ref) : mean {EXPERT_MEAN_REF:.2f} | bonus {EXPERT_BONUS_REF:.1f}%")

        import pickle
        with open(f'dagger_trees_final_{fb_mode}.pkl', 'wb') as f:
            pickle.dump({'rolling_tree': dagger_rolling_tree,
                         'scoring_tree': dagger_scoring_tree,
                         'feature_cols_rolling': feature_rich,
                         'feature_cols_scoring': feature_rich_exp2,
                         'history': dagger_history,
                         'baseline_reference_only': baseline_result,
                         'fallback_mode': fb_mode,
                         'seed': SEED}, f)
        print(f"DAgger trees saved to dagger_trees_final_{fb_mode}.pkl")

        all_results[fb_mode] = {
            'history': dagger_history,
            'baseline_reference_only': baseline_result,
            'final_eval': final_eval,
        }

    print("\n\n" + "=" * 79)
    print("BOTH RUNS COMPLETE")
    print("=" * 79)
    for fb_mode in FALLBACK_MODES_TO_RUN:
        fe = all_results[fb_mode]['final_eval']
        print(f"  [{fb_mode:>6}] final mean={fe['mean_score']:.2f}  "
              f"bonus={fe['bonus_rate']:.1f}%  fb/gm={fe['fallbacks_per_game']:.2f}  "
              f"-> dagger_results_{fb_mode}.png")
