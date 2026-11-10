"""Project-wide settings: model pairs, paths and episode constants. Change things here, not in scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
POLICY_DIR = ROOT / "policies"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

PROMPTS_FILE = DATA_DIR / "prompts.jsonl"

# Draft and target must share a tokenizer (same model family).
PAIRS = {
    "qwen": ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    "smol": ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
}

ALGO_NAMES = {"vi": "Value Iteration", "qlearn": "Q-learning", "dqn": "DQN", "reinforce": "REINFORCE"}

K = 8  # max draft tokens per round; STOP is forced at s = K
MAX_NEW_TOKENS = 128  # answer length per prompt
SEED = 0
