"""All settings and thresholds live here so nothing is hard-coded elsewhere."""
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BASELINE_PATH = DATA_DIR / "baseline.csv"
CURRENT_PATH = DATA_DIR / "current.csv"
METRICS_PATH = DATA_DIR / "model_metrics.json"

# ---------- Data generation ----------
RANDOM_SEED = 42
N_BASELINE_ROWS = 5000
N_CURRENT_ROWS = 5000
TEST_SIZE = 0.2                      # share of baseline held out to score the model

TARGET_COLUMN = "churned"
NUMERIC_FEATURES = ["income", "account_balance", "credit_score",
                    "tenure_months", "age", "support_tickets_90d"]
CATEGORICAL_FEATURES = ["region"]
UNKNOWN_TOKEN = "UNKNOWN"

# ---------- Incident settings (what we deliberately break in current.csv) ----------
BASELINE_INCOME_MISSING_RATE = 0.02
CURRENT_INCOME_MISSING_RATE = 0.20
BALANCE_INFLATION = 1.35
BALANCE_NOISE_STD = 0.05             # 5% random noise on top of the inflation
CURRENT_UNKNOWN_REGION_RATE = 0.08

# ---------- Investigation thresholds ----------
MISSING_INCREASE_THRESHOLD = 0.05    # flag if missing rate rises by more than 5 points
UNKNOWN_INCREASE_THRESHOLD = 0.02    # flag if "UNKNOWN" share rises by more than 2 points
PSI_BINS = 10                        # number of buckets used for drift calculation
PSI_MODERATE = 0.10                  # PSI >= 0.10 -> moderate drift
PSI_MAJOR = 0.25                     # PSI >= 0.25 -> major drift
METRIC_DROP_THRESHOLD = 0.05         # flag a metric if it falls by more than 0.05


# ---------- Day 3: Groq agent ----------
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"   # override with the GROQ_MODEL environment variable
SERVER_SCRIPT = BASE_DIR / "src" / "mcp_server.py"
AGENT_MAX_ITERATIONS = 6             # max LLM calls per investigation (last one must be the report)
GROQ_MAX_OUTPUT_TOKENS = 2000        # upper bound on one LLM reply (includes reasoning tokens)
GROQ_RATE_LIMIT_RETRIES = 3          # free tier has per-minute limits, so wait and retry
GROQ_RETRY_WAIT_SECONDS = 20
MCP_CONNECT_TIMEOUT_SECONDS = 30     # give up if the MCP server does not answer
MAX_TOOL_RESULT_CHARS = 6000         # cap a tool result so it cannot flood the LLM context
MCP_SERVER_ENV = {"OPENBLAS_NUM_THREADS": "1"}


# ---------- Day 4: multi-agent ----------
SPECIALIST_MAX_ITERATIONS = 6        # LLM steps per specialist (the last one must be its answer)
DATA_AGENT_TOOLS = ["get_dataset_stats", "check_data_quality", "detect_drift"]
MODEL_AGENT_TOOLS = ["compare_model_metrics"]


# ---------- Day 5: web UI ----------
UI_MAX_INCIDENT_CHARS = 300          # keep prompts short (also limits abuse on a public demo)
UI_COOLDOWN_SECONDS = 20             # minimum gap between runs from one browser session
EXAMPLE_INCIDENTS = [
    "Churn model recall dropped this week. What happened?",
    "Did anything change in our input data compared to the baseline?",
    "Our churn model's accuracy metrics changed. Has the model itself degraded?",
]