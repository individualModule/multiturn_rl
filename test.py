# First cell - Setup and imports
import sys
import os

# Add project roots to Python path
project_root = "/Users/dicaristic/pyProjects/thesisProject"
clemcore_root = os.path.join(project_root, "clemcore_multiturn_rl")
clembench_root = os.path.join(project_root, "clembench_multiturn_rl")

# Add both paths
for path in [project_root, clemcore_root, clembench_root]:
    if path not in sys.path:
        sys.path.append(path)

# Now add the codenames package path specifically
codenames_path = os.path.join(clembench_root, "codenames")
if codenames_path not in sys.path:
    sys.path.append(codenames_path)

from clemcore.playpen.envs.game_env import GameEnv 
from clemcore.playpen.buffers import StepRolloutBuffer
from codenames.master import CodenamesGameBenchmark
from clemcore.clemgame import GameInstanceIterator, GameSpec
from clemcore.backends import CustomResponseModel

# Create mock models for testing
class MockModel(CustomResponseModel):
    def __init__(self, name="mock"):
        super().__init__(name)
        
    def generate(self, prompt):
        # Mock response generation
        if "Cluegiver" in prompt:
            return "clue: animal 2\ntargets: dog, cat"
        else:
            return "guesses: dog, cat"

# First create a minimal instances.json if it doesn't exist
minimal_experiment = {
    "experiments": [{
        "name": "test_experiment",
        "index": 0,
        "game_instances": [{
            "game_id": 0,
            "board": ["dog", "cat", "fish", "bird", "snake"],
            "assignments": {
                "team": ["dog", "cat"],
                "opponent": ["fish", "bird"],
                "innocent": ["snake"],
                "assassin": []
            }
        }]
    }]
}

instances_path = os.path.join(codenames_path, "in/instances.json")
os.makedirs(os.path.dirname(instances_path), exist_ok=True)
if not os.path.exists(instances_path):
    import json
    with open(instances_path, 'w') as f:
        json.dump(minimal_experiment, f)

# Setup game spec and benchmark
game_spec = GameSpec.from_dict({
    "game_name": "codenames",
    "game_path": codenames_path,
    "players": ["cluegiver", "guesser"]
})

game = CodenamesGameBenchmark(game_spec)

# Load experiments and instances
experiments = game.load_json(instances_path)
task_iterator = GameInstanceIterator(experiments)


# Create models and environment
models = [MockModel("cluegiver"), MockModel("guesser")]
env = GameEnv(game, models, task_iterator)
buffer = StepRolloutBuffer(env)