import gymnasium
from typing import Dict, Any

class EnvRegistry:
    _required_imports = {
        "FlappyBird-v0": "import flappy_bird_gymnasium",
        # Add more environments and their required imports here
    }

    @classmethod
    def make(cls, env_id: str, **kwargs) -> gymnasium.Env:
        """Create environment with auto-import of required packages."""
        # Check if environment needs special import
        if env_id in cls._required_imports:
            try:
                exec(cls._required_imports[env_id])
            except ImportError as e:
                raise ImportError(f"Environment {env_id} requires: {cls._required_imports[env_id]}")
        
        return gymnasium.make(env_id, **kwargs)