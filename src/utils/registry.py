import gymnasium
from typing import Any, Dict
import importlib
from .env_wrappers import FlappyBirdWrapper, TetrisWrapper

class EnvRegistry:
    _required_imports = {
        "FlappyBird-v0": ["flappy_bird_gymnasium"],
        "TetrisA-v0": ["gym_tetris", "nes_py.wrappers", "gym_tetris.actions"],
    }

    _wrappers = {
        "FlappyBird-v0": FlappyBirdWrapper,
        "TetrisA-v0": TetrisWrapper,
    }

    _imported_modules: Dict[str, Any] = {}

    @classmethod
    def _safe_import(cls, module_path: str):
        if module_path not in cls._imported_modules:
            try:
                cls._imported_modules[module_path] = importlib.import_module(module_path)
            except ImportError:
                raise ImportError(f"Required module '{module_path}' could not be imported.")
        return cls._imported_modules[module_path]

    @classmethod
    def make(cls, env_id: str, **kwargs) -> gymnasium.Env:
        if env_id in cls._required_imports:
            for module in cls._required_imports[env_id]:
                cls._safe_import(module)

        if env_id == "TetrisA-v0":
            gym_tetris = cls._imported_modules["gym_tetris"]
            nes_wrappers = cls._imported_modules["nes_py.wrappers"]
            movement = cls._imported_modules["gym_tetris.actions"].SIMPLE_MOVEMENT
            # movement = SIMPLE_MOVEMENT
            env = gym_tetris.make(env_id)
            env = nes_wrappers.JoypadSpace(env, movement)
        else:
            env = gymnasium.make(env_id, **kwargs)

        wrapper_cls = cls._wrappers.get(env_id)
        if wrapper_cls is not None:
            env = wrapper_cls(env)

        return env
    

SIMPLE_MOVEMENT = [
    # ['NOOP'],d
    # ['A'],
    # ['B'],
    ['right'],
    ['left'],
    ['down'],
]