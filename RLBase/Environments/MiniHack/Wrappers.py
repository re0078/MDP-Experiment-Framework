
# ADD WRAPPERS HERE





WRAPPING_TO_WRAPPER = {

}


DEFAULT_GOAL_POSITIONS_BY_SEED: dict[int, tuple[int, int]] = {
    # seed: (goal_x, goal_y) // (column idx, row idx)
    10: (4, 4),
    12: (66, 7),
    20: (27, 17),
    30: (46, 16),
    37: (36, 5),
    44: (67, 7),
    49: (26, 3),
    50: (4, 3),
    52: (21, 5),
    57: (52, 14),
    60: (19, 13),
}
# good seeds: 10, 20, 30, 37, 44, 49, 50, 52, 57, 60

import copy
import importlib
import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from gymnasium.vector import VectorWrapper

import minihack

from nle import nethack
from nle.env.tasks import NetHackStaircase

class OneHotCharsWrapper(ObservationWrapper):
    def __init__(self, env, char_vocab=(" ", "-", "|", "#", ".", "<", ">", "@", "+")):
        super().__init__(env)
        self.char_vocab = char_vocab
        # Reserve last index for unknown
        self.char_to_idx = {c: i for i, c in enumerate(char_vocab)}
        self.unknown_idx = len(char_vocab)

        char_shape = env.observation_space["chars"].shape  # (H, W)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=char_shape + (len(char_vocab) + 1,),  # +1 for unknown
            dtype=np.int8,
        )

    def observation(self, obs):
        chars = obs["chars"]  # (H, W) integers (ASCII codes)
        # Convert ASCII -> characters
        char_array = np.vectorize(chr)(chars)
        # Map characters to indices (default → unknown_idx)
        idx_array = np.vectorize(self.char_to_idx.get)(char_array, self.unknown_idx)
        # One-hot encode
        one_hot = np.eye(len(self.char_vocab) + 1, dtype=np.int8)[idx_array]
        return one_hot
    
class FixedSeedWrapper(gym.Wrapper):
    """Always reset MiniHack with the same seed so the map layout is identical."""
    def __init__(self, env, seed: int):
        super().__init__(env)
        self._seed = int(seed)

    # Gymnasium API: match the signature exactly
    def reset(self, *, seed=None, options=None):
        # Always force the same seed (ignore caller's seed)
        base = self.env.unwrapped
        base.seed(self._seed)
        return self.env.reset(seed=self._seed, options=options)

class MovementActionWrapper(gym.Wrapper):
    """
    Restrict the action space to only movement actions in MiniHack/NLE.
    """
    def __init__(self, env):
        super().__init__(env)
        base = self.env.unwrapped
        # All 8 compass directions
        movement_actions = list(nethack.CompassDirection)
        base.actions = movement_actions

class MiniHackWrap(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        seed: int | None = None,
        view_size: int = 9,
        step_reward: float = -1.0,
        goal_reward: float = 1.0,
        goal_chars: tuple[str, ...] = (">",),
        use_chars: bool = True,
        one_hot: bool = True,
        n_char_classes: int = 256,
        include_dxdy: bool = False,
        char_vocab: tuple[str, ...] | None = None,
        include_other_class: bool = True,
        include_goal_direction: bool = True,
        goal_positions_by_seed: dict[int, tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.env = env
        self._env_ctor = self._infer_env_ctor(env)
        self.seed_ = seed
        self.view_size = int(view_size)
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        self.goal_chars = tuple(goal_chars)
        self.use_chars = bool(use_chars)
        self.one_hot = bool(one_hot)
        self.n_char_classes = int(n_char_classes)
        self.include_dxdy = bool(include_dxdy)
        self.include_other_class = bool(include_other_class)
        self.include_goal_direction = bool(include_goal_direction)

        # Precompute eye for one-hot to avoid reallocs
        self._eye_chars = None
        self._lut_chars = None  # maps ASCII code -> compact index
        self._char_vocab = None
        self._other_char = "?"
        if self.one_hot and self.use_chars:
            # Default compact vocabulary tailored for MiniHack-Corridor
            # Visible chars: space, '-', '|', '#', '.', '<', '>', '@'
            if char_vocab is None:
                char_vocab = (" ", "-", "|", "#", ".", "<", ">", "@", "+", "-")
            self._char_vocab = tuple(char_vocab)
            # Build ASCII code list and LUT to compact indices
            vocab_codes = np.array([ord(c) for c in char_vocab], dtype=np.int64)
            vocab_size = len(vocab_codes) + (1 if self.include_other_class else 0)
            self._eye_chars = np.eye(vocab_size, dtype=np.float32)
            # LUT spans 256 codes; default to 'other' index if enabled else 0
            other_idx = vocab_size - 1 if self.include_other_class else 0
            lut = np.full(256, other_idx, dtype=np.int64)
            for i, code in enumerate(vocab_codes):
                lut[code] = i
            self._lut_chars = lut
        self.render_mode = self.env.render_mode

        # Delegate action space to underlying env
        self.action_space = env.action_space
        self.action_space = gym.spaces.discrete.Discrete(self.action_space.n - 3) # excluding open, kick, and search
        self._last_action_dim = self._infer_last_action_dim()
        self.last_action: int | None = None

        # Goal position bookkeeping (optional seed overrides)
        self.goal_positions_by_seed: dict[int, tuple[int, int]] = {}
        if DEFAULT_GOAL_POSITIONS_BY_SEED:
            self.goal_positions_by_seed.update(DEFAULT_GOAL_POSITIONS_BY_SEED)
        if goal_positions_by_seed:
            for key, value in goal_positions_by_seed.items():
                try:
                    seed_key = int(key)
                    gx, gy = value
                    self.goal_positions_by_seed[seed_key] = (int(gx), int(gy))
                except (TypeError, ValueError):
                    warnings.warn(
                        f"Invalid goal position entry for seed {key!r}: {value!r} (expected iterable of two ints)",
                        RuntimeWarning,
                    )
        self._goal_position: tuple[int, int] | None = None

        # Infer observation space from env.observation_space without resetting.
        # This avoids triggering MiniHack/NLE reset during VectorEnv construction,
        # which can cause low-level crashes (SIGFPE) in some environments.
        self.last_obs = None
        self._last_info: dict[str, object] | None = None
        self._last_done_flag = False

        # Determine crop subspace shape
        crop_subspace = None
        try:
            obs_space = getattr(self.env, "observation_space", None)
            if isinstance(obs_space, spaces.Dict):
                if self.use_chars and "chars_crop" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["chars_crop"]
                elif (not self.use_chars) and "glyphs_crop" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["glyphs_crop"]
                elif self.use_chars and "chars" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["chars"]
                elif "glyphs" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["glyphs"]
        except Exception:
            crop_subspace = None

        if crop_subspace is not None and hasattr(crop_subspace, "shape"):
            crop_shape = tuple(int(x) for x in crop_subspace.shape)
        else:
            # Fallback: assume square crop of view_size if unknown
            crop_shape = (int(self.view_size), int(self.view_size))
        self._crop_shape = crop_shape
        self._crop_size = int(np.prod(crop_shape)) if len(crop_shape) > 0 else 0

        # Compute flattened feature size
        if self.one_hot and self.use_chars:
            vocab_size = self._eye_chars.shape[0] if self._eye_chars is not None else int(self.n_char_classes)
            flat_size = int(np.prod(crop_shape)) * int(vocab_size)
        else:
            flat_size = int(np.prod(crop_shape))
        if self.include_dxdy:
            flat_size += 2
        if self.include_goal_direction:
            flat_size += 4
        if self._last_action_dim:
            flat_size += self._last_action_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
        )

        self.spec = getattr(self.env, "spec", None)

        self.steps = 0

        # Initialize environment state immediately
        # so that last_obs and internal counters are set.
        # This mirrors other env wrappers in the repo.
        self.reset(seed=self.seed_)

    # ----- Pickling helpers -----
    @staticmethod
    def _safe_copy_kwargs(kwargs):
        if not kwargs:
            return {}
        try:
            return copy.deepcopy(kwargs)
        except Exception:
            return dict(kwargs)

    def _infer_env_ctor(self, env: gym.Env | None) -> dict[str, object] | None:
        if env is None:
            return None
        spec = getattr(env, "spec", None)
        ctor: dict[str, object] = {
            "id": None,
            "kwargs": {},
            "entry_point": None,
        }
        if spec is not None:
            ctor["id"] = getattr(spec, "id", None)
            entry_point = getattr(spec, "entry_point", None)
            if isinstance(entry_point, str):
                ctor["entry_point"] = entry_point
            elif entry_point is not None:
                qualname = getattr(entry_point, "__qualname__", getattr(entry_point, "__name__", None))
                if qualname is not None:
                    ctor["entry_point"] = f"{entry_point.__module__}.{qualname}"
                else:
                    ctor["entry_point"] = str(entry_point)
            ctor["kwargs"] = self._safe_copy_kwargs(getattr(spec, "kwargs", None))
        else:
            ctor["entry_point"] = f"{env.__module__}.{env.__class__.__qualname__}"
        return ctor

    @staticmethod
    def _call_env_method(env: gym.Env | None, names: tuple[str, ...], *args, **kwargs):
        if env is None:
            return False, None
        for name in names:
            method = getattr(env, name, None)
            if callable(method):
                try:
                    return True, method(*args, **kwargs)
                except TypeError:
                    continue
        return False, None

    def _infer_last_action_dim(self) -> int:
        if isinstance(self.action_space, spaces.Discrete):
            return int(self.action_space.n)
        return 0

    def _capture_env_state(self, env: gym.Env | None):
        candidates = (
            ("clone_state", ("restore_state", "set_state")),
            ("clone_full_state", ("restore_full_state", "restore_state", "set_state")),
            ("get_state", ("set_state", "restore_state")),
            ("state_dict", ("load_state_dict", "set_state_dict")),
        )
        for getter, setters in candidates:
            call_variants = (
                ((), {}),
                ((), {"include_information": True}),
            )
            for args, kwargs in call_variants:
                success, snapshot = self._call_env_method(env, (getter,), *args, **kwargs)
                if success and snapshot is not None:
                    return {
                        "getter": getter,
                        "setters": setters,
                        "snapshot": snapshot,
                        "call_kwargs": kwargs,
                    }
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is env:
            unwrapped = None
        if unwrapped is not None:
            data = self._capture_env_state(unwrapped)
            if data is not None:
                data.setdefault("unwrap_count", 0)
                data["unwrap_count"] += 1
                return data
        return None

    def _restore_env_state(self, env: gym.Env, payload: dict[str, object]) -> bool:
        if not payload:
            return False
        unwrap_count = payload.get("unwrap_count", 0)
        target_env = env
        for _ in range(int(unwrap_count)):
            target_env = getattr(target_env, "unwrapped", target_env)
        snapshot = payload.get("snapshot")
        setters = payload.get("setters", ())
        for setter in setters:
            success, _ = self._call_env_method(target_env, (setter,), snapshot)
            if success:
                return True
        extra_setters = (
            "restore_state",
            "set_state",
            "restore_full_state",
            "load_state_dict",
            "from_state",
        )
        for setter in extra_setters:
            success, _ = self._call_env_method(target_env, (setter,), snapshot)
            if success:
                return True
        return False

    def __getstate__(self):
        state = self.__dict__.copy()
        env = state.pop("env", None)
        pickled_env = None
        if env is not None:
            pickled_env = {
                "ctor": copy.deepcopy(self._env_ctor) if self._env_ctor else None,
                "state": self._capture_env_state(env),
                "done": bool(self._last_done_flag),
                "info": copy.deepcopy(self._last_info) if self._last_info is not None else None,
            }
        state["_pickled_env"] = pickled_env
        return state

    def __setstate__(self, state):
        pickled_env = state.pop("_pickled_env", None)
        self.__dict__.update(state)

        env = None
        ctor = None
        if pickled_env:
            ctor = pickled_env.get("ctor")
        if ctor and ctor.get("id"):
            env_kwargs = ctor.get("kwargs") or {}
            env = gym.make(ctor["id"], **env_kwargs)
        elif ctor and ctor.get("entry_point"):
            entry_point = ctor["entry_point"]
            if isinstance(entry_point, str) and entry_point:
                if ":" in entry_point:
                    module_name, qualname = entry_point.split(":", 1)
                else:
                    module_name, _, qualname = entry_point.rpartition(".")
                try:
                    module = importlib.import_module(module_name)
                    env_cls = getattr(module, qualname)
                    env = env_cls(**(ctor.get("kwargs") or {}))
                except (ModuleNotFoundError, AttributeError):
                    env = None
        if env is None:
            raise RuntimeError("Failed to reconstruct MiniHack environment during unpickling.")

        if self.seed_ is not None:
            try:
                env.unwrapped.seed(self.seed_)
            except Exception:
                pass

        restored = False
        if pickled_env and pickled_env.get("state"):
            restored = self._restore_env_state(env, pickled_env["state"])
        if not restored:
            try:
                reset_result = env.reset(seed=self.seed_)
            except TypeError:
                reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, _ = reset_result
            else:
                obs = reset_result
            self.last_obs = obs
            self.steps = 0
            self._last_info = {}
            self._last_done_flag = False
        else:
            if "done" in (pickled_env or {}):
                self._last_done_flag = bool(pickled_env.get("done", False))
            if "info" in (pickled_env or {}):
                self._last_info = pickled_env.get("info")

        self.env = env
        self._env_ctor = self._infer_env_ctor(env)
        self.action_space = env.action_space
        self._last_action_dim = self._infer_last_action_dim()
        self.spec = getattr(env, "spec", None)
        self._goal_position = None
        if isinstance(getattr(self, "last_obs", None), dict):
            self._goal_position = self._resolve_goal_position(self.last_obs)

    # ----- Observation helpers -----
    def _grid(self, obs) -> np.ndarray:
        # Prefer built-in cropped keys
        if self.use_chars and "chars_crop" in obs:
            return obs["chars_crop"]
        if (not self.use_chars) and "glyphs_crop" in obs:
            return obs["glyphs_crop"]
        # Fallbacks if ever needed
        if self.use_chars and "chars" in obs:
            return obs["chars"]
        return obs.get("glyphs", obs.get("chars"))

    def _agent_xy(self, obs) -> tuple[int, int]:
        chars = obs.get("chars")
        if isinstance(chars, np.ndarray):
            ys, xs = np.where(chars == ord("@"))
            if xs.size:
                return int(xs[0]), int(ys[0])
        bl = obs.get("blstats")
        if bl is None:
            return 0, 0
        return int(bl[0]), int(bl[1])

    def _global_goal_delta(self, obs) -> tuple[float, float]:
        """ Compute dx,dy using full observation (not the crop).
        dx = agent_x - goal_x, dy = agent_y - goal_y
        """
        ax, ay = self._agent_xy(obs)
        goal = self._goal_position
        if goal is None:
            goal = self._locate_goal_in_chars(obs)
            self._goal_position = goal
        if goal is None:
            return 0.0, 0.0
        gx, gy = goal
        return float(ax - gx), float(ay - gy)

    def _locate_goal_in_chars(self, obs) -> tuple[int, int] | None:
        chars = obs.get("chars")
        if not isinstance(chars, np.ndarray):
            return None
        targets = [ord(c) for c in self.goal_chars]
        ys, xs = np.where(np.isin(chars, targets))
        if xs.size == 0:
            return None
        ax, ay = self._agent_xy(obs)
        dists = np.abs(xs - ax) + np.abs(ys - ay)
        idx = int(np.argmin(dists))
        return int(xs[idx]), int(ys[idx])

    def _resolve_goal_position(self, obs) -> tuple[int, int] | None:
        seed_key: int | None = None
        if self.seed_ is not None:
            try:
                seed_key = int(self.seed_)
            except (TypeError, ValueError):
                seed_key = None
        if seed_key is not None and seed_key in self.goal_positions_by_seed:
            return self.goal_positions_by_seed[seed_key]
        return self._locate_goal_in_chars(obs)

    def _goal_direction(self, obs) -> np.ndarray:
        compass = np.zeros(4, dtype=np.float32)
        goal = self._goal_position
        if goal is None:
            goal = self._resolve_goal_position(obs)
            self._goal_position = goal
        if goal is None:
            return compass
        ax, ay = self._agent_xy(obs)
        gx, gy = goal
        if gy < ay:
            compass[0] = 1.0  # up
        if gx < ax:
            compass[1] = 1.0  # left
        if gx > ax:
            compass[2] = 1.0  # right
        if gy > ay:
            compass[3] = 1.0  # down
        return compass

    def _encode_crop(self, crop: np.ndarray) -> np.ndarray:
        # One-hot encode characters if requested
        if self.one_hot and self.use_chars:
            flat_codes = crop.astype(np.int64).reshape(-1)
            if self._lut_chars is not None:
                mapped = self._lut_chars[np.clip(flat_codes, 0, 255)]
                oh = self._eye_chars[mapped]
            else:
                # Fallback to 256-way one-hot
                flat_codes = np.clip(flat_codes, 0, self.n_char_classes - 1)
                oh = self._eye_chars[flat_codes]
            return oh.reshape(-1)
        # Fallback to raw values
        return crop.astype(np.float32).flatten()

    def _build_observation(self, obs) -> np.ndarray:
        grid = self._grid(obs)
        crop = grid  # always crop view
        # For dx,dy compute using full observation (global goal location)
        dx = dy = 0.0
        if self.include_dxdy:
            dx, dy = self._global_goal_delta(obs)
        parts = [self._encode_crop(crop)]
        if self.include_dxdy:
            parts.append(np.array([dx, dy], dtype=np.float32))
        if self.include_goal_direction:
            parts.append(self._goal_direction(obs))
        if self._last_action_dim:
            parts.append(self._last_action_one_hot())
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _last_action_one_hot(self) -> np.ndarray:
        if self._last_action_dim <= 0:
            return np.empty(0, dtype=np.float32)
        vec = np.zeros(self._last_action_dim, dtype=np.float32)
        if self.last_action is not None and 0 <= self.last_action < self._last_action_dim:
            vec[int(self.last_action)] = 1.0
        return vec

    def print_one_hot_observation(self, encoded_obs: np.ndarray) -> None:
        """Decode a flattened one-hot observation back to characters and print."""
        if not (self.one_hot and self.use_chars):
            raise RuntimeError("print_one_hot_observation requires character one-hot observations.")
        if self._eye_chars is None or self._char_vocab is None:
            raise RuntimeError("Character vocabulary has not been initialized; cannot decode observation.")

        obs = np.asarray(encoded_obs, dtype=np.float32)
        if self._last_action_dim:
            if obs.size < self._last_action_dim:
                raise ValueError("Encoded observation is too short to contain last action components.")
            obs = obs[:-self._last_action_dim]
        if self.include_goal_direction:
            if obs.size < 4:
                raise ValueError("Encoded observation is too short to contain goal direction components.")
            obs = obs[:-4]
        if self.include_dxdy:
            if obs.size < 2:
                raise ValueError("Encoded observation is too short to contain dx/dy components.")
            obs = obs[:-2]

        vocab_size = self._eye_chars.shape[0]
        if obs.size % vocab_size != 0:
            raise ValueError(
                "Encoded observation length is not divisible by the vocabulary size; cannot reshape to crop grid."
            )

        if self._crop_size == 0:
            raise ValueError("Crop size is zero; cannot decode observation")

        flattened = obs.reshape(-1, vocab_size)
        indices = np.argmax(flattened, axis=1)

        chars = list(self._char_vocab)
        if self.include_other_class:
            chars.append(self._other_char)
        if len(chars) != vocab_size:
            raise ValueError("Character mapping length does not match vocabulary size.")

        char_grid = np.array(chars, dtype="<U1")[indices]
        try:
            char_grid = char_grid.reshape(self._crop_shape)
        except ValueError as exc:
            raise ValueError("Cannot reshape decoded characters to crop shape.") from exc

        for row in char_grid:
            print("".join(row.tolist()))

    # ----- Gym API -----
    def reset(self, *, seed: int | None = None, options=None):
        # traceback.print_stack()
        self.steps = 0
        self.last_action = None
        if seed is not None:
            self.seed_ = seed
        else:
            seed = self.seed_
        self.env.unwrapped.seed(seed)
        obs, info = self.env.reset(options=options)
        self.last_obs = obs
        self._last_info = info
        self._last_done_flag = False
        self._goal_position = None
        self._goal_position = self._resolve_goal_position(obs)
        return self._build_observation(obs), info

    def get_observation(self):
        return self._build_observation(self.last_obs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.last_obs = obs
        self._last_info = info
        if self._last_action_dim:
            try:
                self.last_action = int(action)
            except (TypeError, ValueError):
                self.last_action = None

        reward = self.goal_reward if terminated and info["end_status"] == NetHackStaircase.StepStatus.TASK_SUCCESSFUL else self.step_reward

        # CAVIAT: No option implementation
        info.update({"action_size": 1, "steps": self.steps})
        done = bool(terminated) or bool(truncated)
        self._last_done_flag = done
        return self._build_observation(obs), reward, bool(terminated), bool(truncated), info

    def get_observation_space(self):
        return self.observation_space.shape[0]

    def get_action_space(self):
        return self.action_space.n

    def is_over(self) -> bool:
        # Prefer underlying env-provided signal if available
        underlying = getattr(self.env, "is_over", None)
        if callable(underlying):
            try:
                result = underlying()
                if result is not None:
                    return bool(result)
            except Exception:
                pass

        if self._last_done_flag:
            return True

        # Fallback: check whether agent currently stands on the goal tile
        if self.last_obs is not None:
            try:
                obs = self.last_obs
                chars = obs.get("chars") if isinstance(obs, dict) else None
                if chars is not None:
                    ax, ay = self._agent_xy(obs)
                    targets = [ord(c) for c in self.goal_chars]
                    if 0 <= ay < chars.shape[0] and 0 <= ax < chars.shape[1]:
                        if chars[int(ay), int(ax)] in targets:
                            return True
            except Exception:
                return False

        return False

    def render(self):
        return self.env.render()

    def seed(self, seed: int):
        self.seed_ = seed
        self.env.unwrapped.seed(seed)
        self.env.reset()

    def close(self):
        return self.env.close()




WRAPPING_TO_WRAPPER = {
    "MainWrapper": MiniHackWrap,
}
# WRAPPING_TO_WRAPPER = {
#     "OneHotChars": OneHotCharsWrapper,
#     "FixedSeed": FixedSeedWrapper,
#     "MovementAction": MovementActionWrapper,
# }