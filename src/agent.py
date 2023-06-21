from stable_baselines3 import PPO
import pathlib
from action.custom_action import CustomActionParser

class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        print("Path = ", str(_path))
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "n_envs": 1,
        }

        try:
            self.actor = PPO.load(str(_path) + '/best_agent', device='cpu', custom_objects=custom_objects)
            self.actor.training = False
            self.parser = CustomActionParser()
        except Exception as e:
            print("Model failed to load due to the following exception: ", e)

    def act(self, state):
        action = self.actor.predict(state, deterministic=True)
        return action[0]

        # x = self.parser.parse_actions(action[0], state)
        # return x[0]
