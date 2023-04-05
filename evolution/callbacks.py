from stable_baselines3.common.callbacks import BaseCallback


class TrainerVisualizer(BaseCallback):
    """ Visualization for training process. User only for gym.Env subclasses """

    def _on_step(self) -> bool:
        self.model.env.envs[0].render()
        return True
