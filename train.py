import hydra
from omegaconf import DictConfig
import torch
import os

from trainers.archer_trainer import ArcherPlayPen
from clemcore.clemgame.registry import GameRegistry
from clemcore.backends import ModelRegistry, BackendRegistry
from clemcore.backends import ModelSpec
from modelling.archer_critic import DoubleCritic



def load_checkpoint(checkpoint_path, trainer):
    """
    Load a checkpoint and restore the trainer's state.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        trainer (ArcherPlayPen): The trainer instance to restore.

    Returns:
        int: The iteration to resume training from.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    # Restore model and optimizer states
    trainer.policy.load_state_dict(checkpoint["policy_state_dict"])
    trainer.critic.load_state_dict(checkpoint["critic_state_dict"])
    trainer.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
    trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    trainer.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])

    # Restore other parameters
    trainer.best_metric = checkpoint["best_metric"]
    trainer.cfg = checkpoint["config"]

    print(f"Checkpoint loaded from {checkpoint_path}, resuming from iteration {checkpoint['iteration']}")
    return checkpoint["iteration"]

# to be reworked, hydra not doing well
# don't load teacher if teacher is not needed
def initialize_game_and_models(cfg: DictConfig):
    """Initialize game registry, model registry, backend registry, and load models."""
    # Initialize game registry
    game_registry = GameRegistry.from_directories_and_cwd_files()

    # Initialize model registry and backend registry
    model_registry = ModelRegistry.from_packaged_and_cwd_files()
    backend_registry = BackendRegistry.from_packaged_and_cwd_files()

    # get model specs
    learner_spec = ModelSpec.from_string(cfg.game.learner.model_name)
    teacher_spec = ModelSpec.from_string(cfg.game.teacher.model_name)

    # Load learner and teacher model specs
    learner_spec = model_registry.get_first_model_spec_that_unify_with(learner_spec)
    teacher_spec = model_registry.get_first_model_spec_that_unify_with(teacher_spec)

    # Load learner and teacher models
    learner_backend = backend_registry.get_backend_for(learner_spec.backend)
    learner = learner_backend.get_model_for(learner_spec)
    learner.set_gen_args(temperature = cfg.game.learner.temperature, max_tokens=cfg.game.learner.max_tokens)

    teacher_backend = backend_registry.get_backend_for(teacher_spec.backend)
    teacher = teacher_backend.get_model_for(teacher_spec)
    teacher.set_gen_args(temperature = cfg.game.teacher.temperature, max_tokens=cfg.game.teacher.max_tokens)

    return game_registry, learner, teacher

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    


    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize game registry and models
    game_registry, learner, teacher = initialize_game_and_models(cfg)

    # Initialize components
    critic = DoubleCritic(
        in_dim=cfg.model.critic.hidden_dims[0],
        out_dim=1,
        critic_lm=cfg.model.critic.critic_lm,
        device=device
    )
    target_critic = DoubleCritic(
        in_dim=cfg.model.critic.hidden_dims[0],
        out_dim=1,
        critic_lm=cfg.model.critic.critic_lm,
        device=device
    )

    critic_optimizer = hydra.utils.instantiate(cfg.optimizer.critic, params=critic.parameters())
    actor_optimizer = hydra.utils.instantiate(cfg.optimizer.actor, params=learner.model.parameters())
    critic_loss = hydra.utils.instantiate(cfg.loss.critic)
    actor_loss = hydra.utils.instantiate(cfg.loss.actor)

    # Initialize trainer
    trainer = ArcherPlayPen(
        learner=learner,
        teacher=teacher,
        critic=critic,
        target_critic=target_critic,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        critic_loss=critic_loss,
        actor_loss=actor_loss,
        rollout_iterations=cfg.trainer.rollout_iterations,
        cfg=cfg
    )
    

    # Load checkpoint if specified
    # checkpoint_path = cfg.get("checkpoint_path", None)
    # start_iteration = 0
    # if checkpoint_path and os.path.exists(checkpoint_path):
    #     start_iteration = load_checkpoint(checkpoint_path, trainer)


    # Start training
    trainer.learn_interactive(game_registry)

if __name__ == "__main__":
    main()





