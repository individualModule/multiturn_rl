import hydra
from omegaconf import DictConfig
import torch
from trainers.archer_trainer import ArcherPlayPen
from clemcore_multiturn_rl.clemcore.clemgame.registry import GameRegistry
from clemcore_multiturn_rl.clemcore.backends.model_registry import ModelRegistry, BackendRegistry
from modelling.archer_critic import CriticNetwork

# to be reworked, hydra not doing well

# don't load teacher if teacher is not needed
def initialize_game_and_models(cfg: DictConfig):
    """Initialize game registry, model registry, backend registry, and load models."""
    # Initialize game registry
    game_registry = GameRegistry.from_directories_and_cwd_files()

    # Initialize model registry and backend registry
    model_registry = ModelRegistry.from_packaged_and_cwd_files()
    backend_registry = BackendRegistry.from_packaged_and_cwd_files()

    # Load learner and teacher model specs
    learner_spec = model_registry.get_first_model_spec_that_unify_with(cfg.game.learner)
    teacher_spec = model_registry.get_first_model_spec_that_unify_with(cfg.game.teacher)

    # Load learner and teacher models
    learner_backend = backend_registry.get_backend_for(learner_spec.backend)
    learner = learner_backend.get_model_for(learner_spec)
    learner.set_gen_args(max_tokens=100, temperature=0.0)

    teacher_backend = backend_registry.get_backend_for(teacher_spec.backend)
    teacher = teacher_backend.get_model_for(teacher_spec)
    teacher.set_gen_args(max_tokens=100, temperature=0.0)

    return game_registry, learner, teacher

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    
    # Initialize game registry and models
    game_registry, learner, teacher = initialize_game_and_models(cfg)

    # Initialize components
    critic = CriticNetwork(
        in_dim=cfg.model.critic.hidden_dims[0],
        out_dim=1,
        is_q_critic=True
    )
    target_critic = CriticNetwork(
        in_dim=cfg.model.critic.hidden_dims[0],
        out_dim=1,
        is_q_critic=True
    )
    critic_optimizer = hydra.utils.instantiate(cfg.optimizer.critic, params=critic.parameters())
    actor_optimizer = hydra.utils.instantiate(cfg.optimizer.actor, params=learner.parameters())
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
    
    # Start training
    trainer.learn_interactive(game_registry)

if __name__ == "__main__":
    main()





