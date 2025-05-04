import hydra
from omegaconf import DictConfig
import torch
from trainers.archer_trainer import ArcherPlayPen
from clemcore_multiturn_rl.clemcore.clemgame.registry import GameRegistry

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    
    # Initialize components
    critic_optimizer = hydra.utils.instantiate(cfg.optimizer.critic)
    actor_optimizer = hydra.utils.instantiate(cfg.optimizer.actor)
    critic_loss = hydra.utils.instantiate(cfg.loss.critic)
    actor_loss = hydra.utils.instantiate(cfg.loss.actor)
    
    # Initialize teacher and learner models
    teacher = hydra.utils.instantiate(cfg.game.teacher)
    learner = hydra.utils.instantiate(cfg.game.learner)
    
    # Get game registry and spec
    game_registry = GameRegistry()
    
    trainer = ArcherPlayPen(
        learner=learner,
        teacher=teacher,
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





