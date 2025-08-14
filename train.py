import hydra
from omegaconf import DictConfig
import torch
from torch import nn
import os
import pickle

from peft import LoraConfig, get_peft_model
from trainers.archer_trainer import ArcherPlayPen
from clemcore.clemgame.registry import GameRegistry
from clemcore.backends import ModelRegistry, BackendRegistry
from clemcore.backends import ModelSpec
from modelling.archer_critic import DoubleCritic
from clemcore.playpen import BatchReplayBuffer



def load_checkpoint(checkpoint_path, trainer, lora_config):
    """
    Load a checkpoint and restore the trainer's state (DP or single GPU).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # (Re)attach LoRA if not already present
    if not hasattr(trainer.learner.model, "peft_config"):
        trainer.learner.model = get_peft_model(trainer.learner.model, lora_config)

    # Unwrap helper
    def unwrap(m):
        return m.module if isinstance(m, nn.DataParallel) else m

    # Load LoRA (policy) weights
    learner_ref = unwrap(trainer.learner.model)
    learner_ref.load_state_dict(checkpoint["lora_state_dict"], strict=False)

    # Load critics
    unwrap(trainer.critic).load_state_dict(checkpoint["critic_state_dict"])
    unwrap(trainer.target_critic).load_state_dict(checkpoint["target_critic_state_dict"])

    # Optimizers (safe even if shapes match after DP wrap)
    trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    trainer.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])

    # Misc
    trainer.best_metric = checkpoint.get("best_metric", float("-inf"))
    # (Optional) you could merge cfg instead of overwrite
    trainer.cfg = checkpoint.get("config", trainer.cfg)

    iteration = checkpoint.get("iteration", 0)
    print(f"Loaded checkpoint {checkpoint_path} (iteration {iteration})")
    return iteration

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
    # Load learner and teacher model specs
    learner_spec = model_registry.get_first_model_spec_that_unify_with(learner_spec)
    # Load learner and teacher models
    learner_backend = backend_registry.get_backend_for(learner_spec.backend)
    learner = learner_backend.get_model_for(learner_spec)
    learner.set_gen_args(temperature = cfg.game.learner.temperature, max_tokens=cfg.game.learner.max_tokens)

    if cfg.game.teacher.model_name:
        teacher_spec = ModelSpec.from_string(cfg.game.teacher.model_name)
        teacher_spec = model_registry.get_first_model_spec_that_unify_with(teacher_spec)
        teacher_backend = backend_registry.get_backend_for(teacher_spec.backend)
        teacher = teacher_backend.get_model_for(teacher_spec)
        teacher.set_gen_args(temperature = cfg.game.teacher.temperature, max_tokens=cfg.game.teacher.max_tokens)
    else:
        teacher = None

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

        # Initialize LoRA for the policy model
    lora_config = LoraConfig(
            r=cfg.lora.r,  # Rank of the low-rank matrices
            lora_alpha=cfg.lora.alpha,  # Scaling factor
            lora_dropout=cfg.lora.dropout,  # Dropout for LoRA
            bias=cfg.lora.bias
        )
    
    learner.model = get_peft_model(learner.model, lora_config)

    for name, param in learner.model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True  # Enable gradients for LoRA parameters
        else:
            param.requires_grad = False  # Freeze other parameters   
    
    print("\nLoRA Parameter Status:")
    for name, param in learner.model.named_parameters():
        if 'lora' in name: print(f"{name}: requires_grad={param.requires_grad}, shape={list(param.shape)}")


    if getattr(cfg.trainer, "data_parallel", False) and torch.cuda.device_count() > 1:
        print(f"Enabling DataParallel over {torch.cuda.device_count()} GPUs")
        # Wrap learner policy (LoRA model)
        learner.model = nn.DataParallel(learner.model)
        # Wrap critics
        critic = nn.DataParallel(critic)
        target_critic = nn.DataParallel(target_critic)

    # (Assertions must unwrap .module when DP is used)
    def iter_named(model):
        return model.module.named_parameters() if isinstance(model, nn.DataParallel) else model.named_parameters()



    # Add assertions to verify LoRA parameters are trainable
    lora_params_count = sum(1 for n, p in iter_named(learner.model) if 'lora' in n)
    trainable_lora_params = sum(1 for n, p in iter_named(learner.model) if 'lora' in n and p.requires_grad)
    non_lora_trainable = sum(1 for n, p in iter_named(learner.model) if 'lora' not in n and p.requires_grad)
    assert trainable_lora_params > 0
    assert trainable_lora_params == lora_params_count
    assert non_lora_trainable == 0

    print(f"\nVerified: {trainable_lora_params} LoRA parameters are trainable, all other parameters are frozen")
    critic_optimizer = hydra.utils.instantiate(cfg.optimizer.critic,
                                               params=(critic.module if isinstance(critic, nn.DataParallel) else critic).parameters())
    actor_optimizer = hydra.utils.instantiate(cfg.optimizer.actor,
                                              params=learner.model.module.parameters() if isinstance(learner.model, nn.DataParallel) else learner.model.parameters())
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
        cfg=cfg,
        game_registry = game_registry
    )
    

    # Load checkpoint if specified
    # checkpoint_path = cfg.get("checkpoint_path", None)
    # start_iteration = 0
    # if checkpoint_path and os.path.exists(checkpoint_path):
    #     start_iteration = load_checkpoint(checkpoint_path, trainer)

    # start_iter = load_checkpoint('/home/users/dristic/project/Archer/checkpoints/latest_checkpoint_exp_4.pt', trainer, lora_config)
    # Load buffer if exists
    # buffer_path = os.path.join("checkpoints", "latest_buffer.pkl")
    # buffer_path = '/home/users/dristic/project/Archer/checkpoints/latest_buffer_exp_4.pkl'
    # for regular training       
    # start_iter = 0
    # buffer = None
    # Start training
    # trainer.learn_interactive(game_registry, start_iteration=start_iter, buffer_path=buffer_path)
    trainer.learn_interactive(game_registry)
if __name__ == "__main__":
    main()





