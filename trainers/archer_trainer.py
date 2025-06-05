"""
Writing this code for:

1) Interactive training
2) Off-policy (collect rollouts -> train critic -> on-policy actor updates)
3) No branching

- We need rewards (success/fail/abort)
- dataloader for trajectory -> so that we can shuffle random steps and not learn on continuous trajectories.
- continuous trajectory vs shuffled steps should be an ablation - see the effects and compare them.


- in the original code, sample function only takes a single observation - not the entire trajectory.
"""

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import wandb
import hydra
from omegaconf import DictConfig
import os 

from clemcore.playpen import BasePlayPenMultiturnTrajectory, make_env, StepRolloutBuffer, ReplayBuffer, GameRecordCallback, RolloutProgressCallback
from clemcore.clemgame import GameRegistry, GameSpec
from modelling.archer_critic import ArcherAgent, CriticNetwork
from dataloaders.playpen_dataloader import StepRolloutDataset, FlatBufferDataset, custom_collate_fn

class ArcherPlayPen(BasePlayPenMultiturnTrajectory):
    def __init__(self, learner, teacher, critic, target_critic,
                 critic_optimizer, actor_optimizer,
                 critic_loss, actor_loss, rollout_iterations,
                 cfg: DictConfig):
        
        super().__init__(learner, teacher)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game_spec = None
        self.cfg = cfg  # Fix: Assign cfg to self.cfg

        # lora parameters
        self.policy = learner
        # Initialize Archer components

        self.critic = critic
        self.target_critic = target_critic
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.critic_loss = critic_loss
        self.actor_loss = actor_loss
        self.agent = ArcherAgent(self.policy, self.critic, self.target_critic, self.cfg.trainer.gamma)
        
        # Load parameters from config
        self.critic_epochs = self.cfg.trainer.critic_epochs
        self.actor_epochs = self.cfg.trainer.actor_epochs

        self.forPlayer = self.cfg.game.learner.name
        self.rollout_steps = self.cfg.trainer.rollout_steps # trajectory count 
        self.rollout_iterations = self.cfg.trainer.rollout_iterations
        self.eval_instances = self.cfg.trainer.eval_instances # name of eval instances file
        self.eval_rollout_steps = self.cfg.trainer.eval_rollout_steps # trajectory count
        self.eval_frequency = self.cfg.trainer.eval_frequency
        self.eval_episodes = self.cfg.trainer.eval_episodes
        self.step_size = self.cfg.trainer.step_size # number of sample datapoints per epoch
        self.batch_size = self.cfg.trainer.batch_size
        self.num_workers = self.cfg.trainer.num_workers
        self.max_grad_norm = self.cfg.trainer.max_grad_norm
        self.tau = self.cfg.trainer.tau
        self.warmup_iterations = self.cfg.trainer.warmup_iters
        self.scaling_factor = self.cfg.trainer.scaling_factor
        self.scale_reward = self.cfg.trainer.scale_reward

        # buffer definition and parameters
        self.is_replay_buffer = self.cfg.trainer.is_replay_buffer

        self.add_callback(GameRecordCallback())
        self.add_callback(RolloutProgressCallback(self.rollout_steps))

        self.best_metric = float('-inf')

        # Initialize wandb with config
        wandb.init(project=self.cfg.project_name,
                   name=self.cfg.run_name,
                   group=self.cfg.group,
                  config=dict(self.cfg))

    def learn_interactive(self, game_registry: GameRegistry):
        # Select game spec you want to train on
        self.game_spec = game_registry.get_game_specs_that_unify_with(self.cfg.game.spec_name)[0]
        
        # Create environment and buffer
        with make_env(self.game_spec, [self.learner, self.teacher]) as env:
            if self.is_replay_buffer:
                # sample size should be equal to the steps sampled.
                # need to figure this one out. How many items in the buffer and on how many items do we train? 
                buffer = ReplayBuffer(env, buffer_size=self.rollout_steps*15, sample_size=self.step_size)
            else:
                buffer = StepRolloutBuffer(env)

            # self._collect_rollouts(env, self.rollout_steps, buffer) 
            self._train(buffer, env)
            # buffer.reset()

    def _monitor_lora_gradients(self):
        """Monitor gradients and parameters of LoRA adapters."""
        lora_params = {}
        lora_grads = {}
        
        # Collect LoRA parameter names, values, and gradients
        for name, param in self.policy.model.named_parameters():
            if 'lora' in name and param.requires_grad:
                lora_params[name] = param.data.norm().item()
                if param.grad is not None:
                    lora_grads[name] = param.grad.norm().item()
                else:
                    lora_grads[name] = 0.0
        
        # Log average gradient norm and parameter norm
        avg_grad_norm = sum(lora_grads.values()) / max(len(lora_grads), 1)
        avg_param_norm = sum(lora_params.values()) / max(len(lora_params), 1)
        
        # Log to wandb
        metrics = {
            "lora/avg_grad_norm": avg_grad_norm,
            "lora/avg_param_norm": avg_param_norm,
            "lora/active_params": len(lora_params)
        }
        
        # Sample some specific layers to track in detail
        for name, grad in list(lora_grads.items())[:5]:  # Track first 5 LoRA layers
            metrics[f"lora_grad/{name}"] = grad
        
        return metrics
    
    def _evaluate_policy(self, current_iteration=None):
        with torch.no_grad():
            with make_env(self.game_spec, [self.learner, self.teacher], instances_name=self.eval_instances) as eval_env:
                eval_buffer = StepRolloutBuffer(eval_env)

                # Collect rollouts for evaluation
                self._collect_rollouts(
                    game_env=eval_env,
                    rollout_steps=self.eval_rollout_steps,
                    rollout_buffer=eval_buffer,
                    forPlayer=self.forPlayer,
                    eval=True
                )

            eval_trajectories = eval_buffer.sample_trajectories()

            # Initialize metrics

            total_episode_scores = []
            total_response_scores = []
            
            per_episode_response_sum = []

            min_episode_score = float('inf')
            max_episode_score = float('-inf')
            
            # Process each trajectory
            for trajectory in eval_trajectories:
                episode_score = 0

                trajectory_response_sum = 0
                for step in trajectory:
                    # Accumulate response scores for turn-level rewards
                    if step['done']:
                        response_score = step['info'].get('episode_score', 0)
                    else:
                        response_score = step['info'].get('response_score', 0)

                    total_response_scores.append(response_score)
                    trajectory_response_sum += response_score
                    # Update episode score if available
                    episode_score = step['info'].get('episode_score', episode_score)

                # Track episode-level metrics
                total_episode_scores.append(episode_score)
                per_episode_response_sum.append(trajectory_response_sum)
                min_episode_score = min(min_episode_score, episode_score)
                max_episode_score = max(max_episode_score, episode_score)

            # Calculate metrics

            metrics = {
                'eval/average_episode_reward': sum(total_episode_scores) / len(total_episode_scores) if total_episode_scores else 0,
                'eval/average_turn_reward': sum(total_response_scores) / len(total_response_scores) if total_response_scores else 0,
                'eval/average_accumulated_reward': sum(per_episode_response_sum)/len(per_episode_response_sum) if per_episode_response_sum else 0,
                'eval/min_reward': min_episode_score if total_episode_scores else 0,
                'eval/max_reward': max_episode_score if total_episode_scores else 0
            }

            # if current_iteration is not None:
            #     metrics['iteration'] = current_iteration

            # Log metrics to wandb
            wandb.log(metrics)
            
            eval_buffer.reset()
            
        return metrics

    
    def _save_checkpoint(self, iteration, eval_metrics, buffer=None):
        """Save training checkpoint if the metric improves."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Use a key metric to determine if this is the best checkpoint
        current_metric = eval_metrics['eval/average_accumulated_reward']  # Example: average reward
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint.pt")
            
            checkpoint = {
                "iteration": iteration,
                "policy_state_dict": self.policy.model.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "target_critic_state_dict": self.target_critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "config": self.cfg,
                "best_metric": self.best_metric
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Best checkpoint saved at {checkpoint_path} with metric: {current_metric:.2f}")

            if buffer:
                buffer.save_buffer(checkpoint_path)

    def _train(self, buffer, env):

        # Training loop
        for iteration in range(self.rollout_iterations):
            torch.cuda.empty_cache() # empty cache ocassionally
            # Collect trajectories
            self._collect_rollouts(game_env = env,
                                   rollout_steps = self.rollout_steps,
                                   rollout_buffer = buffer,
                                   forPlayer = self.forPlayer ) # use this also to collect eval data
            # Run evaluation if it's time
            if iteration % self.eval_frequency == 0:
                eval_metrics = self._evaluate_policy(current_iteration=iteration)
                print(f"Initial evaluation:", 
                    f"Average Reward: {eval_metrics['eval/average_episode_reward']:.2f},",
                    f"Avg Turn Reward: {eval_metrics['eval/average_turn_reward']:.2f}",
                    f"Avg accumulated reward: {eval_metrics['eval/average_accumulated_reward']}")
                
                # Save checkpoint if evaluation metrics improve
                self._save_checkpoint(iteration, eval_metrics, buffer=buffer)            
            # Get stored trajectories

            critic_metrics = self._update_critic(self.critic_epochs,
                                                  scaled_reward=self.scale_reward, scaling_factor=self.scaling_factor, buffer=buffer)

            if iteration >= self.warmup_iterations:
                actor_metrics = self._update_actor(self.actor_epochs, buffer=buffer)
            else:
                actor_metrics = {"actor/avg_loss": None, "actor/epochs": 0}  # Placeholder metrics for warmup
                
            # Log iteration metrics
            wandb.log({
                    "iteration": iteration,
                    **critic_metrics,
                    **actor_metrics
                })

            # replay buffer has no reset - pop mechanism (oldest samples are popped)
            if not self.is_replay_buffer:
                buffer.reset()


        # Final evaluation after training
        print("Running final evaluation...")
        final_eval_metrics = self._evaluate_policy(current_iteration=self.rollout_iterations)
        print(f"Final evaluation:",
            f"Average Reward: {final_eval_metrics['eval/average_episode_reward']:.2f},",
            f"Avg Turn Reward: {final_eval_metrics['eval/average_turn_reward']:.2f}",
            f"Avg accumulated reward: {final_eval_metrics['eval/average_accumulated_reward']}")
        
        # Optionally save a final checkpoint
        self._save_checkpoint(self.rollout_iterations, final_eval_metrics)

    def _update_critic(self, critic_epochs, scaled_reward=False, scaling_factor=100.0, buffer=None):
        """
        Update the critic network.

        Args:
            critic_epochs: Number of epochs to train the critic.
            dataloader: DataLoader for training data.
            scaled_reward: If True, scale the rewards by the scaling_factor.
            scaling_factor: The factor by which to scale the rewards.
        """
        epoch_losses = []
        
        for e in range(critic_epochs):
            torch.cuda.empty_cache() # empty cache ocassionally
            epoch_loss = 0
            num_batches = 0

            dataset = FlatBufferDataset(buffer.sample_steps())
            if dataset is None or len(dataset) == 0:
                raise ValueError("Dataset is empty after maximum retries. Please check data preparation.")

            print("Dataset size:", len(dataset))

            dataloader = DataLoader(
                                    dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    collate_fn=custom_collate_fn
                                )

            for batch in tqdm(dataloader):
                batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                self.critic_optimizer.zero_grad()

                # Scale rewards if scaled_reward is True
                # TODO - need to implement this in the actor as well
                if scaled_reward:
                    batch['reward'] = batch['reward'] / scaling_factor

                q1, q2, v1, v2 = self.agent.get_critic_values(batch['obs'], batch['action'])
                target_q1, target_q2 = self.agent.compute_target_q(batch['obs'])
                target_v1, target_v2 = self.agent.compute_target_v(batch['next_obs'],
                                                                batch['action'],
                                                                batch['reward'],
                                                                batch['done'])

                loss = self.critic_loss(q1, q2, v1, v2,
                                        target_v1, target_v2,
                                        target_q1, target_q2)

                loss.backward()
                clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.critic_optimizer.step()
                self.agent.soft_target_update(self.target_critic, self.critic, self.tau)
                

                with torch.no_grad():
                    # Log batch metrics
                    metrics = {
                        "critic/loss": loss.item(),
                        "critic/q1_mean": q1.mean().item(),
                        "critic/q2_mean": q2.mean().item(),
                        "critic/v1_mean": v1.mean().item(),
                        "critic/v2_mean": v2.mean().item(),
                        "critic/q1_min": q1.min().item(),
                        "critic/q1_max": q1.max().item(),
                        "critic/q2_min": q2.min().item(),
                        "critic/q2_max": q2.max().item(),
                        "critic/v1_min": v1.min().item(),
                        "critic/v1_max": v1.max().item(),
                        "critic/v2_min": v2.min().item(),
                        "critic/v2_max": v2.max().item(),
                        "critic/epoch": e
                    }
                wandb.log(metrics)
                
                epoch_loss += loss.item()
                num_batches += 1
            
            epoch_losses.append(epoch_loss / num_batches)
        
        return {
            "critic/avg_loss": sum(epoch_losses) / len(epoch_losses),
            "critic/epochs": critic_epochs
        }


    def _update_actor(self, actor_epochs, buffer):
        epoch_losses = []
        
        for e in range(actor_epochs):
            torch.cuda.empty_cache() # empty cache ocassionally

            # sample data
            dataset = FlatBufferDataset(buffer.sample_steps())
            if dataset is None or len(dataset) == 0:
                raise ValueError("Dataset is empty after maximum retries. Please check data preparation.")

            print("Dataset size:", len(dataset))

            dataloader = DataLoader(
                                    dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    collate_fn=custom_collate_fn
                                )

            epoch_loss = 0
            num_batches = 0
            for batch in tqdm(dataloader):
                batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

                self.actor_optimizer.zero_grad()
                with torch.no_grad():
                    pi_action = self.agent.get_policy_action(batch['obs'])
                    q1, q2, v1, v2 = self.agent.get_critic_values(batch['obs'], pi_action, detach_model=True)
                    #take minumum of q and minimum of v
                    q = torch.minimum(q1, q2)
                    v = torch.minimum(v1, v2)

                    advantages = self.agent.compute_advantages(q, v)

                logprobs = self.agent.get_log_prob(batch['obs'],
                                                   pi_action)
                
                loss = self.actor_loss(advantages, logprobs)
                loss.backward()

                clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                lora_grad_metrics = self._monitor_lora_gradients()
                # Log batch metrics
                with torch.no_grad():
                    metrics = {
                        "actor/loss": loss.item(),
                        "actor/advantages_mean": advantages.mean().item(),
                        "actor/logprobs_mean": logprobs.mean().item(),
                        "actor/logprobs_min": logprobs.min().item(),
                        "actor/logprobs_max": logprobs.max().item(),
                        "actor/q1_mean": q1.mean().item(),
                        "actor/q2_mean": q2.mean().item(),
                        "actor/v1_mean": v1.mean().item(),
                        "actor/v2_mean": v2.mean().item(),
                        "actor/q1_min": q1.min().item(),
                        "actor/q1_max": q1.max().item(),
                        "actor/q2_min": q2.min().item(),
                        "actor/q2_max": q2.max().item(),
                        "actor/v1_min": v1.min().item(),
                        "actor/v1_max": v1.max().item(),
                        "actor/v2_min": v2.min().item(),
                        "actor/v2_max": v2.max().item(),
                        "actor/epoch": e
                    }
                    
                    metrics.update(lora_grad_metrics)  # Add to existing metrics dictionary

                wandb.log(metrics)
                
                epoch_loss += loss.item()
                num_batches += 1
            
            epoch_losses.append(epoch_loss / num_batches)
        
        return {
            "actor/avg_loss": sum(epoch_losses) / len(epoch_losses),
            "actor/epochs": actor_epochs
        }

