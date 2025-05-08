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

from clemcore.playpen import BasePlayPen, make_env, StepRolloutBuffer, GameRecordCallback, RolloutProgressCallback
from clemcore.clemgame import GameRegistry, GameSpec
from modelling.archer_critic import ArcherAgent, CriticNetwork
from dataloaders.playpen_dataloader import StepRolloutDataloader

class ArcherPlayPen(BasePlayPen):
    def __init__(self, learner, teacher, critic, target_critic,
                 critic_optimizer, actor_optimizer,
                 critic_loss, actor_loss, rollout_iterations,
                 cfg: DictConfig):
        
        super().__init__(learner, teacher)
        
        # Initialize Archer components
        self.policy = learner
        self.critic = critic
        self.target_critic = target_critic
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.critic_loss = critic_loss
        self.actor_loss = actor_loss
        self.agent = ArcherAgent(self.policy, self.critic, self.target_critic)
        self.cfg = cfg  # Fix: Assign cfg to self.cfg

        # Load parameters from config
        self.rollout_steps = self.cfg.trainer.rollout_steps
        self.rollout_iterations = self.cfg.trainer.rollout_iterations
        self.eval_frequency = self.cfg.trainer.eval_frequency
        self.eval_episodes = self.cfg.trainer.eval_episodes
        self.batch_size = self.cfg.trainer.batch_size
        self.num_workers = self.cfg.trainer.num_workers
        self.max_grad_norm = self.cfg.trainer.max_grad_norm
        self.tau = self.cfg.trainer.tau
        
        self.add_callback(GameRecordCallback())
        self.add_callback(RolloutProgressCallback(self.rollout_steps))

        self.best_metric = float('-inf')

        # Initialize wandb with config
        wandb.init(project=self.cfg.project_name,
                  config=dict(self.cfg))

    def learn_interactive(self, game_registry: GameRegistry):
        # Select game spec you want to train on
        game_spec = game_registry.get_game_specs_that_unify_with(self.cfg.game.spec_name)[0]
        
        # Create environment and buffer
        with make_env(game_spec, [self.learner, self.teacher]) as env:
            buffer = StepRolloutBuffer(env)

            # self._collect_rollouts(env, self.rollout_steps, buffer) 
            self._train(buffer, env)
            # buffer.reset()

    def _evaluate_policy(self, env, current_iteration=None):
        """Run evaluation episodes and return metrics."""
        total_rewards = []
        success_count = 0
        
        # Disable gradients for evaluation
        with torch.no_grad():
            for _ in range(self.eval_episodes):
                episode_reward = 0
                obs = env.reset()
                done = False
                
                while not done:
                    # Get action from policy (without exploration noise)
                    action = self.agent.get_policy_action(obs, deterministic=True)
                    
                    # Step environment
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    # Track success based on environment info
                    if done and info.get('success', False):
                        success_count += 1
                
                total_rewards.append(episode_reward)
        
        # Compute metrics
        metrics = {
            'eval/average_reward': sum(total_rewards) / len(total_rewards),
            'eval/success_rate': success_count / self.eval_episodes,
            'eval/min_reward': min(total_rewards),
            'eval/max_reward': max(total_rewards)
        }
        
        if current_iteration is not None:
            metrics['iteration'] = current_iteration
            
        # Log to wandb
        wandb.log(metrics)
        
        return metrics
    
    def _save_checkpoint(self, iteration, eval_metrics):
        """Save training checkpoint if the metric improves."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Use a key metric to determine if this is the best checkpoint
        current_metric = eval_metrics['eval/average_reward']  # Example: average reward
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint.pt")
            
            checkpoint = {
                "iteration": iteration,
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "target_critic_state_dict": self.target_critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "config": self.cfg,
                "best_metric": self.best_metric
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Best checkpoint saved at {checkpoint_path} with metric: {current_metric:.2f}")


    def _train(self, buffer, env):
        # Run initial evaluation
        print("Running initial evaluation...")
        eval_metrics = self._evaluate_policy(env, current_iteration=0)
        print(f"Initial evaluation:", 
              f"Average Reward: {eval_metrics['eval/average_reward']:.2f},",
              f"Success Rate: {eval_metrics['eval/success_rate']:.2%}")

        # need to be trained in epochs
        # usually we do N epochs, for critic and M for actor (in paper 50 vs 3)
        # they usually do Y sample iterations (2000)
        # they also do warmup rounds with no actor updates

        # in the paper, they don't train on all of the data that is in the buffer. - they randomly sample
        # use that as an ablation potentially

        # Training loop
        for iteration in range(self.rollout_iterations):
            # Collect trajectories
            self._collect_rollouts(env, self.rollout_steps, buffer) # use this also to collect eval data
            
            # Run evaluation if it's time
            if iteration % self.eval_frequency == 0:
                eval_metrics = self._evaluate_policy(env, current_iteration=iteration)
                print(f"Iteration {iteration} evaluation:", 
                      f"Average Reward: {eval_metrics['eval/average_reward']:.2f},",
                      f"Success Rate: {eval_metrics['eval/success_rate']:.2%}")
                
                # Save checkpoint if evaluation metrics improve
                self._save_checkpoint(iteration, eval_metrics)

            # Get stored trajectories
            dataset = StepRolloutDataloader(buffer.trajectories)
            dataloader = DataLoader(
                                    dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_workers
                                )

            critic_metrics = self._update_critic(self.critic_epochs, dataloader)
            actor_metrics = self._update_actor(self.actor_epochs, dataloader)
            
            # Log iteration metrics
            wandb.log({
                "iteration": iteration,
                **critic_metrics,
                **actor_metrics
            })

            buffer.reset()


    def _update_critic(self, critic_epochs, dataloader):
        epoch_losses = []
        
        for e in range(critic_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in tqdm(dataloader):
                self.critic_optimizer.zero_grad()

                q1, q2, v1, v2 = self.agent.get_critic_values(batch['observation'], batch['action'])
                target_q1, target_q2 = self.agent.compute_target_q(batch['observation'])
                target_v1, target_v2 = self.agent.compute_target_v(batch['next_observation'],
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
                
                # Log batch metrics
                metrics = {
                    "critic/loss": loss.item(),
                    "critic/q1_mean": q1.mean().item(),
                    "critic/q2_mean": q2.mean().item(),
                    "critic/v1_mean": v1.mean().item(),
                    "critic/v2_mean": v2.mean().item(),
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

    def _update_actor(self, actor_epochs, dataloader):
        epoch_losses = []
        
        for e in range(actor_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in tqdm(dataloader):
                self.actor_optimizer.zero_grad()

                pi_action = self.agent.get_policy_action(batch['observation'])
                q1, q2, v1, v2 = self.agent.get_critic_values(batch['observation'], pi_action)
                
                #take minumum of q and minimum of v
                q = torch.minimum(q1, q2)
                v = torch.minumum(v1, v2)

                advantages = self.agent.compute_advantages(q, v)

                logprobs = self.agent.get_log_prob(batch['observation'],
                                                   pi_action)
                
                loss = self.actor_loss(advantages, logprobs)
                loss.backward()

                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Log batch metrics
                metrics = {
                    "actor/loss": loss.item(),
                    "actor/advantages_mean": advantages.mean().item(),
                    "actor/logprobs_mean": logprobs.mean().item(),
                    "actor/epoch": e
                }
                wandb.log(metrics)
                
                epoch_loss += loss.item()
                num_batches += 1
            
            epoch_losses.append(epoch_loss / num_batches)
        
        return {
            "actor/avg_loss": sum(epoch_losses) / len(epoch_losses),
            "actor/epochs": actor_epochs
        }

