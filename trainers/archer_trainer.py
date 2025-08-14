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
import pickle

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from tqdm import tqdm
import wandb
import hydra
from omegaconf import DictConfig
from huggingface_hub import login
import os 

from clemcore.playpen import EvalBatchRollout, BatchRollout, make_batch_env, make_eval_env, BatchRolloutBuffer, BatchReplayBuffer, GameRecordCallback, RolloutProgressCallback
from clemcore.clemgame import GameRegistry
from modelling.archer_critic import ArcherAgent
from dataloaders.playpen_dataloader import FlatBufferDataset, custom_collate_fn
from utils.utils import save_lora_state_dict
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file

hf_token = os.getenv("HF_TOKEN")
hf_username = os.getenv("HF_USERNAME")

class ArcherPlayPen(BatchRollout):
    def __init__(self, learner, teacher, critic, target_critic,
                 critic_optimizer, actor_optimizer,
                 critic_loss, actor_loss, rollout_iterations,
                 cfg: DictConfig, game_registry):
        
        super().__init__(learner, teacher)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game_spec = None
        self.cfg = cfg  # Fix: Assign cfg to self.cfg
        # lora parameters
        # self.policy = learner
        # switched all self.policy to self.learner
        # Initialize Archer components
        self.learner_name = cfg.game.learner.name
        self.teacher_name = cfg.game.teacher.name

        self.critic = critic
        self.target_critic = target_critic
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.critic_loss = critic_loss
        self.actor_loss = actor_loss
        self.agent = ArcherAgent(self.learner, self.critic, self.target_critic, self.cfg.trainer.gamma)
        
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
        self.actor_batch_size = self.cfg.trainer.actor_batch_size
        self.critic_batch_size = self.cfg.trainer.critic_batch_size
        self.num_workers = self.cfg.trainer.num_workers
        self.max_grad_norm = self.cfg.trainer.max_grad_norm
        self.critic_max_grad_norm = self.cfg.trainer.critic_max_grad_norm
        self.actor_grad_accum_steps = self.cfg.trainer.actor_grad_accum_steps
        self.critic_grad_accum_steps = self.cfg.trainer.critic_grad_accum_steps
        self.tau = self.cfg.trainer.tau
        self.warmup_iterations = self.cfg.trainer.warmup_iters
        self.scaling_factor = self.cfg.trainer.scaling_factor
        self.scale_reward = self.cfg.trainer.scale_reward
        self.inference_batch_size = self.cfg.trainer.inference_batch_size
        self.buffer_size = self.cfg.trainer.buffer_size
        self.evaluator = ArcherEval(learner, teacher, cfg, game_registry)
        self.lora_save_every = cfg.trainer.save_every
        # buffer definition and parameters
        self.is_replay_buffer = self.cfg.trainer.is_replay_buffer

        self.add_callback(GameRecordCallback(top_dir=f"playpen/{self.cfg.run_name}"))
        self.add_callback(RolloutProgressCallback(self.rollout_steps))

        self.best_metric = float('-inf')

        # Initialize wandb with config
        wandb.init(project=self.cfg.project_name,
                   name=self.cfg.run_name,
                   group=self.cfg.group,
                  config=dict(self.cfg))

    def learn_interactive(self, game_registry: GameRegistry, start_iteration=0, buffer_path=None):
        # Select game spec you want to train on
        self.game_spec = game_registry.get_game_specs_that_unify_with(self.cfg.game.spec_name)[0]
        players = [self.learner, self.teacher] if self.teacher else [self.learner]
        # Create environment and buffer
        with make_batch_env(self.game_spec, players, shuffle_instances = True, batch_size = self.inference_batch_size) as env:
            if buffer_path is not None:
                rollout_buffer = BatchReplayBuffer(env, buffer_size=self.buffer_size, sample_size=self.step_size)
                rollout_buffer.load_buffer(buffer_path)
                print('buffer loaded successfully!')
                print(len(rollout_buffer.trajectories))
            else:
                if self.is_replay_buffer:
                    # sample size should be equal to the steps sampled.
                    # need to figure this one out. How many items in the buffer and on how many items do we train? 
                    rollout_buffer = BatchReplayBuffer(env, buffer_size=self.buffer_size, sample_size=self.step_size)
                else:
                    rollout_buffer = BatchRolloutBuffer(env)

            # self._collect_rollouts(env, self.rollout_steps, buffer) 
            self._train(rollout_buffer, env, start_iteration=start_iteration)
            # buffer.reset()
    
    def _train(self, buffer, env, start_iteration=0):
        # Training loop
        for iteration in range(start_iteration, self.rollout_iterations):
            torch.cuda.empty_cache() # empty cache ocassionally
            # Collect trajectories
            rollout_metrics = self._collect_rollouts(game_env = env,
                                   rollout_steps = self.rollout_steps,
                                   rollout_buffer = buffer,
                                   forPlayer = self.forPlayer ) # use this also to collect eval data
            wandb.log(rollout_metrics)
            # Run evaluation if it's time
            if iteration % self.eval_frequency == 0 and (iteration == 0 or iteration > self.warmup_iterations):
                eval_metrics = self._evaluate_policy(current_iteration=iteration)
                print(f"Initial evaluation:", 
                    f"Average Reward: {eval_metrics['eval/average_episode_reward']:.2f},",
                    f"Avg Turn Reward: {eval_metrics['eval/average_turn_reward']:.2f}",
                    f"Avg accumulated reward: {eval_metrics['eval/average_accumulated_reward']}")
                
                # Save checkpoint if evaluation metrics improve
                self._save_checkpoint(iteration, eval_metrics, buffer=buffer)            
            # Get stored trajectories

            print(len(buffer.steps))
            critic_metrics = self._update_critic(self.critic_epochs,
                                                  scaled_reward=self.scale_reward, scaling_factor=self.scaling_factor, buffer=buffer)

            if iteration >= self.warmup_iterations:
                print("!NOT WARMUP!")
                actor_metrics = self._update_actor(self.actor_epochs,
                                                   scaled_reward=self.scale_reward,
                                                   scaling_factor=self.scaling_factor,
                                                    buffer=buffer)
            else:
                actor_metrics = {"actor/avg_loss": None, "actor/epochs": 0}  # Placeholder metrics for warmup
                
            # Log iteration metrics
            wandb.log({
                    "iteration": iteration,
                    **critic_metrics,
                    **actor_metrics
                })
            if iteration >= self.warmup_iterations:
                self.lora_save_every_n(iteration)
                
            # save checkpoint every iter
            self._save_checkpoint(iteration, buffer=buffer)            
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

            # with open("critic_dataset.pkl", "wb") as f:
            #     pickle.dump(dataset, f)
            # print("Dataset saved to critic_dataset.pkl")


            dataloader = DataLoader(
                                    dataset,
                                    batch_size=self.critic_batch_size,
                                    shuffle=True,
                                    collate_fn=custom_collate_fn
                                )

            for inx, batch in enumerate(tqdm(dataloader)):
                batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                # self.critic_optimizer.zero_grad()
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
                print(q1.requires_grad)
                print(v1.requires_grad)
                print(target_q1.requires_grad)
                print(target_v1.requires_grad)

                loss = self.critic_loss(q1, q2, v1, v2,
                                        target_v1, target_v2,
                                        target_q1, target_q2)

                loss.backward()
                # Log gradient norms
                grad_norms = {}
                for name, param in self.critic.named_parameters():
                    if param.grad is not None:
                        grad_norms[f"critic_grad/{name}"] = param.grad.norm().item()


                if (inx+1) % self.critic_grad_accum_steps == 0 or (inx+1) == len(dataloader):
                    clip_grad_norm_(self.critic.parameters(), self.critic_max_grad_norm)
                    self.critic_optimizer.step()
                    self.agent.soft_target_update(self.target_critic, self.critic, self.tau)
                    self.critic_optimizer.zero_grad()

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
                            "critic/q1_std": q1.std().item(),
                            "critic/q2_std": q2.std().item(),
                            "critic/v1_std": v1.std().item(),
                            "critic/v2_std": v2.std().item(),
                            "critic/epoch": e,
                            "target/target_q1_mean": target_q1.mean().item(),
                            "target/target_q2_mean": target_q2.mean().item(),
                            "target/target_v1_mean": target_v1.mean().item(),
                            "target/target_v2_mean": target_v2.mean().item(),
                            "target/target_q1_min": target_q1.min().item(),
                            "target/target_q1_max": target_q1.max().item(),
                            "target/target_q2_min": target_q2.min().item(),
                            "target/target_q2_max": target_q2.max().item(),
                            "target/target_v1_min": target_v1.min().item(),
                            "target/target_v1_max": target_v1.max().item(),
                            "target/target_v2_min": target_v2.min().item(),
                            "target/target_v2_max": target_v2.max().item(),
                            "target/target_q1_std": target_q1.std().item(),
                            "target/target_q2_std": target_q2.std().item(),
                            "target/target_v1_std": target_v1.std().item(),
                            "target/target_v2_std": target_v2.std().item(),

                        }
                    metrics.update(grad_norms)  # Add gradient norms to metrics

                    wandb.log(metrics)
                
                epoch_loss += loss.item()
                num_batches += 1
            
            epoch_losses.append(epoch_loss / num_batches)
        
        return {
            "critic/avg_loss": sum(epoch_losses) / len(epoch_losses),
            "critic/epochs": critic_epochs
        }

    def _update_actor(self, actor_epochs, scaled_reward=False, scaling_factor=100.0, buffer=None):
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
                                    batch_size=self.actor_batch_size,
                                    shuffle=True,
                                    collate_fn=custom_collate_fn
                                )

            epoch_loss = 0
            num_batches = 0
            for inx, batch in enumerate(tqdm(dataloader)):
                batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

                # fetches both logprob and actions in a batch

                if scaled_reward:
                    batch['reward'] = batch['reward'] / scaling_factor


                pi_action, logprobs = self.agent.get_policy_action(batch['obs'], get_logprob=True) 
                q1, q2, v1, v2 = self.agent.get_critic_values(batch['obs'], pi_action, detach_model=True)
                #take minumum of q and minimum of v
                q = torch.minimum(q1, q2)
                v = torch.minimum(v1, v2)

                advantages = self.agent.compute_advantages(q, v)

                # logprobs = self.agent.get_log_prob(batch['obs'],
                #                                    pi_action)
                print(logprobs.requires_grad)  # Should be True
                print(advantages.requires_grad)  # Should be True
                loss = self.actor_loss(advantages, logprobs)
                loss.backward()
                if (inx+1) % self.actor_grad_accum_steps == 0 or (inx+1) == len(dataloader):

                    clip_grad_norm_(self.learner.model.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    lora_grad_metrics = self._monitor_lora_gradients()
                    self.actor_optimizer.zero_grad()

                    # Log batch metrics
                    with torch.no_grad():
                        metrics = {
                            "actor/loss": loss.item(),
                            "actor/advantages_mean": advantages.mean().item(),
                            "actor/advantages_min": advantages.min().item(),
                            "actor/advantages_max": advantages.max().item(),
                            "actor/advantages_std": advantages.std().item(),
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
                            "actor/q1_std": q1.std().item(),
                            "actor/q2_std": q2.std().item(),
                            "actor/v1_std": v1.std().item(),
                            "actor/v2_std": v2.std().item(),
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

    def _evaluate_policy(self, current_iteration=None):
        metrics = self.evaluator.evaluate()
        # Log overall metrics to wandb
        wandb.log(metrics)

        return metrics

    def _monitor_lora_gradients(self):
        """Monitor gradients and parameters of LoRA adapters."""
        base = self._unwrap(self.learner.model)
        lora_params, lora_grads = {}, {}
        for name, param in base.named_parameters():
            if 'lora' in name and param.requires_grad:
                lora_params[name] = param.data.norm().item()
                lora_grads[name] = (param.grad.norm().item()
                                    if param.grad is not None else 0.0)
        avg_grad_norm = sum(lora_grads.values()) / max(len(lora_grads), 1)
        avg_param_norm = sum(lora_params.values()) / max(len(lora_params), 1)
        metrics = {
            "lora/avg_grad_norm": avg_grad_norm,
            "lora/avg_param_norm": avg_param_norm,
            "lora/active_params": len(lora_params)
        }
        for name, grad in list(lora_grads.items())[:5]:
            metrics[f"lora_grad/{name}"] = grad
        return metrics
    
    def _save_checkpoint(self, iteration, eval_metrics=None, buffer=None):
        """Save training checkpoint if the metric improves.
        
        Probably does not work well - must revisit and ensure it loads properly.
        """
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "iteration": iteration,
            "lora_state_dict": self._lora_state_dict(),
            "critic_state_dict": self._unwrap(self.critic).state_dict(),
            "target_critic_state_dict": self._unwrap(self.target_critic).state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "config": self.cfg,
            "best_metric": self.best_metric
        }

        if eval_metrics:
            current_metric = eval_metrics['eval/average_accumulated_reward']
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                best_checkpoint_path = os.path.join(checkpoint_dir, f"{self.cfg.run_name}_best_checkpoint.pt")
                torch.save(checkpoint, best_checkpoint_path)
                print(f"Best checkpoint saved at {best_checkpoint_path} with metric: {current_metric:.2f}")
                if buffer:
                    best_buffer_path = os.path.join(checkpoint_dir, f"{self.cfg.run_name}_best_buffer.pkl")
                    buffer.save_buffer(best_buffer_path, default_name = False)
                    print(f"Best buffer saved at {best_buffer_path}")

        else:
        # Use a key metric to determine if this is the best checkpoint
                
            latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
            torch.save(checkpoint, latest_checkpoint_path)
            print(f"New checkpoint saved at {latest_checkpoint_path}")
            if buffer:
                latest_buffer_path = os.path.join(checkpoint_dir, "latest_buffer.pkl")
                buffer.save_buffer(latest_buffer_path, default_name=False)


            with open(os.path.join(checkpoint_dir, "latest_checkpoint.txt"), "w") as f:
                f.write(str(iteration))


    def _unwrap(self, model):
        return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model

    def _lora_state_dict(self):
        base = self._unwrap(self.learner.model)
        # Filter only LoRA params (name contains 'lora')
        return {k: v for k, v in base.state_dict().items() if 'lora' in k}

    def lora_save_every_n(self, iteration):
        if self.lora_save_every and iteration > 0 and iteration % self.lora_save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            lora_path = os.path.join(
                "checkpoints",
                f"{self.cfg.run_name}_lora_dict_iter_{iteration}.pt"
            )
            save_lora_state_dict(self.learner.model, lora_path)
            print(f"Saved LoRA state dict at {lora_path}")

            # Load credentials from environment variables
            hf_token = os.getenv("HF_TOKEN")
            hf_username = os.getenv("HF_USERNAME")
            if not hf_token or not hf_username:
                print("HF_TOKEN or HF_USERNAME not set in environment. Skipping push to Hugging Face Hub.")
                return

            repo_id = f"{hf_username}/{self.cfg.run_name}_lora_adapter"

            # Authenticate (only needs to be done once per session)
            login(token=hf_token)

            # If using transformers' PEFT or similar, use push_to_hub
            try:
                self.learner.model.push_to_hub(
                    repo_id=repo_id,
                    commit_message=f"LoRA adapter at iteration {iteration}",
                    use_temp_dir=True
                )
                print(f"Pushed LoRA adapter to Hugging Face Hub: {repo_id}")
            except Exception as e:
                print(f"Failed to push to Hugging Face Hub: {e}")


class ArcherEval(EvalBatchRollout):
    def __init__(self, learner, teacher, cfg, game_registry):
        """
        Evaluation class for ArcherPlayPen.
        Args:
            learner: The learner model.
            teacher: The teacher model.
            cfg: Configuration dictionary.
        """

        # need to reorder stuff here so that the total steps == total eval instances.
        super().__init__(learner, teacher)
        self.cfg = cfg
        self.forPlayer = cfg.game.learner.name
        self.learner_name = cfg.game.learner.name
        self.teacher_name = cfg.game.teacher.name
        # self.eval_rollout_steps = self.cfg.trainer.eval_rollout_steps
        self.eval_instances = cfg.trainer.eval_instances
        self.batch_size = cfg.trainer.inference_batch_size
        self.game_registry = game_registry
        # Add evaluation-specific callbacks
        self.add_callback(GameRecordCallback(top_dir=f"eval_results/{self.cfg.run_name}", store_instance=True))
        self.game_spec = game_registry.get_game_specs_that_unify_with(self.cfg.game.spec_name)[0]
        players = [self.learner, self.teacher] if self.teacher else [self.learner]
        with make_eval_env(self.game_spec, players, shuffle_instances = False, instances_name=self.eval_instances, batch_size=self.batch_size) as self.eval_env:
            self.eval_buffer = BatchRolloutBuffer(self.eval_env)

        self.add_callback(RolloutProgressCallback(self.eval_env.get_rollout_length()))

    def evaluate(self):
        """
        Perform evaluation and return metrics.
        Args:
            game_registry: The game registry containing game specifications.
        Returns:
            A dictionary of evaluation metrics.
        """
        # set temp to 0
        self.learner.set_gen_args(temperature=0, max_tokens=self.cfg.game.learner.max_tokens)

        # Collect rollouts for evaluation
        self._collect_rollouts(
                game_env=self.eval_env,
                rollout_buffer=self.eval_buffer,
                forPlayer=self.forPlayer,
            )

            # Process evaluation trajectories
        eval_trajectories = self.eval_buffer.sample_trajectories()
        metrics = self._process_trajectories(eval_trajectories)

        self.eval_buffer.reset()
        # revert temp back to default
        self.learner.set_gen_args(temperature = self.cfg.game.learner.temperature,
                                   max_tokens=self.cfg.game.learner.max_tokens)

        
        return metrics

    def _process_trajectories(self, trajectories):
        """
        Process evaluation trajectories to compute metrics.
        Args:
            trajectories: List of trajectories collected during evaluation.
        Returns:
            A dictionary of computed metrics.
        """
        total_episode_scores = []
        total_response_scores = []
        per_episode_response_sum = []
        game_length = []

        success_count = 0
        aborted_count = 0
        lost_count = 0

        for trajectory in trajectories:
            if not trajectory:
                continue
            episode_score = 0
            trajectory_response_sum = 0
            game_length.append(len(trajectory))

            for step in trajectory:
                if step['done']:
                    response_score = step['info'].get('episode_score', 0)
                else:
                    response_score = step['info'].get('response_score', 0)

                total_response_scores.append(response_score)
                trajectory_response_sum += response_score
                episode_score = step['info'].get('episode_score', episode_score)

            total_episode_scores.append(episode_score)
            per_episode_response_sum.append(trajectory_response_sum)

            instance_info = trajectory[-1]['info']
            if instance_info['success']:
                print('success')
                success_count += 1
            elif instance_info['aborted']:
                print('aborted')
                aborted_count += 1
            elif instance_info['lost']:
                print('lost')
                lost_count += 1

        print(f"Total evaluated episodes: {len(total_episode_scores)}")
        metrics = {
            'eval/average_episode_reward': sum(total_episode_scores) / len(total_episode_scores) if total_episode_scores else 0,
            'eval/average_turn_reward': sum(total_response_scores) / len(total_response_scores) if total_response_scores else 0,
            'eval/average_accumulated_reward': sum(per_episode_response_sum) / len(per_episode_response_sum) if per_episode_response_sum else 0,
            'eval/success_count': success_count,
            'eval/aborted_count': aborted_count,
            'eval/lost_count': lost_count,
            'eval/avg_game_length': sum(game_length) / len(game_length) if game_length else 0,
            'eval/min_episode_reward': min(total_episode_scores) if total_episode_scores else 0,
            'eval/max_episode_reward': max(total_episode_scores) if total_episode_scores else 0,
            'eval/std_episode_reward': torch.std(torch.tensor(total_episode_scores, dtype=torch.float32)).item() if total_episode_scores else 0,
            'eval/min_turn_reward': min(total_response_scores) if total_response_scores else 0,
            'eval/max_turn_reward': max(total_response_scores) if total_response_scores else 0,
            'eval/std_turn_reward': torch.std(torch.tensor(total_response_scores, dtype=torch.float32)).item() if total_response_scores else 0,
            'eval/min_accumulated_reward': min(per_episode_response_sum) if per_episode_response_sum else 0,
            'eval/max_accumulated_reward': max(per_episode_response_sum) if per_episode_response_sum else 0,
            'eval/std_accumulated_reward': torch.std(torch.tensor(per_episode_response_sum, dtype=torch.float32)).item() if per_episode_response_sum else 0,
            'eval/min_game_length': min(game_length) if game_length else 0,
            'eval/max_game_length': max(game_length) if game_length else 0,
            'eval/std_game_length': torch.std(torch.tensor(game_length, dtype=torch.float32)).item() if game_length else 0,
        }

        return metrics