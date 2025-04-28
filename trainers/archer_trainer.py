"""
Writing this code for:

1) Interactive training
2) Off-policy (collect rollouts -> train critic -> on-policy actor updates)
3) No branching

- We need rewards (success/fail/abort)
- dataloader for trajectory -> so that we can shuffle random steps and not learn on continuous trajectories.
- continuous trajectory vs shuffled steps should be an ablation - see the effects and compare them.

"""



from clemcore_multiturn_rl.clemcore.playpen import BasePlayPen, make_env, StepRolloutBuffer, GameRecordCallback
from clemcore_multiturn_rl.clemcore.clemgame import GameRegistry, GameSpec
from modelling.archer_critic import ArcherAgent, CriticNetwork

class ArcherPlayPen(BasePlayPen):
    def __init__(self, learner, teacher):
        super().__init__(learner, teacher)
        
        # Initialize Archer components
        self.policy = learner  # Your clemcore model
        self.critic = CriticNetwork(...)  # Initialize critic network
        self.target_critic = CriticNetwork(...)  # Initialize target critic
        self.agent = ArcherAgent(self.policy, self.critic, self.target_critic)
        
    def learn_interactive(self, game_registry: GameRegistry):
        # Select game spec you want to train on
        game_spec = game_registry.get_game_specs_that_unify_with(...)[0]
        
        # Create environment and buffer
        with make_env(game_spec, [self.learner, self.teacher]) as env:
            buffer = StepRolloutBuffer(env)
            self.add_callback(GameRecordCallback())
            
            # Training loop
            for episode in range(...):
                # Collect trajectories
                self._collect_rollouts(env, ..., buffer)

                # Get stored trajectories
                trajectories = buffer.trajectories
                
                # should turn the trajectories into a dataloader -> we want to break continuity so that the model doesn't overfit

                for trajectory in trajectories:
                    for step in trajectory:
                        # Get relevant data from step
                        obs = step['context'] 
                        action = step['response']
                        reward = step['info'].get('reward', 0)
                        next_obs = next(trajectory)['context']
                        done = step['done']
                        
                        # Update Archer agent
                        self.agent.compute_target_q(obs)
                        self.agent.compute_target_v(next_obs, action, reward, done)
                        # Add other Archer training logic
                
                buffer.reset()