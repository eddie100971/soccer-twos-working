from mappo.mappo_trainer import MAPPOTrainer
from mappo.ppo_model import PolicyNormal
from mappo.ppo_model import CriticNet
from mappo.ppo_model import ActorNet
from mappo.ppo_model import Opponent
from mappo.ppo_agent import PPOAgent
from soccer_twos.wrappers import EnvType
import soccer_twos
import numpy as np
from nashpy import Game
import torch
import sys  
import os
import wandb

device = torch.device('cpu')
#'cuda:0' if torch.cuda.is_available() else 

def load_env(env_loc):
    """
    Initializes the UnityEnviornment and corresponding environment variables
    based on the running operating system.

    Arguments:
        env_loc: A string designating unity environment directory.

    Returns:
        env: A UnityEnvironment used for Agent evaluation and training.
        num_agents: Integer number of Agents to be trained in environment.
        state_size: Integer number of possible states.
        action_size: Integer number of possible actions.
    """

    # Initialize unity environment, return message if error thrown.
    env = soccer_twos.make(render=False, termination_mode="ALL", time_scale=80)

    # Extract state dimensionality from env.
    state_size = env.observation_space.shape[0]

    # Extract action dimensionality and number of agents from env.
    action_size = env.action_space.shape[0]
    num_agents = 4

    # Display relevant environment information.
    print('\nNumber of Agents: {}, State Size: {}, Action Size: {}\n'.format(
        num_agents, state_size, action_size))

    return env, num_agents, state_size, action_size

def create_opponent(state_size, action_size, actor_fc1_units=512,
                 actor_fc2_units=256, agent_ix=None, epoch=None):
    if epoch is not None:
        path = os.path.join(os.getcwd(), "saved_files", f"actor_agent_{agent_ix}_episode_{epoch}.pth") 
    else:
        path = None
    
    return Opponent(state_size, action_size, actor_fc1_units, actor_fc2_units, path)
    

def create_agent(state_size, action_size, actor_fc1_units=512,
                 actor_fc2_units=256, actor_lr=1e-4, critic_fc1_units=512,
                 critic_fc2_units=256, critic_lr=1e-4, gamma=0.99,
                 num_updates=10, max_eps_length=1500, eps_clip=0.3,
                 critic_loss=0.5, entropy_bonus=0.01, batch_size=256, agent_ix = 0, 
                 use_sd=False, sd_delta=.5):
    """
    This function creates an agent with specified parameters for training.

    Arguments:
        state_size: Integer number of possible states.
        action_size: Integer number of possible actions.
        actor_fc1_units: An integer number of units used in the first FC
            layer for the Actor object.
        actor_fc2_units: An integer number of units used in the second FC
            layer for the Actor object.
        actor_lr: A float designating the learning rate of the Actor's
            optimizer.
        critic_fc1_units: An integer number of units used in the first FC
            layer for the Critic object.
        critic_fc2_units: An integer number of units used in the second FC
            layer for the Critic object.
        critic_lr: A float designating the learning rate of the Critic's
            optimizer.
        gamma: A float designating the discount factor.
        num_updates: Integer number of updates desired for every
            update_frequency steps.
        max_eps_length: An integer for maximum number of timesteps per
            episode.
        eps_clip: Float designating range for clipping surrogate objective.
        critic_loss: Float designating initial Critic loss.
        entropy_bonus: Float increasing Actor's tendency for exploration.
        batch_size: An integer for minibatch size.

    Returns:
        agent: An Agent object used for training.
    """

    # Create Actor/Critic networks based on designated parameters.
    actor_net = ActorNet(state_size, action_size, actor_fc1_units,
                         actor_fc2_units).to(device)
    critic_net = CriticNet(state_size, critic_fc1_units, critic_fc2_units)\
        .to(device)

    # Create copy of Actor/Critic networks for action prediction.
    actor_net_old = ActorNet(state_size, action_size, actor_fc1_units,
                             actor_fc2_units).to(device)
    critic_net_old = CriticNet(state_size, critic_fc1_units, critic_fc2_units)\
        .to(device)
    actor_net_old.load_state_dict(actor_net.state_dict())
    critic_net_old.load_state_dict(critic_net.state_dict())

    # Create PolicyNormal objects containing both sets of Actor/Critic nets.
    actor_critic = PolicyNormal(actor_net, critic_net, agent_ix)
    actor_critic_old = PolicyNormal(actor_net_old, critic_net_old, agent_ix)

    # Initialize optimizers for Actor and Critic networks.
    actor_optimizer = torch.optim.Adam(
        actor_net.parameters(),
        lr=actor_lr
    )
    critic_optimizer = torch.optim.Adam(
        critic_net.parameters(),
        lr=critic_lr
    )

    # Create and return PPOAgent with relevant parameters.
    agent = PPOAgent(
        device=device,
        actor_critic=actor_critic,
        actor_critic_old=actor_critic_old,
        gamma=gamma,
        num_updates=num_updates,
        eps_clip=eps_clip,
        critic_loss=critic_loss,
        entropy_bonus=entropy_bonus,
        batch_size=batch_size,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        use_sd=use_sd,
        sd_delta=sd_delta
    )

    return agent


def create_trainer(env, agents, opponents, save_dir, state_size=0, action_size=0, update_frequency=5000,
                   max_eps_length=1500, score_window_size=100, use_PSRO=False):
    """
    Initializes trainer to train agents in specified environment.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        agents: Agent objects used for training.
        save_dir: Path designating directory to save resulting files.
        update_frequency: An integer designating the step frequency of
            updating target network parameters.
        max_eps_length: An integer for maximum number of timesteps per
            episode.
        score_window_size: Integer window size used in order to gather
            max mean score to evaluate environment solution.

    Returns:
        trainer: A MAPPOTrainer object used to train agents in environment.
    """
    if(use_PSRO):
        trainer = PSRO(
        env=env,
        agents=agents,
        opponents=opponents,
        score_window_size=score_window_size,
        max_episode_length=max_eps_length,
        update_frequency=update_frequency,
        save_dir=save_dir
        # action_size=action_size,
        # state_size=state_size,
        # rollout_length= 100
    )
    else:
    # Initialize MAPPOTrainer object with relevant arguments.
        trainer = MAPPOTrainer(
            env=env,
            agents=agents,
            opponents=opponents,
            score_window_size=score_window_size,
            max_episode_length=max_eps_length,
            update_frequency=update_frequency,
            save_dir=save_dir
        )

    return trainer


def train_agents(env, trainer, n_episodes=100, target_score=0.5,
                 score_window_size=100):
    """
    This function carries out the training process with specified trainer.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        trainer: A MAPPOTrainer object used to train agents in environment.
        n_episodes: An integer for maximum number of training episodes.
        target_score: An float max mean target score to be achieved over
            the last score_window_size episodes.
        score_window_size: The integer number of past episode scores
            utilized in order to calculate the current mean scores.
    """

    # Train the agent for n_episodes.
    for i_episode in range(1, n_episodes + 1):

        # Step through the training process.
        trainer.step()
        
        trainer.print_status()
        
        # Print status of training every 100 episodes.
        if i_episode % 100 == 0:
            scores = np.max(trainer.score_history, axis=1).tolist()
            trainer.print_status()

        # If target achieved, print and plot reward statistics.
        '''
        mean_reward = np.max(
            trainer.score_history[-score_window_size:], axis=1
        ).mean()
        if mean_reward >= target_score:
            print('Environment is solved.')
            env.close()
            trainer.print_status()
            trainer.plot()
            trainer.save()
            break
        '''
    trainer.save()

def train_agents_sp(env, trainer, n_episodes=100, target_score=0.5,
                score_window_size=100, epochs = 85):
    """
    This function carries out the training process with specified trainer.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        trainer: A MAPPOTrainer object used to train agents in environment.
        n_episodes: An integer for maximum number of training episodes.
        target_score: An float max mean target score to be achieved over
            the last score_window_size episodes.
        score_window_size: The integer number of past episode scores
            utilized in order to calculate the current mean scores.
    """

    # Train the agent for n_episodes.
    for epoch in range(epochs):
        for i_episode in range(1, n_episodes + 1):

            # Step through the training process.
            trainer.step()
            
            trainer.print_status()
            
            # Print status of training every 100 episodes.
            if i_episode % 100 == 0:
                scores = np.max(trainer.score_history, axis=1).tolist()
                trainer.print_status()

            # If target achieved, print and plot reward statistics.
            '''
            mean_reward = np.max(
                trainer.score_history[-score_window_size:], axis=1
            ).mean()
            if mean_reward >= target_score:
                print('Environment is solved.')
                env.close()
                trainer.print_status()
                trainer.plot()
                trainer.save()
                break
            '''
        trainer.save()
        trainer.opponents = [trainer.opponents[o].child(os.path.join("saved_files", f"actor_agent_{o}_episode_{(epoch + 1)*n_episodes}.pth")) for o in (0,1)]
        print(trainer.opponents)
        print(f'{epoch + 1} / {epochs} epochs done')      


class PSRO(MAPPOTrainer):
    def __init__(self, env, agents, score_window_size, max_episode_length,
                 update_frequency, save_dir, opponents):#, #action_size, state_size, rollout_length):
        super().__init__(env, agents, score_window_size, max_episode_length,
                 update_frequency, save_dir, opponents)

        self.population1 = [agents]
        self.population2 = [opponents]
        self.utilities = [[]]

        self.agent_args = (336, 3)
        self.rollout_length = 100



    def rollout(self) -> int: # gets utility of two policies
        """
        Runs several episodes and averages the rewards

        Returns:
            utility: average utility of each episode
        """
        
        # Initialize list to hold reward values at each timestep.
        utility = 0

        for _ in range(self.rollout_length):
            utility += self.step()
        
        return utility / self.rollout_length
    
    def run(self, epochs):

        for i in range(len(self.population1)):
            self.agents = self.population1[i]
            for j in range(len(self.population2)):
                self.opponents = self.population2[j]
                if len(self.utilities) >= i:
                    self.utilities[i].append(self.rollout())
                else:
                    self.utilities.append([self.rollout()])

        populations = (self.population1, self.population2)
        for _ in range(epochs):
            for i, population in enumerate(populations):
                #instantiate current team
                self.agents = (create_agent(*self.agent_args, agent_ix=0), create_agent(*self.agent_args, agent_ix=1))
                ## rollouts
                nash = Game(np.array([np.array(self.utilities[j]) for j in range(len(self.utilities))])).support_enumeration().__next__()
                dis = (torch.distributions.Categorical(torch.tensor(nash[0])), torch.distributions.Categorical(torch.tensor(nash[1])))
                for j in range(self.rollout_length):
                    #sample opponent policy
                    #print(self.utilities)
                    #print(np.asarray([np.array(self.utilities[i]) for i in range(len(self.utilities))]))
                    #replace self.utilities with this:
                    self.opponents = populations[~i][dis[i].sample()] # samples a policy from the opponent
                    #runs and trains the episode
                    self.step()

                ##add policy to population
                population.append(self.agents)
                self.print_status()

            print("Udating Utilit Table")
            print("Epoch:", _+1)
            ## Update Utility Table
            for i in range(len(self.population1)):
                self.agents = self.population1[i]
                for j in range(len(self.population2)):
                    self.opponents = self.population2[j]
                    if len(self.utilities) == i: # new row
                        self.utilities.append([self.rollout()])
                    elif len(self.utilities[i]) == j: # existing row
                        self.utilities[i].append(self.rollout())
            
            for i in self.utilities:
                print(i)
        return Game(np.array([np.array(self.utilities[i]) for i in range(len(self.utilities))])).support_enumeration().__next__()



if __name__ == '__main__':
    wandb.login()
    run = wandb.init(project="soccerTwos")
    # Set the project where this run will be logged
    # Track hyperparameters and run metadata

    # Initialize environment, extract state/action dimensions and num agents.
    env, num_agents, state_size, action_size = load_env(os.getcwd())
    print(state_size, action_size)
    # Initialize agents for training.
    #agents = [create_agent(state_size, action_size, agent_ix=_) for _ in range(num_agents)]
    #opponents = [create_opponent(state_size, action_size, epoch=None, agent_ix=i) for i in range(num_agents)]
    agents = [create_agent(state_size, action_size, agent_ix=_, use_sd=True, sd_delta=.25) for _ in range(2)]
    opponents = [create_opponent(state_size, action_size, epoch=None, agent_ix=i) for i in range(2,4)]
    # Create MAPPOTrainer object to train agents.
    save_dir = os.path.join(os.getcwd(), r'saved_files')
    trainer = create_trainer(env, agents, opponents, save_dir, state_size, action_size, use_PSRO=False)

    # Train agent in specified environment.
    train_agents_sp(env, trainer, n_episodes=100)
    #print(trainer.run(6))
    print("Finished Training")
    
