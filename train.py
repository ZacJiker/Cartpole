import gym

from tqdm import tqdm
from argparse import ArgumentParser
from reinforce import ReinforceAgent, PolicyNetwork

def run_training(env, agent, total_num_episodes=1000):
    reward_over_episode = []

    tqdm.write('Training the agent...')

    for _ in tqdm(range(total_num_episodes)):
        obs, _ = env.reset()
        done = False

        rewards = 0

        while not done:
            # Get action and log probability
            action = agent.sample_action(obs)
            # Take action
            obs, reward, terminated, truncated, _ = env.step(action)
            # Store reward
            rewards += reward
            # Check if episode is terminated or truncated
            done = terminated or truncated
        
        # Store reward per episode
        reward_over_episode.append(rewards)
        # Store reward on agent
        agent.rewards.append(rewards)
        # Update policy
        agent.update()

    tqdm.write('Training finished!')

    # Display the reward per episode
    agent.plot_reward_per_episode(reward_over_episode)

    # Answer if I want to save the model
    save = input('Do you want to save the model? (y/n): ')
    if save == 'y' or save == 'Y':
        # Save the model with the current timestamp
        agent.save_model()
    # Close environment
    env.close()

def run_test(env, agent):
    agent.load_model(args.model)
    obs, _ = env.reset()
    done = False

    while not done:
        action = agent.sample_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()

    env.close()

if __name__ == '__main__':
    argspars = ArgumentParser()
    argspars.add_argument('--model', type=str, default='model.h5')
    argspars.add_argument('--num_episodes', type=int, default=1000)
    
    args = argspars.parse_args()

    env = gym.make('CartPole-v1', render_mode='human' if args.model else 'rgb_array')

    # Hyperparameters
    observation_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.n

    # Initialize agent and policy
    policy = PolicyNetwork(observation_space_dims, action_space_dims)
    agent = ReinforceAgent(observation_space_dims, action_space_dims)

    # Run training or test
    run_test(env, agent) if args.model else run_training(env, agent)