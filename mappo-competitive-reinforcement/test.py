'''
import soccer_twos
env = soccer_twos.make(render=True)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space.shape)

team0_reward = 0
team1_reward = 0
obs, reward, done, info = env.step(
        {
            0: env.action_space.sample(),
            1: env.action_space.sample(),
            2: env.action_space.sample(),
            3: env.action_space.sample(),
        }
    )
obs, reward, done, info = env.reset()

print("Initial Observation Space")
print(obs)

print("Initial Reward")
print(reward)
# for i in range(2):
#     obs, reward, done, info = env.step(
#         {
#             0: env.action_space.sample(),
#             1: env.action_space.sample(),
#             2: env.action_space.sample(),
#             3: env.action_space.sample(),
#         }
#     )
#     # print(obs)
#     team0_reward += reward[0] + reward[1]
#     team1_reward += reward[2] + reward[3]
#     if done["__all__"]:
#         print("Total Reward: ", team0_reward, " x ", team1_reward)
#         team0_reward = 0
#         team1_reward = 0
#         env.reset()
'''
import soccer_twos
from soccer_twos.side_channels import EnvConfigurationChannel
env_channel = EnvConfigurationChannel()
env = soccer_twos.make(env_channel=env_channel, render=True)

obs, reward, done, info = env.reset()
print("Initial Observation Space")
print(reward)
print(obs)
env_channel.set_parameters(
    ball_state={
        "position": [1, -1],
        "velocity": [-1.2, 3],
    },
    players_states={
        3: {
            "position": [-5, 10],
            "rotation_y": 45,
            "velocity": [5, 0],
        }
    }
)
actions = {i:[1,1,1] if i < 2 else [0,0,0] for i in range(4)}
print(actions)
'''
obs, reward, done, info = env.step({
    0: [0,0,0],
    1: [0,0,0],             
    2: [0,0,0],
    3: [0,0,0],
})
'''
print("Initial Observation Space")
print(obs)
