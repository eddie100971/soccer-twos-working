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
env.reset()
env_channel.set_parameters(
    ball_state={
        "position": [0, 0],
        "velocity": [100, 0]
    },
    
    players_states = {
        1: {
            "position": [0, 0],
            "rotation_y": 0,
            "velocity": [0, 0],
        }
    }
)
for i in range(20):
    obs, reward, done, info = env.step({
        0: [0,0,0],
        1: [0,0,0],             
        2: [0,0,0],
        3: [0,0,0],
    })
    print(reward)

'''
#testing rewards
for i in range(10):
    obs, reward, done, info = env.step({
        0: env.action_space.sample(),
        1: env.action_space.sample(),             
        2: env.action_space.sample(),
        3: env.action_space.sample(),
    })
    print(reward)
'''

'''
print("Initial Observation Space")
print(obs)
'''
