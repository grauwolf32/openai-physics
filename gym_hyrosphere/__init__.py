from gym.envs.registration import register

register(
    id='hyrosphere-v0',
    entry_point='gym_hyrosphere.envs:HyrospherePhysicsEnv',
)
