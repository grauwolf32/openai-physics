from gym.envs.registration import register

register(
    id='hyrosphere-v0',
    entry_point='gym_hyrosphere.envs:HyrospherePhysicsEnv',
)

register(
    id='linearsphere-v0',
    entry_point='gym_hyrosphere.envs:LinearspherePhysicsEnv',
)