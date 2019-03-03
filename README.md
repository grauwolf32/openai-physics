**Setup**
sudo pip3 inatall -e .

**Learn model**
 python3 -m baselines.run --alg=acktr --network=mlp --env=hyrosphere-v0 --num_timesteps=10e4 --save_path=~/openai-physics/hyrosphere-acktr --load_path=~/openai-physics/hyrosphere-acktr 

**Play model**
 python3 -m baselines.run --alg=acktr --network=mlp --env=hyrosphere-v0 --num_timesteps=0  --load_path=~/openai-physics/hyrosphere-acktr  --play
