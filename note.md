

python main.py --config configs/Pakistan/DQL/NOTE.yaml     --random     "training.lr=0.001,0.002,0.005,0.01,0.02"     "training.gamma=0.9,0.95,0.99"     "training.batch_size=128,256,512,1024"     "training.epsilon=0.1,0.15,0.2,0.3"     "training.update_freq=64,128,256"     "training.target_update_freq=64,128,256"    --num_workers 16  --n_samples 64