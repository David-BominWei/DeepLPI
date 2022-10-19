rm /home/wbm001/deeplpi/DeepLPI/output/tensorboard/*
rm /home/wbm001/deeplpi/DeepLPI/output/model/*

nohup python -u "/home/wbm001/deeplpi/DeepLPI/scripts/DeepLPI_6165_Kd_Classification.py" > "/home/wbm001/deeplpi/DeepLPI/output/nohup/network.out" 2>&1 &
nohup tensorboard --logdir "/home/wbm001/deeplpi/DeepLPI/output/tensorboard" > "/home/wbm001/deeplpi/DeepLPI/output/nohup/tensorboard.out" 2>&1 &

tail -fn 50 /home/wbm001/deeplpi/DeepLPI/output/nohup/network.out