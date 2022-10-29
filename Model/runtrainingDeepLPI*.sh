rm /home/wbm001/deeplpi/DeepLPI/output/tensorboard/*
rm /home/wbm001/deeplpi/DeepLPI/output/model/*0.pth.tar

nohup python -u "/home/wbm001/deeplpi/DeepLPI/Model/DeepLPI6165v9b3-Davis-Reg.py" > "/home/wbm001/deeplpi/DeepLPI/output/nohup/network.out" 2>&1 &

killall tensorboard
nohup tensorboard --logdir "/home/wbm001/deeplpi/DeepLPI/output/tensorboard" > "/home/wbm001/deeplpi/DeepLPI/output/nohup/tensorboard.out" 2>&1 &

tail -fn 50 /home/wbm001/deeplpi/DeepLPI/output/nohup/network.out