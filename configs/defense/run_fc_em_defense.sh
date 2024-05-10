cd ../..

# replace {custom poison path} with corresponding poison path in each .yaml file

python main.py --exp-config configs/defense/pgd.yaml --gpu-id 0 --eps 0.08

python main.py --exp-config configs/defense/trades.yaml --gpu-id 0 --eps 0.08

python main.py --exp-config configs/defense/mixup.yaml --gpu-id 0

python main.py --exp-config configs/defense/rsmix.yaml --gpu-id 0

python main.py --exp-config configs/defense/if_defense.yaml --gpu-id 0