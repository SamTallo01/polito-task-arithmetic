# 1) finetune on all datasets
# 2) eval single task
# 3) task addition

python finetune.py \
--data-location= C:\Users\gade_\Documents\MSC-ComputerEngineering\AdvancedMachineLearning\datasets \
--save=C:\Users\gade_\Documents\MSC-ComputerEngineering\AdvancedMachineLearning\results \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0

python eval_single_task.py \
--data-location=C:\Users\gade_\Documents\MSC-ComputerEngineering\AdvancedMachineLearning\datasets \
--save=C:\Users\gade_\Documents\MSC-ComputerEngineering\AdvancedMachineLearning\results \

python eval_task_addition.py \
--data-location=C:\Users\gade_\Documents\MSC-ComputerEngineering\AdvancedMachineLearning\datasets \
--save=C:\Users\gade_\Documents\MSC-ComputerEngineering\AdvancedMachineLearning\results \