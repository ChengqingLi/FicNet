# training codes (including test)
python trainer.py -batch 64  -data_dir  ../datasets/ -dataset car -gpu 1 -extra_dir your_run -temperature_attn 2.0 -lamb 1.5 -shot 5 -milestones 50 60 70
# direct test
python test.py -data_dir  ../datasets/ -dataset car  -extra_dir your_run

# parameters
'batch': 64,
 'data_dir': '../fewdata/',
 'dataset': 'car',
 'extra_dir': 'your_car',
 'gamma': 0.05,
 'gpu': '1',
 'lamb': 1.5,
 'lr': 0.1,
 'max_epoch': 80,
 'milestones': [50, 60, 70],
 'no_wandb': False,
 'query': 15,
 'save_all': False,
 'seed': 1,
 'self_method': 'scr',
 'shot': 5,
 'temperature': 0.2,
 'temperature_attn': 1.0,
 'test_episode': 1200,
 'val_episode': 200,
 'way': 5}
use gpu: [1]
manual seed: 1

# format of dataset
   images
	000001.jpg
	000002.jpg
	000003.jpg
	......
	016185.jpg
   split 
	train.csv
	val.csv
	test.csv
