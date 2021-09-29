import subprocess
import os

mkdirs = lambda x: os.path.exists(x) or os.makedirs(x)
runs = ".\\runs"
models =".\\Models"
datasets = ".\\Datasets"
results = '.\\results'
mkdirs(results)

num_of_tests = str(10)

# inference models
for model_name in os.listdir(runs):
    for task in os.listdir(os.path.join(runs,model_name)):
        for data in os.listdir(os.path.join(runs,model_name,task)):
            for x in os.listdir(os.path.join(runs, model_name, task, data)):
                if (x.endswith('.pth') and (not x.startswith('latest')) and x[0].isdigit()):
                    os.rename(os.path.join(runs, model_name, task, data,x),os.path.join(runs, model_name, task, data,'iter_'+x))
            num_of_oters = max([int(x.split('_')[1]) for x in os.listdir(os.path.join(runs,model_name,task,data)) if (x.endswith('.pth') and (not x.startswith('latest')) and x.split('_')[1][0].isdigit())])
            for iteration in range(num_of_oters, num_of_oters+8000, 8000):
                model_path = os.path.join(models, model_name,'test.py')
                results_path = os.path.join(results, model_name,task)
                checkpoints_dir = os.path.join(runs,model_name,task)
                data_root = os.path.join(datasets, data+'_'+task)
                load_iter = str(iteration)
                mkdirs(results_path)

                commands = ['python', model_path,\
                                    '--checkpoints_dir', checkpoints_dir, \
                                    '--name', data, \
                                    '--dataroot', data_root, \
                                    '--load_iter', load_iter, \
                                    '--num_test', '10', \
                                    '--crop_size', '128', \
                                    '--batch_size', '128', \
                                    '--display_winsize', '128', \
                                    '--load_size', '128', \
                                    '--no_flip'\
                                    ]
                if model_name.lower().startswith('bi'):
                    commands += ['--netD','basic_128_multi',\
                                '--netD2','basic_128_multi',\
                                '--netE', 'resnet_128',\
                                '--netG', 'unet_128',\
                                '--no_flip',\
                                '--nz','64',\
                                '--no_encode', \
                                '--n_samples', '5', \
                                '--model','bicycle_gan', \
                                 '--results_dir', os.path.join(results_path,data)
                                 ]
                    subprocess.run(commands,shell=True)
                elif model_name.lower().startswith('cy'):
                    commands += ['--netD', 'basic', \
                                 '--netG', 'unet_128', \
                                 '--model', 'cycle_gan',\
                                 '--results_dir', results_path
                                 ]
                    subprocess.run(commands,shell=True)
                elif model_name.lower().startswith('pix'):
                    commands += ['--netG', 'unet_128', \
                                 '--model', 'pix2pix',\
                                 '--results_dir', results_path
                                 ]
                    subprocess.run(commands,shell=True)
    # sample from model