This file containing the metrics we use:

FID
LPIPS
Inception_Score
PSNR
SSIM
CFID
CAFID
LR_consistence

In order to use those metrics some extra liblaries installation requered:
PLease execute the following commands:

    %pip install tensorflow_gan

    !git clone https://github.com/alexlee-gk/lpips-tensorflow.git
    %cd lpips-tensorflow
    !tf_upgrade_v2   --infile  /content/lpips-tensorflow/lpips_tf.py --outfile  /content/lpips-tensorflow/lpips_tf.py
    !tf_upgrade_v2   --infile  /content/lpips-tensorflow/export_to_tensorflow.py --outfile  /content/lpips-tensorflow/export_to_tensorflow.py
    !tf_upgrade_v2   --infile  /content/lpips-tensorflow/setup.py --outfile  /content/lpips-tensorflow/setup.py
    %pip install -r requirements.txt
    
Than in order to use those liblaries do follows:  
sys.path.append("../lpips-tensorflow/")
import lpips_tf
import tensorflow_gan as tfg
import tensorflow as tf
