import subprocess
img_path = 'data/Replica/single_miriel_sep8/results/depth%06d.png'
#img_path = 'outputs/rvc_pretrain_sep8/limg%06d.jpg'
save_path = 'sep8_raw.mp4'
subprocess.run(args=f'ffmpeg -framerate 60 -i {img_path} -y {save_path}', shell=True)