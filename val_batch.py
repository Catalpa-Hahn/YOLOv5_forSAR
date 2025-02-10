import subprocess
import platform
import pathlib

def main():
    models = ['best_300epochs.pt', 'best_400plus_epochs.pt']
    # 选择dataset文件
    plt = platform.system() # 判断操作系统类型
    if plt == 'Windows':
        dataset_path = './data/SARDet-100K_local.yaml'
    elif plt == 'Linux':
        dataset_path = './data/SARDet-100K.yaml'
    else:
        dataset_path = './data/SARDet-100K.yaml'

    # 运行测试
    for i in range(len(models)):
        print('正在测试模型{}...'.format(models[i]))
        subprocess.run(['python', 'val.py',
                        '--data', dataset_path,
                        '--weights', './weights/{}'.format(models[i]),
                        '--batch-size', '32',
                        '--imgsz', '640',
                        '--conf-thres', '0.001',
                        '--iou-thres', '0.6',
                        '--task', 'test'
                        ])
    print('测试结束！')

if __name__ == '__main__':
    main()