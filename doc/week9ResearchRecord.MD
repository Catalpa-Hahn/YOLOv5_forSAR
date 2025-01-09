# 第9周工作记录

## 2024.12.31
运行中断，报错：  
RuntimeError: Caught RuntimeError in pin memory thread for device 0.  
RuntimeError: CUDA error: out of memory  
本次输出详见文件runs/exp2/YOLOv5第一次正式训练输出.pdf

解决：将参数resume设置为True，并将预训练权重参数weight改为最后一代训练权重last.pt后，重新启动训练，实现接续训练。

## 2025-01-04
第一次117轮训练完成，实现在SARDet-100K测试集上的预测。
