import torch
import torch.nn as nn

# 定义一个只包含ReLU激活函数的模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

# 创建一个模型实例
model = Model()

# 创建一个随机的输入张量
x = torch.randn(1, 3, 224, 224)

# 将模型保存为PyTorch模型
torch.save(model.state_dict(), 'model.pth')

# 将模型导出为ONNX格式
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
