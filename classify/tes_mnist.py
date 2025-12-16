import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from models.cnn import Net
#from toonnx import to_onnx

use_cuda = True
model = Net(10)
model.load_state_dict(torch.load('output/params_3.pth'))
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

img = cv2.imread('test.jpg')
img = cv2.resize(img, (28, 28))
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0)

if use_cuda and torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))

# 计算概率
probabilities = torch.nn.functional.softmax(prediction, dim=1)
pred = torch.max(prediction, 1)[1]

# 输出结果
print("预测值:", prediction.detach().numpy())
print("预测类别:", pred.item())
print("\n概率分布:")
for i in range(10):
    print(f"数字{i}: {probabilities[0][i].item():.4f}")

# 概率可视化，使用matplotlib
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"read photo\n predict: {pred.item()}")
plt.axis('off')

plt.subplot(1, 2, 2)
x = range(10)
plt.bar(x, probabilities[0].detach().numpy())
plt.title("probability distribution")
plt.xlabel("number")
plt.ylabel("probability")
plt.xticks(x)
plt.ylim(0, 1)

# 高亮预测结果
plt.gca().patches[pred.item()].set_color('red')

plt.tight_layout()
plt.show()

# 原有显示
cv2.imshow("image", img)
cv2.waitKey(0)