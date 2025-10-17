import torch
import torch.onnx
import torch.nn as nn
from pathlib import Path


class CIFAR10ModelV_CNN_With_BN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units), 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units), 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_3=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)



        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*4*4, 
                      out_features=output_shape)
        )
    
    def forward(self, x):
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))


MODEL_WEIGHTS_PATH = "models/model_6_weights.pth" # Model ağırlıklarını kaydettiğin .pth dosyasının yolu
ONNX_MODEL_PATH = "models/cifar10_model_v2.onnx" # Çıktı olarak üreteceğimiz ONNX modelinin yolu
HIDDEN_UNITS = 128 

# 1. Modeli Başlat (Ağırlıklar olmadan, sadece mimari)
# Not: Buradaki class adı, senin notebook'taki class adınla aynı olmalı
model = CIFAR10ModelV_CNN_With_BN(input_shape=3,
                                  hidden_units=HIDDEN_UNITS,
                                  output_shape=10) # 10 sınıfımız var



model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))



model.eval() 

# (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 32, 32) 

# 5. ONNX'e Çevir
torch.onnx.export(model,               # Çalıştırılacak model
                  dummy_input,         # Modelin girdi tensörü
                  ONNX_MODEL_PATH,     # Modelin kaydedileceği yer
                  input_names=['input'],   # Modelin girdi adı
                  output_names=['output'], # Modelin çıktı adı
                  dynamic_axes={'input' : {0 : 'batch_size'}, # Batch size'ın dinamik olmasına izin ver
                                'output' : {0 : 'batch_size'}})

