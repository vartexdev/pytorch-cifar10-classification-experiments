import torch
import torch.onnx
# model_6'nın mimarisini (class tanımını) buraya kopyalaman gerekiyor
# Tıpkı notebook'taki gibi:
# class CIFAR10ModelV_CNN_With_BN(nn.Module):
#    ... (tüm class kodunu buraya yapıştır) ...

# --- Ayarlar ---
MODEL_WEIGHTS_PATH = "models/model_6_weights.pth" # Model ağırlıklarını kaydettiğin .pth dosyasının yolu
ONNX_MODEL_PATH = "models/cifar10_model_v2.onnx" # Çıktı olarak üreteceğimiz ONNX modelinin yolu
HIDDEN_UNITS = 128 # En iyi modeli (Deneme 5) 128 unit ile eğitmiştik

# 1. Modeli Başlat (Ağırlıklar olmadan, sadece mimari)
print("Model mimarisi oluşturuluyor...")
# Not: Buradaki class adı, senin notebook'taki class adınla aynı olmalı
model = CIFAR10ModelV_CNN_With_BN(input_shape=3,
                                  hidden_units=HIDDEN_UNITS,
                                  output_shape=10) # 10 sınıfımız var

# 2. Kayıtlı Ağırlıkları Modele Yükle
print(f"'{MODEL_WEIGHTS_PATH}' adresinden ağırlıklar yükleniyor...")
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))

# 3. Modeli "Inference" (Çıkarım) Moduna Al
# Bu, BatchNorm ve Dropout gibi katmanları kapatır. Dönüşüm için şarttır.
model.eval() 

# 4. Sahte Bir Girdi Tensörü (Dummy Input) Oluştur
# ONNX, modelin içinden bir kere veri geçirmemizi (trace) ister.
# Girdinin boyutu, modelin eğitimde beklediği boyutla aynı olmalı:
# (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 32, 32) 

# 5. ONNX'e Çevir!
print(f"Modele '{ONNX_MODEL_PATH}' adresine çevriliyor...")
torch.onnx.export(model,               # Çalıştırılacak model
                  dummy_input,         # Modelin girdi tensörü
                  ONNX_MODEL_PATH,     # Modelin kaydedileceği yer
                  input_names=['input'],   # Modelin girdi adı
                  output_names=['output'], # Modelin çıktı adı
                  dynamic_axes={'input' : {0 : 'batch_size'}, # Batch size'ın dinamik olmasına izin ver
                                'output' : {0 : 'batch_size'}})

print("Dönüşüm tamamlandı.")