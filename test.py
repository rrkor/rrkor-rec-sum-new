import sounddevice as sd
print(sd.query_devices())
print("Default input:", sd.query_devices(sd.default.device[0])['name'])


import torch
print(torch.backends.mps.is_available())  # Должно вернуть True
print(torch.backends.mps.is_built())      # Должно вернуть True