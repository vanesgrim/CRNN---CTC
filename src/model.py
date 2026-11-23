from __future__ import annotations

import torch
import torch.nn as nn


class CNNExtractor(nn.Module):
    """
    Paso 3 del documento: Extractor convolucional tipo VGG con BatchNorm en todos los bloques
    para estabilizar el entrenamiento con Adam + CTCLoss.
    Entrada: (N, 1, 32, 128)
    Salida:  (N, 512, 1, 32)
    """

    def __init__(self):
        super().__init__()

        # Bloque 1: conv 3x3 (64) + BN + ReLU + maxpool 2x2 -> (N, 64, 16, 64)
        self.bloque1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Bloque 2: conv 3x3 (128) + BN + ReLU + maxpool 2x2 -> (N, 128, 8, 32)
        self.bloque2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Bloque 3: 2x (conv 3x3 (256) + BN + ReLU) + maxpool 2x1 -> (N, 256, 4, 32)
        self.bloque3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # Bloque 4: 2x (conv 3x3 (512) + BN + ReLU) + maxpool 2x1 -> (N, 512, 2, 32)
        self.bloque4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # Bloque 5: conv 2x1 (512) + BN + ReLU, sin relleno -> (N, 512, 1, 32)
        self.bloque5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, mostrar_formas: bool = False) -> torch.Tensor:
        if mostrar_formas:
            print(f"Entrada (N, C, H, W): {tuple(x.shape)}")

        x = self.bloque1(x)
        if mostrar_formas:
            print(f"Después Bloque 1:     {tuple(x.shape)}  (pool 2x2: H y W a la mitad)")

        x = self.bloque2(x)
        if mostrar_formas:
            print(f"Después Bloque 2:     {tuple(x.shape)}  (pool 2x2: W llega a T=32)")

        x = self.bloque3(x)
        if mostrar_formas:
            print(f"Después Bloque 3:     {tuple(x.shape)}  (pool 2x1: solo baja H)")

        x = self.bloque4(x)
        if mostrar_formas:
            print(f"Después Bloque 4:     {tuple(x.shape)}  (pool 2x1: solo baja H)")

        x = self.bloque5(x)
        if mostrar_formas:
            print(f"Después Bloque 5:     {tuple(x.shape)}  (conv 2x1 sin padding: H=1)")

        return x


def mapa_a_secuencia(features: torch.Tensor) -> torch.Tensor:
    """
    Paso 4 del documento:
    Convierte (N, 512, 1, 32) -> (N, 32, 512)
    """
    return features.squeeze(2).permute(0, 2, 1)


class CRNN(nn.Module):
    """
    Arquitectura CRNN completa (Pasos 3, 4 y 5 del documento):
      CNN (VGG + BN) -> mapa_a_secuencia -> BiLSTM (2 capas, 256/dir) -> Linear (37 clases)
    Entrada: (N, 1, 32, 128)
    Salida:  (N, 32, 37)  [logits crudos por paso de tiempo y clase]
    """

    def __init__(self, num_classes: int = 37):
        super().__init__()
        self.cnn = CNNExtractor()

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor, mostrar_formas: bool = False) -> torch.Tensor:
        mapa = self.cnn(x, mostrar_formas=mostrar_formas)
        secuencia = mapa_a_secuencia(mapa)
        if mostrar_formas:
            print(f"Después mapa->sec.:   {tuple(secuencia.shape)}  (N, T=32, 512)")

        rnn_out, _ = self.rnn(secuencia)
        if mostrar_formas:
            print(f"Después BiLSTM:       {tuple(rnn_out.shape)}  (N, T=32, 256*2=512)")

        logits = self.fc(rnn_out)
        if mostrar_formas:
            print(f"Logits finales:       {tuple(logits.shape)}  (N, T=32, C=37 clases)")

        return logits