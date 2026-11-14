import torch
from ultralytics import YOLO

if __name__ == "__main__":
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    print(f"Treinando no dispositivo: {dispositivo}")

    # Baixa e carrega o modelo base do YOLOv11
    model = YOLO("yolo11n.pt")

    # Inicia o treinamento usando o seu data.yaml
    model.train(
        data="data.yaml",
        epochs=35,
        imgsz=640,
        batch=16,
        device=dispositivo,
        hsv_h=0.0,  # Preserva as cores reais (verde vs. vermelho)
        hsv_s=0.3,
        hsv_v=0.3,
        fliplr=0.5,
        project="runs_tomate",
        name="modelo_maturacao",
    )

    # Avalia o modelo treinado nas imagens inéditas da pasta test/
    metricas = model.val(data="data.yaml", split="test")
    print(f"\nTreino finalizado! mAP50 no teste: {metricas.box.map50:.4f}")