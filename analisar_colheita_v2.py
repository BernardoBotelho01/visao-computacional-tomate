"""Analisa fotos com o YOLO11n treinado neste projeto."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
PESOS = RAIZ / "modelo_treinado" / "weights" / "best.pt"
EXTENSOES = {".jpg", ".jpeg", ".png", ".webp"}
# id 0 = maduro, id 1 = verde. Critérios escolhidos nas 80 fotos de validação.
PERFIS = ("original", "ajustado")
IOU_MESMA_CLASSE = 0.45
IOU_OUTRA_CLASSE = 0.72
LIMIAR_MADURO_NOVO = 0.24
LIMIAR_VERDE_NOVO = 0.10
LIMIAR_APOIO_VERDE = 0.04
IOU_APOIO_VERDE = 0.45
PESOS_APOIO = PESOS.with_name("last.pt")
CORES = ((0, 60, 230), (40, 170, 40))


def conferir_classes(modelo):
    names = modelo.names
    names = dict(enumerate(names)) if isinstance(names, list) else names
    names = {int(k): " ".join(str(v).lower().split()) for k, v in names.items()}
    if names != {0: "ripe tomato", 1: "unripe tomato"}:
        raise ValueError(f"Pesos incompatíveis com as classes do projeto: {names}")


def sobreposicao(caixa, caixas):
    a = np.maximum(caixa[:2], caixas[:, :2])
    b = np.minimum(caixa[2:4], caixas[:, 2:4])
    inter = np.prod(np.maximum(0.0, b - a), axis=1)
    area1 = np.prod(np.maximum(0.0, caixa[2:4] - caixa[:2]))
    area2 = np.prod(np.maximum(0.0, caixas[:, 2:4] - caixas[:, :2]), axis=1)
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def combinar_vistas(a,b,min_iou=.5,isolada=.65):
    """Combina caixas da mesma classe e reduz o peso de detecções de uma só vista."""
    a=a[np.argsort(-a[:,4],kind='stable')]
    b=b[np.argsort(-b[:,4],kind='stable')]
    usados=np.zeros(len(b),bool)
    c=[]
    for x in a:
        ok=np.flatnonzero(~usados & (b[:,5]==x[5]))
        ious=sobreposicao(x,b[ok])
        if ok.size and ious.max() >= min_iou:
            idx=ok[ious.argmax()]; y=b[idx]; usados[idx]=True
            d=(x if x[4]>=y[4] else y).copy()
            d[:4]=(x[:4]*x[4]+y[:4]*y[4])/(x[4]+y[4])
            d[4]=max(x[4],y[4])
        else:
            d=x.copy();d[4]*=isolada
        c.append(d)
    d=b[~usados].copy();d[:,4]*=isolada
    return np.concatenate((np.asarray(c).reshape(-1,6),d))


def selecionar(candidatas, limiares, iou, duplicata=None, max_det=300):
    """Uma lista final de caixas para a contagem e a visualização."""
    a = np.asarray(candidatas, dtype=np.float32).reshape(-1, 6)
    a = a[a[:, 4] >= np.asarray(limiares)[a[:, 5].astype(int)]]
    ordem = np.argsort(-a[:, 4], kind="stable")
    manter = []
    while ordem.size and len(manter) < max_det:
        atual, resto = ordem[0], ordem[1:]
        manter.append(atual)
        iguais = a[resto, 5] == a[atual, 5]
        ov = sobreposicao(a[atual], a[resto])
        cortar = iguais & (ov > iou)
        if duplicata is not None:
            cortar |= (~iguais) & (ov > duplicata)
        ordem = resto[~cortar]
    return a[manter]


def prever(modelo, fotos, imgsz=640, device=None, bruto=False, conf=0.01, iou=0.75, batch=4):
    return modelo.predict(
        source=fotos, imgsz=imgsz, conf=conf, iou=iou, device=device,
        rect=False, agnostic_nms=False, verbose=False, batch=batch,
        max_det=450 if bruto else 300,
    )


def caixas_resultado(resultado):
    return resultado.boxes.data.cpu().numpy().copy().reshape(-1, 6)


def resultado_colheita(maduros, verdes):
    total = maduros + verdes
    if not total:
        return 0.0, "NENHUM TOMATE DETECTADO"
    return maduros * 100.0 / total, (
        "PRONTO PARA COLHEITA" if maduros * 5 >= total * 3 else "AGUARDAR MATURACAO"
    )


def desenhar(img, caixas, nome):
    import cv2

    img = img.copy()
    alt, larg = img.shape[:2]
    maduros = int(np.count_nonzero(caixas[:, 5] == 0))
    verdes = len(caixas) - maduros
    taxa, status = resultado_colheita(maduros, verdes)
    for x1, y1, x2, y2, conf, cls in caixas:
        x1 = max(0, min(larg - 1, round(float(x1))))
        x2 = max(0, min(larg - 1, round(float(x2))))
        y1 = max(0, min(alt - 1, round(float(y1))))
        y2 = max(0, min(alt - 1, round(float(y2))))
        classe = int(cls)
        cor = CORES[classe]
        texto = f"{('Maduro', 'Verde')[classe]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
        (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        x = max(0, min(x1, larg - w - 5))
        y = y1 - 5 if y1 >= h + 8 else min(alt - 2, y1 + h + 5)
        cv2.rectangle(img, (x, max(0, y - h - 3)), (min(larg - 1, x + w + 4), y + 2), cor, -1)
        cv2.putText(img, texto, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    # Painel externo à foto. O painel antigo podia encobrir caixas e frutos no topo.
    linhas = (f"Total: {len(caixas)} | Maduros: {maduros} | Verdes: {verdes}",
              f"Maduros detectados: {taxa:.1f}% | {status}")
    painel = np.full((66, larg, 3), 24, dtype=np.uint8)
    for i, linha in enumerate(linhas):
        escala = 0.55
        largura_texto = cv2.getTextSize(linha, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)[0][0]
        escala *= min(1, max(1, larg - 18) / max(1, largura_texto))
        cv2.putText(painel, linha, (8, 25 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, escala,
                    (245, 245, 245) if i == 0 else (130, 222, 235), 1, cv2.LINE_AA)
    return np.vstack((painel, img)), (nome, len(caixas), maduros, verdes, f"{taxa:.2f}%", status)


def ler_imagem(path):
    import cv2

    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrada", type=Path, default=RAIZ / "test" / "images", help="Foto ou pasta de fotos.")
    ap.add_argument("--saida", type=Path, default=RAIZ / "resultados_tomate_v2")
    ap.add_argument("--pesos", type=Path, default=PESOS)
    ap.add_argument("--pesos-apoio", type=Path, default=PESOS_APOIO)
    ap.add_argument("--perfil", choices=PERFIS, default="ajustado")
    ap.add_argument("--device", help="Ex.: cpu ou 0. Padrão: automático.")
    args = ap.parse_args()

    from ultralytics import YOLO
    import cv2

    if not args.entrada.exists():
        ap.error(f"Entrada não encontrada: {args.entrada}")
    if not args.pesos.is_file():
        ap.error(f"Arquivo de pesos não encontrado: {args.pesos}")
    if args.perfil == "ajustado" and not args.pesos_apoio.is_file():
        ap.error(f"Pesos auxiliares não encontrados: {args.pesos_apoio}. Use --perfil original para analisar só com best.pt.")
    fotos = sorted(p for p in args.entrada.iterdir() if p.is_file() and p.suffix.lower() in EXTENSOES) if args.entrada.is_dir() else [args.entrada]
    if not fotos:
        ap.error(f"Nenhuma imagem encontrada em {args.entrada}")
    args.saida.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(str(args.pesos))
    conferir_classes(modelo)
    auxiliar = YOLO(str(args.pesos_apoio)) if args.perfil == "ajustado" else None
    if auxiliar is not None:
        conferir_classes(auxiliar)
    linhas = []
    for foto in fotos:
        img = ler_imagem(foto)
        if img is None:
            print(f"Imagem ilegível: {foto}")
            continue
        caixas = detectar(modelo, [img], args.perfil, auxiliar, args.device)[0]

        # Limpeza final: reduz caixas duplicadas de maduros sem alterar os verdes.
        if args.perfil == "ajustado":
            maduros = caixas[caixas[:, 5] == 0]
            verdes = caixas[caixas[:, 5] == 1]
            maduros = selecionar(maduros, (0.0, 0.0), iou=0.30)
            caixas = np.concatenate((maduros, verdes), axis=0)

        painel, linha = desenhar(img, caixas, foto.name)
        ok, buf = cv2.imencode(".jpg", painel)
        if not ok:
            raise OSError(f"Falha ao gerar imagem para {foto.name}")
        nome_saida = f"analise_{foto.name}" if foto.suffix.lower() in {".jpg", ".jpeg"} else f"analise_{foto.name}.jpg"
        buf.tofile(args.saida / nome_saida)
        linhas.append(linha)

    if not linhas:
        ap.error("Nenhuma imagem foi processada.")
    with (args.saida / "relatorio_tomates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(("Arquivo", "Total_Tomates", "Maduros", "Verdes", "Taxa_Colheita", "Status"))
        writer.writerows(linhas)
    print(f"{len(linhas)} imagens processadas. Resultados em {args.saida.resolve()}")


def obter_candidatas(modelo, fotos, device=None, espelhar=False):
    fontes = [np.ascontiguousarray(img[:, ::-1]) for img in fotos] if espelhar else fotos
    caixas = [caixas_resultado(r) for r in prever(modelo, fontes, device=device, bruto=True)]
    if espelhar:
        for c, img in zip(caixas, fotos, strict=True):
            c[:, [0, 2]] = img.shape[1] - c[:, [2, 0]]
    return caixas


def tem_apoio(caixa, outras):
    ious = sobreposicao(caixa, outras)
    return bool(np.any((outras[:, 4] >= LIMIAR_APOIO_VERDE) & (ious >= IOU_APOIO_VERDE)))


def complementar(base, principal, espelhada, auxiliar):
    """Preserva a saída original e só acrescenta caixas sem sobreposição excessiva."""
    v_principal = principal[principal[:, 5] == 1]
    v_espelhada = espelhada[espelhada[:, 5] == 1]
    v_auxiliar = auxiliar[auxiliar[:, 5] == 1]
    verdes = [
        x for x in v_principal
        if x[4] >= LIMIAR_VERDE_NOVO and tem_apoio(x, v_espelhada) and tem_apoio(x, v_auxiliar)
    ]
    maduros = combinar_vistas(principal[principal[:, 5] == 0], auxiliar[auxiliar[:, 5] == 0])
    maduros = selecionar(maduros, (LIMIAR_MADURO_NOVO, 0.25), IOU_MESMA_CLASSE)
    final = list(base)
    for caixa in np.concatenate((np.asarray(verdes).reshape(-1, 6), maduros)):
        a = np.asarray(final).reshape(-1, 6)
        limites = np.where(a[:, 5] == caixa[5], IOU_MESMA_CLASSE, IOU_OUTRA_CLASSE)
        if not np.any(sobreposicao(caixa, a) > limites):
            final.append(caixa)
    return np.asarray(final, dtype=np.float32).reshape(-1, 6)


def detectar(modelo, fotos, perfil, auxiliar=None, device=None):
    base = [caixas_resultado(r) for r in prever(modelo, fotos, device=device, conf=0.25, iou=0.45)]
    if perfil == "original":
        return base
    if auxiliar is None:
        raise ValueError("O perfil ajustado precisa dos pesos auxiliares.")
    candidatas = obter_candidatas(modelo, fotos, device=device)
    espelhadas = obter_candidatas(modelo, fotos, device=device, espelhar=True)
    apoio = obter_candidatas(auxiliar, fotos, device=device)
    return [complementar(*x) for x in zip(base, candidatas, espelhadas, apoio, strict=True)]


if __name__ == "__main__":
    main()
