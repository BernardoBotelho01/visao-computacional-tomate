# Contexto do Projeto de Visão Computacional

## Stack e Ambiente
- Linguagem: Python 3 (ambiente Windows, execução via terminal PowerShell).
- Biblioteca principal: Ultralytics YOLOv8, OpenCV (`cv2`) e manipulação de datasets exportados do Roboflow.
- Sempre gere código Python compatível com Windows (use `pathlib.Path` em vez de caminhos fixos com barras invertidas).

## Estrutura e Regras de Segurança de Arquivos
- Os diretórios `resultados_tomate/`, `datasets/`, `runs/` e arquivos de pesos (`.pt`, `.onnx`) contêm dados binários pesados ou resultados já processados.
- Nunca altere, mova ou apague imagens, anotações YOLO (`.txt`) ou arquivos de pesos `.pt` sem pedir confirmação explícita antes.
- Ao sugerir comandos de terminal, utilize sintaxe compatível com PowerShell.

## Protocolo de Auditoria Visual e Erros de Contagem
Sempre que o usuário anexar imagens de `resultados_tomate/` para conferir a contagem:
1. **Contagem independente:** Conte quantos tomates reais estão visíveis na foto antes de considerar os rótulos desenhados e compare com o total de *bounding boxes* geradas pelo modelo.
2. **Classificação da causa do erro:** Para cada divergência encontrada, aponte explicitamente qual fenômeno ocorreu e em qual região da imagem (ex.: canto superior esquerdo, centro, fundo):
   - **Falso Negativo (contagem a menor):** Tomates ignorados por oclusão parcial (atrás de folhas/galhos), sombra forte, frutos verdes com baixo contraste em relação à folhagem ou frutos muito pequenos perdidos na redução de resolução (`imgsz`).
   - **Falso Positivo (contagem a maior):** Folhas enroladas, sépalas, reflexos de luz ou elementos do fundo marcados incorretamente como tomate.
   - **Duplicidade / Falha de NMS:** O mesmo tomate contado duas vezes com caixas sobrepostas (limiar `iou` muito alto) ou dois tomates encostados em cacho fundidos em uma única caixa (limiar `iou` muito baixo).
   - **Efeito de Augmentation (Roboflow):** Cortes (*crop*), rotações ou alterações de brilho identificáveis pelos sufixos `.rf.<hash>` que deixaram frutos cortados nas bordas da imagem.
3. **Ação corretiva vinculada ao código:** Sempre relacione o erro visual identificado aos parâmetros do `model.predict()` (`conf`, `iou`, `imgsz`, `agnostic_nms`) ou à necessidade de reforçar exemplos específicos no dataset de treino.

## Convenções de Código e Análise
- Ao modificar scripts de inferência ou treinamento, preserve os caminhos de entrada/saída existentes e comente alterações nos hiperparâmetros (`conf`, `iou`, `imgsz`).
- Sempre responda em português do Brasil de forma direta e técnica.