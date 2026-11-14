# Projeto Tomate — como rodar a detecção

O ZIP corrigido contém o script de detecção v2, os pesos já treinados, as fotos e anotações do projeto e os resultados anteriores. Não é necessário treinar novamente.

## Primeiro teste no Windows

1. Extraia o ZIP. No VS Code, abra a pasta `Projeto_Tomate` de dentro do ZIP, onde estão `analisar_colheita_v2.py` e `data.yaml`.
2. No terminal do VS Code (PowerShell), se as bibliotecas ainda não estiverem instaladas, rode:

   ```powershell
   py -m pip install -r requirements.txt
   ```

3. Teste a detecção nas 49 fotos que já vieram no projeto:

   ```powershell
   py analisar_colheita_v2.py
   ```

   As fotos com caixas e o `relatorio_tomates.csv` ficam em `resultados_tomate_v2`. O comando antigo `py analisar_colheita.py` também abre o analisador atualizado. Os resultados históricos em `resultados_tomate` não são sobrescritos por padrão.

## Testar as suas fotos

Coloque as fotos originais, sem caixas desenhadas, na pasta `fotos_teste`. Depois rode:

```powershell
py analisar_colheita_v2.py --entrada ".\fotos_teste" --saida ".\teste_externo"
```

As imagens e o CSV ficam em `teste_externo`. Para testar outra foto, adicione-a a `fotos_teste` e repita o comando. O CSV é atualizado a cada execução. Feche o CSV no Excel antes de executar novamente se ele estiver bloqueado.

## Comparar com a detecção original

```powershell
py analisar_colheita_v2.py --entrada ".\fotos_teste" --saida ".\teste_original" --perfil original
```

O perfil padrão aplica uma limpeza final de caixas sobrepostas para tomates maduros. O perfil original não aplica esse filtro e usa somente `best.pt`. Nenhum dos dois garante contagem exata em fotos novas. Para conferir, compare as caixas com cada fruto visível na foto original.
