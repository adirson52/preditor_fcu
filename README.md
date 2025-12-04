# Preditor FCU v3 - Relatório de Sensibilidade e Ablação

Este repositório contém o código fonte e o relatório final gerado para o projeto **Preditor FCU v3**, focado na análise de sensibilidade e ablação de modelos de Machine Learning (EBM - Explainable Boosting Machine) para predição de áreas de risco urbano.

## Estrutura do Repositório

- **`index.html`**: O relatório final interativo (HTML), pronto para visualização e deploy (ex: Vercel, GitHub Pages).
- **`assets/`**: Imagens e gráficos utilizados no relatório (curvas de calibração, elasticidade, etc.).
- **`scripts/`**: Código Python utilizado para processar os logs, calcular métricas avançadas e gerar o relatório.
  - `consolidate_results.py`: Script principal que lê os logs das rodadas (Runs), consolida os resultados e gera o HTML.
  - `advanced_evaluation.py`: Script auxiliar para cálculo de métricas avançadas (AUPRC, Recall@k, Estabilidade de Drivers).
  - `package_for_vercel.py`: Utilitário para empacotar o relatório e assets em uma pasta pronta para deploy.

## Visualização

Para visualizar o relatório, basta abrir o arquivo `index.html` em qualquer navegador moderno.

[Clique aqui para ver o relatório online](https://seu-usuario.github.io/preditor-fcu-relatorio/) *(Substitua pelo link real após o deploy)*

## Como Reproduzir os Resultados

Para regenerar o relatório a partir dos logs brutos (não incluídos neste repo devido ao tamanho), siga as instruções no arquivo [MANUAL.md](MANUAL.md).

## Tecnologias

- **Python 3.10+**
- **Pandas** (Manipulação de dados)
- **Matplotlib/Seaborn** (Visualização)
- **HTML5/CSS3** (Relatório Interativo)

## Autoria

Desenvolvido como parte do projeto Preditor FCU v3.
