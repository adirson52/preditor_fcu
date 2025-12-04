# Manual de Uso e Reprodução

Este manual descreve como utilizar os scripts contidos na pasta `scripts/` para regenerar o relatório de sensibilidade.

## Pré-requisitos

1. **Python 3.10+** instalado.
2. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

3. **Dados Brutos (Logs):**
    Os scripts esperam que a estrutura de pastas dos logs de treinamento esteja presente no diretório pai ou configurada corretamente no script.

    Estrutura esperada (exemplo):

    ```
    /preditor_fcu_v3/
    ├── log_v3_95_std/
    ├── log_v3_80_std/
    ├── ...
    ├── log_v3_60_noint/ (Run 8)
    └── scripts/
        └── consolidate_results.py
    ```

    *Nota: Os logs brutos não estão incluídos neste repositório.*

## Scripts

### 1. `consolidate_results.py`

Este é o script principal. Ele varre as pastas de logs definidas na variável `RUNS`, extrai métricas (AUC, Brier), contagens de features e matrizes de importância (Drivers), e gera o arquivo `comparativo_sensibilidade.html`.

**Como rodar:**

```bash
cd scripts
python consolidate_results.py
```

O arquivo HTML será gerado na raiz do projeto (ou onde configurado).

### 2. `advanced_evaluation.py`

Este script calcula métricas adicionais que não são geradas durante o treino padrão, como AUPRC, Recall@1% e Estabilidade de Drivers. Ele gera arquivos CSV e imagens na pasta `comparativo_avancado/`.

**Como rodar:**

```bash
cd scripts
python advanced_evaluation.py
```

### 3. `package_for_vercel.py`

Este script pega o `comparativo_sensibilidade.html` gerado e todas as imagens referenciadas, e cria uma pasta `deploy_vercel` (ou similar) com uma estrutura plana (`index.html` + `assets/`), ideal para deploy estático.

**Como rodar:**

```bash
cd scripts
python package_for_vercel.py
```

## Customização

Para adicionar novas Runs ou alterar os escopos analisados, edite as listas `RUNS` e `SCOPES` no início do arquivo `consolidate_results.py`.
