import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import json
import logging

# Importando funções do core para replicar o split exato
from core.data_preparation import load_and_clean_data, criar_grid_espacial

# Configuração
BASE_DIR = Path(r"C:\Users\adirs\Downloads\preditor_fcu-20251203T221625Z-1-001\preditor_fcu_v3").resolve()
INPUT_FILE = BASE_DIR / "dados" / "pnui_x_ibge.geoparquet"
OUTPUT_DIR = BASE_DIR / "comparativo_avancado"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Runs para analisar
RUNS_TO_ANALYZE = [
    ("Run 1 (0.95, Std)", "log_v3_95_std"),
    ("Run 6 (0.80, NoInt)", "log_v3_80_noint"),
    ("Run 7 (0.70, NoInt)", "log_v3_70_noint"),
    ("Run 8 (0.60, NoInt)", "log_v3_60_noint")
]

SCOPES = ["BRASIL", "Belo Horizonte", "Brasília", "Juazeiro do Norte", "Marabá", "Porto Alegre", "Recife"]

def get_test_ids(scope, input_file):
    """
    Replica a lógica de split do 03_treinamento_modelo.py para recuperar os IDs do conjunto de teste.
    """
    # 1. Carrega e Limpa (Exatamente como no treino)
    df = load_and_clean_data(input_file, scope=scope)
    
    # 2. Cria Grid Espacial (Exatamente como no treino: size=20)
    groups = criar_grid_espacial(df, grid_size=20)
    
    # 3. Split (Exatamente como no treino: test_size=0.25, random_state=42)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["FCU"], groups=groups))
    
    # Retorna os IDs do teste
    return df.iloc[test_idx]["ID"].astype(str).values

def calculate_auprc(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

def calculate_recall_at_k(y_true, y_prob, k_percent):
    k = int(len(y_true) * (k_percent / 100))
    if k == 0: return 0.0
    
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False)
    
    top_k = df.head(k)
    recall = top_k['true'].sum() / df['true'].sum()
    return recall

def calculate_precision_at_k(y_true, y_prob, k_percent):
    k = int(len(y_true) * (k_percent / 100))
    if k == 0: return 0.0
    
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False)
    
    top_k = df.head(k)
    precision = top_k['true'].sum() / k
    return precision

def plot_calibration_curve(y_true, y_prob, run_name, scope, output_path):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=f'{run_name}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfeitamente Calibrado')
    plt.xlabel('Probabilidade Predita Média')
    plt.ylabel('Fração de Positivos (FCU)')
    plt.title(f'Curva de Calibração - {scope}\n{run_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def get_drivers_from_txt(log_dir, scope):
    txt_path = log_dir / "relatorios" / f"relatorio_analise_{scope}.txt"
    drivers = {}
    if not txt_path.exists(): return drivers
    
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        start_idx = -1
        for i, line in enumerate(lines):
            if "Fatores de Influência (EBM)" in line:
                start_idx = i + 3
                break
        
        if start_idx != -1:
            for i in range(start_idx, len(lines)):
                line = lines[i].strip()
                if not line or "──────" in line:
                    if line and "──────" in line: break
                    continue
                if "feature_pretty" in line: continue

                parts = line.rsplit(maxsplit=1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        score = float(parts[1])
                        drivers[name] = score
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Erro ao ler {txt_path}: {e}")
    return drivers

def analyze_driver_stability(runs):
    """
    Conta a frequência de drivers importantes (Score > 0.2) em todos os runs e escopos.
    """
    print("Analisando estabilidade dos drivers...")
    driver_counts = {}
    
    for run_name, folder_name in runs:
        log_dir = BASE_DIR / folder_name
        for scope in SCOPES:
            drivers = get_drivers_from_txt(log_dir, scope)
                    
            # Filtra drivers relevantes
            for feature, score in drivers.items():
                if abs(score) > 0.2: # Limiar de relevância
                    driver_counts[feature] = driver_counts.get(feature, 0) + 1
    
    # Cria DataFrame
    df_stab = pd.DataFrame(list(driver_counts.items()), columns=['Driver', 'Frequency'])
    df_stab = df_stab.sort_values('Frequency', ascending=False)
    
    # Salva
    df_stab.to_csv(OUTPUT_DIR / "driver_stability.csv", index=False)
    return df_stab

def main():
    results = []
    
    # 1. Recupera IDs de Teste para BRASIL (Global)
    print("Recuperando IDs de Teste (BRASIL)...")
    test_ids_brasil = get_test_ids("BRASIL", INPUT_FILE)
    print(f"Total Teste BRASIL: {len(test_ids_brasil)}")
    
    # 2. Recupera IDs de Teste para cada Polo (Local)
    test_ids_local = {}
    for scope in SCOPES:
        if scope == "BRASIL": continue
        print(f"Recuperando IDs de Teste ({scope})...")
        test_ids_local[scope] = get_test_ids(scope, INPUT_FILE)

    for run_name, folder_name in RUNS_TO_ANALYZE:
        print(f"Processando {run_name}...")
        
        log_dir = BASE_DIR / folder_name
        master_path = log_dir / "output_final" / "output_final_master.geoparquet"
        
        if not master_path.exists():
            print(f"Arquivo não encontrado: {master_path}")
            continue
            
        df_master = gpd.read_parquet(master_path)
        df_master["ID"] = df_master["ID"].astype(str).str.strip()
        
        # --- ANÁLISE GLOBAL (BRASIL) ---
        # Filtra apenas o conjunto de TESTE do Brasil
        df_brasil_test = df_master[df_master["ID"].isin(test_ids_brasil)].copy()
        
        if df_brasil_test.empty:
            print("ERRO: DataFrame de teste vazio para BRASIL.")
            continue
            
        # Limpeza de NaNs
        df_brasil_test = df_brasil_test.dropna(subset=["FCU", "prob_fcu_BRASIL"])
        
        y_true_br = df_brasil_test["FCU"]
        y_prob_br = df_brasil_test["prob_fcu_BRASIL"]
        
        auprc_global = calculate_auprc(y_true_br, y_prob_br)
        recall_1pct = calculate_recall_at_k(y_true_br, y_prob_br, 1.0)
        recall_5pct = calculate_recall_at_k(y_true_br, y_prob_br, 5.0)
        prec_1pct = calculate_precision_at_k(y_true_br, y_prob_br, 1.0)
        
        # Plot Calibração Global
        plot_calibration_curve(y_true_br, y_prob_br, run_name, "BRASIL", 
                             OUTPUT_DIR / f"calib_{folder_name.split('_')[-2]}_BRASIL.png")

        # --- ANÁLISE LOCAL (Média dos Polos) ---
        auprc_locals = []
        
        for scope in SCOPES:
            if scope == "BRASIL": continue
            
            # Filtra apenas o conjunto de TESTE do Polo
            ids_polo = test_ids_local[scope]
            df_polo_test = df_master[df_master["ID"].isin(ids_polo)].copy()
            
            safe_scope = scope.replace(' ', '_')
            col_prob = f"prob_fcu_{safe_scope}"
            
            if col_prob not in df_polo_test.columns:
                continue
                
            df_polo_test = df_polo_test.dropna(subset=["FCU", col_prob])
            
            if len(df_polo_test) > 0:
                val = calculate_auprc(df_polo_test["FCU"], df_polo_test[col_prob])
                auprc_locals.append(val)
                
                # Plot Calibração Local (Ex: Recife, Porto Alegre)
                if scope in ["Recife", "Porto Alegre"]:
                     plot_calibration_curve(df_polo_test["FCU"], df_polo_test[col_prob], run_name, scope, 
                                          OUTPUT_DIR / f"calib_{folder_name.split('_')[-2]}_{safe_scope}.png")

        avg_auprc_local = np.mean(auprc_locals) if auprc_locals else 0.0
        
        results.append({
            "Run": run_name,
            "AUPRC (Global)": auprc_global,
            "Recall@1% (Global)": recall_1pct,
            "Recall@5% (Global)": recall_5pct,
            "Precision@1% (Global)": prec_1pct,
            "AUPRC (Média Local)": avg_auprc_local
        })

    # Salva Métricas
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "advanced_metrics.csv", index=False)
    print("Métricas salvas em advanced_metrics.csv")
    
    # Estabilidade Drivers
    analyze_driver_stability(RUNS_TO_ANALYZE)

if __name__ == "__main__":
    main()
