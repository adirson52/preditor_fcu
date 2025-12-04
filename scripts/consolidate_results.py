import json
import pandas as pd
from pathlib import Path
import re
import glob

# Configuração
BASE_DIR = Path(r"C:\Users\adirs\Downloads\preditor_fcu-20251203T221625Z-1-001\preditor_fcu_v3")
OUTPUT_FILE = BASE_DIR / "comparativo_sensibilidade.txt"

RUNS = [
    ("Run 1 (0.95, Std)", "log_v3_95_std"),
    ("Run 2 (0.80, Std)", "log_v3_80_std"),
    ("Run 3 (0.70, Std)", "log_v3_70_std"),
    ("Run 4 (0.60, Std)", "log_v3_60_std"),
    ("Run 5 (0.95, NoInt)", "log_v3_95_noint"),
    ("Run 6 (0.80, NoInt)", "log_v3_80_noint"),
    ("Run 7 (0.70, NoInt)", "log_v3_70_noint"),
    ("Run 8 (0.60, NoInt)", "log_v3_60_noint"),
]

SCOPES = [
    "Juazeiro_do_Norte",
    "Brasília",
    "Belo_Horizonte",
    "Marabá",
    "Recife",
    "Porto_Alegre",
    "BRASIL"
]

def get_metrics(log_dir, scope):
    try:
        with open(log_dir / "modelos" / f"metrics_{scope}.json", "r") as f:
            data = json.load(f)
        return data.get("auc", 0), data.get("brier_score", 0)
    except FileNotFoundError:
        try:
            log_path = log_dir / "treinamento" / f"log_treino_{scope}.txt"
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            match = re.search(r"AUC=(\d+\.\d+), Brier=(\d+\.\d+)", content)
            if match:
                return float(match.group(1)), float(match.group(2))
        except FileNotFoundError:
            pass
        return None, None

def get_feature_count(log_dir, scope):
    try:
        with open(log_dir / "features_config" / f"selected_features_{scope}.json", "r") as f:
            data = json.load(f)
        return len(data)
    except FileNotFoundError:
        try:
            log_path = log_dir / "treinamento" / f"log_treino_{scope}.txt"
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            match = re.search(r"Shapes: Treino=\(\d+, (\d+)\)", content)
            if match:
                return int(match.group(1))
        except FileNotFoundError:
            pass
        return 0

def get_full_drivers_matrix(log_dir, scope):
    txt_path = log_dir / "relatorios" / f"relatorio_analise_{scope}.txt"
    drivers = {}
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
                    if line and "──────" in line:
                        break
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
    except FileNotFoundError:
        pass
    return drivers

def get_distribution_html(log_dir):
    csv_path = log_dir / "output_final" / "relatorios_visuais" / "table_distribution_comparative.csv"
    try:
        df = pd.read_csv(csv_path)
        return df.to_html(index=False, border=0, classes='table table-striped')
    except FileNotFoundError:
        return "<p>Tabela não encontrada.</p>"

VAR_PRETTY_MAP = {
    "media_banheiros_hab": "Densidade Sanitária (Banheiros/Hab)",
    "banheiro_exclusivo": "Acesso Sanitário Exclusivo",
    "entorno_pavimentada_sim": "Infraestrutura Viária (Pavimentada)",
    "agua_outra_soma": "Abastecimento Hídrico Alternativo",
    "censo_entorno_caminhao": "Fluxo Logístico Pesado",
    "AREA_KM2": "Área da Célula (km²)",
    "n_cel_s": "Densidade Demográfica Relativa",
    "entorno_arborizacao_sim": "Índice de Arborização Urbana",
    "entorno_arborizacao_nao": "Déficit de Arborização",
    "esgoto_adequado_soma": "Saneamento Adequado",
    "esgoto_inadequado_soma": "Saneamento Inadequado",
    "fisico_declividade_media": "Declividade Média do Terreno",
    "entorno_iluminacao_nao": "Déficit de Iluminação Pública",
    "entorno_bueiro_nao": "Déficit de Drenagem Pluvial",
    "entorno_via_leve": "Acessibilidade Viária (Leve)",
    "entorno_calcada_nao": "Déficit de Calçadas",
    "banheiro_multiplo": "Instalações Sanitárias Compartilhadas",
    "censo_lixo_cacamba": "Coleta de Resíduos (Caçamba)",
    "rate_banheiro_exclusivo": "Taxa de Exclusividade Sanitária",
    "entorno_obstaculo_sim": "Obstrução de Vias",
    "entorno_obstaculo_nao": "Ausência de Obstrução Viária",
    "V06004": "Renda Média Domiciliar",
    "cad_renda_media_pc": "Renda Média Per Capita (CadÚnico)",
    "cad_qtde_pessoas": "Total de Pessoas (CadÚnico)",
    "prob_fcu": "Probabilidade FCU (Local)",
    "prob_fcu_BRASIL": "Probabilidade FCU (Global)",
    "VB002_rate": "Taxa de Banheiro Exclusivo",
    "VB003_rate": "Taxa de Banheiro Compartilhado",
    "V00398_rate": "Taxa de Lixo em Caçamba",
    "VL001_rate": "Taxa de Coleta de Lixo Regular",
    "VL002_rate": "Taxa de Lixo Irregular",
    "VE001_rate": "Taxa de Esgoto Adequado",
    "VE002_rate": "Taxa de Esgoto Inadequado",
    "V06001_rate": "Taxa de Responsável por Domicílio",
    "V00314_rate": "Taxa de Esgoto em Rio/Vala",
    "VA002_rate": "Taxa de Abastecimento de Água Alternativo",
    "V00309_rate": "Taxa de Esgoto na Rede Geral",
    "V0001": "Total de Domicílios Ocupados",
    "V0005": "Total de Moradores",
    "V0007": "Total de Domicílios Particulares",
    "cad_ilum_eletrica_s_medidor": "Iluminação Elétrica sem Medidor (CadÚnico)",
    "cad_calcamento_inexistente": "Calçamento Inexistente (CadÚnico)",
    "lixo_irregular_soma": "Lixo Irregular (Soma)",
    "entorno_via_pesada": "Vias Pesadas (Caminhões/Ônibus)",
    "lixo_regular_bin": "Coleta de Lixo Regular (Binário)",
    "censo_esgoto_rio": "Esgoto em Rio/Vala (Censo)",
    "censo_resp_domicilio": "Responsável pelo Domicílio (Censo)",
    "fisico_indice_forma_medio": "Índice de Forma Médio (Físico)",
    "agua_rede_bin": "Água da Rede (Binário)",
    "cad_ilum_precario_soma": "Iluminação Precária (CadÚnico)",
    "fisico_pct_vias_50m": "Densidade de Vias (50m)",
    "cad_pessoas_por_dom": "Média de Pessoas por Domicílio (CadÚnico)",
    "cad_renda_media_pc_fam": "Renda Média Familiar Per Capita (CadÚnico)",
    "cad_qtde_pessoas_fam": "Tamanho da Família (CadÚnico)",
    "V00238_rate": "Taxa de Banheiro (V00238)",
    "entorno_bueiro_sim": "Presença de Bueiro",
    "censo_esgoto_rede": "Esgoto na Rede (Censo)",
    "entorno_calcada_sim": "Presença de Calçada",
    "entorno_pavimentada_nao": "Déficit de Pavimentação"
}

def get_pretty_name(feature_name):
    if "&" in feature_name:
        parts = feature_name.split("&")
        pretty_parts = [VAR_PRETTY_MAP.get(p.strip(), p.strip()) for p in parts]
        return " & ".join(pretty_parts)
    return VAR_PRETTY_MAP.get(feature_name, feature_name)

def calculate_ablation_impact():
    comparisons = [
        ("0.95", "Run 1 (0.95, Std)", "Run 5 (0.95, NoInt)"),
        ("0.80", "Run 2 (0.80, Std)", "Run 6 (0.80, NoInt)"),
        ("0.70", "Run 3 (0.70, Std)", "Run 7 (0.70, NoInt)"),
        ("0.60", "Run 4 (0.60, Std)", "Run 8 (0.60, NoInt)"),
    ]
    
    html = []
    html.append("<p>Comparativo direto entre modelos com interações (Standard) e sem interações (NoInt) para avaliar o ganho de performance.</p>")
    html.append("<table><caption>Tabela 5. Impacto das Interações na Performance (AUC)</caption>")
    html.append("<tr><th>Threshold</th><th>AUC (Com Interações)</th><th>AUC (Sem Interações)</th><th>Delta</th><th>Impacto</th></tr>")
    
    for thresh, run_std, run_noint in comparisons:
        folder_std = next(f for r, f in RUNS if r == run_std)
        folder_noint = next(f for r, f in RUNS if r == run_noint)
        
        auc_std, _ = get_metrics(BASE_DIR / folder_std, "BRASIL")
        auc_noint, _ = get_metrics(BASE_DIR / folder_noint, "BRASIL")
        
        if auc_std and auc_noint:
            delta = auc_std - auc_noint
            impact = "Positivo" if delta > 0 else "Negativo/Neutro"
            color = "green" if delta > 0 else "red"
            html.append(f"<tr><td>{thresh}</td><td>{auc_std:.4f}</td><td>{auc_noint:.4f}</td><td style='color:{color}'><b>{delta:+.4f}</b></td><td>{impact}</td></tr>")
            
    html.append("</table>")
    html.append("<div class='source'>Fonte: Elaboração própria a partir dos dados do Preditor FCU v3.</div>")
    return "\n".join(html)

def generate_html_report(output_file):
    html = []
    html.append("""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Relatório Final: Preditor FCU v3</title>
        <style>
            body { font-family: 'Arial', sans-serif; line-height: 1.5; color: #000; max-width: 210mm; margin: 0 auto; padding: 20px; background-color: #fff; }
            h1 { text-align: center; font-size: 16pt; font-weight: bold; margin-bottom: 40px; margin-top: 0; text-transform: uppercase; }
            h2 { font-size: 14pt; font-weight: bold; margin-top: 30px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
            h3 { font-size: 12pt; font-weight: bold; margin-top: 20px; }
            p { text-align: justify; font-size: 11pt; margin-bottom: 10px; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 9pt; }
            caption { caption-side: top; text-align: center; font-weight: bold; margin-bottom: 5px; font-size: 10pt; }
            th { border-top: 1px solid #000; border-bottom: 1px solid #000; padding: 4px; text-align: center; font-weight: bold; line-height: 1.1; }
            td { padding: 2px 4px; text-align: center; border: none; line-height: 1.2; }
            tr:last-child td { border-bottom: 1px solid #000; }
            .heatmap-cell { color: #000; font-weight: bold; border: 1px solid #fff; }
            .source { font-size: 8pt; text-align: left; margin-bottom: 20px; }
            .footer { text-align: center; font-size: 9pt; margin-top: 50px; color: #666; }
            .calibration-plot { width: 30%; display: inline-block; margin: 1%; }
            .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; margin-bottom: 10px; }
            .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 10px 16px; transition: 0.3s; font-size: 10pt; font-weight: bold; }
            .tab button:hover { background-color: #ddd; }
            .tab button.active { background-color: #ccc; }
            .tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; animation: fadeEffect 1s; }
            @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
            
            /* Tabs Elasticidade */
            .tablinks-elast { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 10pt; font-weight: bold; }
            .tablinks-elast:hover { background-color: #ddd; }
            .tablinks-elast.active { background-color: #ccc; }
            .content-elast { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; animation: fadeEffect 1s; }
        </style>
        <script>
            function openTab(evt, tabId, contentClass, btnClass) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName(contentClass);
                for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }
                tablinks = document.getElementsByClassName(btnClass);
                for (i = 0; i < tablinks.length; i++) { tablinks[i].className = tablinks[i].className.replace(" active", ""); }
                document.getElementById(tabId).style.display = "block";
                evt.currentTarget.className += " active";
            }
            window.onload = function() {
                if(document.getElementsByClassName("tablinks-dist").length > 0) document.getElementsByClassName("tablinks-dist")[0].click();
                if(document.getElementsByClassName("tablinks-drivers").length > 0) document.getElementsByClassName("tablinks-drivers")[0].click();
                if(document.getElementsByClassName("tablinks-calib").length > 0) document.getElementsByClassName("tablinks-calib")[0].click();
                if(document.getElementsByClassName("tablinks-elast").length > 0) document.getElementsByClassName("tablinks-elast")[0].click();
            };
        </script>
    </head>
    <body>
    <h1>Relatório Final de Análise de Sensibilidade<br>Preditor FCU v3</h1>
    <p style="text-align: center;">Data de Geração: """ + pd.Timestamp.now().strftime("%d/%m/%Y %H:%M") + """</p>
    <p>Este relatório consolida os resultados de oito rodadas de treinamento (Runs) do modelo Preditor FCU v3.</p>
    """)

    # 1. Resumo Métricas
    html.append("<h2>1. Resumo de Métricas (Escopo: BRASIL)</h2>")
    html.append("<table><caption>Tabela 1. Desempenho Global dos Modelos</caption>")
    html.append("<tr><th>Run</th><th>Variáveis Selecionadas</th><th>AUC (ROC)</th><th>Brier Score</th></tr>")
    for run_name, folder_name in RUNS:
        log_dir = BASE_DIR / folder_name
        auc, brier = get_metrics(log_dir, "BRASIL")
        count = get_feature_count(log_dir, "BRASIL")
        if auc is not None:
            html.append(f"<tr><td style='text-align:left'>{run_name}</td><td>{count}</td><td>{auc:.4f}</td><td>{brier:.4f}</td></tr>")
        else:
            html.append(f"<tr><td style='text-align:left'>{run_name}</td><td>-</td><td>-</td><td>-</td></tr>")
    html.append("</table>")
    html.append("<div class='source'>Fonte: Elaboração própria.</div>")

    # 2. Tabelas de Distribuição
    html.append("<h2>2. Distribuição de Probabilidades (Local vs Global)</h2>")
    html.append("<div class='tab'>")
    for i, (run_name, _) in enumerate(RUNS, 1):
        safe_id = f"DistRun{i}"
        label = run_name.split('(')[0].strip()
        html.append(f"<button class='tablinks-dist' onclick=\"openTab(event, '{safe_id}', 'content-dist', 'tablinks-dist')\">{label}</button>")
    html.append("</div>")
    
    for i, (run_name, folder_name) in enumerate(RUNS, 1):
        safe_id = f"DistRun{i}"
        html.append(f"<div id='{safe_id}' class='content-dist tabcontent'>")
        html.append(f"<h3>{run_name}</h3>")
        log_dir = BASE_DIR / folder_name
        dist_html = get_distribution_html(log_dir)
        if "Tabela não encontrada" in dist_html:
             html.append("<p>Dados de distribuição não disponíveis.</p>")
        else:
            dist_html = dist_html.replace('<table border="1" class="dataframe">', f'<table><caption>Tabela 2.{i}. Distribuição - {run_name}</caption>')
            dist_html = dist_html.replace('<thead>', '<thead>').replace('<th>', '<th style="text-align:center">')
            html.append(dist_html)
            html.append("<div class='source'>Fonte: Elaboração própria.</div>")
        html.append("</div>")

    # 3. Matriz de Drivers
    html.append("<h2>3. Matriz de Drivers de Risco (Completa)</h2>")
    html.append("<div class='tab'>")
    for i, (run_name, _) in enumerate(RUNS, 1):
        safe_id = f"DriverRun{i}"
        label = run_name.split('(')[0].strip()
        html.append(f"<button class='tablinks-drivers' onclick=\"openTab(event, '{safe_id}', 'content-drivers', 'tablinks-drivers')\">{label}</button>")
    html.append("</div>")
    
    for i, (run_name, folder_name) in enumerate(RUNS, 1):
        safe_id = f"DriverRun{i}"
        html.append(f"<div id='{safe_id}' class='content-drivers tabcontent'>")
        html.append(f"<h3>{run_name}</h3>")
        
        all_drivers_data = {}
        all_feature_names = set()
        for scope in SCOPES:
            log_dir = BASE_DIR / folder_name
            drivers = get_full_drivers_matrix(log_dir, scope)
            pretty_drivers = {get_pretty_name(k): v for k, v in drivers.items()}
            all_drivers_data[scope] = pretty_drivers
            all_feature_names.update(pretty_drivers.keys())
            
        feature_max_scores = {}
        for feat in all_feature_names:
            scores = [abs(all_drivers_data.get(s, {}).get(feat, 0.0)) for s in SCOPES]
            feature_max_scores[feat] = max(scores)
            
        sorted_features = sorted(list(all_feature_names), key=lambda x: feature_max_scores[x], reverse=True)
        
        if not sorted_features:
            html.append("<p>Dados não disponíveis para este Run.</p>")
        else:
            html.append(f"<table><caption>Tabela 3.{i}. Matriz de Importância - {run_name}</caption>")
            html.append("<tr><th style='text-align:left'>Variável</th>" + "".join([f"<th>{scope.replace('_', ' ')}</th>" for scope in SCOPES]) + "</tr>")
            for feature in sorted_features:
                html.append(f"<tr><td style='text-align:left'>{feature}</td>")
                for scope in SCOPES:
                    val = all_drivers_data.get(scope, {}).get(feature, 0.0)
                    max_val = 1.5
                    norm = min(abs(val) / max_val, 1.0)
                    g = int(255 * (1 - norm))
                    b = int(200 * (1 - norm))
                    bg_color = f"rgb(255, {g}, {b})"
                    text_color = "#fff" if norm > 0.6 else "#000"
                    html.append(f"<td class='heatmap-cell' style='background-color: {bg_color}; color: {text_color}'>{val:.2f}</td>")
                html.append("</tr>")
            html.append("</table>")
            html.append("<div class='source'>Fonte: Elaboração própria.</div>")
        html.append("</div>")

    # 3.1 Curvas de Elasticidade (Run 8)
    html.append("<h2>3.1 Curvas de Elasticidade (Run 8)</h2>")
    html.append("<p>As curvas de elasticidade (ou funções de forma) ilustram como a probabilidade de risco (Score EBM) varia em função dos valores de cada variável preditora, mantendo as demais constantes. Elas permitem visualizar a natureza da relação (linear, não-linear, limiar) e a direção do efeito.</p>")
    
    run8_folder = next((f for r, f in RUNS if "Run 8" in r), None)
    if run8_folder:
        elasticity_dir = BASE_DIR / run8_folder / "relatorios_finais"
        if elasticity_dir.exists():
            elast_by_scope = {}
            for img_path in elasticity_dir.glob("elasticidade_*.png"):
                fname = img_path.name
                found_scope = None
                for s in SCOPES:
                    safe_s = s.replace(" ", "_")
                    if f"elasticidade_{safe_s}_" in fname:
                        found_scope = s
                        break
                if found_scope:
                    if found_scope not in elast_by_scope: elast_by_scope[found_scope] = []
                    feature_raw = fname.replace(f"elasticidade_{safe_s}_", "").replace(".png", "")
                    feature_pretty = get_pretty_name(feature_raw)
                    rel_path = f"{run8_folder}/relatorios_finais/{fname}"
                    elast_by_scope[found_scope].append((feature_pretty, rel_path))
            
            sorted_scopes = sorted(elast_by_scope.keys())
            if "BRASIL" in sorted_scopes:
                sorted_scopes.remove("BRASIL")
                sorted_scopes.insert(0, "BRASIL")
            
            html.append("<div class='tab'>")
            for i, scope in enumerate(sorted_scopes, 1):
                safe_id = f"ElastScope{i}"
                html.append(f"<button class='tablinks-elast' onclick=\"openTab(event, '{safe_id}', 'content-elast', 'tablinks-elast')\">{scope}</button>")
            html.append("</div>")
            
            for i, scope in enumerate(sorted_scopes, 1):
                safe_id = f"ElastScope{i}"
                imgs = elast_by_scope[scope]
                html.append(f"<div id='{safe_id}' class='content-elast tabcontent'>")
                html.append(f"<h4>{scope}</h4>")
                html.append("<div style='display:flex; flex-wrap:wrap; justify-content:center;'>")
                for feat, src in imgs:
                    html.append(f"<div class='calibration-plot' style='width:45%; margin:5px;'><img src='{src}' style='width:100%'><br><small>{feat}</small></div>")
                html.append("</div>")
                html.append("</div>")
            
            html.append("<p><strong>Análise (Run 8):</strong> Na Run 8 (0.60, NoInt), observamos que as curvas de elasticidade confirmam a consistência física e socioeconômica do modelo. Variáveis de infraestrutura (como saneamento e pavimentação) exibem forte decaimento do risco conforme a qualidade aumenta, enquanto indicadores de precariedade (déficit de arborização, densidade sanitária) mostram crescimento monotônico do risco.</p>")
            html.append("<div class='source'>Fonte: Elaboração própria.</div>")
        else:
            html.append("<p>Imagens de elasticidade não encontradas.</p>")

    # 4. Análises Avançadas
    html.append("<h2>4. Análises Avançadas e Robustez</h2>")
    adv_metrics_path = BASE_DIR / "comparativo_avancado" / "advanced_metrics.csv"
    if adv_metrics_path.exists():
        df_adv = pd.read_csv(adv_metrics_path)
        html.append("<h3>4.1 Métricas de Eventos Raros (AUPRC e Recall@k)</h3>")
        cols = ['Run', 'AUPRC (Global)', 'Recall@1% (Global)', 'Precision@1% (Global)', 'Recall@5% (Global)', 'AUPRC (Média Local)']
        cols = [c for c in cols if c in df_adv.columns]
        html.append("<table><caption>Tabela 4.1. Performance em Eventos Raros</caption>")
        html.append("<tr>" + "".join([f"<th>{c}</th>" for c in cols]) + "</tr>")
        for _, row in df_adv.iterrows():
            html.append("<tr>")
            for c in cols:
                val = row[c]
                if isinstance(val, (int, float)): html.append(f"<td>{val:.4f}</td>")
                else: html.append(f"<td style='text-align:left'>{val}</td>")
            html.append("</tr>")
        html.append("</table>")
        html.append("<div class='source'>Fonte: Elaboração própria.</div>")

        html.append("<h3>4.2 Curvas de Calibração</h3>")
        all_calib_imgs = sorted(glob.glob(str(BASE_DIR / "comparativo_avancado" / "calib_*.png")))
        calib_by_run = {}
        for img_path in all_calib_imgs:
            rel_path = Path(img_path).name
            run_id = ""
            scope_name = ""
            if "95" in rel_path: run_id = "Run 1 (0.95, Std)"; scope_name = rel_path.split("95_")[1]
            elif "60" in rel_path and "std" in rel_path: run_id = "Run 4 (0.60, Std)"; scope_name = rel_path.split("60_")[1]
            elif "80" in rel_path: run_id = "Run 6 (0.80, NoInt)"; scope_name = rel_path.split("80_")[1]
            elif "70" in rel_path: run_id = "Run 7 (0.70, NoInt)"; scope_name = rel_path.split("70_")[1]
            elif "60" in rel_path and "noint" in rel_path: run_id = "Run 8 (0.60, NoInt)"; scope_name = rel_path.split("60_")[1]
            elif "60" in rel_path: run_id = "Run 8 (0.60, NoInt)"; scope_name = rel_path.split("60_")[1] 
            else: continue 
            scope_name = scope_name.replace(".png", "").replace("_", " ")
            if run_id not in calib_by_run: calib_by_run[run_id] = []
            calib_by_run[run_id].append((scope_name, f"comparativo_avancado/{rel_path}"))

        html.append("<div class='tab'>")
        for i, run_id in enumerate(calib_by_run.keys(), 1):
            safe_id = f"CalibRun{i}"
            label = run_id.split('(')[0].strip()
            html.append(f"<button class='tablinks-calib' onclick=\"openTab(event, '{safe_id}', 'content-calib', 'tablinks-calib')\">{label}</button>")
        html.append("</div>")
        for i, (run_id, imgs) in enumerate(calib_by_run.items(), 1):
            safe_id = f"CalibRun{i}"
            html.append(f"<div id='{safe_id}' class='content-calib tabcontent'>")
            html.append(f"<h4>{run_id}</h4>")
            html.append("<div style='text-align:center'>")
            for scope, img_src in imgs:
                html.append(f"<div class='calibration-plot'><img src='{img_src}' style='width:100%'><br><small>{scope}</small></div>")
            html.append("</div>")
            html.append("</div>")
        html.append("<div class='source'>Fonte: Elaboração própria.</div>")

    # 4.3 Estabilidade Drivers
    driver_stab_path = BASE_DIR / "comparativo_avancado" / "driver_stability.csv"
    if driver_stab_path.exists():
        df_stab = pd.read_csv(driver_stab_path)
        html.append("<h3>4.3 Estabilidade dos Drivers (Consistência Global)</h3>")
        html.append("<p>A partir das matrizes de importância EBM por run e por escopo, foi calculada a frequência com que cada variável apresentou Score > 0,2. Esse critério de corte permite identificar os drivers realmente robustos, que emergem de forma recorrente sob diferentes thresholds de correlação e presença/ausência de interações.</p>")
        html.append("<table><caption>Tabela 4.3. Drivers Mais Robustos</caption>")
        html.append("<tr><th>Rank</th><th>Variável</th><th>Frequência</th><th>Interpretação</th></tr>")
        interpretations = {
            "fisico_pct_app_30m": "Restrição ambiental (APP)",
            "fisico_declividade_media": "Topografia acidentada",
            "entorno_arborizacao_nao": "Déficit de arborização (indicador ambiental)",
            "banheiro_exclusivo": "Saneamento/Infraestrutura consolidada",
            "entorno_pavimentada_nao": "Precariedade viária",
            "entorno_via_leve": "Acessibilidade local",
            "censo_moradores_total": "Densidade populacional",
            "media_banheiros_hab": "Densidade sanitária"
        }
        for i, row in df_stab.head(10).iterrows():
            driver = row['Driver']
            pretty_driver = get_pretty_name(driver)
            interp = interpretations.get(driver, "-")
            if interp == "-":
                for k, v in interpretations.items():
                    if k in driver: interp = v; break
            html.append(f"<tr><td>{i+1}º</td><td style='text-align:left'>{pretty_driver}</td><td>{row['Frequency']}</td><td style='text-align:left'>{interp}</td></tr>")
        html.append("</table>")
        html.append("<div class='source'>Fonte: Elaboração própria.</div>")

    # 5. Ablação
    html.append("<h2>5. Análise de Ablação (Impacto das Interações)</h2>")
    html.append(calculate_ablation_impact())

    # 6. Conclusão
    html.append("<h2>6. Conclusão e Recomendação</h2>")
    html.append("""
    <p>Em termos de desempenho global, o modelo sem interações com limiar de correlação 0,80 (Run 6) apresenta a maior AUC. No entanto, o modelo com limiar 0,60 (Run 8) exibe Brier Score ligeiramente menor e comportamento de calibração mais estável. Na prática, ambos os modelos são equivalentes em desempenho, e a escolha entre eles deve considerar o equilíbrio desejado entre parcimônia e calibração.</p>
    <p><strong>Recomendação:</strong> Neste relatório, adotamos a <strong>Run 8 (0.60, NoInt)</strong> como configuração padrão do Preditor FCU v3, por oferecer o melhor compromisso entre desempenho, calibração e estabilidade dos drivers de risco, além de ser um modelo mais parcimonioso (menos variáveis).</p>
    """)

    # Apêndice
    html.append("<h2>Apêndice A. Glossário de Variáveis</h2>")
    html.append("<table><caption>Tabela A.1. Dicionário de Dados</caption>")
    html.append("<tr><th style='text-align:left'>Nome Técnico</th><th style='text-align:left'>Nome Descritivo</th></tr>")
    for k, v in sorted(VAR_PRETTY_MAP.items()):
        html.append(f"<tr><td style='text-align:left'>{k}</td><td style='text-align:left'>{v}</td></tr>")
    html.append("</table>")

    html.append("<div class='footer'>Relatório gerado automaticamente pelo sistema Preditor FCU v3.</div>")
    html.append("</body></html>")
    
    with open(BASE_DIR / "comparativo_sensibilidade.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"HTML salvo em: {BASE_DIR / 'comparativo_sensibilidade.html'}")

def main():
    generate_html_report(BASE_DIR / "comparativo_sensibilidade.html")

if __name__ == "__main__":
    main()
