import os
import shutil
import re
from pathlib import Path

def package_report():
    base_dir = Path.cwd()
    output_dir = base_dir / "deploy_vercel"
    assets_dir = output_dir / "assets"
    
    # Clean up previous build
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True)
    assets_dir.mkdir()
    
    html_file = base_dir / "comparativo_sensibilidade.html"
    if not html_file.exists():
        print("Erro: Relatório HTML não encontrado.")
        return

    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find all images
    # Matches src='...' or src="..."
    images = re.findall(r"src=['\"](.*?)['\"]", content)
    
    new_content = content
    
    print(f"Encontradas {len(images)} imagens para copiar.")
    
    for img_rel_path in images:
        # img_rel_path is relative to base_dir in the original HTML context
        # e.g. "comparativo_avancado/calib_60_BRASIL.png"
        # or "log_v3_60_noint/relatorios_finais/..."
        
        src_path = base_dir / img_rel_path
        
        if src_path.exists():
            # Flatten filename to avoid subdirectories in assets
            # e.g. calib_60_BRASIL.png
            # But we might have name collisions if we flatten too much?
            # Let's keep unique names by replacing slashes with underscores if needed, 
            # or just use the original filename if unique.
            # Safe approach: hash or keep structure?
            # Vercel likes flat or simple structure.
            # Let's replace path separators with underscores for the new filename
            
            # Normalize path separators
            safe_name = str(img_rel_path).replace("/", "_").replace("\\", "_")
            
            dst_path = assets_dir / safe_name
            shutil.copy2(src_path, dst_path)
            
            # Update HTML content
            # We need to replace the original path with "assets/safe_name"
            # Be careful with replace() if multiple images share substrings, but here paths are usually distinct.
            new_content = new_content.replace(img_rel_path, f"assets/{safe_name}")
            
        else:
            print(f"Aviso: Imagem não encontrada: {src_path}")
            
    # Save new HTML
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print(f"Pacote criado com sucesso em: {output_dir}")
    print("Conteúdo:")
    print(f" - index.html")
    print(f" - assets/ ({len(list(assets_dir.glob('*')))} arquivos)")

if __name__ == "__main__":
    package_report()
