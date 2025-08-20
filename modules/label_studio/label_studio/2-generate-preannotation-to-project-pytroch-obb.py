import requests
import time
import json
from tqdm import tqdm
import os
import io
import numpy as np
import tempfile
from PIL import Image
from ultralytics import YOLO

# Modo de debug - definir como True para processar apenas a primeira imagem
DEBUG_MODE = False

# Configuração da API
project_id = 2
api_base = f"https://internal-label.cogtive.com/api/projects/{project_id}/tasks/"
headers = {
    "Authorization": "Token a27c0b74353fe5f041d9d54d8323d8dfb3457c64"
}

# Conjunto para verificar IDs de tarefas já processadas
processed_ids = set()

# Definir o tamanho da página e quantidade de tarefas a baixar com base no modo
if DEBUG_MODE:
    print("\n=== MODO DEBUG ATIVADO - Baixando apenas uma tarefa ===\n")
    page_size = 1
    max_tasks = 1
else:
    page_size = 100
    max_tasks = None  # Sem limite

page = 1
all_tasks = []
page_count = 0

# Primeira requisição para descobrir o número total de tarefas
print(f"Obtendo informações sobre o projeto {project_id}...")

# Configurar parâmetros conforme a documentação da API
params = {
    'page': 1,
    'page_size': 1,  # Apenas para obter o count total
}

response = requests.get(api_base, headers=headers, params=params)
response.raise_for_status()
data = response.json()

# Verificar resposta e obter contagem total se disponível
if isinstance(data, dict):
    total_count = data.get("count")
    if total_count:
        print(f"Total oficial de tarefas: {total_count}")
    else:
        total_count = 13052  # Valor informado pelo usuário
        print(f"Usando contagem estimada: {total_count}")
else:
    total_count = 13052  # Valor informado pelo usuário
    print(f"Usando contagem estimada: {total_count}")

# Configurar barra de progresso
if DEBUG_MODE:
    pbar = tqdm(total=1, desc="Baixando tarefas")
else:
    pbar = tqdm(total=total_count, desc="Baixando tarefas")

# Começar a paginação para obter todas as tarefas
while True:
    page_count += 1
    
    # Configurar parâmetros para esta página
    params = {
        'page': page,
        'page_size': page_size,
    }
    
    try:
        print(f"Requisitando página {page} (tamanho: {page_size})")
        response = requests.get(api_base, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Extrair tarefas da resposta
        if isinstance(data, list):
            tasks = data
            has_next = bool(tasks)  # Se tem tarefas, assumimos que pode ter próxima página
        else:
            tasks = data.get("results", [])
            # Verificar se tem próxima página
            has_next = data.get("next") is not None
        
        # Se não há tarefas, terminar
        if not tasks:
            print("Nenhuma tarefa retornada nesta página. Finalizando.")
            break
            
        # Processar tarefas e verificar duplicatas
        new_tasks_count = 0
        for task in tasks:
            task_id = task.get('id')
            if task_id not in processed_ids:
                processed_ids.add(task_id)
                all_tasks.append(task)
                new_tasks_count += 1
                
        print(f"Página {page}: {new_tasks_count} novas tarefas adicionadas")
        
        # Atualizar barra de progresso
        pbar.update(new_tasks_count)
        
        # Verificar se já baixamos o número máximo de tarefas (modo debug)
        if DEBUG_MODE or (max_tasks and len(all_tasks) >= max_tasks):
            print("Número máximo de tarefas atingido. Finalizando.")
            break
        
        # Verificar se devemos continuar para a próxima página
        if not has_next or new_tasks_count == 0:
            print("Não há mais páginas para buscar. Finalizando.")
            break
            
        # Avançar para a próxima página
        page += 1
        
        # Pequena pausa para não sobrecarregar a API
        time.sleep(0.1)
        
    except requests.exceptions.RequestException as e:
        print(f"\nErro ao fazer requisição: {e}")
        # Se for um erro temporário, esperar e tentar novamente
        if response.status_code in (429, 500, 502, 503, 504):
            wait_time = 5
            print(f"Erro temporário. Tentando novamente em {wait_time} segundos...")
            time.sleep(wait_time)
            continue
        else:
            break

pbar.close()

# Mostrar resultados finais
print(f"\nTotal de tarefas coletadas: {len(all_tasks)}")
print(f"Total de tarefas únicas (por ID): {len(processed_ids)}")
print(f"Total de páginas acessadas: {page_count}")

if len(all_tasks) > 0:
    print("\nDetalhes estatísticos:")
    
    # Estatísticas sobre predições e anotações
    with_predictions = sum(1 for task in all_tasks if task.get("predictions"))
    with_annotations = sum(1 for task in all_tasks if task.get("annotations"))
    
    print(f"Tarefas com predições: {with_predictions} ({with_predictions/len(all_tasks)*100:.1f}%)")
    print(f"Tarefas com anotações: {with_annotations} ({with_annotations/len(all_tasks)*100:.1f}%)")
    
    print("\nDetalhes das primeiras 5 tarefas:")
    print("-" * 80)
    
    for i, task in enumerate(all_tasks[:min(5, len(all_tasks))], 1):
        has_predictions = "Sim" if task.get("predictions") else "Não"
        has_annotations = "Sim" if task.get("annotations") else "Não"
        image_path = task['data'].get('image', 'N/A')
        
        print(f"{i}. Task ID: {task['id']}")
        print(f"   Imagem: {image_path}")
        print(f"   Tem predições: {has_predictions}")
        print(f"   Tem anotações: {has_annotations}")
        print("-" * 80)

# ========== ADICIONAR PREDIÇÕES COM MODELO YOLO STANDARD ==========

# Caminho para o modelo YOLO (ajuste para seu modelo de detecção padrão)
model_path = "/home/diego/2TB/yolo/Trains/runs/cogtive_betterbeef_7.0_obb_416_sgd/weights/best.pt"

# Verificar se o modelo existe
if not os.path.exists(model_path):
    print(f"Erro: Modelo não encontrado no caminho {model_path}")
    exit(1)

# Função para baixar uma imagem a partir da URL
def download_image(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))

# Função para converter OBB (Ultralytics) para o formato Label Studio (polygonlabels)
def convert_yolo_obb_to_labelstudio(results, image_width, image_height):
    annotations = []
    
    # Verificar se o resultado não é vazio
    if not results or len(results) == 0:
        print("Nenhuma detecção encontrada nos resultados.")
        return annotations
    
    for result in results:
        obb = getattr(result, 'obb', None)
        if obb is None or not hasattr(obb, 'xyxyxyxy') or obb.xyxyxyxy is None or len(obb.xyxyxyxy) == 0:
            print("[WARN] Resultado sem OBB (xyxyxyxy).")
            continue
        names = getattr(result, 'names', None)
        n_preds = len(obb.xyxyxyxy)
        for i in range(n_preds):
            # Extração robusta dos 4 vértices (pode vir como (8,) ou (4,2))
            item = obb.xyxyxyxy[i]
            try:
                if hasattr(item, 'cpu'):
                    arr = item.cpu().numpy()
                else:
                    arr = np.array(item)
            except Exception:
                arr = np.array(item)

            if arr.ndim == 2 and arr.shape == (4, 2):
                pts_list = arr.reshape(-1).tolist()  # 8 valores
            elif arr.size == 8:
                pts_list = arr.reshape(-1).tolist()
            else:
                # Fallback: tentar tolist e achatar
                tl = item.tolist() if hasattr(item, 'tolist') else list(item)
                pts_list = np.array(tl).reshape(-1).tolist()

            conf = float(obb.conf[i]) if hasattr(obb, 'conf') else 1.0
            cls_id = int(obb.cls[i]) if hasattr(obb, 'cls') else 0

            if isinstance(names, dict):
                cls_name_raw = names.get(cls_id, str(cls_id))
            elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                cls_name_raw = names[cls_id]
            else:
                cls_name_raw = str(cls_id)
            class_name = str(cls_name_raw).lower()

            # Converter para percentuais 0-100 e clamp
            points = []
            if len(pts_list) != 8:
                print(f"[WARN] OBB com tamanho inesperado: len={len(pts_list)}. Ignorando.")
                continue
            for j in range(0, 8, 2):
                x = max(0.0, min(100.0, (float(pts_list[j]) / image_width) * 100.0))
                y = max(0.0, min(100.0, (float(pts_list[j + 1]) / image_height) * 100.0))
                points.append([x, y])

            label_studio_format = {
                "id": f"result_{i+1}",
                "type": "polygonlabels",
                "value": {
                    "points": points,
                    "polygonlabels": [class_name]
                },
                "score": conf,
                "from_name": "label",
                "to_name": "image",
                "original_width": image_width,
                "original_height": image_height
            }
            annotations.append(label_studio_format)
    return annotations

# Função para enviar predições para o Label Studio
def send_prediction_to_labelstudio(task_id, predictions, headers):
    url = f"https://internal-label.cogtive.com/api/predictions/"
    data = {
        "task": task_id,
        "model_version": "yolov8_obb_best",
        "result": predictions
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code >= 200 and response.status_code < 300:
        return True, response.json()
    else:
        return False, f"Erro {response.status_code}: {response.text}"

# Função para download e visualização da imagem no modo debug
def debug_visualize(image_path, image, results):
    """Salva imagem original e com anotações para fins de debug"""
    import os
    import matplotlib.pyplot as plt
    
    # Criar diretório de debug se não existir
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Salvar a imagem original
    original_path = os.path.join(debug_dir, "original.jpg")
    image.save(original_path)
    print(f"Imagem original salva em: {original_path}")
    
    # Salvar a imagem com anotações (usa a função de plot do YOLO)
    if results and len(results) > 0:
        # Plotar resultados e salvar
        plot_result = results[0].plot()  # Retorna imagem numpy com anotações
        annotated_path = os.path.join(debug_dir, "annotated.jpg")
        plt.figure(figsize=(12, 8))
        plt.imshow(plot_result)
        plt.axis('off')
        plt.savefig(annotated_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        print(f"Imagem com anotações salva em: {annotated_path}")
        
        # Imprimir detalhes das detecções
        print("\n===== DETALHES DAS DETECÇÕES =====")
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            print(f"Número de detecções: {len(results[0].boxes.xyxy)}")
            print(f"Classes detectadas: {results[0].boxes.cls.tolist()}")
            print(f"Nomes das classes: {[results[0].names[c.item()] for c in results[0].boxes.cls]}")
            print(f"Confiança: {results[0].boxes.conf.tolist()}")
            print(f"Caixas (xyxy): {results[0].boxes.xyxy.tolist()}")
        else:
            print("Nenhuma detecção encontrada")
    else:
        print("Nenhum resultado para visualização")
        
    return original_path, debug_dir

# Filtrar tarefas sem predições
tasks_without_predictions = [t for t in all_tasks if not t.get("predictions")]
print(f"Encontradas {len(tasks_without_predictions)} tarefas sem predições.")

if tasks_without_predictions:
    # Perguntar se deseja processar as tarefas
    if True:
        # Carregar o modelo YOLO
        print(f"Carregando o modelo YOLO de {model_path}...")
        model = YOLO(model_path)
        print("Modelo carregado com sucesso!")
        
        # Processar tarefas sem predições
        success_count = 0
        failed_count = 0
        
        # Em modo debug, processar apenas a primeira tarefa
        if DEBUG_MODE:
            tasks_to_process = tasks_without_predictions[:1]
            print("\n=== MODO DEBUG ATIVADO - Processando apenas a primeira tarefa ===\n")
        else:
            tasks_to_process = tasks_without_predictions
            pbar_predict = tqdm(total=len(tasks_to_process), desc="Processando predições")

        for task in tasks_to_process:
            task_id = task["id"]
            image_url = task.get("data", {}).get("image")
            
            print(f"\nProcessando tarefa {task_id}...")
            print(f"URL da imagem: {image_url}")
            
            if not image_url:
                print(f"Task {task_id}: Imagem não encontrada.")
                failed_count += 1
                if not DEBUG_MODE:
                    pbar_predict.update(1)
                continue
            
            # Se a URL for relativa, adicionar o domínio
            if not image_url.startswith("http"):
                image_url = f"https://internal-label.cogtive.com{image_url}"
            
            try:
                print(f"Baixando imagem de {image_url}...")
                # Baixar a imagem
                image = download_image(image_url, headers)
                image_np = np.array(image)
                
                # Obter dimensões da imagem
                image_height, image_width = image_np.shape[:2]
                print(f"Dimensões da imagem: {image_width}x{image_height}")
                
                # Temporariamente salvar a imagem para processamento
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    image.save(temp_path)
                
                print(f"Executando modelo YOLO (OBB) em {temp_path}...")
                # Fazer predições com o modelo YOLO (OBB)
                results = model(temp_path, verbose=DEBUG_MODE)
                
                # No modo debug, visualizar a imagem e os resultados
                if DEBUG_MODE:
                    original_path, debug_dir = debug_visualize(temp_path, image, results)
                    print(f"Arquivos de debug salvos em {debug_dir}")
                
                # Remover arquivo temporário
                os.unlink(temp_path)
                
                # Converter resultados YOLO OBB para formato Label Studio (polígonos)
                print("Convertendo resultados OBB para formato Label Studio (polygonlabels)...")
                annotations = convert_yolo_obb_to_labelstudio(results, image_width, image_height)
                
                if not annotations:
                    print(f"Task {task_id}: Nenhuma detecção encontrada.")
                    failed_count += 1
                    if not DEBUG_MODE:
                        pbar_predict.update(1)
                    continue
                
                # No modo debug, mostrar as anotações formatadas
                if DEBUG_MODE:
                    print("\n==== ANOTAÇÕES FORMATADAS PARA LABEL STUDIO ====")
                    import json
                    print(json.dumps(annotations, indent=2))
                
                # Enviar predições para o Label Studio
                print(f"Enviando predições para a tarefa {task_id}...")
                success, response = send_prediction_to_labelstudio(task_id, annotations, headers)
                
                if success:
                    print(f"Task {task_id}: Predições enviadas com sucesso.")
                    success_count += 1
                else:
                    print(f"Task {task_id}: Falha ao enviar predições: {response}")
                    failed_count += 1
                
                # Pequena pausa para não sobrecarregar a API
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Task {task_id}: Erro ao processar: {str(e)}")
                if DEBUG_MODE:
                    import traceback
                    traceback.print_exc()  # Mostrar traceback completo em modo debug
                failed_count += 1
            
            if not DEBUG_MODE:
                pbar_predict.update(1)
        
        if not DEBUG_MODE:
            pbar_predict.close()
            
        # Mostrar resultados finais
        print("\nResultados do processamento:")
        print(f"Total de tarefas processadas: {len(tasks_to_process)}")
        print(f"Sucesso: {success_count}")
        print(f"Falha: {failed_count}")
else:
    print("Não há tarefas sem predições para processar.")

# Opcionalmente, permitir salvar os resultados em um arquivo para análise futura
save_results = input("Deseja salvar os resultados em um arquivo JSON? (s/n): ")
if save_results.lower() == "s":
    output_file = "tasks_results.json"
    with open(output_file, "w") as f:
        json.dump(all_tasks, f, indent=2)
    print(f"Resultados salvos em {output_file}")
