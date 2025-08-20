import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import json
from tqdm import tqdm
import os
import numpy as np
import onnxruntime as ort
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
 

# Modo de debug - definir como True para processar apenas a primeira imagem
DEBUG_MODE = False

# Número de threads para processamento paralelo (apenas quando DEBUG_MODE=False)
MAX_THREADS = int(os.getenv('LS_MAX_THREADS', '8'))

# Configuração da API
project_id = 25
api_base = f"https://internal-label.cogtive.com/api/projects/{project_id}/tasks/"
headers = {
    # Prefira variável de ambiente; mantém valor padrão como fallback
    "Authorization": f"Token {os.getenv('LABEL_STUDIO_TOKEN', 'a27c0b74353fe5f041d9d54d8323d8dfb3457c64')}"
}


# Conjunto para verificar IDs de tarefas já processadas
processed_ids = set()

# Definir o tamanho da página e quantidade de tarefas a baixar com base no modo
if DEBUG_MODE:
    print("\n=== MODO DEBUG ATIVADO - Baixando apenas uma tarefa ===\n")
    page_size = 1
    max_tasks = 1
else:
    # Aumentar o page_size para reduzir round-trips (ajuste se API suportar)
    page_size = 500
    max_tasks = None  # Sem limite

# Preferir filtrar no servidor tarefas sem anotação, se suportado pela API
SERVER_FILTER_NO_ANNOTATIONS = True

# Criar sessão HTTP reutilizável com retries/backoff
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"])  # Retry em GET/POST
adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Sessão thread-local para uso no pool
_thread_local = threading.local()

def get_thread_session():
    if getattr(_thread_local, 'session', None) is None:
        s = requests.Session()
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return _thread_local.session

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

# Aplicar filtro de servidor, se habilitado
if SERVER_FILTER_NO_ANNOTATIONS:
    # Filtro comum no Label Studio para tarefas sem anotações
    params['annotations__isnull'] = 'true'

response = session.get(api_base, headers=headers, params=params)
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
    if SERVER_FILTER_NO_ANNOTATIONS:
        params['annotations__isnull'] = 'true'

    try:
        print(f"Requisitando página {page} (tamanho: {page_size})")
        response = session.get(api_base, headers=headers, params=params)
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

        # Pausa mínima opcional (reduzida) – rely on retries para backoff
        if DEBUG_MODE:
            time.sleep(0.05)

    except requests.exceptions.RequestException as e:
        print(f"\nErro ao fazer requisição: {e}")
        # Session com retries já tenta novamente; sair em falhas não recuperáveis
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

# Caminho para o modelo YOLO
model_path = "/home/diego/Downloads/yolov5s-kavak.onnx"

# Label mapping para o modelo Kavak (baseado na transcripção fornecida)
KAVAK_LABELS = {
    1: "license_plate",
    2: "person",
    3: "prisma",
    4: "car"
}

# Função para pré-processamento da imagem para YOLOv5 (array in-memory)
def preprocess_image_array(img_bgr, input_size=640):
    """Pré-processa imagem (numpy BGR) para inferência YOLOv5 ONNX"""
    # Converter para RGB, manter dimensões originais
    original_height, original_width = img_bgr.shape[:2]
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Redimensionar mantendo aspect ratio
    scale = min(input_size / original_width, input_size / original_height)
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))

    # Redimensionar
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Padding para chegar ao tamanho exato (somente direita e inferior)
    pad_width = input_size - new_width
    pad_height = input_size - new_height
    padded = cv2.copyMakeBorder(
        resized, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # Normalizar e converter para formato NCHW
    input_tensor = padded.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC para CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Adicionar batch dimension

    return input_tensor, original_width, original_height, scale

# Função para pós-processamento das saídas YOLOv5
def postprocess_yolo_output(output, original_width, original_height, scale, conf_threshold=0.25, iou_threshold=0.45):
    """Pós-processa saídas YOLOv5 ONNX para extrair detecções (vetorizado + NMS OpenCV)"""
    # YOLOv5 output shape: (1, 25200, 9) onde 9 = 4 (bbox) + 1 (objectness) + 4 (classes)
    predictions = output[0]  # (N, 9)
    if predictions.size == 0:
        return []

    boxes_cxcywh = predictions[:, :4]
    objectness = predictions[:, 4]
    class_scores = predictions[:, 5:]

    # Filtrar por objectness
    mask = objectness >= conf_threshold
    if not np.any(mask):
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    objectness = objectness[mask]
    class_scores = class_scores[mask]

    # Melhor classe por predição
    class_ids = np.argmax(class_scores, axis=1) + 1  # IDs começam em 1
    best_class_scores = class_scores[np.arange(class_scores.shape[0]), class_ids - 1]
    confidences = best_class_scores * objectness

    # Filtrar por confidence final
    mask_conf = confidences >= conf_threshold
    if not np.any(mask_conf):
        return []

    boxes_cxcywh = boxes_cxcywh[mask_conf]
    confidences = confidences[mask_conf]
    class_ids = class_ids[mask_conf]

    # Converter para x1y1x2y2 na imagem original
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = (cx - w / 2.0) / scale
    y1 = (cy - h / 2.0) / scale
    x2 = (cx + w / 2.0) / scale
    y2 = (cy + h / 2.0) / scale

    # Clipping
    x1 = np.clip(x1, 0, original_width)
    y1 = np.clip(y1, 0, original_height)
    x2 = np.clip(x2, 0, original_width)
    y2 = np.clip(y2, 0, original_height)

    # Preparar para NMS do OpenCV (x, y, w, h)
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    boxes_xywh_int = boxes_xywh.astype(np.int32).tolist()
    confidences_list = confidences.astype(float).tolist()

    # NMS (classe-agnóstica)
    indices = cv2.dnn.NMSBoxes(boxes_xywh_int, confidences_list, score_threshold=conf_threshold, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return []
    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    else:
        indices = [int(i) if not isinstance(i, (list, tuple)) else int(i[0]) for i in indices]

    detections = []
    for idx in indices:
        bx1, by1, bw, bh = boxes_xywh[idx]
        det = {
            'box': [float(bx1), float(by1), float(bx1 + bw), float(by1 + bh)],
            'confidence': float(confidences[idx]),
            'class_id': int(class_ids[idx]),
            'class_name': KAVAK_LABELS.get(int(class_ids[idx]), 'unknown')
        }
        detections.append(det)

    # Ordenar por confiança descrescente
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections

# Função para aplicar Non-Maximum Suppression
def apply_nms(detections, iou_threshold):
    """Aplica NMS usando OpenCV (mantido por compatibilidade, não utilizado no caminho vetorizado)."""
    if not detections:
        return []
    boxes = []
    scores = []
    for d in detections:
        x1, y1, x2, y2 = d['box']
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        scores.append(float(d['confidence']))
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return []
    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    else:
        indices = [int(i) if not isinstance(i, (list, tuple)) else int(i[0]) for i in indices]
    return [detections[i] for i in indices]

# Função para calcular IoU
def calculate_iou(box1, box2):
    """Calcula IoU entre duas boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calcular interseção
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calcular união
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

# Função para enviar ANOTAÇÕES para o Label Studio (em vez de predições)
def send_annotation_to_labelstudio(session, task_id, annotations, headers):
    """Cria uma anotação real para a task no Label Studio.

    Endpoint: POST /api/tasks/{task_id}/annotations/
    Payload mínimo: {"result": [...]} onde result é a lista no formato do Label Studio
    """
    url = f"https://internal-label.cogtive.com/api/tasks/{task_id}/annotations/"
    data = {
        "result": annotations,
        # Campos opcionais que podem ser úteis:
        # "ground_truth": False,
        # "was_cancelled": False,
        # "lead_time": 0,
    }

    response = session.post(url, json=data, headers=headers)
    if 200 <= response.status_code < 300:
        return True, response.json()
    else:
        return False, f"Erro {response.status_code}: {response.text}"

# Verificar se o modelo existe
if not os.path.exists(model_path):
    print(f"Erro: Modelo não encontrado no caminho {model_path}")
    exit(1)

# Carregar o modelo ONNX com otimizações e provider disponível
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = ['CPUExecutionProvider']
if 'CUDAExecutionProvider' in ort.get_available_providers():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
input_name = ort_session.get_inputs()[0].name

# Função de processamento de uma única task (para uso com ThreadPool)
def process_task(task):
    try:
        task_id = task["id"]
        image_url = task.get("data", {}).get("image")

        if not image_url:
            if DEBUG_MODE:
                print(f"Task {task_id}: Imagem não encontrada.")
            return False

        if not image_url.startswith("http"):
            image_url = f"https://internal-label.cogtive.com{image_url}"

        sess = get_thread_session()

        # Download e decodificação
        resp = sess.get(image_url, headers=headers)
        resp.raise_for_status()
        np_arr = np.frombuffer(resp.content, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            if DEBUG_MODE:
                print(f"Task {task_id}: Falha ao decodificar a imagem")
            return False

        # Pré-processamento
        input_tensor, original_width, original_height, scale = preprocess_image_array(img_bgr, input_size=640)

        # Inferência
        outputs = ort_session.run(None, {input_name: input_tensor})

        # Pós-processamento
        detections = postprocess_yolo_output(outputs[0], original_width, original_height, scale)

        # Montar anotações
        annotations = []
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['box']
            class_name = detection['class_name'].upper()
            confidence = detection['confidence']

            x_percent = (x1 / original_width) * 100
            y_percent = (y1 / original_height) * 100
            width_percent = ((x2 - x1) / original_width) * 100
            height_percent = ((y2 - y1) / original_height) * 100

            label_studio_format = {
                "id": f"result_{i+1}",
                "type": "rectanglelabels",
                "value": {
                    "x": max(0, x_percent),
                    "y": max(0, y_percent),
                    "width": width_percent,
                    "height": height_percent,
                    "rectanglelabels": [class_name]
                },
                "score": confidence,
                "from_name": "label",
                "to_name": "image",
                "original_width": original_width,
                "original_height": original_height
            }
            annotations.append(label_studio_format)

        # Enviar anotações
        ok, _ = send_annotation_to_labelstudio(sess, task_id, annotations, headers)
        if not ok and DEBUG_MODE:
            print(f"Task {task_id}: Falha ao enviar anotações")
        return ok
    except Exception as e:
        if DEBUG_MODE:
            import traceback
            print(f"Task {task.get('id')}: Erro ao processar: {e}")
            traceback.print_exc()
        return False

# Filtrar tarefas sem anotações
tasks_without_annotations = [t for t in all_tasks if not t.get("annotations")]
print(f"Encontradas {len(tasks_without_annotations)} tarefas sem anotações.")

if tasks_without_annotations:
    # Perguntar se deseja processar as tarefas
    if True:
        # Processar tarefas sem anotações
        success_count = 0
        failed_count = 0

        # Em modo debug, processar apenas a primeira tarefa
        if DEBUG_MODE:
            tasks_to_process = tasks_without_annotations[:1]
            print("\n=== MODO DEBUG ATIVADO - Processando apenas a primeira tarefa ===\n")
        else:
            tasks_to_process = tasks_without_annotations
            pbar_predict = tqdm(total=len(tasks_to_process), desc="Processando anotações")

        if DEBUG_MODE:
            for task in tasks_to_process:
                if process_task(task):
                    success_count += 1
                else:
                    failed_count += 1
                # Progresso simples no debug
                print(f"Progresso: {success_count + failed_count}/{len(tasks_to_process)}")
        else:
            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                futures = {executor.submit(process_task, task): task for task in tasks_to_process}
                for future in as_completed(futures):
                    ok = future.result()
                    if ok:
                        success_count += 1
                    else:
                        failed_count += 1
                    pbar_predict.update(1)

        if not DEBUG_MODE:
            pbar_predict.close()

        # Mostrar resultados finais
        print("\nResultados do processamento:")
        print(f"Total de tarefas processadas: {len(tasks_to_process)}")
        print(f"Sucesso: {success_count}")
        print(f"Falha: {failed_count}")
else:
    print("Não há tarefas sem anotações para processar.")

# Opcionalmente, permitir salvar os resultados em um arquivo para análise futura
if DEBUG_MODE:
    save_results = input("Deseja salvar os resultados em um arquivo JSON? (s/n): ")
    if save_results.lower() == "s":
        output_file = "tasks_results.json"
        with open(output_file, "w") as f:
            json.dump(all_tasks, f, indent=2)
        print(f"Resultados salvos em {output_file}")
