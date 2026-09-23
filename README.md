omni-os-v2/
├── docker-compose.yml          # Orquestração da camada cognitiva
├── shared/                     # Código comum (partilhado via symlinks ou pacotes locais)
│   ├── config/                 # Onde ficará o state_routing.json
│   └── models/                 # schemas.py e graph_models.py
├── daemon/                     # O Host Daemon (Corre diretamente no SO)
│   ├── requirements.txt        # Dependências locais (pyautogui, mss, pywinctl)
│   ├── main.py                 # O loop principal (Orchestrator) que acabámos de criar
│   ├── execution/              # driver.py 
│   └── core/                   # window.py, capture.py, router.py
└── services/                   # A Camada Cognitiva (Contentores Docker)
    ├── planner/                # Microserviço Fast Path (Gemini Flash)
    │   ├── Dockerfile
    │   └── main.py             # API (ex: FastAPI) que recebe o estado e devolve o plano
    └── mapper/                 # Microserviço Slow Path (Gemini Pro + EasyOCR)
        ├── Dockerfile
        └── main.py             # API que processa a imagem e atualiza o GraphDB