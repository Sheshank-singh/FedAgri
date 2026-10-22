import os
import sys
import uvicorn

if __name__ == "__main__":
    banner = r"""
  ______        _     _             _ 
 |  ____|      | |   / \   __ _ _ _(_)
 | |__ ___   __| |  / _ \ / _` | '__| |
 |  __/ _ \ / _` | / ___ \ (_| | |  | |
 |_|  \___/ \__,_|/_/   \_\__, |_|  |_|
                          |___/        
 =======================================================
  FedAgri — Federated Learning Platform for Smart Agriculture
  Connecting Edge Agritech to Distributed AI
 =======================================================
"""
    print(banner)
    print("  -> Web Dashboard:    http://localhost:8000")
    print("  -> REST & WS API:    http://localhost:8000/docs")
    print("  -> Hardware Engine:  PyTorch with Auto GPU/CPU Detection")
    print("  -> Press Ctrl+C to terminate.")
    print(" =======================================================\n")

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
