"""JSON HTTP inference service with one process-local model; optional serve extra."""


def create_app(policy):
    import threading
    import numpy as np
    from fastapi import FastAPI, HTTPException
    app = FastAPI(title="OpenVLA QuIC")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ready", "model": policy.model.__class__.__name__}

    @app.post("/act")
    def act(payload: dict):
        instruction = payload.get("instruction")
        try:
            image = np.asarray(payload.get("image"))
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="image must be a rectangular RGB array")
        if not isinstance(instruction, str) or not instruction.strip():
            raise HTTPException(status_code=422, detail="instruction must be a nonempty string")
        if (image.ndim != 3 or image.shape[-1] != 3 or 0 in image.shape
                or not np.issubdtype(image.dtype, np.integer) or np.any(image < 0) or np.any(image > 255)):
            raise HTTPException(status_code=422, detail="image must be an integer RGB array [H,W,3] in [0,255]")
        with lock:
            action = np.asarray(policy.predict(image.astype(np.uint8), instruction), dtype=np.float32)
        if (action.ndim not in (1, 2) or action.shape[-1] != 7
                or (action.ndim == 2 and action.shape[0] < 1) or not np.isfinite(action).all()):
            raise HTTPException(status_code=500, detail="model returned invalid action values")
        return {"action": action.tolist()}
    return app


def serve(host, adapter=None, *, bind="127.0.0.1", port=8000):
    import uvicorn
    from .runtime import load_policy
    policy = load_policy(host, adapter)
    uvicorn.run(create_app(policy), host=bind, port=port, workers=1)
