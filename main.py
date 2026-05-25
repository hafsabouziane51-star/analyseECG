
import numpy as np
import neurokit2 as nk
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
import uuid
import pandas as pd
import io
app = FastAPI(title="Clinical Holter Batch Engine v1.0")

# In-memory database for demo (Use Redis/PostgreSQL for production)
tasks_db = {}

class HolterStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    results: Optional[Dict] = None

# --- Core Batch Logic ---
def process_holter_task(task_id: str, signal: np.array, fs: int):
    tasks_db[task_id]["status"] = "processing"
    
    # 1. Configuration: 1-minute windows (60 seconds)
    window_size = 60 * fs
    total_samples = len(signal)
    num_windows = total_samples // window_size
    
    batch_results = []
    
    try:
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            segment = signal[start:end]
            
            # Clinical Processing per window
            try:
                # Clean and find peaks
                cleaned = nk.ecg_clean(segment, sampling_rate=fs)
                _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
                
                # Metrics for this minute
                r_peaks = info["ECG_R_Peaks"]
                if len(r_peaks) > 5:
                    rr_intervals = np.diff(r_peaks) / fs
                    bpm = 60 / np.mean(rr_intervals)
                    hrv = nk.hrv_time(info, sampling_rate=fs)
                    
                    batch_results.append({
                        "minute": i,
                        "bpm": round(float(bpm), 1),
                        "sdnn": round(float(hrv["HRV_SDNN"].iloc[0]), 2),
                        "sqi": round(float(nk.ecg_quality(cleaned, sampling_rate=fs)), 3)
                    })
            except:
                continue # Skip noisy segments (common in Holter)

            # Update Progress
            tasks_db[task_id]["progress"] = round(((i + 1) / num_windows) * 100, 1)

        # 2. Global Aggregation (Clinical 24h Summary)
        all_bpms = [r["bpm"] for r in batch_results]
        tasks_db[task_id]["results"] = {
            "summary": {
                "mean_hr": round(np.mean(all_bpms), 1),
                "max_hr": max(all_bpms),
                "min_hr": min(all_bpms),
                "total_minutes_analyzed": len(batch_results)
            },
            "trends": batch_results
        }
        tasks_db[task_id]["status"] = "completed"

    except Exception as e:
        tasks_db[task_id]["status"] = f"failed: {str(e)}"

# --- API Endpoints ---

@app.post("/upload-holter")
async def upload_holter(background_tasks: BackgroundTasks, file: UploadFile = File(...), fs: int = 250):
    """
    Upload a large CSV or Binary Holter file.
    """
    task_id = str(uuid.uuid4())
    
    # Read file into memory (For very large files, use Dask or stream from disk)
    content = await file.read()
    try:
        # Assuming CSV for this example
        df = pd.read_csv(io.BytesIO(content))
        signal = df.iloc[:, 0].values # First column as ECG
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid file format. Expected CSV.")

    tasks_db[task_id] = {"status": "queued", "progress": 0, "results": None}
    
    # Start background processing
    background_tasks.add_task(process_holter_task, task_id, signal, fs)
    
    return {"task_id": task_id, "message": "Holter analysis started in background."}

@app.get("/status/{task_id}", response_model=HolterStatus)
async def get_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return {**tasks_db[task_id], "task_id": task_id}