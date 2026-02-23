"""
watchdog_service.py — Background service to monitor data/sensor_logs.csv
and automatically trigger re-ingestion into the FAISS vector store.
"""

import time
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest import ingest

class SensorLogHandler(FileSystemEventHandler):
    def __init__(self, target_file):
        self.target_file = os.path.abspath(target_file)
        self.last_run = 0
        self.cooldown = 3  # Seconds to wait between triggers

    def on_modified(self, event):
        if event.is_directory:
            return
        
        if os.path.abspath(event.src_path) == self.target_file:
            current_time = time.time()
            if current_time - self.last_run > self.cooldown:
                print(f"\n🔔 Detected change in {os.path.basename(self.target_file)}")
                print("🔄 Triggering automatic re-ingestion...")
                try:
                    ingest()
                    self.last_run = time.time()
                    print("✅ Re-ingestion complete.\n")
                except Exception as e:
                    print(f"❌ Error during ingestion: {e}\n")

if __name__ == "__main__":
    target = "data/sensor_logs.csv"
    watch_dir = "data"
    
    if not os.path.exists(watch_dir):
        print(f"❌ Directory {watch_dir} not found.")
        sys.exit(1)

    event_handler = SensorLogHandler(target)
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=False)
    
    print(f"🕵️  FailureRadar Watchdog started.")
    print(f"👀 Monitoring: {target}")
    print("Press Ctrl+C to stop.")
    
    try:
        observer.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
