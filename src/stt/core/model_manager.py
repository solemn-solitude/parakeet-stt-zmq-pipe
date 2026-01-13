"""Model manager with lazy loading and timeout-based deallocation."""
import logging
import threading
import time
from typing import Optional

import nemo.collections.asr as nemo_asr


logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the ASR model lifecycle with lazy loading and timeout."""
    
    def __init__(self, model_name: str, timeout_minutes: int = 10):
        """Initialize the model manager.
        
        Args:
            model_name: Name of the NeMo ASR model to load
            timeout_minutes: Minutes of inactivity before model deallocation
        """
        self.model_name = model_name
        self.timeout_seconds = timeout_minutes * 60
        
        self._model: Optional[nemo_asr.models.ASRModel] = None
        self._last_used_time: float = 0
        self._lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        
        logger.info(
            f"ModelManager initialized: model={model_name}, "
            f"timeout={timeout_minutes} minutes"
        )
    
    def start_monitoring(self) -> None:
        """Start the background thread that monitors model timeout."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_timeout,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Model timeout monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Model timeout monitoring stopped")
    
    def get_model(self) -> nemo_asr.models.ASRModel:
        """Get the ASR model, loading it if necessary.
        
        Returns:
            The loaded ASR model
            
        Raises:
            RuntimeError: If model loading fails
        """
        with self._lock:
            if self._model is None:
                self._load_model()
            
            # Update last used time
            self._last_used_time = time.time()
            return self._model
    
    def _load_model(self) -> None:
        """Load the ASR model."""
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()
        
        try:
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            self._last_used_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def _deallocate_model(self) -> None:
        """Deallocate the model to free memory."""
        with self._lock:
            if self._model is not None:
                logger.info("Deallocating model due to timeout")
                self._model = None
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
                # If CUDA is available, clear cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("CUDA cache cleared")
                except ImportError:
                    pass
                
                logger.info("Model deallocated successfully")
    
    def _monitor_timeout(self) -> None:
        """Background thread that monitors inactivity and deallocates model."""
        check_interval = 60  # Check every minute
        
        while self._running:
            time.sleep(check_interval)
            
            if not self._running:
                break
            
            with self._lock:
                if self._model is not None:
                    idle_time = time.time() - self._last_used_time
                    
                    if idle_time >= self.timeout_seconds:
                        logger.info(
                            f"Model idle for {idle_time/60:.1f} minutes, "
                            f"deallocating..."
                        )
                        self._deallocate_model()
    
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        with self._lock:
            return self._model is not None
    
    def force_reload(self) -> None:
        """Force reload of the model."""
        with self._lock:
            logger.info("Forcing model reload")
            self._deallocate_model()
            self._load_model()
