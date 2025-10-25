import torch
from typing import List, Optional, Union, Dict, Any
import logging
from .base_orchestrator import BaseOrchestrator

logger = logging.getLogger(__name__)


class PostprocessingOrchestrator(BaseOrchestrator[torch.Tensor, torch.Tensor]):
    """
    Orchestrates postprocessing with parallelization and pipelining.
    
    Handles super-resolution, enhancement, style transfer, and other postprocessing operations
    that are applied to generated images after diffusion.
    """
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, max_workers: int = 4, pipeline_ref: Optional[Any] = None):
        # Postprocessing: 50ms timeout for quality-critical operations like upscaling
        super().__init__(device, dtype, max_workers, timeout_ms=20.0, pipeline_ref=pipeline_ref)

        # Postprocessing-specific state
        self._last_input_tensor = None
        self._last_processed_result = None
        self._current_input_tensor = None  # For BaseOrchestrator fallback logic

        # Cache for pipeline-aware processor detection (avoid hot path checks)
        self._processors_cache_key = None
        self._has_sync_required_cache = False


    def process_pipelined(self, 
                        input_tensor: torch.Tensor,
                        postprocessors: List[Any],
                        *args, **kwargs) -> torch.Tensor:
        """
        Process input with intelligent pipelining.
        
        Overrides base method to store current input tensor for fallback logic.
        """
        # Store current input for fallback logic
        self._current_input_tensor = input_tensor
        
        # RACE CONDITION FIX: Check if there are actually enabled processors
        # Filter to only enabled processors (same logic as _get_ordered_processors)
        enabled_processors = [p for p in postprocessors if getattr(p, 'enabled', True)] if postprocessors else []
        
        if not enabled_processors:
            return input_tensor
        
        # Call parent implementation
        return super().process_pipelined(input_tensor, postprocessors, *args, **kwargs)
    
    def _should_use_sync_processing(self, *args, **kwargs) -> bool:
        """
        Determine if synchronous processing should be used instead of pipelined.

        Checks for pipeline-aware postprocessors (feedback, temporal, etc.) that require
        synchronous processing to avoid temporal artifacts and ensure access to previous
        pipeline outputs.

        Args:
            *args: Arguments from process_pipelined call (postprocessors is first argument)
            **kwargs: Keyword arguments

        Returns:
            True if pipeline-aware postprocessors detected, False otherwise
        """
        # Extract postprocessors from args - they're the first argument after input_tensor
        if len(args) < 1:
            return False

        postprocessors = args[0]  # postprocessors is first arg after input_tensor
        return self._check_pipeline_aware_cached(postprocessors)
    
    def process_sync(self, 
                   input_tensor: torch.Tensor,
                   postprocessors: List[Any],
                   *args, **kwargs) -> torch.Tensor:
        """
        Process tensor through postprocessors synchronously.
        
        Args:
            input_tensor: Input tensor to postprocess (typically from diffusion output)
            postprocessors: List of postprocessor instances
            *args, **kwargs: Additional arguments for postprocessors
            
        Returns:
            Postprocessed tensor
        """
        if not postprocessors:
            return input_tensor
        
        # Use same stream context as background processing for consistency
        original_stream = self._set_background_stream_context()
        try:
            # Sequential application of postprocessors
            current_tensor = input_tensor
            for postprocessor in postprocessors:
                if postprocessor is not None:
                    current_tensor = self._apply_single_postprocessor(current_tensor, postprocessor)
            
            return current_tensor
        finally:
            self._restore_stream_context(original_stream)
    
    def _process_frame_background(self, 
                                input_tensor: torch.Tensor,
                                postprocessors: List[Any],
                                *args, **kwargs) -> Dict[str, Any]:
        """
        Process a frame in the background thread.
        
        Implementation of BaseOrchestrator._process_frame_background for postprocessing.
        
        Returns:
            Dictionary containing processing results and status
        """
        try:
            # Set CUDA stream for background processing
            original_stream = self._set_background_stream_context()
            
            if not postprocessors:
                return {
                    'result': input_tensor,
                    'status': 'success'
                }
            
            # Check for cache hit (same input tensor)
            cache_hit = False
            if (self._last_input_tensor is not None and self._last_processed_result is not None):
                if input_tensor.device == self._last_input_tensor.device:
                    # Same device - direct comparison
                    cache_hit = torch.equal(input_tensor, self._last_input_tensor)
                else:
                    # Different devices - move cached tensor to input device for comparison
                    cached_on_input_device = self._last_input_tensor.to(device=input_tensor.device, dtype=input_tensor.dtype)
                    cache_hit = torch.equal(input_tensor, cached_on_input_device)
            
            if cache_hit:
                return {
                    'result': self._last_processed_result,  # Return previously processed result
                    'status': 'success',
                    'cache_hit': True
                }
            
            # Update cache with current input tensor
            self._last_input_tensor = input_tensor.clone()
            
            # Process postprocessors in parallel if multiple, sequential if single
            if len(postprocessors) > 1:
                result = self._process_postprocessors_parallel(input_tensor, postprocessors)
            else:
                result = self._apply_single_postprocessor(input_tensor, postprocessors[0])
            
            # Cache the processed result for future cache hits
            self._last_processed_result = result
            
            return {
                'result': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"PostprocessingOrchestrator: Background processing failed: {e}")
            return {
                'result': input_tensor,  # Return original on error
                'error': str(e),
                'status': 'error'
            }
        finally:
            # Restore original CUDA stream
            self._restore_stream_context(original_stream)
        
    def _process_postprocessors_parallel(self, 
                                       input_tensor: torch.Tensor, 
                                       postprocessors: List[Any]) -> torch.Tensor:
        """
        Process multiple postprocessors in parallel.
        
        Note: This applies postprocessors sequentially for now, but could be extended
        to support parallel processing for independent postprocessors in the future.
        
        Args:
            input_tensor: Input tensor to process
            postprocessors: List of postprocessor instances
            
        Returns:
            Processed tensor
        """
        # For now, apply sequentially since most postprocessors are dependent
        # Future enhancement: Detect independent postprocessors and run in parallel
        current_tensor = input_tensor
        for postprocessor in postprocessors:
            if postprocessor is not None:
                current_tensor = self._apply_single_postprocessor(current_tensor, postprocessor)
        
        return current_tensor
    
    def _apply_single_postprocessor(self, 
                                  input_tensor: torch.Tensor, 
                                  postprocessor: Any) -> torch.Tensor:
        """
        Apply a single postprocessor to the input tensor.
        
        Handles normalization conversion between VAE output range [-1,1] and 
        processor input range [0,1], then converts back to VAE range.
        
        Args:
            input_tensor: Input tensor from VAE (range [-1,1])
            postprocessor: Postprocessor instance
            
        Returns:
            Processed tensor in VAE range [-1,1]
        """
        try:
            # Ensure tensor is on correct device and dtype
            processed_tensor = input_tensor.to(device=self.device, dtype=self.dtype)
            
            logger.debug(f"_apply_single_postprocessor: Converting tensor from VAE range [-1,1] to processor range [0,1]")
            processor_input = (processed_tensor / 2.0 + 0.5).clamp(0, 1)
            
            # Apply postprocessor
            if hasattr(postprocessor, 'process_tensor'):
                # Prefer tensor processing if available
                processor_output = postprocessor.process_tensor(processor_input)
            elif hasattr(postprocessor, 'process'):
                # Fallback to general process method
                processor_output = postprocessor.process(processor_input)
            elif callable(postprocessor):
                # Treat as callable
                processor_output = postprocessor(processor_input)
            else:
                logger.warning(f"PostprocessingOrchestrator: Unknown postprocessor type: {type(postprocessor)}")
                return processed_tensor
            
            # Ensure result is a tensor
            if isinstance(processor_output, torch.Tensor):
                # CRITICAL: Convert back from processor output range [0,1] to VAE input range [-1,1]
                logger.debug(f"_apply_single_postprocessor: Converting result from processor range [0,1] back to VAE range [-1,1]")
                result = (processor_output - 0.5) * 2.0  # Convert [0,1] -> [-1,1]
                
                return result.to(device=self.device, dtype=self.dtype)
            else:
                logger.warning(f"PostprocessingOrchestrator: Postprocessor returned non-tensor: {type(processor_output)}")
                return processed_tensor
                
        except Exception as e:
            logger.error(f"PostprocessingOrchestrator: Postprocessor failed: {e}")
            return input_tensor  # Return original on error
    
    def _check_pipeline_aware_cached(self, postprocessors: List[Optional[Any]]) -> bool:
        """
        Efficiently check for pipeline-aware postprocessors using caching.

        Only performs expensive isinstance checks when postprocessor list actually changes.
        Pipeline-aware postprocessors (feedback, temporal, etc.) need synchronous processing
        to avoid temporal artifacts and ensure access to previous pipeline outputs.

        Args:
            postprocessors: List of postprocessor instances

        Returns:
            True if any postprocessor requires sync processing, False otherwise
        """
        # Create cache key from postprocessor identities
        cache_key = tuple(id(p) for p in postprocessors)

        # Return cached result if postprocessors haven't changed
        if cache_key == self._processors_cache_key:
            return self._has_sync_required_cache

        # Postprocessors changed - recompute and cache
        self._processors_cache_key = cache_key
        self._has_sync_required_cache = False

        try:
            # Check for the mixin or class attribute first
            for prep in postprocessors:
                if prep is not None and getattr(prep, 'requires_sync_processing', False):
                    self._has_sync_required_cache = True
                    break
        except Exception:
            # Fallback: check for specific known pipeline-aware postprocessors
            try:
                from .processors.feedback import FeedbackPreprocessor
                from .processors.color_correction_feedback import ColorCorrectionFeedbackPreprocessor
                from .processors.feedback_transform import FeedbackTransformPreprocessor
                for prep in postprocessors:
                    if isinstance(prep, (FeedbackPreprocessor, ColorCorrectionFeedbackPreprocessor, FeedbackTransformPreprocessor)):
                        self._has_sync_required_cache = True
                        break
            except Exception:
                # Final fallback on class name check without importing
                for prep in postprocessors:
                    if prep is not None:
                        class_name = prep.__class__.__name__.lower()
                        if any(name in class_name for name in ['feedback', 'temporal']):
                            self._has_sync_required_cache = True
                            break

        return self._has_sync_required_cache

    def clear_cache(self) -> None:
        """Clear postprocessing cache"""
        self._last_input_tensor = None
        self._last_processed_result = None
        # Clear processor cache when clearing other caches
        self._processors_cache_key = None
        self._has_sync_required_cache = False


