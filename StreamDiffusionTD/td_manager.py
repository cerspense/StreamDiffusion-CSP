"""
TouchDesigner StreamDiffusion Manager

Core bridge between TouchDesigner and the LivePeer StreamDiffusion fork.
Handles configuration, streaming loop, and parameter updates.
"""

import os
import sys
import time
import platform
import threading
import logging
from typing import Dict, Any, Optional, Union
from multiprocessing import shared_memory
import numpy as np
import torch
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('TouchDesignerManager')

# Add StreamDiffusion to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from streamdiffusion.config import load_config, create_wrapper_from_config
from streamdiffusion import StreamDiffusionWrapper
from frame_capture_debug import FrameCaptureDebug
from synthetic_input_generator import SyntheticInputGenerator


class TouchDesignerManager:
    """
    Main manager class that bridges TouchDesigner with StreamDiffusion fork.
    
    Key differences from your old version:
    1. Uses new fork's config system and unified parameter updates
    2. Maintains same SharedMemory/Syphon interface patterns
    3. Leverages new fork's caching and performance improvements
    """
    
    def __init__(self, config: Union[str, Dict[str, Any]], input_mem_name: str, output_mem_name: str):
        self.input_mem_name = input_mem_name
        self.output_mem_name = output_mem_name
        self.osc_handler = None  # Will be set later
        
        # Handle both config dict (new) and config path (legacy compatibility)
        if isinstance(config, dict):
            print("Using pre-merged configuration dictionary")
            self.config = config
            self.config_path = None
        else:
            # Legacy: Load configuration using new fork's config system
            print(f"Loading configuration from: {config}")
            self.config = load_config(config)
            self.config_path = config
        
        # Extract TD-specific settings
        self.td_settings = self.config.get('td_settings', {})
        
        # Platform detection (same as your current version)
        self.is_macos = platform.system() == 'Darwin'
        self.stream_method = "syphon" if self.is_macos else "shared_mem"
        
        # Initialize StreamDiffusion wrapper using new fork
        print("Creating StreamDiffusion wrapper...")
        self.wrapper: StreamDiffusionWrapper = create_wrapper_from_config(self.config)
        
        # Memory interfaces (will be initialized in start_streaming)
        self.input_memory: Optional[shared_memory.SharedMemory] = None
        self.output_memory: Optional[shared_memory.SharedMemory] = None
        self.control_memory: Optional[shared_memory.SharedMemory] = None
        self.control_processed_memory: Optional[shared_memory.SharedMemory] = None  # For pre-processed ControlNet output
        self.ipadapter_memory: Optional[shared_memory.SharedMemory] = None
        self.syphon_handler = None
        
        # Streaming state
        self.streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.paused = False
        self.process_frame = False
        self.frame_acknowledged = False
        
        # State tracking for logging (only log changes)
        self._last_paused_state = False
        self._frames_processed_in_pause = 0
        
        # ControlNet and IPAdapter state
        self.ipadapter_update_requested = False
        self.control_mem_name = self.input_mem_name + '-cn'
        self.control_processed_mem_name = self.input_mem_name + '-cn-processed'  # For pre-processed ControlNet output
        self.ipadapter_mem_name = self.input_mem_name + '-ip'
        
        # Track live IPAdapter scale from OSC updates
        self._current_ipadapter_scale = self.config.get('ipadapters', [{}])[0].get('scale', 1.0)
        
        # Performance tracking
        self.frame_count = 0
        self.total_frame_count = 0  # Total frames processed (for OSC)
        self.start_time = time.time()
        self.fps_smoothing = 0.9  # For exponential moving average
        self.current_fps = 0.0
        self.last_frame_output_time = 0.0  # Track actual frame output timing
        
        # OSC notification flags
        self._sent_processed_cn_name = False

        # Mode tracking (img2img or txt2img)
        self.mode = self.config.get('mode', 'img2img')
        print(f"Initialized in {self.mode} mode")

        # Frame capture debug (if enabled)
        self.frame_capturer = FrameCaptureDebug(
            enabled=self.config.get('debug_capture_frames', False),
            skip_frames=30,
            capture_count=4
        )

        # Synthetic input generator (for frame capture testing)
        self.synthetic_input = None
        if self.frame_capturer.enabled:
            self.synthetic_input = SyntheticInputGenerator(
                width=self.config.get('width', 576),
                height=self.config.get('height', 448),
                frequency=4.0,
                speed=0.02
            )

        print("TouchDesigner Manager initialized successfully")
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update StreamDiffusion parameters using new fork's unified system.
        
        This replaces the scattered parameter updates in your old version
        with a single, atomic update call that handles caching efficiently.
        """
        try:
            # Track IPAdapter scale changes from OSC
            if 'ipadapter_config' in params and 'scale' in params['ipadapter_config']:
                self._current_ipadapter_scale = params['ipadapter_config']['scale']
                logger.info(f"📊 Updated IPAdapter scale to: {self._current_ipadapter_scale}")
            
            # Filter out invalid parameters that wrapper doesn't accept
            valid_params = ['num_inference_steps', 'guidance_scale', 'delta', 't_index_list', 'seed',
                           'prompt_list', 'negative_prompt', 'prompt_interpolation_method', 'normalize_prompt_weights',
                           'seed_list', 'seed_interpolation_method', 'normalize_seed_weights',
                           'controlnet_config', 'ipadapter_config', 'image_preprocessing_config',
                           'image_postprocessing_config', 'latent_preprocessing_config', 
                           'latent_postprocessing_config', 'use_safety_checker', 'safety_checker_threshold']
            
            filtered_params = {k: v for k, v in params.items() if k in valid_params}
            
            # Use the new fork's unified parameter update system
            self.wrapper.update_stream_params(**filtered_params)
            # Removed noisy parameter logging
        except Exception as e:
            print(f"Error updating parameters: {e}")
    
    def start_streaming(self) -> None:
        """Initialize memory interfaces and start the streaming loop"""
        if self.streaming:
            print("Already streaming!")
            return

        try:
            self._initialize_memory_interfaces()

            # Wire up frame capturer to pipeline (if enabled)
            if self.frame_capturer.enabled:
                self.wrapper.stream.frame_capturer = self.frame_capturer
                print("[FRAME CAPTURE] Linked to pipeline")

            self.streaming = True

            # Start streaming in separate thread (non-blocking)
            self.stream_thread = threading.Thread(target=self._streaming_loop, daemon=True)
            self.stream_thread.start()

            print(f"Streaming started - Method: {self.stream_method}")

        except Exception as e:
            print(f"Failed to start streaming: {e}")
            raise
    
    def stop_streaming(self) -> None:
        """Stop streaming and cleanup resources"""
        if not self.streaming:
            return
            
        print("Stopping streaming...")
        self.streaming = False
        
        # Wait for streaming thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
        
        # Cleanup memory interfaces
        self._cleanup_memory_interfaces()
        
        print("Streaming stopped")
    
    def _initialize_memory_interfaces(self) -> None:
        """Initialize platform-specific memory interfaces"""
        width = self.config['width']
        height = self.config['height']
        
        if self.is_macos:
            # Initialize Syphon (using existing syphon_utils.py)
            try:
                from syphon_utils import SyphonUtils
                
                self.syphon_handler = SyphonUtils(
                    sender_name=self.output_mem_name,
                    input_name=self.input_mem_name,
                    control_name=None,
                    width=width,
                    height=height,
                    debug=False
                )
                self.syphon_handler.start()
                print(f"Syphon initialized - Input: {self.input_mem_name}, Output: {self.output_mem_name}")
                
            except Exception as e:
                print(f"Failed to initialize Syphon: {e}")
                raise
        else:
            # Initialize SharedMemory (same pattern as your current version)
            try:
                # Input memory (from TouchDesigner)
                self.input_memory = shared_memory.SharedMemory(name=self.input_mem_name)
                print(f"Connected to input SharedMemory: {self.input_mem_name}")
                
                # Output memory (to TouchDesigner) - try to connect first, create if not exists
                frame_size = width * height * 3  # RGB
                try:
                    # Try to connect to existing memory first (like main_sdtd.py)
                    self.output_memory = shared_memory.SharedMemory(name=self.output_mem_name)
                    print(f"Connected to existing output SharedMemory: {self.output_mem_name}")
                except FileNotFoundError:
                    # Create new if doesn't exist
                    self.output_memory = shared_memory.SharedMemory(
                        name=self.output_mem_name, 
                        create=True, 
                        size=frame_size
                    )
                    print(f"Created new output SharedMemory: {self.output_mem_name}")
                
                # ControlNet memory (per-frame updates)
                try:
                    self.control_memory = shared_memory.SharedMemory(name=self.control_mem_name)
                    print(f"Connected to ControlNet SharedMemory: {self.control_mem_name}")
                except FileNotFoundError:
                    print(f"ControlNet SharedMemory not found: {self.control_mem_name} (will create if needed)")
                    self.control_memory = None
                
                # ControlNet processed output memory (to send pre-processed image back to TD)
                # Create output memory for processed ControlNet with -cn-processed suffix
                control_processed_mem_name = self.output_mem_name + '-cn-processed'
                try:
                    # Try to connect to existing memory first
                    self.control_processed_memory = shared_memory.SharedMemory(name=control_processed_mem_name)
                    print(f"Connected to existing ControlNet processed output SharedMemory: {control_processed_mem_name}")
                except FileNotFoundError:
                    # Create new if doesn't exist
                    self.control_processed_memory = shared_memory.SharedMemory(
                        name=control_processed_mem_name,
                        create=True,
                        size=frame_size  # Same size as main output
                    )
                    print(f"Created new ControlNet processed output SharedMemory: {control_processed_mem_name}")
                
                # IPAdapter memory (OSC-triggered updates)
                try:
                    self.ipadapter_memory = shared_memory.SharedMemory(name=self.ipadapter_mem_name)
                    print(f"Connected to IPAdapter SharedMemory: {self.ipadapter_mem_name}")
                except FileNotFoundError:
                    print(f"IPAdapter SharedMemory not found: {self.ipadapter_mem_name} (will create if needed)")
                    self.ipadapter_memory = None
                
            except Exception as e:
                print(f"Failed to initialize SharedMemory: {e}")
                raise
    
    def _cleanup_memory_interfaces(self) -> None:
        """Cleanup memory interfaces"""
        try:
            if self.syphon_handler:
                self.syphon_handler.stop()
                self.syphon_handler = None
                
            if self.input_memory:
                self.input_memory.close()
                self.input_memory = None
                
            if self.output_memory:
                self.output_memory.close()
                self.output_memory.unlink()  # Delete the shared memory
                self.output_memory = None
                
            if self.control_memory:
                self.control_memory.close()
                self.control_memory = None
                
            if self.control_processed_memory:
                self.control_processed_memory.close()
                self.control_processed_memory.unlink()  # Delete the shared memory
                self.control_processed_memory = None
                
            if self.ipadapter_memory:
                self.ipadapter_memory.close()
                self.ipadapter_memory = None
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def _streaming_loop(self) -> None:
        """
        Main streaming loop - processes frames as fast as possible.
        
        Key insight: The wrapper.img2img() call runs as fast as it can.
        You don't manually control the speed - the wrapper handles timing internally.
        """
        print("Starting streaming loop...")
        
        frame_time_accumulator = 0.0
        fps_report_interval = 1.0  # Report FPS every second
        
        while self.streaming:
            try:
                # CRITICAL: Check pause state FIRST before any processing
                if self.paused:
                    if not self.process_frame:
                        time.sleep(0.001)  # Brief sleep if paused and no frame requested
                        continue
                    else:
                        # Reset acknowledgment flag when starting frame processing
                        self.frame_acknowledged = False
                        self._frames_processed_in_pause += 1
                        
                        # Log useful info every 10th frame 
                        if self._frames_processed_in_pause % 10 == 0:
                            logger.info(f"🎬 PAUSE: Frame {self._frames_processed_in_pause} | FPS: {self.current_fps:.1f}")
                        
                        # CRITICAL: Reset process_frame IMMEDIATELY after confirming we're processing
                        # This must happen INSIDE the pause block, not outside
                        self.process_frame = False
                else:
                    # In continuous mode, we don't need to reset process_frame
                    pass
                
                loop_start = time.time()

                # Process through StreamDiffusion - mode determines which method to call
                if self.mode == "txt2img":
                    # txt2img mode: generate from prepared noise (respects seed)
                    # Use init_noise instead of random noise to respect seed parameter
                    stream = self.wrapper.stream
                    batch_size = 1

                    # Check if seed is -1 (randomize every frame)
                    # Get current seed from config (updated via OSC)
                    current_seed = self.config.get('seed', -1)

                    if current_seed == -1:
                        # Generate fresh random noise every frame (random seed)
                        adjusted_noise = torch.randn(
                            (batch_size, 4, stream.latent_height, stream.latent_width),
                            device=stream.device,
                            dtype=stream.dtype
                        )
                    else:
                        # Generate noise from seed (consistent image every frame)
                        generator = torch.Generator(device=stream.device)
                        generator.manual_seed(current_seed)
                        adjusted_noise = torch.randn(
                            (batch_size, 4, stream.latent_height, stream.latent_width),
                            device=stream.device,
                            dtype=stream.dtype,
                            generator=generator
                        )

                    # Apply latent preprocessing hooks before prediction
                    adjusted_noise = stream._apply_latent_preprocessing_hooks(adjusted_noise)

                    # Call predict_x0_batch with prepared noise
                    x_0_pred_out = stream.predict_x0_batch(adjusted_noise)

                    # Apply latent postprocessing hooks
                    x_0_pred_out = stream._apply_latent_postprocessing_hooks(x_0_pred_out)

                    # Store for latent feedback processors
                    stream.prev_latent_result = x_0_pred_out.detach().clone()

                    # Decode to image
                    x_output = stream.decode_image(x_0_pred_out).detach().clone()

                    # Apply image postprocessing hooks
                    x_output = stream._apply_image_postprocessing_hooks(x_output)

                    # Postprocess to desired output format
                    output_image = self.wrapper.postprocess_image(x_output, output_type=self.wrapper.output_type)
                else:
                    # img2img mode: get input frame and process
                    input_image = self._get_input_frame()
                    if input_image is None:
                        time.sleep(0.001)  # Brief sleep if no input
                        continue

                    # FRAME CAPTURE: Capture raw input frame
                    if self.frame_capturer.should_capture():
                        self.frame_capturer.capture_stage("input", input_image, stage_order=0)

                    # Convert input from uint8 [0,255] to float [0,1] like main_sdtd.py
                    if input_image.dtype == np.uint8:
                        input_image = input_image.astype(np.float32) / 255.0

                    # Process ControlNet frame (per-frame if enabled)
                    self._process_controlnet_frame()

                    # Process IPAdapter frame (only on OSC request)
                    self._process_ipadapter_frame()

                    # Transform input image
                    output_image = self.wrapper.img2img(input_image)

                # FRAME CAPTURE: Capture final output frame
                if self.frame_capturer.should_capture():
                    self.frame_capturer.capture_stage("output", output_image, stage_order=7)

                # Send output frame to TouchDesigner
                self._send_output_frame(output_image)
                
                # Calculate FPS based on actual frame output timing (not loop timing)
                frame_output_time = time.time()
                if self.last_frame_output_time > 0:
                    frame_interval = frame_output_time - self.last_frame_output_time
                    instantaneous_fps = 1.0 / frame_interval if frame_interval > 0 else 0.0
                    # Smooth the FPS calculation
                    self.current_fps = self.current_fps * self.fps_smoothing + instantaneous_fps * (1 - self.fps_smoothing)
                self.last_frame_output_time = frame_output_time
                
                # Update frame counters
                self.frame_count += 1
                self.total_frame_count += 1

                # FRAME CAPTURE: Increment frame counter (must be after processing)
                self.frame_capturer.increment_frame()

                # Auto-shutdown after frame capture completes
                if self.frame_capturer.enabled and self.frame_capturer.is_complete:
                    print("\n[FRAME CAPTURE] Complete! Stopping streaming...")
                    self.streaming = False
                    break
                
                # Update performance metrics for display
                loop_end = time.time()
                loop_time = loop_end - loop_start
                frame_time_accumulator += loop_time
                
                # Send OSC messages EVERY FRAME (like main_sdtd.py - critical for TD connection!)
                if self.osc_handler:
                    self.osc_handler.send_message('/framecount', self.total_frame_count)
                    # Only send frame_ready in continuous mode, NOT in pause mode
                    if not self.paused:
                        self.osc_handler.send_message('/frame_ready', self.total_frame_count)
                    
                    # Send processed ControlNet memory name (only once at startup or when needed)
                    if not self._sent_processed_cn_name and self.control_processed_memory:
                        processed_cn_name = self.output_mem_name + '-cn-processed'
                        self.osc_handler.send_message('/stream-info/controlnet-processed-name', processed_cn_name)
                        self._sent_processed_cn_name = True
                
                # Report FPS periodically using the correct frame output FPS
                if frame_time_accumulator >= fps_report_interval:
                    # Clear the line properly and write new status
                    status_line = f"Streaming | FPS: {self.current_fps:.1f} | Frame: {self.frame_count}"
                    print(f"\r{' ' * 80}\r{status_line}", end='', flush=True)
                    
                    # Reset counters
                    frame_time_accumulator = 0.0
                    self.frame_count = 0
                
                # Send FPS EVERY FRAME (like main_sdtd.py) - report actual measured FPS
                if self.osc_handler and self.current_fps > 0:
                    self.osc_handler.send_message('/stream-info/fps', self.current_fps)
                
                # Handle frame acknowledgment for pause mode synchronization
                if self.paused:
                    # Send frame completion signal to TouchDesigner
                    if self.osc_handler:
                        self.osc_handler.send_message('/frame_ready', self.total_frame_count)
                    
                    # Wait for TouchDesigner to acknowledge it has processed the frame
                    acknowledgment_timeout = time.time() + 5.0  # 5 second timeout
                    while not self.frame_acknowledged and time.time() < acknowledgment_timeout:
                        time.sleep(0.001)  # Small sleep to prevent busy waiting
                    
                    if not self.frame_acknowledged:
                        logger.warning(f"⚠️  TIMEOUT: No /frame_ack after 5s (frame {self._frames_processed_in_pause})")
                    
                    # Reset for next frame
                    self.frame_acknowledged = False
                
            except Exception as e:
                print(f"\\nError in streaming loop: {e}")
                # Continue streaming unless explicitly stopped
                if self.streaming:
                    time.sleep(0.1)  # Brief pause before retrying
                else:
                    break
        
        print("\\nStreaming loop ended")
    
    def _get_input_frame(self) -> Optional[np.ndarray]:
        """Get input frame from TouchDesigner (platform-specific) or synthetic generator"""
        try:
            # SYNTHETIC INPUT MODE: Use generated fractal noise for frame capture testing
            if self.synthetic_input is not None:
                return self.synthetic_input.generate_frame()

            # NORMAL MODE: Get from TouchDesigner
            if self.is_macos and self.syphon_handler:
                # Get frame from Syphon
                frame = self.syphon_handler.capture_input_frame()
                return frame

            elif self.input_memory:
                # Get frame from SharedMemory
                width = self.config['width']
                height = self.config['height']

                # Create numpy array view of shared memory
                frame = np.ndarray(
                    (height, width, 3),
                    dtype=np.uint8,
                    buffer=self.input_memory.buf
                )

                return frame.copy()  # Copy to avoid memory issues
            
            return None
            
        except Exception as e:
            print(f"Error getting input frame: {e}")
            return None
    
    def _send_output_frame(self, output_image: Union[Image.Image, torch.Tensor, np.ndarray]) -> None:
        """Send output frame to TouchDesigner (platform-specific)"""
        try:
            # Convert output to numpy array if needed
            if isinstance(output_image, Image.Image):
                frame_np = np.array(output_image)
            elif isinstance(output_image, torch.Tensor):
                frame_np = output_image.cpu().numpy()
                if frame_np.shape[0] == 3:  # CHW -> HWC
                    frame_np = np.transpose(frame_np, (1, 2, 0))
            else:
                frame_np = output_image
                
            # Ensure proper scaling like main_sdtd.py (line 916)
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            elif frame_np.dtype != np.uint8:
                frame_np = frame_np.astype(np.uint8)
            
            if self.is_macos and self.syphon_handler:
                # Send via Syphon
                self.syphon_handler.send_frame(frame_np)
                
            elif self.output_memory:
                # Send via SharedMemory
                output_array = np.ndarray(
                    frame_np.shape, 
                    dtype=np.uint8, 
                    buffer=self.output_memory.buf
                )
                np.copyto(output_array, frame_np)
                
        except Exception as e:
            print(f"Error sending output frame: {e}")
    
    def _send_processed_controlnet_frame(self, processed_tensor: Optional[torch.Tensor]) -> None:
        """Send processed ControlNet frame to TouchDesigner via shared memory"""
        if not self.control_processed_memory or processed_tensor is None:
            return
            
        try:
            # Convert tensor to numpy array (similar to _send_output_frame)
            if isinstance(processed_tensor, torch.Tensor):
                frame_np = processed_tensor.cpu().numpy()
                
                # Handle tensor format: CHW -> HWC
                if frame_np.ndim == 4 and frame_np.shape[0] == 1:
                    frame_np = frame_np.squeeze(0)  # Remove batch dimension
                if frame_np.ndim == 3 and frame_np.shape[0] == 3:
                    frame_np = np.transpose(frame_np, (1, 2, 0))  # CHW -> HWC
            else:
                frame_np = processed_tensor
                
            # Ensure proper scaling (0-1 range to 0-255)
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            elif frame_np.dtype != np.uint8:
                frame_np = frame_np.astype(np.uint8)
            
            # Send via SharedMemory
            if self.control_processed_memory:
                output_array = np.ndarray(
                    frame_np.shape, 
                    dtype=np.uint8, 
                    buffer=self.control_processed_memory.buf
                )
                np.copyto(output_array, frame_np)
                
        except Exception as e:
            print(f"Error sending processed ControlNet frame: {e}")
    
    def pause_streaming(self) -> None:
        """Pause streaming (frame processing continues only on process_frame requests)"""
        if not self.paused:  # Only log state changes
            self.paused = True
            self._frames_processed_in_pause = 0  # Reset counter

            logger.info("🔴 STREAMING PAUSED - waiting for /process_frame commands")
    
    def resume_streaming(self) -> None:
        """Resume streaming (continuous frame processing)"""
        if self.paused:  # Only log state changes
            self.paused = False
            logger.info(f"🟢 STREAMING RESUMED - processed {self._frames_processed_in_pause} frames while paused")
    
    def process_single_frame(self) -> None:
        """Process a single frame when paused"""
        if self.paused:
            self.process_frame = True
            # Only log occasionally to reduce spam
        else:
            logger.warning("⚠️  process_single_frame called but streaming is not paused")
    
    def acknowledge_frame(self) -> None:
        """Acknowledge frame processing completion (called by TouchDesigner)"""
        self.frame_acknowledged = True
    
    def _process_controlnet_frame(self) -> None:
        """Process ControlNet frame data (per-frame updates)"""
        if not self.control_memory or not self.config.get('use_controlnet', False):
            return
            
        try:
            width = self.config['width']
            height = self.config['height']
            
            # Read ControlNet frame from shared memory
            control_frame = np.ndarray(
                (height, width, 3), 
                dtype=np.uint8, 
                buffer=self.control_memory.buf
            )
            
            # Convert to float [0,1] and pass to wrapper
            if control_frame.dtype == np.uint8:
                control_frame = control_frame.astype(np.float32) / 255.0
            
            # Update ControlNet image in wrapper (index 0 for first ControlNet)
            self.wrapper.update_control_image(0, control_frame)
            
            # IMPORTANT: After processing, extract the pre-processed image and send it back to TD
            # The processed image is now available in the controlnet module
            try:
                if (hasattr(self.wrapper, 'stream') and 
                    hasattr(self.wrapper.stream, '_controlnet_module') and
                    self.wrapper.stream._controlnet_module is not None):
                    
                    controlnet_module = self.wrapper.stream._controlnet_module
                    
                    # Check if we have processed images available
                    if (hasattr(controlnet_module, 'controlnet_images') and 
                        len(controlnet_module.controlnet_images) > 0 and
                        controlnet_module.controlnet_images[0] is not None):
                        
                        processed_tensor = controlnet_module.controlnet_images[0]
                        
                        # Send the processed ControlNet image back to TouchDesigner
                        self._send_processed_controlnet_frame(processed_tensor)
                        
            except Exception as processed_error:
                logger.debug(f"Could not extract processed ControlNet image: {processed_error}")
            
        except Exception as e:
            logger.error(f"Error processing ControlNet frame: {e}")
    
    def _process_ipadapter_frame(self) -> None:
        """Process IPAdapter frame data (OSC-triggered updates only)"""
        if not self.ipadapter_memory:
            logger.debug(f"❌ IPAdapter SharedMemory not connected: {self.ipadapter_mem_name}")
            return
        if not self.config.get('use_ipadapter', False):
            logger.debug("❌ IPAdapter disabled in config")
            return
            
        if not self.ipadapter_update_requested:
            return
            
        try:
            width = self.config['width']
            height = self.config['height']
            
            # Read IPAdapter frame from shared memory
            ipadapter_frame = np.ndarray(
                (height, width, 3), 
                dtype=np.uint8, 
                buffer=self.ipadapter_memory.buf
            )
            
            # Convert to float [0,1] and pass to wrapper
            if ipadapter_frame.dtype == np.uint8:
                ipadapter_frame = ipadapter_frame.astype(np.float32) / 255.0
            
            # Update IPAdapter config FIRST (like img2img example)
            # Use live scale value updated by OSC, not static config
            current_scale = self._current_ipadapter_scale
            logger.info(f"📊 Using live IPAdapter scale: {current_scale}")
            self.wrapper.update_stream_params(ipadapter_config={'scale': current_scale})
            
            # THEN update the style image (following img2img pattern)
            self.wrapper.update_style_image(ipadapter_frame)
            
            # CRITICAL FIX: Trigger prompt re-blending to apply the new IPAdapter embeddings
            # The embedding hooks only run during _apply_prompt_blending, so we must trigger it
            # after updating the IPAdapter image to concatenate the new embeddings with prompts
            if hasattr(self.wrapper.stream._param_updater, '_current_prompt_list') and \
               self.wrapper.stream._param_updater._current_prompt_list:
                # Get current interpolation method or use default
                interpolation_method = getattr(self.wrapper.stream._param_updater, 
                                             '_last_prompt_interpolation_method', 'slerp')
                # Re-apply prompt blending to trigger embedding hooks
                self.wrapper.stream._param_updater._apply_prompt_blending(interpolation_method)
                logger.info(f"🔄 Re-applied prompt blending to incorporate new IPAdapter embeddings")
            
            # Debug: Check if embeddings were cached
            cached_embeddings = self.wrapper.stream._param_updater.get_cached_embeddings("ipadapter_main")
            if cached_embeddings is not None:
                logger.info(f"✅ IPAdapter config+image updated! Scale: {current_scale}, Embeddings: {cached_embeddings[0].shape}")
            else:
                logger.error("❌ IPAdapter embeddings NOT cached!")
            
            # Reset the update flag
            self.ipadapter_update_requested = False
            logger.info("🎨 IPAdapter config and image updated from SharedMemory")
            
        except Exception as e:
            logger.error(f"Error processing IPAdapter frame: {e}")
    
    def request_ipadapter_update(self) -> None:
        """Request IPAdapter image update on next frame (called via OSC)"""
        self.ipadapter_update_requested = True
        logger.info("📨 IPAdapter update requested")

    def set_mode(self, mode: str) -> None:
        """
        Switch between txt2img and img2img modes.

        Args:
            mode: Either "txt2img" or "img2img"
        """
        if mode not in ['txt2img', 'img2img']:
            logger.error(f"Invalid mode: {mode}. Must be 'txt2img' or 'img2img'")
            return

        if mode == self.mode:
            logger.info(f"Already in {mode} mode")
            return

        logger.info(f"Switching from {self.mode} to {mode} mode")
        self.mode = mode

        # txt2img mode requirements
        if mode == 'txt2img':
            logger.warning("⚠️  txt2img mode requires cfg_type='none' - verify your config!")

    def get_stream_state(self) -> Dict[str, Any]:
        """Get current streaming state and parameters"""
        return {
            'streaming': self.streaming,
            'paused': self.paused,
            'fps': self.current_fps,
            'frame_count': self.frame_count,
            'stream_method': self.stream_method,
            'wrapper_state': self.wrapper.get_stream_state() if hasattr(self.wrapper, 'get_stream_state') else {},
            'controlnet_connected': self.control_memory is not None,
            'controlnet_processed_connected': self.control_processed_memory is not None,
            'ipadapter_connected': self.ipadapter_memory is not None
        }