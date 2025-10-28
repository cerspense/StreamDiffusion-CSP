"""
TouchDesigner StreamDiffusion Main Entry Point

Minimal main script leveraging the full LivePeer StreamDiffusion fork capabilities.
Replaces the complex main_sdtd.py with a cleaner, config-driven approach.

Reads configuration from td_config.yaml (single source of truth)
"""

import os
import sys
import json
import signal
import argparse

# Add StreamDiffusion to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from td_manager import TouchDesignerManager
from td_osc_handler import OSCParameterHandler


class StreamDiffusionTD:
    """Main application class"""

    def __init__(self, debug_capture_frames: bool = False):
        print("StreamDiffusionTD v3.0.0 - LivePeer Fork Edition")
        print("=" * 60)

        # Load YAML config - single source of truth
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_config_path = os.path.join(script_dir, "td_config.yaml")

        from streamdiffusion.config import load_config
        yaml_config = load_config(yaml_config_path)

        # Add debug frame capture flag to config
        yaml_config['debug_capture_frames'] = debug_capture_frames

        # Get TouchDesigner-specific settings from YAML
        td_settings = yaml_config.get('td_settings', {})
        input_mem = td_settings.get('input_mem_name', 'input_stream')

        # Make output memory name unique (prevents conflicts & different resolutions)
        import time
        base_output_name = td_settings.get('output_mem_name', 'sd_to_td')
        output_mem = f"{base_output_name}_{int(time.time())}"

        # Get OSC ports from YAML
        listen_port = td_settings.get('osc_transmit_port', 8247)  # Python listens
        transmit_port = td_settings.get('osc_receive_port', 8248)  # Python transmits

        # DEBUG MODE: Use different ports to avoid conflicts with running TouchDesigner
        if debug_capture_frames:
            listen_port = 9999  # Different port for debug mode
            transmit_port = 9998
            print(f"[DEBUG MODE] Using alternate OSC ports: Listen {listen_port}, Transmit {transmit_port}")
        
        # DEBUG: Print the actual config being loaded
        print("=" * 80)
        print("ACTUAL CONFIG BEING LOADED:")
        print("=" * 80)
        import json
        print(json.dumps(yaml_config, indent=2))
        print("=" * 80)
        
        # Initialize core manager with clean YAML config
        self.manager = TouchDesignerManager(yaml_config, input_mem, output_mem)
        
        # Initialize OSC handler after manager
        self.osc_handler = OSCParameterHandler(
            manager=self.manager,
            main_app=self,  # Pass main app for shutdown handling
            listen_port=listen_port,
            transmit_port=transmit_port
        )
        
        # Now set OSC handler in manager
        self.manager.osc_handler = self.osc_handler
        
        # Application shutdown state
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"OSC: Listen {listen_port} -> Transmit {transmit_port}")
        print(f"Memory: {input_mem} -> {output_mem}")
        print(f"Platform: {self.manager.stream_method}")
        print("=" * 60)
    
    
    def start(self):
        """Start the application"""
        try:
            # Start OSC communication
            self.osc_handler.start()
            
            print("Ready! Send /start_streaming via OSC or manually call manager.start_streaming()")
            print("Tips:")
            print("   - Use /prompt_list for prompt blending")
            print("   - Use /controlnets for multi-ControlNet")
            print("   - Parameters are batched for optimal performance")
            print("   - Ctrl+C to exit gracefully")
            print()
            
            # Send initial OSC status messages to TouchDesigner
            self.osc_handler.send_message("/server_active", 1)
            self.osc_handler.send_message('/stream-info/output-name', self.manager.output_mem_name)
            
            # Auto-start streaming (matches your current main_sdtd.py behavior)  
            self.manager.start_streaming()
            
            # Keep main thread alive
            self._wait_for_shutdown()
            
        except KeyboardInterrupt:
            print("\\nInterrupted by user")
        except Exception as e:
            print(f"ERROR: {e}")
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        print("Shutting down...")

        try:
            self.manager.stop_streaming()
            self.osc_handler.stop()
            print("Shutdown complete")
        except Exception as e:
            print(f"Shutdown error: {e}")
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print(f"\\nReceived signal {sig}")
        self.shutdown()
        sys.exit(0)

    def request_shutdown(self):
        """Request application shutdown (called by OSC /stop command)"""
        print("\\nStop command received via OSC")
        self.shutdown_requested = True
    
    def _wait_for_shutdown(self):
        """Wait for shutdown signal"""
        try:
            # Keep main thread alive
            while not self.shutdown_requested:
                # Could add periodic status reporting here
                import time
                time.sleep(0.1)  # Check shutdown flag more frequently
                
                # Optional: Print status periodically
                # status = self.manager.get_stream_state()
                # if status['streaming']:
                #     print(f"\\r🎬 Streaming | FPS: {status['fps']:.1f} | Frames: {status['frame_count']}", end='', flush=True)
                
        except KeyboardInterrupt:
            raise


def main():
    """Main entry point - reads from td_config.yaml"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='StreamDiffusion TouchDesigner Backend')
    parser.add_argument(
        '--debug-capture-frames',
        action='store_true',
        help='Enable frame capture debug mode (skips 30 frames, captures 4 frames at all pipeline stages)'
    )
    args = parser.parse_args()

    # Create and start application (no longer needs stream_config.json)
    app = StreamDiffusionTD(debug_capture_frames=args.debug_capture_frames)
    app.start()


if __name__ == "__main__":
    main()