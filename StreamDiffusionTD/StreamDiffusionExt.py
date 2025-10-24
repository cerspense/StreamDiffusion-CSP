"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""
import os
import subprocess
import socket
import json
import time
import traceback
from functools import reduce
import ast
import datetime
import numpy as np
import shutil
import re
from urllib.parse import urlparse
from TDStoreTools import StorageManager
import TDFunctions as TDF
import platform
import asyncio
import requests
import base64
import copy

# ============================================================================
# CONSTANTS
# ============================================================================

# Rate limiting for Daydream API parameter updates
# 12 FPS = 83ms between requests (matches HTML implementation)
PARAM_UPDATE_FPS = 12
PARAM_UPDATE_DELAY = 1.0 / PARAM_UPDATE_FPS  # 0.083 seconds

# ============================================================================

param_json_map = {
    "Modelid": "model_id_or_path",
    "Tindexlist": "t_index_list",
    "Promptdict0concept": "prompt",
    "Negprompt": "negative_prompt",
    "Framesize": "frame_buffer_size",
    "Width": "width",
    "Height": "height",
    "Acceleration": "acceleration",
    "Denoisebatch": "use_denoising_batch",
    "Seeddict0seedval": "seed",
    "Cfgtype": "cfg_type",
    "Guidancescale": "guidance_scale",
    "Delta": "delta",
    "Addnoise": "do_add_noise",
    "Imagefilter": "enable_similar_image_filter",
    "Filterthresh": "similar_image_filter_threshold",
    "Maxskipframe": "similar_image_filter_max_skip_frame",
    "Oscinport": "osc_in_port",
    "Oscoutport": "osc_out_port",
    "Sdmode":"sdmode",
    "Customlcm" : "lcm_lora_id",
    "Streamoutname" : "input_mem_name",
    # "Enablefx" : "use_cached_attn", #Fx
    "Gpuid": "gpu_id",
    "Forcemodeltype": "sd_model_type",   
    "Limitfps": "max_fps",
    "Hfcache": "hf_cache",
    "Ipadapterenable": "use_ipadapter",
    "Ipadapterscale": "ipadapter_scale",
}
model_preset_params = [
    "Modelid",
    "Customlcm",
    "Loradictblock",
    "Uselora",
    "Usecustomlcm",
    "Acceleration",
    "Tindexblock",
    "Skiplcm",
    "Width",
    "Height",
    # "Textualinvblock",
]


class StreamDiffusionExt: 
    """
    StreamDiffusionExt description
    """
    def __init__(self, ownerComp):
        """
        Initializes the StarterExt class. Sets the owner component and calls the functions to create basic parameters and setup tables.
        """
        self.ownerComp = ownerComp
        self.message_box_open = False
        self.last_opened_time = time.time()
        self.logger = op('Logger').ext.Logger

        self.setup_project_info_table()
        # self.setup_par_details_table()
        self.Showbuiltin()
        self.Sethfcache()

        self.ownerComp.par.Streamactive = False
        self.ownerComp.par.Serveractive = False

        self.last_stop_time = time.time()
        self.Basefolder(force = True)
        self.ownerComp.par.Log3.label = ''
        self.ismac = platform.system() == 'Darwin'

        # Initialize shared memory change detection
        self.setup_shared_memory_change_detection()

        # Loopback synchronization variables
        self.last_processframe_time = 0
        self.processframe_debounce_ms = 40

        # Feedback safe state tracking
        self.last_feedback_safe_state = None
        self.feedback_safe_paused_backend = False  # Track if we paused the backend
        self.last_feedback_safe_disable_time = 0  # Track when feedback safe was last disabled

        self.Apikey()
        self.logger.log("StreamDiffusionTD Loaded", level='INFO')

        # Initialize CN menu based on current model
        self.update_cn_id_menus()
        
    def get_version(self):
        return self.ownerComp.par.Txversion.eval()

    def update_cn_id_menus(self):
        """
        Updates CN sequence block Id and Preprocessor parameter menus based on backend and model.
        Sets FULL HuggingFace IDs in menuNames (what gets sent to backend).
        Auto-updates preprocessor if Autopreprocess is enabled.
        """
        if not self.ownerComp.par.Autoupdatecn.eval():
            return
        backend = self.ownerComp.par.Backend.eval()

        if backend == 'Daydream':
            daydream_model = self.ownerComp.par.Daydreammodel.eval()

            # Daydream backend - use official API model IDs
            if daydream_model == "prompthero/openjourney-v4":  # SD 1.5
                menu_names = [
                    'lllyasviel/control_v11f1p_sd15_depth',
                    'lllyasviel/control_v11f1e_sd15_tile',
                    'lllyasviel/control_v11p_sd15_canny'
                ]
                menu_labels = ['Depth', 'Tile', 'Canny']

            elif daydream_model == "stabilityai/sd-turbo":  # SD 2.1
                menu_names = [
                    'thibaud/controlnet-sd21-openpose-diffusers',
                    'thibaud/controlnet-sd21-hed-diffusers',
                    'thibaud/controlnet-sd21-canny-diffusers',
                    'thibaud/controlnet-sd21-depth-diffusers',
                    'thibaud/controlnet-sd21-color-diffusers'
                ]
                menu_labels = ['OpenPose', 'HED', 'Canny', 'Depth', 'Color']

            elif daydream_model == "stabilityai/sdxl-turbo":  # SDXL
                menu_names = [
                    'xinsir/controlnet-depth-sdxl-1.0',
                    'xinsir/controlnet-canny-sdxl-1.0',
                    'xinsir/controlnet-tile-sdxl-1.0'
                ]
                menu_labels = ['Depth', 'Canny', 'Tile']

            else:
                menu_names = [
                    'xinsir/controlnet-canny-sdxl-1.0',
                    'xinsir/controlnet-depth-sdxl-1.0'
                ]
                menu_labels = ['Canny (SDXL)', 'Depth (SDXL)']
        else:
            # Local backend - LivePeer fork supports these models
            current_model = self.ownerComp.par.Modelid.eval().lower()
            config_type = self.get_config_type(current_model)

            if config_type and config_type.startswith("sdxl"):
                menu_names = [
                    'xinsir/controlnet-canny-sdxl-1.0',
                    'xinsir/controlnet-depth-sdxl-1.0',
                    'xinsir/controlnet-openpose-sdxl-1.0',
                    'xinsir/controlnet-scribble-sdxl-1.0',
                    'xinsir/controlnet-tile-sdxl-1.0',
                    'diffusers/controlnet-depth-sdxl-1.0'
                ]
                menu_labels = ['Canny', 'Depth', 'OpenPose', 'Scribble', 'Tile', 'Depth (Official)']
            elif config_type == "sd21":
                menu_names = [
                    'thibaud/controlnet-sd21-canny-diffusers',
                    'thibaud/controlnet-sd21-depth-diffusers',
                    'thibaud/controlnet-sd21-openpose-diffusers',
                    'thibaud/controlnet-sd21-hed-diffusers',
                    'thibaud/controlnet-sd21-scribble-diffusers',
                    'thibaud/controlnet-sd21-lineart-diffusers',
                    'thibaud/controlnet-sd21-normalbae-diffusers',
                    'thibaud/controlnet-sd21-zoedepth-diffusers',
                    'thibaud/controlnet-sd21-ade20k-diffusers',
                    'thibaud/controlnet-sd21-color-diffusers'
                ]
                menu_labels = ['Canny', 'Depth', 'OpenPose', 'HED', 'Scribble', 'Lineart', 'Normal', 'ZoeDepth', 'Segmentation', 'Color']
            else:
                menu_names = [
                    'lllyasviel/control_v11p_sd15_canny',
                    'lllyasviel/control_v11f1p_sd15_depth',
                    'lllyasviel/control_v11p_sd15_openpose',
                    'lllyasviel/sd-controlnet-scribble',
                    'lllyasviel/control_v11p_sd15_softedge',
                    'lllyasviel/control_v11f1e_sd15_tile'
                ]
                menu_labels = ['Canny v1.1', 'Depth v1.1', 'OpenPose v1.1', 'Scribble', 'Soft Edge', 'Tile']

        # Update each block in the CN sequence
        for block in self.ownerComp.par.Cn.sequence:
            block.par.Id.menuNames = menu_names
            block.par.Id.menuLabels = menu_labels

            # Auto-update preprocessor if Autopreprocess is enabled
            auto_preprocess_enabled = hasattr(self.ownerComp.par, 'Autopreprocess') and self.ownerComp.par.Autopreprocess.eval()

            if auto_preprocess_enabled:
                has_preprocessor = hasattr(block.par, 'Preprocessor')

                if has_preprocessor:
                    current_id = block.par.Id.eval()
                    detected_preprocessor = self.get_preprocessor_for_controlnet(current_id, backend)
                    block.par.Preprocessor.val = detected_preprocessor

        # Update dynamic CN parameters if Autoupdatecnpars is enabled
        self.update_cn_dynamic_parameters()

        # self.logger.log(f"Updated CN Id menus for {backend}: {menu_labels}", level='DEBUG')

    def Backend(self):
        """Called when Backend parameter changes (TD automatic callback)"""
        self.update_cn_id_menus()
        if self.ownerComp.par.Backend.eval() == 'Daydream':
            self.ownerComp.par.Cnheader.label = 'ControlNet [ Daydream: TOPin1 + Preprocessor ]'
        else:
            self.ownerComp.par.Cnheader.label = 'ControlNet [ Local: TOPin2 + Preprocessor ]'

    def Daydreammodel(self):
        """Called when Daydreammodel parameter changes (TD automatic callback)"""
        self.update_cn_id_menus()

    def Updatesettings(self):
        # print(f"\n\nUpdatesettings called")
        """
        Sends the four settings (Prompt, Delta, Guidance Scale, and Negative Prompt) to the OSC Out DAT.
        """
        if self.ownerComp.par.Streamactive:
            # Sending the delta value
            delta_value = self.ownerComp.par.Delta.eval()
            self.send_parameter_update('delta', delta_value)
            # Sending the guidance scale value
            guidance_scale_value = self.ownerComp.par.Guidancescale.eval()
            self.send_parameter_update('guidance_scale', guidance_scale_value)
            # Sending the negative prompt
            # negative_prompt_address = '/negative_prompt'
            # negative_prompt_message = self.ownerComp.par.Negprompt.eval()
            # osc_out.sendOSC(negative_prompt_address, [negative_prompt_message])
            self.Tindexblock()
            self.set_interpolation()  
            if self.ownerComp.par.Backend.eval() == 'Daydream':
                self.Cnblock()
            # self.Promptblock()
            self.Limitfps()

    def gather_full_config_for_daydream(self):
        """
        Gathers all current configuration parameters for Daydream stream creation.
        Returns complete pipeline_params dictionary based on TouchDesigner settings.
        """
        self.logger.log('Gathering full configuration for Daydream stream creation', level='INFO')

        config = {}

        try:
            # Daydream base model selection
            daydream_model = self.ownerComp.par.Daydreammodel.eval()

            # Determine correct pipeline_id based on model and IP adapter settings (from API docs)
            pipeline_map = {
                "stabilityai/sd-turbo": "pip_SD-turbo",  # TEMPORARY: Use old working pipeline - pip_qpUgXycjWF6YMeSL
                "stabilityai/sdxl-turbo": "pip_SDXL-turbo",
                "prompthero/openjourney-v4": "pip_SD15"
            }
            pipeline_id = pipeline_map.get(daydream_model, "pip_SD-turbo")

            # Check if Face ID is enabled for SDXL models (changes pipeline to faceid version)
            if (daydream_model == "stabilityai/sdxl-turbo" and
                hasattr(self.ownerComp.par, 'Ipfaceid') and
                self.ownerComp.par.Ipfaceid.eval()):
                pipeline_id = "pip_SDXL-turbo-faceid"

            config = {}
            config['model_id'] = daydream_model

            # Core generation parameters - will be set from prompt_list below
            config['negative_prompt'] = self.ownerComp.par.Negprompt.eval()
            config['guidance_scale'] = self.ownerComp.par.Guidancescale.eval()
            config['delta'] = self.ownerComp.par.Delta.eval()
            config['width'] = self.ownerComp.par.Width.eval()
            config['height'] = self.ownerComp.par.Height.eval()

            # CRITICAL: t_index_list from TouchDesigner blocks (prevents 30s reload)
            t_index_list = []
            for block in self.ownerComp.par.Tindexblock.sequence:
                step_value = block.par.Step.eval()
                t_index_list.append(step_value)
            config['t_index_list'] = t_index_list

            # Gather prompt data for stream creation (API expects 'prompt', not 'prompt_list')
            prompt_list = []
            for block in self.ownerComp.par.Promptdict.sequence:
                concept = block.par.Concept.eval()
                weight = block.par.Weight.eval()
                prompt_list.append([concept, weight])

            # Set main prompt for stream creation (API only wants 'prompt', not 'prompt_list')
            if prompt_list:
                config['prompt'] = prompt_list  # Use the weighted prompt list format
            else:
                config['prompt'] = [["a beautiful landscape", 1.0]]  # Fallback in correct format

            # NOTE: NOT setting 'prompt_list' to avoid duplicate fields in stream creation

            # Gather seed
            config['seed'] = self.ownerComp.par.Seed.eval() if hasattr(self.ownerComp.par, 'Seed') else 0

            # Skip LoRA configuration completely for troubleshooting
            # lora_dict = {}
            # if hasattr(self.ownerComp.par, 'Loradictblock'):
            #     for block in self.ownerComp.par.Loradictblock.sequence:
            #         lora_path = block.par.Lorapath.eval()
            #         lora_weight = block.par.Weight.eval()
            #         # Only include valid LoRA paths and non-zero weights
            #         if (lora_path and
            #             lora_path.strip() and
            #             lora_path != 'select_lora_from_dropdown' and
            #             lora_weight != 0):
            #             lora_dict[lora_path] = lora_weight
            # config['lora_dict'] = lora_dict

            # Gather ControlNet configuration - read directly from parameters
            controlnets = []
            if hasattr(self.ownerComp.par, 'Cn'):
                for block in self.ownerComp.par.Cn.sequence:
                    model_id = block.par.Id.eval()
                    cn_weight = block.par.Weight.eval()
                    cn_enabled = block.par.Enable.eval()
                    preprocessor = block.par.Preprocessor.eval() if hasattr(block.par, 'Preprocessor') else None

                    if cn_enabled and model_id and model_id != "none":
                        controlnet_config = {
                            "model_id": model_id,
                            "conditioning_scale": cn_weight,
                            "preprocessor": preprocessor if preprocessor else "canny",
                            "preprocessor_params": {},
                            "enabled": True,
                            "control_guidance_start": 0,
                            "control_guidance_end": 1
                        }
                        controlnets.append(controlnet_config)
            config['controlnets'] = controlnets

            # IP Adapter configuration - EXCLUDE for SD-turbo completely!
            if daydream_model == "stabilityai/sd-turbo":
                # SD-turbo does NOT support IP adapter - don't include it at all
                self.logger.log('SD-turbo detected - EXCLUDING ip_adapter from config', level='DEBUG')
            elif daydream_model in ["prompthero/openjourney-v4", "stabilityai/sdxl-turbo"]:
                # Only SD1.5 and SDXL support IP adapter
                if hasattr(self.ownerComp.par, 'Ipadapterenable') and self.ownerComp.par.Ipadapterenable.eval():
                    # Ensure scale is always a valid number
                    ip_adapter_scale = 0.7  # Default value
                    if hasattr(self.ownerComp.par, 'Ipadapterscale'):
                        scale_value = self.ownerComp.par.Ipadapterscale.eval()
                        if scale_value is not None and scale_value != "":
                            ip_adapter_scale = float(scale_value)

                    # Check if Face ID is enabled for SDXL-turbo-faceid pipeline
                    is_faceid = (daydream_model == "stabilityai/sdxl-turbo" and
                                hasattr(self.ownerComp.par, 'Ipfaceid') and
                                self.ownerComp.par.Ipfaceid.eval())

                    if is_faceid:
                        ip_adapter_scale = 1.0  # FaceID default scale
                        config['ip_adapter'] = {
                            "enabled": True,
                            "scale": ip_adapter_scale,
                            "type": "faceid",
                            "weight_type": "linear"
                        }
                        self.logger.log(f'IP Adapter (FaceID) enabled for {daydream_model}: scale={ip_adapter_scale}', level='DEBUG')
                    else:
                        # Regular SDXL IP adapter - NO type/weight_type fields!
                        config['ip_adapter'] = {
                            "enabled": True,
                            "scale": ip_adapter_scale
                        }
                        self.logger.log(f'IP Adapter enabled for {daydream_model}: scale={ip_adapter_scale}', level='DEBUG')
                else:
                    # CRITICAL: Explicitly disable IP adapter (API defaults to enabled for SDXL)
                    # API may require scale field even when disabled
                    config['ip_adapter'] = {
                        "enabled": False,
                        "scale": 0.0
                    }
                    self.logger.log(f'IP Adapter explicitly disabled for {daydream_model}', level='DEBUG')

            # Heavy parameters that trigger pipeline reload if not set at creation
            config['num_inference_steps'] = 50  # StreamDiffusion default
            config['acceleration'] = 'tensorrt'
            config['use_denoising_batch'] = True
            config['do_add_noise'] = True
            config['use_lcm_lora'] = True
            config['lcm_lora_id'] = 'latent-consistency/lcm-lora-sdv1-5'
            # Note: use_safety_checker parameter removed - not supported by current API

            # Image filtering (if available)
            if hasattr(self.ownerComp.par, 'Imagefilter'):
                config['enable_similar_image_filter'] = self.ownerComp.par.Imagefilter.eval()
            if hasattr(self.ownerComp.par, 'Filterthresh'):
                config['similar_image_filter_threshold'] = self.ownerComp.par.Filterthresh.eval()
            if hasattr(self.ownerComp.par, 'Maxskipframe'):
                config['similar_image_filter_max_skip_frame'] = int(self.ownerComp.par.Maxskipframe.eval())

            self.logger.log(f'Gathered configuration with {len(config)} parameters for pipeline {pipeline_id}', level='INFO')
            self.logger.log(f't_index_list: {config["t_index_list"]}', level='DEBUG')
            if config.get('controlnets'):
                self.logger.log(f'ControlNets: {len(config["controlnets"])} configured', level='DEBUG')

            # DETAILED DEBUG LOGGING - Show exactly what we're sending to API
            self.logger.log(f'=== FULL CONFIG DEBUG for {daydream_model} ===', level='DEBUG')
            self.logger.log(f'Pipeline ID: {pipeline_id}', level='DEBUG')
            self.logger.log(f'Model ID: {config.get("model_id")}', level='DEBUG')
            self.logger.log(f'IP Adapter in config: {"ip_adapter" in config}', level='DEBUG')
            if 'ip_adapter' in config:
                self.logger.log(f'IP Adapter config: {config["ip_adapter"]}', level='DEBUG')
            self.logger.log(f'ControlNets count: {len(config.get("controlnets", []))}', level='DEBUG')
            # self.logger.log(f'LoRA dict: {config.get("lora_dict", {})}', level='DEBUG')
            self.logger.log(f'=== END CONFIG DEBUG ===', level='DEBUG')

            return {
                'pipeline_id': pipeline_id,
                'pipeline_params': config
            }

        except Exception as e:
            self.logger.log(f'Error gathering full config for Daydream: {str(e)}', level='ERROR')
            return {}

    def handle_daydream_message(self, address, args):
        """
        Handle incoming OSC messages from Daydream manager.
        Called from oscin1_callbacks.py for /daydream/ addresses.
        """

        
        if address == '/daydream/request_config':
            self.logger.log('Daydream requested full configuration - sending now...', level='INFO')
            self.send_full_config_to_daydream()
        elif address == '/daydream/status':
            status_msg = args[0] if args else ''
            self.logger.log(f'Daydream status: {status_msg}', level='INFO')
        elif address == '/daydream/streamId':
            stream_id = args[0] if args else ''
            self.logger.log(f'Received Daydream Stream ID: {stream_id}', level='INFO')
        elif address == '/daydream/ingestUrl':
            ingest_url = args[0] if args else ''
            self.update_videostreamout_top(ingest_url)
        elif address == '/daydream/playbackUrl':
            playback_url = args[0] if args else ''
            self.update_webrender_top(playback_url)

    def send_osc_messages(self, message_dict):
        """
        Sends OSC messages to multiple addresses.

        Parameters:
        message_dict (dict): A dictionary where keys are OSC addresses and values are the messages to be sent.
        """
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')

            for address, message in message_dict.items():
                osc_out.sendOSC(address, [message])

    def send_parameter_update(self, param_name, value, update_type='single'):
        """
        Unified parameter update - routes to OSC or WebServer based on backend

        Args:
            param_name (str): Parameter name (e.g. 'delta', 'guidance_scale')
            value: Parameter value
            update_type (str): 'single', 'json', 'list', 'command'
        """
        if not self.ownerComp.par.Streamactive:
            self.logger.log(f"DEBUG: send_parameter_update({param_name}) - Streamactive is False", level='INFO')
            return False

        backend = self.ownerComp.par.Backend.eval()
        # self.logger.log(f"DEBUG: send_parameter_update({param_name}) - Backend: {backend}", level='INFO')

        if backend.lower() == 'local':
            # Local backend: send via OSC to local Python process
            return self._send_osc_parameter(param_name, value, update_type)

        elif backend == 'Daydream':
            # Check if we should use WebServer (web mode) or OSC (legacy mode)
            webserver = op('daydream_webserver')
            if webserver and op('daydream_web_status')['active_client', 1].val:
                # Web mode: send via WebServer to browser for Daydream API
                return self._send_web_parameter(param_name, value, update_type)
            else:
                # Legacy mode: send via OSC to external Python process
                return self._send_osc_parameter(param_name, value, update_type)

        return False

    def _send_osc_parameter(self, param_name, value, update_type):
        """Send parameter via OSC - simplified version matching working implementation"""
        try:
            osc_out = op('oscout1')
            if not osc_out:
                self.logger.log(f"ERROR: oscout1 not found for parameter {param_name}", level='ERROR')
                return False

            address = f'/{param_name}'
            # self.logger.log(f"Sending OSC: {address} = {str(value)[:100]} (type: {update_type})", level='INFO')

            if update_type == 'command':
                osc_out.sendOSC(address, [])
            elif update_type == 'json':
                import json
                osc_out.sendOSC(address, [json.dumps(value)])
            elif update_type == 'list':
                osc_out.sendOSC(address, value)
            else:  # single
                osc_out.sendOSC(address, [value])

            # self.logger.log(f"OSC sent successfully: {address}", level='INFO')
            return True
        except Exception as e:
            self.logger.log(f"OSC send failed for {param_name}: {str(e)}", level='ERROR')
            return False


    def _cache_web_parameter(self, param_name, value, update_type):
        """Cache parameter in full API-ready format (non-blocking, TouchDesigner safe)"""
        try:
            # Debug what's being cached
            if param_name == 'controlnets':
                self.logger.log(f"Caching controlnets: {value}", level='DEBUG')
            # Initialize full parameter cache if not exists - stores complete API config
            if 'web_full_params_cache' not in self.ownerComp.storage:
                # Initialize with current full configuration in API format
                self._initialize_full_params_cache()

            if 'web_param_last_send_time' not in self.ownerComp.storage:
                self.ownerComp.storage['web_param_last_send_time'] = 0

            # Get full params cache (API-ready format)
            full_cache = self.ownerComp.storage['web_full_params_cache']

            # Map parameter name to API format and update specific field
            param_name_mapping = {
                'slerp': 'prompt_interpolation_method',
                'prompt_list': 'prompt',
                'seed_list': 'seed',
                # Skip parameters that don't exist in API
                'max_fps': None,
                'disable_cached_attn': None,
                # Skip LoRA parameters completely
                'lora_weights': None,
                'lora_dict': None,
                # IP Adapter nested parameters - special handling below
                'ipadapter_scale': 'ip_adapter.scale',
                'ipadapter_enable': 'ip_adapter.enabled',
                'ipadapter_enabled': 'ip_adapter.enabled'
            }

            api_param_name = param_name_mapping.get(param_name, param_name)
            if api_param_name is None:
                return True  # Skip unsupported parameters

            # Handle nested IP adapter parameters
            if api_param_name.startswith('ip_adapter.'):
                nested_key = api_param_name.split('.')[1]  # Get 'scale' or 'enabled'

                # Ensure ip_adapter object exists with required fields (API requires both enabled and scale)
                if 'ip_adapter' not in full_cache:
                    full_cache['ip_adapter'] = {
                        'enabled': bool(self.ownerComp.par.Ipadapterenable.eval()),
                        'scale': float(self.ownerComp.par.Ipadapterscale.eval())
                    }

                # Update the specific field with type conversion
                if nested_key == 'scale':
                    full_cache['ip_adapter'][nested_key] = float(value)
                elif nested_key == 'enabled':
                    full_cache['ip_adapter'][nested_key] = bool(value)
                else:
                    full_cache['ip_adapter'][nested_key] = value

                self.logger.log(f"Cached ip_adapter: {full_cache['ip_adapter']}", level='DEBUG')
                return True

            # Convert JSON strings back to arrays for weighted lists
            if param_name in ['prompt_list', 'seed_list'] and isinstance(value, str):
                try:
                    import json
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    self.logger.log(f"Failed to parse {param_name} JSON, using as string", level='WARNING')


            # Update the specific parameter in full cache
            full_cache[api_param_name] = value
            return True

        except Exception as e:
            self.logger.log(f"Failed to cache parameter {param_name}: {e}", level='ERROR')
            return False

    def _initialize_full_params_cache(self):
        """Initialize full parameter cache with current TouchDesigner configuration"""
        try:
            # Get current full config and flatten it for API
            full_config = self.gather_full_config_for_daydream()

            # Extract flattened parameters from nested structure
            if 'pipeline_params' in full_config:
                api_params = full_config['pipeline_params'].copy()
            else:
                api_params = full_config.copy()

            # Remove nested structures that don't belong in API params
            api_params.pop('pipeline_id', None)


            # Store in API-ready format
            self.ownerComp.storage['web_full_params_cache'] = api_params

        except Exception as e:
            self.logger.log(f"Failed to initialize full params cache: {e}", level='ERROR')
            # Fallback to empty cache
            self.ownerComp.storage['web_full_params_cache'] = {}

    def clear_params_cache(self):
        """Clear parameter cache - call when model changes"""
        self.ownerComp.storage['web_full_params_cache'] = {}
        self.ownerComp.storage['web_cache_last_sent'] = {}
        self.logger.log("Cleared parameter cache due to model change", level='DEBUG')

    def _send_web_parameter(self, param_name, value, update_type):
        """Legacy function - now just caches parameters for batched sending"""
        return self._cache_web_parameter(param_name, value, update_type)

    def check_web_parameters(self):
        """Send ONLY DYNAMIC parameters with rate limiting (called from execute DAT onFrameEnd)"""
        import json
        import time

        try:
            # Check if we have a full parameter cache
            if 'web_full_params_cache' not in self.ownerComp.storage:
                return True  # No parameters to send

            full_cache = self.ownerComp.storage['web_full_params_cache']
            if not full_cache:
                return True  # Empty cache

            # Check if parameters have changed since last send
            if 'web_cache_last_sent' not in self.ownerComp.storage:
                self.ownerComp.storage['web_cache_last_sent'] = {}

            last_sent_cache = self.ownerComp.storage['web_cache_last_sent']
            if full_cache == last_sent_cache:
                return True  # No changes, don't send

            # Rate limiting: Slow down to 12 FPS (83ms) to avoid API rate limits
            # API was returning "Rate limit exceeded" at 12 FPS
            PARAM_UPDATE_DELAY = 0.083  # 83ms = 12 FPS
            current_time = time.time()
            last_send_time = self.ownerComp.storage.get('web_param_last_send_time', 0)
            time_since_last = current_time - last_send_time

            if time_since_last < PARAM_UPDATE_DELAY:
                # Rate limited - log occasionally to avoid spam
                if int(current_time * 10) % 10 == 0:  # Log every 100ms
                    self.logger.log(f"Rate limited: {int(time_since_last * 1000)}ms since last send (need {int(PARAM_UPDATE_DELAY * 1000)}ms)", level='DEBUG')
                return True  # Too soon, wait for next frame

            # Check if stream is ready for parameter updates
            status_table = op('daydream_web_status')
            if not status_table:
                return False

            stream_id = status_table['stream_id', 1].val
            stream_state = status_table['stream_state', 1].val if status_table else 'unknown'
            is_active = status_table['is_active', 1].val if status_table else False

            if not stream_id:
                # Clear cache if no stream
                self.ownerComp.storage['web_full_params_cache'] = {}
                return False

            # Check if we're actually streaming
            if not self.ownerComp.par.Streamactive.eval():
                # Don't send parameters when not streaming
                return True

            # Get API key
            api_key = self._load_daydream_key()
            if not api_key:
                self.logger.log("No API key for cached parameter updates", level='ERROR')
                return False

            # API URL for PATCH request
            api_url = f"https://api.daydream.live/v1/streams/{stream_id}"

            # CRITICAL: Only send CHANGED parameters (not all dynamic parameters)
            # Compare current cache with last sent cache to find what changed
            dynamic_params = {}

            # Dynamic parameter keys that can be updated without reload
            DYNAMIC_PARAM_KEYS = ['prompt', 'guidance_scale', 'delta', 'num_inference_steps',
                                   't_index_list', 'seed', 'prompt_interpolation_method']

            # Only send parameters that CHANGED
            for key in DYNAMIC_PARAM_KEYS:
                if key in full_cache:
                    # Check if value changed
                    if key not in last_sent_cache or full_cache[key] != last_sent_cache[key]:
                        dynamic_params[key] = full_cache[key]

            # Special handling for controlnets - only send if changed
            if 'controlnets' in full_cache and isinstance(full_cache['controlnets'], list):
                # Check if controlnets changed
                if 'controlnets' not in last_sent_cache or full_cache['controlnets'] != last_sent_cache['controlnets']:
                    dynamic_params['controlnets'] = full_cache['controlnets']

            # IP adapter scale - only send if changed
            if 'ip_adapter' in full_cache:
                if 'ip_adapter' not in last_sent_cache or full_cache['ip_adapter'] != last_sent_cache['ip_adapter']:
                    dynamic_params['ip_adapter'] = full_cache['ip_adapter']

            # IP adapter style image URL - ONLY send if changed (not on every update!)
            # This parameter changes the embedding, takes ~0.5s to process
            if 'ip_adapter_style_image_url' in full_cache:
                if 'ip_adapter_style_image_url' not in last_sent_cache or full_cache['ip_adapter_style_image_url'] != last_sent_cache['ip_adapter_style_image_url']:
                    dynamic_params['ip_adapter_style_image_url'] = full_cache['ip_adapter_style_image_url']

            # Build payload with ONLY dynamic parameters
            payload = {
                "params": dynamic_params  # Send ONLY dynamic parameters
            }
            param_data = json.dumps(payload)


            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'x-client-source': 'StreamDiffusionTD'
            }

            # FIRE AND FORGET - No callbacks needed for parameter updates
            try:
                # UPDATE RATE LIMIT TIMESTAMP + CACHE IMMEDIATELY
                self.ownerComp.storage['web_param_last_send_time'] = current_time
                import copy
                self.ownerComp.storage['web_cache_last_sent'] = copy.deepcopy(full_cache)

                # Launch async PATCH with callback to log results
                asyncio_op = self.ownerComp.op('TDAsyncIO')
                if asyncio_op:
                    asyncio_op.Run(
                        self.async_http_patch(api_url, payload, headers, timeout=10),
                        description="PATCH params",
                        completion_callback=self._on_patch_complete
                    )
                return True

            except Exception as e:
                self.logger.log(f"PATCH launch failed: {e}", level='ERROR')
                self.ownerComp.storage['web_param_last_send_time'] = last_send_time
                return False

        except Exception as e:
            self.logger.log(f"Parameter cache check failed: {e}", level='ERROR')
            return False

    def _on_stream_create_complete(self, task):
        """Callback when async POST stream creation completes"""
        try:
            if task.status.value == 'completed':
                response = task.result
                request_context = task.info

                self.logger.log(f"POST Stream Create - Status: {response.status_code}", level='INFO')

                if response.status_code == 201 or response.status_code == 200:
                    # Parse response data
                    response_data = response.json()
                    self.logger.log(f"Stream created: {response_data.get('id')}", level='INFO')

                    # Store stream data in status table
                    status_table = op('daydream_web_status')
                    if status_table:
                        try:
                            status_table['stream_id', 1] = response_data.get('id', '')
                            status_table['whip_url', 1] = response_data.get('whip_url', '')
                            status_table['playback_id', 1] = response_data.get('output_playback_id', '')
                            status_table['output_stream_url', 1] = response_data.get('output_stream_url', '')
                            status_table['stream_key', 1] = response_data.get('stream_key', '')
                            status_table['stream_state', 1] = 'created'
                            status_table['is_active', 1] = True

                            # Send stream_config WebSocket message to HTML browser
                            try:
                                webserver = request_context.get('webserver')
                                client = request_context.get('client')
                                api_key = self._load_daydream_key()

                                if webserver and client:
                                    stream_config_message = {
                                        'type': 'stream_config',
                                        'api_key': api_key,
                                        'data': response_data
                                    }
                                    webserver.webSocketSendText(client, json.dumps(stream_config_message))
                                    self.logger.log(f"Sent stream_config to browser", level='INFO')
                            except Exception as e:
                                self.logger.log(f"Failed to send stream_config: {e}", level='ERROR')

                            # Pulse timers
                            self.pulse_web_timers()

                        except Exception as e:
                            self.logger.log(f"Failed to store stream data: {e}", level='ERROR')
                else:
                    self.logger.log(f"Stream creation failed - HTTP {response.status_code}: {response.text}", level='ERROR')

            elif task.status.value == 'failed':
                self.logger.log(f"Stream creation task failed: {task.error}", level='ERROR')

            elif task.status.value == 'timeout':
                self.logger.log("Stream creation timed out", level='ERROR')

        except Exception as e:
            self.logger.log(f"Error in stream creation callback: {e}", level='ERROR')

    def _on_status_check_complete(self, task):
        """Callback when async GET status check completes"""
        try:
            if task.status.value == 'completed':
                response = task.result
                request_context = task.info

                if response.status_code == 200:
                    # Parse and store status response
                    response_data = response.json()

                    # Store in status table
                    status_table = op('daydream_web_status')
                    if status_table:
                        try:
                            status_table['api_response', 1] = response.text
                            status_table['api_timestamp', 1] = time.time()

                            # Process status data
                            if response_data.get('success') and response_data.get('data'):
                                stream_data = response_data['data']
                                api_state = stream_data.get('state', '')
                                stream_id = stream_data.get('stream_id', '')

                                # Update last poll timestamp
                                from datetime import datetime
                                status_table['last_poll', 1] = datetime.now().isoformat()

                                # Check for orchestrator errors
                                gateway_status = stream_data.get('gateway_status', {})
                                gateway_error = gateway_status.get('error', {})
                                if gateway_error and isinstance(gateway_error, dict):
                                    error_message = gateway_error.get('error_message', '')
                                    if 'no orchestrator' in error_message.lower():
                                        self.logger.log(f"CRITICAL: No orchestrator available!", level='ERROR')
                                        status_table['stream_state', 1] = 'ERROR'
                                        status_table['is_active', 1] = False
                                        self.ownerComp.Stopstream()
                                        return

                                # Extract gateway metrics
                                whep_url = gateway_status.get('whep_url')
                                if whep_url:
                                    status_table['whep_url', 1] = whep_url

                                    # Push WHEP URL to browser if available
                                    try:
                                        webserver = op('daydream_webserver')
                                        active_client = status_table['active_client', 1].val
                                        if webserver and active_client:
                                            video_config = {
                                                'type': 'video_ready',
                                                'whep_url': whep_url,
                                                'stream_id': stream_id,
                                                'state': api_state
                                            }
                                            webserver.webSocketSendText(active_client, json.dumps(video_config))
                                    except Exception as e:
                                        self.logger.log(f"Failed to push WHEP URL: {e}", level='WARNING')

                                # Extract connection quality metrics
                                ingest_metrics = gateway_status.get('ingest_metrics', {})
                                if ingest_metrics:
                                    stats = ingest_metrics.get('stats', {})
                                    conn_quality = stats.get('conn_quality', '')
                                    if conn_quality:
                                        status_table['connection_quality', 1] = conn_quality

                                    # Extract video track stats (packet loss, jitter)
                                    track_stats = stats.get('track_stats', [])
                                    for track in track_stats:
                                        if track.get('type') == 'video':
                                            status_table['packet_loss_pct', 1] = track.get('packet_loss_pct', 0)
                                            status_table['jitter_ms', 1] = round(track.get('jitter', 0), 2)
                                            status_table['rtt_ms', 1] = track.get('rtt', 0)

                                # Extract inference FPS (Daydream output rate)
                                inference_status = stream_data.get('inference_status', {})
                                if inference_status:
                                    inference_fps = inference_status.get('fps', 0)
                                    if inference_fps:
                                        status_table['output_fps', 1] = round(inference_fps, 2)

                                    # Track inference errors
                                    last_error = inference_status.get('last_error')
                                    if last_error:
                                        status_table['last_error', 1] = str(last_error)

                                # Extract input FPS (what Daydream is receiving via WHIP)
                                input_status = stream_data.get('input_status', {})
                                if input_status:
                                    input_fps = input_status.get('fps', 0)
                                    if input_fps:
                                        status_table['daydream_input_fps', 1] = round(input_fps, 2)

                                # Update state
                                if api_state == 'ONLINE' and whep_url:
                                    status_table['stream_state', 1] = 'ONLINE'
                                    status_table['is_active', 1] = True

                                # Mirror metrics to stream_osc_data table for OSC compatibility
                                try:
                                    osc_table = op('stream_osc_data')

                                    if osc_table:
                                        # Update output-name
                                        self._update_osc_table(osc_table, 'output-name', 'Daydream WebRTC')

                                        # Update framecount from frames_received (browser counter)
                                        browser_frames = status_table['frames_received', 1].val
                                        if browser_frames:
                                            self._update_osc_table(osc_table, 'framecount', browser_frames)

                                        # Update FPS (inference output rate)
                                        if inference_fps:
                                            self._update_osc_table(osc_table, 'fps', round(inference_fps, 2))

                                        # Update stream state (use 'ONLINE' not api_state)
                                        if api_state == 'ONLINE' and whep_url:
                                            self._update_osc_table(osc_table, 'stream-state', 'ONLINE')
                                        else:
                                            self._update_osc_table(osc_table, 'stream-state', api_state)

                                        # Update connection quality
                                        if conn_quality:
                                            self._update_osc_table(osc_table, 'connection-quality', conn_quality)

                                        # Update error if exists
                                        if last_error:
                                            self._update_osc_table(osc_table, 'last-error', str(last_error))

                                except Exception as e:
                                    self.logger.log(f"Failed to update stream_osc_data: {e}", level='DEBUG')

                                # Pulse timers
                                self.pulse_web_timers()

                        except Exception as e:
                            self.logger.log(f"Failed to process status response: {e}", level='ERROR')
                else:
                    self.logger.log(f"Status check failed - HTTP {response.status_code}", level='WARNING')

            elif task.status.value == 'failed':
                self.logger.log(f"Status check task failed: {task.error}", level='WARNING')

        except Exception as e:
            self.logger.log(f"Error in status check callback: {e}", level='ERROR')

    def _request_stream_creation(self, webserver, client, api_key):
        """Create a new Daydream stream directly via TDAsyncIO using complete TouchDesigner configuration"""
        import json
        import time

        try:
            # Gather ALL pipeline parameters from TouchDesigner using the complete config function
            config = self.gather_full_config_for_daydream()

            if not config or 'pipeline_id' not in config:
                self.logger.log("Failed to gather configuration for stream creation", level='ERROR')
                return False

            pipeline_id = config['pipeline_id']
            pipeline_params = config['pipeline_params']

            api_url = "https://api.daydream.live/v1/streams"

            payload = {
                "pipeline_id": pipeline_id,  # Correct pipeline_id from gathered config
                "pipeline_params": pipeline_params,  # Send full config at creation (includes controlnets)
                # "name": f"StreamDiffusionTD-{int(time.time())}"
            }

            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'x-client-source': 'StreamDiffusionTD'
            }

            # LOG THE COMPLETE JSON PAYLOAD FOR DEBUGGING
            self.logger.log(f'=== STREAM CREATION PAYLOAD ===', level='DEBUG')
            self.logger.log(f'URL: {api_url}', level='DEBUG')
            self.logger.log(f'Pipeline ID: {pipeline_id}', level='DEBUG')
            self.logger.log(f'Parameters: {len(pipeline_params)} total', level='DEBUG')
            self.logger.log(f'Key params: t_index_list={pipeline_params.get("t_index_list")}, controlnets={len(pipeline_params.get("controlnets", []))}', level='DEBUG')
            self.logger.log(f'=== END PAYLOAD ===', level='DEBUG')

            # Use TDAsyncIO for non-blocking stream creation
            asyncio_op = self.ownerComp.op('TDAsyncIO')
            if not asyncio_op:
                self.logger.log("TDAsyncIO not found - cannot create stream", level='ERROR')
                return False

            # Store context for callback
            request_context = {
                'url': api_url,
                'payload': payload,
                'webserver': webserver,
                'client': client
            }

            task_id = asyncio_op.Run(
                self.async_http_post(api_url, payload, headers, timeout=30),
                description="POST /streams (create)",
                info=request_context,
                completion_callback=lambda task: self._on_stream_create_complete(task)
            )

            self.logger.log(f"Stream creation task launched (ID: {task_id})", level='INFO')
            return True

        except Exception as e:
            self.logger.log(f"Stream creation failed: {e}", level='ERROR')
            return False

    def pulse_web_timers(self):
        """Pulse timers for web Daydream mode - called from web callbacks"""
        timer_stream = op('timer_stream')
        timer_server = op('timer_server')

        if timer_stream:
            timer_stream.par.start.pulse()
            self.logger.log('Timer stream pulsed from web callbacks', level='DEBUG')

        if timer_server:
            timer_server.par.start.pulse()
            self.logger.log('Timer server pulsed from web callbacks', level='DEBUG')

    def trigger_status_check(self):
        """Trigger async status check via TDAsyncIO - called from timer_client"""
        # Check if we should exit fast polling mode
        if self.check_fast_polling_timeout():
            pass  # Already switched to normal mode

        try:
            # Get current stream ID from status table
            status_table = op('daydream_web_status')
            if not status_table:
                return

            stream_id = status_table['stream_id', 1].val
            self.logger.log(f"Status check reading stream_id from table: '{stream_id}'", level='DEBUG')
            if not stream_id or not stream_id.strip():
                self.logger.log("No stream_id in table, skipping status check", level='DEBUG')
                return

            # Get API key
            api_key = self._load_daydream_key()
            if not api_key:
                self.logger.log("No API key for status check", level='WARNING')
                return

            # Use TDAsyncIO for non-blocking status check
            asyncio_op = self.ownerComp.op('TDAsyncIO')
            if not asyncio_op:
                self.logger.log("TDAsyncIO not found for status check", level='ERROR')
                return

            status_url = f"https://api.daydream.live/v1/streams/{stream_id}/status"

            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'x-client-source': 'StreamDiffusionTD'
            }

            try:
                self.logger.log(f"Status check for stream: {stream_id}", level='DEBUG')

                # Store context for callback
                request_context = {
                    'url': status_url,
                    'stream_id': stream_id
                }

                task_id = asyncio_op.Run(
                    self.async_http_get(status_url, headers, timeout=10),
                    description=f"GET /streams/{stream_id}/status",
                    info=request_context,
                    completion_callback=lambda task: self._on_status_check_complete(task)
                )

                self.logger.log(f"Status check task launched (ID: {task_id})", level='DEBUG')
            except Exception as e:
                self.logger.log(f"Failed to launch status check task: {e}", level='ERROR')

        except Exception as e:
            self.logger.log(f"Status check error: {e}", level='ERROR')

    def start_status_polling(self):
        """Start the timer_client for periodic status polling"""
        timer_client = op('timer_client')
        if timer_client:
            # Set timer parameters for 10-second polling interval
            timer_client.par.active = 'running'  # Active while running
            timer_client.par.initialize.pulse()  # Initialize first
            timer_client.par.start.pulse()       # Then start
            timer_client.par.cycle = True        # Enable cycling
            timer_client.par.length = 10         # 10 second intervals
            timer_client.par.lengthunits = 2     # 2 = seconds
            self.logger.log("Started status polling timer (10s intervals)", level='INFO')
        else:
            self.logger.log("Timer client not found for status polling", level='ERROR')

    def stop_status_polling(self):
        """Stop the timer_client status polling"""
        timer_client = op('timer_client')
        if timer_client:
            timer_client.par.gotodone.pulse()  # Go to done state
            self.logger.log("Stopped status polling timer", level='INFO')

    def start_fast_status_polling(self):
        """Start fast status polling (3s intervals) for WHEP URL detection"""
        import time
        timer_client = op('timer_client')
        if timer_client:
            # Set timer for fast 3-second polling (like working manager)
            timer_client.par.active = 'running'
            timer_client.par.cycle = True
            timer_client.par.length = 3         # 3 second intervals for WHEP detection
            timer_client.par.lengthunits = 2     # 2 = seconds
            timer_client.par.delay = 0           # No initial delay for fast mode

            # Apply settings with proper initialization
            timer_client.par.initialize.pulse()
            timer_client.par.start.pulse()

            # Mark that we're in fast polling mode
            self.ownerComp.storage['fast_polling_mode'] = True
            self.ownerComp.storage['fast_polling_start_time'] = time.time()

            self.logger.log("Started FAST status polling for WHEP URL detection (3s intervals)", level='INFO')
        else:
            self.logger.log("Timer client not found for fast status polling", level='ERROR')

    def check_fast_polling_timeout(self):
        """Check if we should switch back to normal 10s polling"""
        import time

        if not self.ownerComp.storage.get('fast_polling_mode', False):
            return False

        start_time = self.ownerComp.storage.get('fast_polling_start_time', 0)
        elapsed = time.time() - start_time

        # Switch to normal polling after 24 seconds (8 attempts × 3s)
        if elapsed > 24:
            self.switch_to_normal_polling()
            return True

        # DON'T switch just because WHEP URL exists - wait for browser video_ready confirmation
        # The browser will trigger switch_to_normal_polling() when WebRTC connection is established

        return False

    def switch_to_normal_polling(self):
        """Switch back to normal 10-second status polling"""
        timer_client = op('timer_client')
        if timer_client:
            timer_client.par.length = 10         # Back to 10 second intervals
            timer_client.par.delay = 1           # 1 second delay for normal mode

            # Apply settings with proper initialization
            timer_client.par.initialize.pulse()
            timer_client.par.start.pulse()

            self.ownerComp.storage['fast_polling_mode'] = False
            self.logger.log("Switched to normal status polling (10s intervals)", level='INFO')

    def _force_process_frame(self, reason):
        """
        Force a process frame command without state checks.
        Used for kickstarting feedback safe loops.
        
        Args:
            reason (str): Reason for forcing the frame (for logging)
        """
        if self.ownerComp.par.Serveractive and self.ownerComp.par.Streamactive:
            message_dict = {'/process_frame': 1}
            self.send_osc_messages(message_dict)
            self.logger.log(f'Forced process frame to {reason}', level='INFO')
        else:
            self.logger.log(f"Cannot force process frame ({reason}) - server/stream not active", level='WARNING')

    def Resolutionchange(self):
        width = self.ownerComp.par.Width.eval()
        height = self.ownerComp.par.Height.eval()
        date = datetime.datetime.now().strftime("%Y%m%d")
        new_name = f"stream_{width}x{height}_{date}"
        # self.ownerComp.par.Streamoutname = new_name
        op('numpy_share_out').par.reinitextensions.pulse()
        op('numpy_share_out_cn').par.reinitextensions.pulse()        
        self.Streamoutname()


    def ImageChanged(self, info=None):
        """
        Callback triggered when StreamDiffusion output image changes.
        This is called from the shared memory extension when new frame content is detected.
        Much more efficient than TOP->CHOP conversion method.
        
        Auto-handles feedback safe mode when enabled.
        """
        try:
            current_timestamp = str(datetime.datetime.now())
            callback_data = {
                'timestamp': current_timestamp,
                'frame_source': 'shared_memory_detection',
                'detection_method': 'lightweight_hash'
            }
            
            # AUTO FEEDBACK SAFE MODE - handle frame-by-frame processing
            is_feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
            is_stream_active = self.ownerComp.par.Streamactive.eval()
            is_manually_paused = self.ownerComp.par.Pausestream.eval()
            
            # Only trigger frame processing if:
            # - Feedback safe mode is enabled
            # - Stream is active  
            # - NOT manually paused (manual pause overrides feedback safe)
            if is_feedback_safe and is_stream_active and not is_manually_paused:
                # Auto-trigger single frame processing (automatic call)
                self.Processframe(is_manual=False)
                # self.logger.log("Auto-triggered feedback safe frame processing", level='DEBUG')
            
            # Handle normal image change callback
            if hasattr(self.ownerComp.par, 'Onimagechange') and self.ownerComp.par.Onimagechange:
                self.ownerComp.DoCallback("onImageChange", callback_data)
                
        except Exception as e:
            self.logger.log(f"Error in ImageChanged callback: {str(e)}", level='ERROR')


    def Promptblock(self):
        """
        Sends the concept and weight from each block in the Promptdict sequence to the OSC Out DAT.
        If par.Normpweights is True, the weights are normalized so that their sum equals the value of par.Totalpweights.
        """
        import json
        prompt_list = []

        # Iterate over each block in the Promptdict sequence to build the list
        total_weight = 0
        for block in self.ownerComp.par.Promptdict.sequence:
            concept = block.par.Concept.eval()  
            weight = block.par.Weight.eval()  
            prompt = [concept, weight]  # Create a list
            prompt_list.append(prompt)  # Append the list to the list
            if self.ownerComp.par.Normpweights:
                total_weight += weight  

        # Normalize the weights if required
        if self.ownerComp.par.Normpweights and total_weight != 0:
            normalization_factor = self.ownerComp.par.Totalpweights / total_weight
            for i in range(len(prompt_list)):
                concept, weight = prompt_list[i]  # Unpack the list
                weight *= normalization_factor
                prompt_list[i] = [concept, weight]  # Create a new list

        # Convert the list to a JSON string
        prompt_list_str = json.dumps(prompt_list)
        # print(prompt_list_str)  

        # Send the JSON string via OSC
        if self.ownerComp.par.Streamactive:
            self.send_parameter_update('prompt_list', prompt_list_str, 'single')

    def Tindexblock(self):
        """
        Sends the t_index list with backend-specific parameter names.
        Local backend expects 't_list', Daydream expects 't_index_list'.
        """
        if self.ownerComp.par.Streamactive:
            t_index_list = []
            for block in self.ownerComp.par.Tindexblock.sequence:
                step_value = block.par.Step.eval()
                t_index_list.append(step_value)
            
            if len(t_index_list) != self.ownerComp.par.Tindexblock.sequence.numBlocks:
                return
            
            backend = self.ownerComp.par.Backend.eval()
            if backend.lower() == 'local':
                # Local backend expects 't_list'
                self.send_parameter_update('t_list', t_index_list, 'list')
            else:
                # Daydream backend expects 't_index_list'
                self.send_parameter_update('t_index_list', t_index_list, 'list')

    def Loradict(self):
        if self.ownerComp.par.Streamactive:
            import json
            lora_dict = {}
            if self.ownerComp.par.Uselora:
                for block in self.ownerComp.par.Loradictblock.sequence:
                    lora_path = block.par.Lorapath.eval()
                    if lora_path != 'select_lora_from_dropdown' and lora_path != '':
                        lora_dict[lora_path] = block.par.Weight.eval()
            self.send_parameter_update('lora_weights', json.dumps(lora_dict), 'single')

    def Textualinvdict(self):
        """
        Sends the textual inversion settings to the OSC Out DAT.
        Similar to Loradict(), but for textual inversions.
        """
        import json
        if self.ownerComp.par.Streamactive:
            ti_dict = {}
            if self.ownerComp.par.Usetextualinv:
                for block in self.ownerComp.par.Textualinvblock.sequence:
                    embed_path = block.par.Embedpath.eval()
                    if embed_path and embed_path != 'select_embedding_from_dropdown':
                        token = block.par.Token.eval()
                        # Expand the path if it's local
                        if not embed_path.startswith('http'):
                            embed_path = tdu.expandPath(embed_path)
                        ti_dict[embed_path] = token if token else None
            if self.ownerComp.par.Backend.eval() == 'Local':
                # Send the dictionary via OSC
                self.send_parameter_update('textual_inversion_dict', json.dumps(ti_dict), 'single')
            # else:
            #     # Send the dictionary via WebServer
            #     self.send_parameter_update('textual_inversion_dict', json.dumps(ti_dict), 'single')

            # if ti_dict:
            #     self.logger.log(f'Sending textual inversion update via OSC: {json.dumps(ti_dict, indent=2)}', level='INFO')

    def Feedbacksafe(self, feedback_safe = None):
        # Handle feedback safe mode toggle - manages backend pause state for frame-by-frame control
        if feedback_safe is None:
            feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
        
        # Only take action if the state has actually changed
        if feedback_safe != self.last_feedback_safe_state:
            if feedback_safe:
                # Enable feedback safe mode - put backend in pause mode for frame-by-frame control
                # Only pause if not already paused by user
                if not self.ownerComp.par.Pausestream.eval():
                    self.Pausestream(pause=True)
                    self.feedback_safe_paused_backend = True  # Remember we paused it
                    self.logger.log("Feedback safe mode enabled - backend paused for automatic frame-by-frame processing", level='INFO')
                    
                    # KICKSTART: Force initial frame to start the feedback safe loop (bypass state checks)
                    run('me.ext.StreamDiffusionExt._force_process_frame("start feedback safe loop")', delayFrames=1, fromOP=self.ownerComp)
                else:
                    self.feedback_safe_paused_backend = False  # User already paused
                    self.logger.log("Feedback safe mode enabled - using existing pause state", level='INFO')
            else:
                # Disable feedback safe mode - resume normal streaming if not manually paused
                is_manually_paused = self.ownerComp.par.Pausestream.eval()
                self.logger.log(f"DEBUG: Disabling feedback safe - feedback_safe_paused_backend={self.feedback_safe_paused_backend}, is_manually_paused={is_manually_paused}", level='DEBUG')
                
                # Always try to resume if not manually paused, regardless of who paused it initially
                if not is_manually_paused:
                    # User is not manually paused, so we should resume streaming
                    if self.ownerComp.par.Serveractive:
                        self.send_parameter_update('play', None, 'command')
                        self.logger.log("Feedback safe mode disabled - sent /play to resume normal streaming", level='INFO')
                        
                        # Also pulse the timers to ensure they don't timeout during transition
                        op('timer_stream').par.start.pulse()
                        op('timer_server').par.start.pulse()
                        self.logger.log("Pulsed timers to prevent timeout during feedback safe disable", level='DEBUG')
                    else:
                        self.logger.log("Server not active - cannot resume streaming", level='WARNING')
                else:
                    # User is manually paused, preserve that state
                    self.logger.log("Feedback safe mode disabled - manual pause state preserved", level='INFO')
                
                # Reset our tracking flag and record disable time
                self.feedback_safe_paused_backend = False
                self.last_feedback_safe_disable_time = time.time()
            
            # Update state tracking
            self.last_feedback_safe_state = feedback_safe

    def Seed(self): 
        """
        Sends the seed settings to the OSC Out DAT as a list of lists, combining weights for identical seeds before normalization.
        """
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')
            seed_weight_accumulator = {}
            for block in self.ownerComp.par.Seeddict.sequence:
                seed_val = block.par.Seedval.eval() 
                weight = block.par.Seedweight.eval()  
                if weight != 0:  
                    if seed_val in seed_weight_accumulator:
                        seed_weight_accumulator[seed_val] += weight  
                    else:
                        seed_weight_accumulator[seed_val] = weight  
            seed_list = [[seed, weight] for seed, weight in seed_weight_accumulator.items()]
            total_weight = sum(weight for _, weight in seed_list)
            if total_weight != 0:
                for i in range(len(seed_list)):
                    seed_val, weight = seed_list[i]
                    weight /= total_weight
                    seed_list[i] = [seed_val, weight]
            blend_multipliers = [1 + 0.5 * np.cos(np.pi * (weight - 0.5)) * (1 + 0.3 * (len(seed_list) - 2)) for _, weight in seed_list]
            avg_blend_multiplier = self.ownerComp.par.Noisemult.eval() * (sum(blend_multipliers) / len(blend_multipliers)) if blend_multipliers else 0
            for i in range(len(seed_list)):
                seed_val, weight = seed_list[i]
                weight *= avg_blend_multiplier
                seed_list[i] = [seed_val, weight]
            import json
            seed_list_str = json.dumps(seed_list)
            self.send_parameter_update('seed_list', seed_list_str, 'single')
            # self.logger.log(f'Sending seed update via OSC: {seed_list_str}', level='INFO')  # Add debug logging

    def Sdmode(self,sdmode = None):
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')
            if sdmode is None:
                sdmode = self.ownerComp.par.Sdmode.eval()
                if sdmode in ['txt2img', 'img2img']:
                    osc_out.sendOSC('/sdmode', [sdmode])

    def Limitfps(self):
        if self.ownerComp.par.Streamactive:
            max_fps_value = self.ownerComp.par.Limitfps.eval()
            self.send_parameter_update('max_fps', max_fps_value)

    def Usecontrolnet(self):
        if self.ownerComp.par.Streamactive:
            backend = self.ownerComp.par.Backend.eval()
            osc_out = op('oscout1')
            use_controlnet_value = self.ownerComp.par.Usecontrolnet.eval()
            
            if backend == 'Daydream':
                # For Daydream, trigger the sequence block function instead
                self.Cnblock()
            else:
                # For local backend, send individual parameter
                osc_out.sendOSC('/use_controlnet', [use_controlnet_value])

    def Streamoutname(self):
        if self.ownerComp.par.Streamactive:
            stream_out_name_value = self.ownerComp.par.Streamoutname.eval()
            self.send_parameter_update('stream_out_name', stream_out_name_value)

    def Negprompt(self):
        if self.ownerComp.par.Streamactive:
            negative_prompt_message = self.ownerComp.par.Negprompt.eval()
            self.send_parameter_update('negative_prompt', negative_prompt_message)


    def set_interpolation(self):
        slerp = self.ownerComp.par.Setinterpolation.eval()
        if self.ownerComp.par.Streamactive:
            self.send_parameter_update('slerp', slerp)

    def update_width_height_from_config(self):
        """
        Updates Width and Height parameters from the dynamically generated YAML config.
        Reads from streamdiffusionTD/td_config.yaml (not the deprecated stream_config DAT).
        """
        import yaml

        base_folder = self.ownerComp.par.Basefolder.eval()
        config_path = os.path.join(base_folder, 'streamdiffusionTD', 'td_config.yaml')

        try:
            # Read from YAML file (the actual source of truth)
            if not os.path.exists(config_path):
                self.logger.log(f"Config file not found at {config_path}", level='WARNING')
                return

            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            if 'width' in config:
                width = config['width']
                if self.ownerComp.par.Width.mode == ParMode.CONSTANT:
                    self.ownerComp.par.Width = width
                self.logger.log(f"Updated Width to {width} from YAML config", level='INFO')

            if 'height' in config:
                height = config['height']
                if self.ownerComp.par.Height.mode == ParMode.CONSTANT:
                    self.ownerComp.par.Height = height
                self.logger.log(f"Updated Height to {height} from YAML config", level='INFO')

        except yaml.YAMLError as e:
            self.logger.log(f"Error parsing YAML config: {e}", level='ERROR')
        except Exception as e:
            self.logger.log(f"Error updating Width and Height from YAML: {e}", level='ERROR')

    def Startstream(self):
        """
        Starts the StreamDiffusion stream by executing a batch file.
        The batch file activates a Python virtual environment and starts the main script.
        Also resets any stuck stream creation locks for Daydream mode.
        """
        # Reset any stuck stream creation locks for Daydream mode
        if self.ownerComp.par.Backend.eval() == 'Daydream':
            # Clear parameter cache to ensure fresh config with correct ControlNet models
            self.clear_params_cache()

            try:
                # Reset the status table stream creation indicator
                status_table = op('daydream_web_status')
                if status_table:
                    current_status = status_table['status', 1].val
                    if current_status == 'Creating stream...':
                        status_table['status', 1] = 'Ready for stream creation'
                        self.logger.log("Reset stuck 'Creating stream...' status on Startstream", level='INFO')

                # Reset stream creation lock using TouchDesigner Storage system
                storage = self.ownerComp.storage.get('stream_creation_in_progress', False)
                if storage:
                    self.ownerComp.storage['stream_creation_in_progress'] = False
                    self.logger.log("Reset stream creation lock via Storage on Startstream", level='INFO')

            except Exception as e:
                self.logger.log(f"Could not reset stream creation locks: {e}", level='DEBUG')

        self.copy_sdtd_code()
        if self.ownerComp.par.Serveractive:
            #send play
            if self.ownerComp.par.Pausestream.mode == ParMode.CONSTANT:
                self.ownerComp.par.Pausestream = False
            self.send_parameter_update('play', None, 'command')
            self.logger.log('Playing stream...', level='INFO')
            return

        backend = self.ownerComp.par.Backend.eval()
        base_folder = self.ownerComp.par.Basefolder.eval()

        if backend == 'Daydream':
            # New web-based Daydream mode - start WebServer and request stream creation
            api_key = self._load_daydream_key()
            if not api_key:
                self.logger.log("Daydream API Key is not set. Please set it in the parameters.", level="ERROR")
                return

            # Initialize status table with all required columns
            self._ensure_daydream_status_table()

            # CLEAR EXISTING STREAM DATA - force new stream creation every time
            status_table = op('daydream_web_status')
            if status_table:
                status_table['stream_id', 1] = ''
                status_table['whip_url', 1] = ''
                status_table['whep_url', 1] = ''
                status_table['playback_id', 1] = ''
                status_table['output_stream_url', 1] = ''
                status_table['status', 1] = 'Starting new stream...'
                status_table['stream_state', 1] = 'unknown'
                status_table['is_active', 1] = False
                status_table['frames_received', 1] = 0
                self.logger.log("Cleared existing stream data - will create new stream", level='INFO')

            # Start the WebServer DAT for web-based Daydream integration
            webserver = op('daydream_webserver')
            frame_sender = op('frame_sender')
            frame_sender.par.active = True
            # Set custom port for Daydream WebServer
            webserver.par.port = 7844
            webserver.par.active = True

            # Configure WebRender TOP to point to WebServer DAT endpoint
            webserver_url = "http://localhost:7844/"
            self.update_webrender_top(webserver_url)
            self.logger.log('Daydream WebServer started on port 7844 - waiting for browser connection...', level='INFO')

            # Mark server as active so parameter updates work
            self.ownerComp.par.Serveractive = True

            # Timer pulsing will be handled by web callbacks when browser connects

            return
        else: # Local backend
            # Check for virtual environment (Local only needs venv or .venv)
            venv_path = os.path.join(base_folder, 'venv')
            dot_venv_path = os.path.join(base_folder, '.venv')
            if os.path.exists(venv_path):
                activate_script_path = os.path.join(venv_path, 'Scripts', 'activate.bat')
            elif os.path.exists(dot_venv_path):
                activate_script_path = os.path.join(dot_venv_path, 'Scripts', 'activate.bat')
            else:
                self.logger.log("Check Install page / Installation not complete - Basefolder parameter must be set to '.../StreamDiffusion'", level="ERROR")
                return

            self.copy_sdtd_code()

            self.update_width_height_from_config()
            python_script_path = 'streamdiffusionTD\\td_main.py'
            generation_command = f'{python_script_path}'
            log_message = 'Launching StreamDiffusionTD (Local)...'


        if not self.ownerComp.par.Visiblewindow:
            full_command = f'cmd.exe /c "{activate_script_path} && python {generation_command}"'
            try:
                subprocess.Popen(full_command, cwd=base_folder, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
                self.logger.log(f'{log_message} (silently.)', level='INFO')
            except Exception as e:
                self.logger.log(f'Error Startstream: Failed to start streaming process without command window. Error: {e}', level='ERROR')
            return

        if platform.system() == 'Windows':
            bat_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'Start_StreamDiffusion.bat')
        elif platform.system() == 'Darwin':
            bat_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'Start_StreamDiffusion.sh')

        debug_cmd = 'pause' if self.ownerComp.par.Debugcmd.eval() else ''
        use_powershell = False
        if hasattr(self.ownerComp.par, 'Powershell'):
            use_powershell = self.ownerComp.par.Powershell.eval()
        
        # Only Local backend reaches here now (Daydream returns early)
        python_command_part = "'streamdiffusionTD\\td_main.py'"

        if use_powershell:
            batch_file_content = f"""
@echo off
cd /d %~dp0
if exist venv (
    PowerShell -Command "& {{& 'venv\\Scripts\\Activate.ps1'; & 'venv\\Scripts\\python.exe' {python_command_part}}}"
) else (
    PowerShell -Command "& {{& '.venv\\Scripts\\Activate.ps1'; & '.venv\\Scripts\\python.exe' {python_command_part}}}"
)
    {debug_cmd}
            """
        else:
            batch_file_content = f"""
            @echo off
            cd /d %~dp0
            if exist venv (
                call venv\\Scripts\\activate.bat
                venv\\Scripts\\python.exe {generation_command}
            ) else (
                call .venv\\Scripts\\activate.bat
                .venv\\Scripts\\python.exe {generation_command}
            )
            {debug_cmd}
            """
        if platform.system() == 'Darwin':
            mac_debug_cmd = 'read -p "Press any key to continue..." key' if self.ownerComp.par.Debugcmd.eval() else ''
            debug_at_start = False
            
            # Only Local backend reaches here now (Daydream returns early)
            mac_python_command = "python streamdiffusionTD/td_main.py"

            batch_file_content = f"""
                #!/bin/sh
                # Unset PYTHONPATH to avoid TD Python interference
                unset PYTHONPATH
                
                cd "$(dirname "$0")"
                
                if {debug_at_start}; then
                    echo "Current directory: $(pwd)"
                    echo "PATH: $PATH"
                    echo "Available Python versions:"
                    which -a python python3
                fi
                
                if [ -d "venv" ]; then
                    source venv/bin/activate
                    if {debug_at_start}; then
                        echo "Using venv at: $(which python)"
                        echo "Python version: $(python --version)"
                        echo "Python packages installed:"
                        pip list
                    fi
                    {mac_python_command}
                elif [ -d ".venv" ]; then
                    source .venv/bin/activate
                    if {debug_at_start}; then
                        echo "Using .venv at: $(which python)"
                        echo "Python version: $(python --version)"
                        echo "Python packages installed:"
                        pip list
                    fi
                    {mac_python_command}
                else
                    source daydream_venv/bin/activate
                    if {debug_at_start}; then
                        echo "Using daydream_venv at: $(which python)"
                        echo "Python version: $(python --version)" 
                        echo "Python packages installed:"
                        pip list
                    fi
                    {mac_python_command}
                fi
                {mac_debug_cmd}
                """
        with open(bat_file_path, 'w') as bat_file:
            bat_file.write(batch_file_content)
        if platform.system() == 'Windows':
            subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=self.ownerComp.par.Basefolder.eval())
            self.logger.log(f'{log_message}', level='INFO')
        elif platform.system() == 'Darwin':
            os.system(f"chmod +x {bat_file_path}")
            subprocess.Popen(['open', '-a', 'Terminal', bat_file_path], cwd=self.ownerComp.par.Basefolder.eval())
            self.logger.log(f'{log_message}', level='INFO')

    def Stopstream(self):
        """
        Sends a /stop command to the OSC server.
        """ 
        if not self.ownerComp.par.Serveractive:
            return
        
        self.send_parameter_update('stop', None, 'command')
        self.logger.log('Stopping server...', level='INFO')
        run("if me.par.Streamactive.mode == ParMode.CONSTANT: me.par.Streamactive = False", delayFrames = 2, fromOP = self.ownerComp)
        run("if me.par.Pausestream.mode == ParMode.CONSTANT: me.par.Pausestream = False", delayFrames = 3, fromOP = self.ownerComp)
        # run("if me.par.Serveractive.mode == ParMode.CONSTANT: me.par.Serveractive = False", delayFrames = 3, fromOP = self.ownerComp)
        self.last_stop_time = time.time()

        # Graceful shutdown for Daydream backend
        if self.ownerComp.par.Backend.eval() == 'Daydream':
            # Clear stream_osc_data table
            try:
                osc_table = op('stream_osc_data')
                if osc_table:
                    self._update_osc_table(osc_table, 'fps', 0)
                    self._update_osc_table(osc_table, 'stream-state', 'OFFLINE')
                    self._update_osc_table(osc_table, 'framecount', 0)
                    self._update_osc_table(osc_table, 'output-name', '')
            except Exception as e:
                self.logger.log(f"Failed to clear stream_osc_data: {e}", level='DEBUG')

            # Stop frame sender immediately (stop sending WebSocket frames)
            frame_sender = op('frame_sender')
            if frame_sender:
                frame_sender.par.active = False

            # Stop WebServer - this is where the 15 second freeze happens
            # The onServerStop callback closes connections but something else is blocking
            op('daydream_webserver').par.active = False

            # Stop WebRender
            op('webrender_daydream').par.active = False

            # Stop timers
            for timer_name in ['timer_client', 'timer_server', 'timer_stream']:
                timer = op(timer_name)
                if timer:
                    timer.par.start.pulse()
                    
            if self.ownerComp.par.Serveractive.mode == ParMode.CONSTANT:
                self.ownerComp.par.Serveractive = False
        else:
            # Non-Daydream: just stop webrender
            op('webrender_daydream').par.active = False


    def Pausestream(self, pause = None):
        """
        Sends a /pause or /play command to the OSC server based on the Pausestream parameter.
        Can accept either a TouchDesigner parameter object or a boolean value.
        Handles feedback safe mode properly.
        """
        self.update_width_height_from_config()
        if pause is None:
            pause = self.ownerComp.par.Pausestream
        if self.ownerComp.par.Serveractive:
            osc_out = op('oscout1')
            # Handle both parameter objects and boolean values
            pause_value = pause.eval() if hasattr(pause, 'eval') else pause
            is_feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
            
            if pause_value:
                # Manual pause - always pause backend
                self.send_parameter_update('pause', None, 'command')
                self.logger.log('Pausing stream...', level='INFO')
            else:
                # Manual unpause - check if we're in feedback safe mode
                if is_feedback_safe:
                    # In feedback safe mode - keep backend paused for frame-by-frame control
                    self.send_parameter_update('pause', None, 'command')
                    self.logger.log('Resuming stream in feedback safe mode (backend stays paused for frame-by-frame control)...', level='INFO')
                    
                    # KICKSTART: Force initial frame to restart the feedback safe loop (bypass state checks)
                    run('me.ext.StreamDiffusionExt._force_process_frame("restart feedback safe loop")', delayFrames=1, fromOP=self.ownerComp)
                else:
                    # Normal mode - resume streaming
                    self.send_parameter_update('play', None, 'command')
                    self.logger.log('Resuming stream...', level='INFO')
        else:
            if self.ownerComp.par.Pausestream.mode == ParMode.CONSTANT: 
                self.ownerComp.par.Pausestream = False

    def Unloadstream(self):
        """
        Sends an /unload command to the OSC server.
        """
        if self.ownerComp.par.Serveractive:
            self.send_parameter_update('unload', None, 'command')
            self.logger.log('Unloading stream...', level='INFO')
            if self.ownerComp.par.Pausestream.mode == ParMode.CONSTANT:
                self.ownerComp.par.Pausestream = False

    def Serveractive(self):
        self.Promptblock()
        if self.ownerComp.par.Serveractive:
            self.ownerComp.par.Startstream.label = "Start Stream"
            self.ownerComp.par.Stopstream.label = "Stop Server"
            op('timer_server').par.start.pulse()
            if not self.ownerComp.par.Streamactive:
                if self.ownerComp.par.Acceleration == 'tensorrt':
                    self.logger.log('Server active... Loading TensorRT Models...', level='INFO')
                else:
                    self.logger.log('Server active... Loading Models...', level='INFO')
                return
        else:
            self.ownerComp.par.Startstream.label = "Start Stream"
            self.ownerComp.par.Stopstream.label = "Stop Server"
            self.logger.log('Server stopped...', level='INFO')
            if self.ownerComp.par.Backend.eval() == 'Daydream':
                # Just update status, but keep stream data for reconnection
                status_table = op('daydream_web_status')
                if status_table:
                    status_table['status', 1] = 'Server stopped'

                op('webrender_daydream').par.active = True

    def Processframe(self, is_manual=True):
        """
        Triggers processing of a single frame when in pause mode.
        Used for manual frame-by-frame control and loopback scenarios.
        
        Args:
            is_manual (bool): True if called manually (button/pulse), False if called automatically (feedback safe)
        """
        current_time = time.time() * 1000  # Convert to milliseconds
        if current_time - self.last_processframe_time < self.processframe_debounce_ms:
            self.logger.log(f"Processframe debounced (last call {current_time - self.last_processframe_time:.1f}ms ago)", level='DEBUG')
            return
        self.last_processframe_time = current_time        
        delay_frames = self.ownerComp.par.Delayframes.eval()
        run(f'me.ext.StreamDiffusionExt.send_processframe({is_manual})',fromOP=self.ownerComp,delayFrames=delay_frames)


    def send_processframe(self, is_manual=True):
        """
        Triggers processing of a single frame when in pause mode.
        Used for manual frame-by-frame control and loopback scenarios.
        
        Args:
            is_manual (bool): True if called manually, False if called automatically
        """

        
        # Check current state
        is_paused = self.ownerComp.par.Pausestream.eval()
        is_feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
        is_stream_active = self.ownerComp.par.Streamactive.eval()
        
        # Determine if we should process based on manual vs automatic call
        should_process = False
        
        if is_stream_active:
            if is_paused and is_feedback_safe:
                # Manually paused + feedback safe: only allow manual process frame calls
                should_process = is_manual
                if not is_manual:
                    self.logger.log("Auto process frame ignored - manually paused (use manual process frame)", level='DEBUG')
            elif is_paused:
                # Only manually paused: allow manual calls
                should_process = is_manual
            elif is_feedback_safe:
                # Only feedback safe (not manually paused): allow automatic calls
                should_process = not is_manual
            else:
                # Neither paused nor feedback safe: check if we're in transition grace period
                time_since_disable = time.time() - self.last_feedback_safe_disable_time
                grace_period = 2.0  # 2 second grace period for queued automatic calls
                
                if not is_manual and time_since_disable < grace_period:
                    # Allow automatic calls during grace period (queued from when feedback safe was enabled)
                    should_process = False  # Still don't process, but don't log as error
                    self.logger.log(f"Auto process frame ignored - feedback safe recently disabled ({time_since_disable:.1f}s ago)", level='DEBUG')
                else:
                    # Normal case: shouldn't be calling process frame
                    should_process = False
        
        if should_process:
            current_timestamp = str(datetime.datetime.now())
            message_dict = {'/process_frame': 1}
            self.send_osc_messages(message_dict)
            
            # Keep timers alive during feedback safe mode
            if is_feedback_safe:
                op('timer_stream').par.start.pulse()
                op('timer_server').par.start.pulse()
            
            call_type = "manual" if is_manual else "automatic"
            # self.logger.log(f"Triggered {call_type} single frame processing", level='DEBUG')
        else:
            call_type = "manual" if is_manual else "automatic"
            self.logger.log(f"Process frame ignored - {call_type} call not appropriate for current state (paused={is_paused}, feedback_safe={is_feedback_safe})", level='DEBUG')

    def onFrameReady(self, frame_count):
        """
        Called when server signals that a frame is ready.
        Send acknowledgment immediately for proper loopback synchronization.
        """
        try:
            current_timestamp = str(datetime.datetime.now())
            callback_data = {
                'timestamp': current_timestamp,
                'frame_count': frame_count,
                'sync_type': 'server_frame_ready'
            }
            
            # self.logger.log(f"✓ RECEIVED /frame_ready signal from server for frame {frame_count}", level='INFO')
            
            # LOOPBACK SYNCHRONIZATION: Send frame acknowledgment when in pause mode, feedback safe mode, 
            # OR during grace period after feedback safe is disabled (backend still expects acknowledgments)
            is_paused = self.ownerComp.par.Pausestream.eval()
            is_feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
            
            # Check if we're in the grace period after feedback safe was disabled
            time_since_disable = time.time() - self.last_feedback_safe_disable_time
            grace_period = 5.0  # Match backend's 5 second acknowledgment timeout
            in_feedback_safe_grace_period = not is_feedback_safe and time_since_disable < grace_period
            
            if is_paused or is_feedback_safe or in_feedback_safe_grace_period:
                success = self.send_frame_acknowledgment()
                if not success:
                    self.logger.log("FAILED to send frame acknowledgment!", level='ERROR')
                elif in_feedback_safe_grace_period:
                    self.logger.log(f"Sent frame acknowledgment during feedback safe grace period ({time_since_disable:.1f}s ago)", level='DEBUG')
            
            # Call frame ready callback if enabled
            if hasattr(self.ownerComp.par, 'Onframeready') and self.ownerComp.par.Onframeready:
                self.ownerComp.DoCallback("onFrameReady", callback_data)
                
        except Exception as e:
            self.logger.log(f"Error in onFrameReady callback: {str(e)}", level='ERROR')

    def Streamactive(self):
        if not self.ownerComp.par.Serveractive:
            self.ownerComp.par.Streamactive = False 
        else:
            if self.ownerComp.par.Streamactive:
                op('timer_stream').par.start.pulse()
                # Enable shared memory change detection when stream starts
                self.enable_shmem_change_detection(enable=True)
        current_timestamp = str(datetime.datetime.now())
        callback_data = {
            'timestamp': current_timestamp,
        }
        if self.ownerComp.par.Streamactive:
            self.Updatestreamname()
            self.Updatesettings()
            self.Promptblock()
            self.Seed()  # Add missing seed synchronization on stream start
            self.Loradict()  # Also add LoRA sync for completeness
            self.Textualinvdict()  # Add textual inversion sync for completeness
            # self.Enablefx()    
            run("me.Usecontrolnet()", fromOP = self.ownerComp, delayFrames = 10)
            run("me.Updatesettings()", fromOP = self.ownerComp, delayFrames = 10)
            run("me.Seed()", fromOP = self.ownerComp, delayFrames = 15)  # Add delayed seed sync as backup
            op('numpy_share_out').par.reinitextensions.pulse()
            op('numpy_share_out_cn').par.reinitextensions.pulse()
            if self.last_stop_time and time.time() - self.last_stop_time > 1:
                if self.ownerComp.par.Pausestream.eval():
                    return
                if self.ownerComp.par.Onstreamstart:
                    self.ownerComp.DoCallback("onStreamStart", callback_data)
                self.logger.log("onStreamStart: StreamDiffusion has begun streaming", f"Called onStreamStart with data: {callback_data}", level='INFO')
        else:
            # Disable shared memory change detection when stream stops
            # self.enable_shmem_change_detection(enable=False)
            pause = self.ownerComp.par.Pausestream.eval() #bool
            server = self.ownerComp.par.Serveractive.eval() #bool

    def onStreamEnd(self, pause, server):
        current_timestamp = str(datetime.datetime.now())
        callback_data = {
            'timestamp': current_timestamp,
            'is_paused': pause,
            'server_status': server,
        }
        
        # FEEDBACK SAFE MODE: If stream ends but we're in feedback safe mode, try to restart processing
        is_feedback_safe = self.ownerComp.par.Feedbacksafe.eval()
        if is_feedback_safe and server and not pause:
            self.logger.log("onStreamEnd: Stream ended in feedback safe mode - attempting to restart frame processing", level='INFO')
            # Try to trigger another frame to keep the feedback loop going
            try:
                self.Processframe(is_manual=False)  # Automatic restart
                self.logger.log("Successfully triggered restart frame in feedback safe mode", level='INFO')
                return  # Don't call the normal onStreamEnd callback - we're restarting
            except Exception as e:
                self.logger.log(f"Failed to restart feedback safe processing: {str(e)}", level='WARNING')
                # Fall through to normal stream end handling
        
        # Normal stream end handling
        if self.ownerComp.par.Onstreamend:
            self.ownerComp.DoCallback("onStreamEnd", callback_data)	
            if pause and server:
                self.logger.log("onStreamEnd: PAUSED.", f"Called onStreamEnd with data: {callback_data}", level='INFO')
            elif not server:
                self.logger.log("onStreamEnd: SERVER SHUTDOWN.", f"Called onStreamEnd with data: {callback_data}", level='INFO')
            else:
                self.logger.log("onStreamEnd: StreamDiffusion has ended streaming.", f"Called onStreamEnd with data: {callback_data}", level='INFO')

    def onServerEnd(self):
        self.logger.log("The server is shut down.", f"logged from onServerEnd", level='INFO')

    # def Enablefx(self):
    #     osc_out = op('oscout1')
    #     self.send_parameter_update('disable_cached_attn', 1-self.ownerComp.par.Enablefx.eval(), 'single')

    def Synccompcreate(self):
        """
        Duplicates the '_content' operator inside self.ownerComp to the parent level and positions it to the left and up a bit.
        """
        content_op = self.ownerComp.op('_content')
        if content_op:
            duplicated_op = self.ownerComp.parent().copy(content_op)
            duplicated_op.nodeX = self.ownerComp.nodeX - 200  # Move left by 200 units
            duplicated_op.nodeY = self.ownerComp.nodeY + 125  # Move up by 125 units
            duplicated_op.allowCooking = 1 
            new_name = duplicated_op.name
            self.logger.log('StreamDiffusionExt: SyncComp created',
                            f'Operator {new_name} duplicated to parent level and repositioned.',
                            level='INFO')
            callback_cmd = f"\tif op('{new_name}'): op('{new_name}').par.Play = 1; run(\"op('{new_name}').par.Play = 0\", delayFrames=1, fromOP=me)"
            callbackdat = op(self.ownerComp.par.Callbackdat)
            current_callback_code = callbackdat.text
            insert_point = current_callback_code.find('def onReceiveFrame(info):') + len('def onReceiveFrame(info):')
            if callback_cmd not in current_callback_code:
                new_callback_code = (current_callback_code[:insert_point] + 
                                     '\n' + callback_cmd + 
                                     current_callback_code[insert_point:])
                callbackdat.text = new_callback_code
                self.logger.log('StreamDiffusionExt: Callback updated',
                                f'Callback command inserted correctly for operator {new_name}.',
                                level='INFO')
            else:
                self.logger.log('StreamDiffusionExt: Callback already exists',
                                f'Callback command for operator {new_name} is already present.',
                                level='INFO')
        else:
            self.logger.log('StreamDiffusionExt: SyncComp creation failed',
                            'No operator named "_content" found inside the component.',
                            level='ERROR')            

    def Updatestreamname(self):
        try:
            source_name = op('stream_osc_data')[1, 'output-name'].val.lower()
            current_Streaminname = self.ownerComp.par.Streaminname.eval()
            # Check if any of the menu options in self.ownerComp.par.Streaminname contain the source_name
            for i, menu_label in enumerate(self.ownerComp.par.Streaminname.menuLabels):
                if source_name in menu_label.lower():
                    # If found, set the par.Streaminname to that
                    self.ownerComp.par.Streaminname = self.ownerComp.par.Streaminname.menuNames[i]
                    break
            # If par.Streaminname has changed, log the change
            if self.ownerComp.par.Streaminname.eval() != current_Streaminname:
                self.logger.log(f'Source Detected. Source name changed to {self.ownerComp.par.Streaminname.eval()}', level='INFO')
        except:
            pass

    def Clonestreamdiffusion(self):
        if self.message_box_open:
            return
        self.message_box_open = True
        repo_url = 'https://github.com/livepeer/StreamDiffusion.git'
        base_folder_param = self.ownerComp.par.Basefolder
        chosen_folder = base_folder_param.eval() if base_folder_param and base_folder_param.eval() else None

        # If no folder chosen yet, ask for one
        if not chosen_folder:
            chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository')
            if not chosen_folder:
                self.message_box_open = False
                return False

        # Check for spaces BEFORE doing anything else
        while ' ' in chosen_folder:
            choice = ui.messageBox('Installation Path Error',
                                f'The selected path contains spaces which will cause errors:\n\n' +
                                f'Selected path: {chosen_folder}\n\n' +
                                'Would you like to choose a different location?',
                                buttons=['Choose New Location', 'Cancel'])
            if choice == 0:  # Choose New Location
                chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository (No Spaces)')
                if not chosen_folder:
                    self.message_box_open = False
                    return False
            else:  # They hit Cancel
                self.message_box_open = False
                return False

        # NOW we can check if it's a git repo and proceed with cloning
        if os.path.isdir(os.path.join(chosen_folder, '.git')):
            # If it is, ask the user what they want to do
            choice = ui.messageBox('Git Repository Detected',
                                "The selected folder is already a Git repository.\n"
                                "What would you like to do?",
                                buttons=['Update Repo', 'Choose New Download Location', 'Cancel'])
            if choice == 0:  # Update Repo
                try:
                    command = ['git', '-C', chosen_folder, 'pull']
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    stdout, stderr = process.communicate()
                    if process.returncode != 0:
                        ui.messageBox('Error', f'Error updating repository:\n{stderr}')
                        self.logger.log(f'rtLCM: Error updating repository:\n{stderr}', level='ERROR')
                        self.message_box_open = False
                        return False
                    else:
                        self.logger.log('Update successful.', level='INFO')
                        self.message_box_open = False
                        return True
                except Exception as e:
                    ui.messageBox('Error', f'Failed to execute the command: {e}')
                    self.logger.log(f'rtLCM: Failed to execute the command: {e}', level='ERROR')
                    self.message_box_open = False
                    return False
            elif choice == 1:  # Pick New Location for Fresh Download
                chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository')
                if chosen_folder is None:
                    self.message_box_open = False
                    return False
                # Check new location for spaces
                while ' ' in chosen_folder:
                    choice = ui.messageBox('Installation Path Error',
                                        f'The selected path contains spaces which will cause errors:\n\n' +
                                        f'Selected path: {chosen_folder}\n\n' +
                                        'Would you like to choose a different location?',
                                        buttons=['Choose New Location', 'Cancel'])
                    if choice == 0:  # Choose New Location
                        chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository (No Spaces)')
                        if not chosen_folder:
                            self.message_box_open = False
                            return False
                    else:  # They hit Cancel
                        self.message_box_open = False
                        return False
            else:  # Cancel or 'x' button
                self.message_box_open = False
                return False

        # If it's not a git repository, or if the user chose to download to a new location, proceed with cloning
        success = self.clone_git_to_folder(repo_url, chosen_folder=chosen_folder, folder_parameter='Basefolder')
        self.message_box_open = False
        if success:
            self.logger.log('Download successful.', level='INFO')
        else:
            self.logger.log('Download failed.', level='ERROR')

    def clone_git_to_folder(self, repo_url, chosen_folder=None, folder_parameter=None):
        if not self.is_git_installed():
            ui.messageBox('Git Not Found', 'Git is not installed on this system. Please install Git.')
            self.logger.log('rtLCM: Git Not Found', level='ERROR')
            return False
        # if not self.is_connected():
        #     ui.messageBox('Error', 'No internet connection detected.')
        #     self.logger.log('rtLCM: No internet connection detected.', level='ERROR')
        #     return False
        if chosen_folder is None:
            chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository')
        if chosen_folder is None:
            return False
        git_folder_name = repo_url.split('/')[-1].replace('.git', '')
        clone_destination = os.path.join(chosen_folder, git_folder_name)
        if (platform.system() == 'Windows'):
            clone_destination = clone_destination.replace('/', '\\')  # Ensure the path uses backslashes
        try:
            command = ['git', 'clone', repo_url, clone_destination]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                ui.messageBox('Error', f'Error cloning repository:\n{stderr}')
                self.logger.log(f'rtLCM: Error cloning repository:\n{stderr}', level='ERROR')
                return False
        except Exception as e:
            ui.messageBox('Error', f'Failed to execute the command: {e}')
            self.logger.log(f'rtLCM: Failed to execute the command: {e}', level='ERROR')
            return False
        if folder_parameter:
            try:
                setattr(self.ownerComp.par, folder_parameter, clone_destination)
            except Exception as e:
                ui.messageBox('Error', f'Failed to set the parameter: {folder_parameter}')
                self.logger.log(f'rtLCM: Failed to set the parameter: {folder_parameter}', level='ERROR')
                return False
        return True
    
    def is_git_installed(self):
        try:
            subprocess.check_output(["git", "--version"])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def is_connected(self):
        try:
            socket.create_connection(("www.google.com", 80))
            return True
        except OSError:
            return False

    def is_git_lfs_installed(self):
        try:
            subprocess.check_output(["git", "lfs", "version"], creationflags=subprocess.CREATE_NO_WINDOW)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_git_lfs(self):
        try:
            subprocess.check_call(["git", "lfs", "install"])
            self.logger.log("Git LFS installed successfully.", level="INFO")
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Failed to install Git LFS. Error: {e}", level="ERROR")

    def Clonelocalmodels(self):
        """
        Clones the specified repositories from Hugging Face into the local models directory.
        """
        if self.ownerComp.par.Basefolder.eval() in ['', None]:
            self.logger.log(f'Warning: Basefolder is not set. Please set the Basefolder parameter to proceed.', level='WARNING')
            return
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        models_folder = os.path.join(base_folder, 'models')
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        model_info = [
            # ('https://huggingface.co/madebyollin/taesd', 'VAE/taesd', 'Customvae'),
            ('https://huggingface.co/latent-consistency/lcm-lora-sdv1-5', 'LCM_LoRA/lcm-lora-sdv1-5', 'Customlcm')
        ]
        for repo_url, model_subpath, param_name in model_info:
            target_folder = os.path.join(models_folder, model_subpath)
            was_cloned = self.clone_repo(repo_url, target_folder)
            self.update_local_model_par(was_cloned, target_folder, param_name)

    def clone_repo(self, repo_url, target_folder):
        """
        Clones a git repository to a specified target folder.
        Returns True if cloning was successful, False otherwise.
        """
        if os.path.exists(os.path.join(target_folder, '.git')):
            self.logger.log(f'Warning: Model download skipped. {target_folder} is already a git repository. Cloning skipped.', level='WARNING')
            return True
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        clone_command = ['git', 'clone', repo_url, target_folder]
        try:
            subprocess.check_call(clone_command)
            self.logger.log(f'Successfully cloned {repo_url} into {target_folder}', level='INFO')
            # Remove the .git/lfs/objects directory to save space
            git_lfs_objects_path = os.path.join(target_folder, '.git', 'lfs', 'objects')
            if os.path.exists(git_lfs_objects_path):
                shutil.rmtree(git_lfs_objects_path)
                self.logger.log(f'Removed Git LFS objects from {target_folder}', level='INFO')
            return True
        except subprocess.CalledProcessError as e:
            self.logger.log(f'Failed to clone {repo_url}. Error: {e}', level='ERROR')
            return False
        except Exception as e:
            self.logger.log(f'Unexpected error occurred while cloning {repo_url}. Error: {e}', level='ERROR')
            return False

    def check_tensorrt_engine_compatibility(self):
        """
        Checks if the current model specifications and parameter settings
        match an existing TensorRT engine.
        
        Returns:
        bool: True if a compatible engine exists, False otherwise.
        str: A message explaining the result.
        """
        if self.ownerComp.par.Acceleration.eval() != 'tensorrt':
            return False, "TensorRT is not selected as the acceleration method."

        engines_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'engines')
        if not os.path.exists(engines_folder):
            return False, "No engines folder found. TensorRT engines need to be built."

        current_config = self.get_current_engine_config()
        engine_pattern = self.generate_engine_pattern(current_config)

        for root, dirs, files in os.walk(engines_folder):
            for subdir in dirs:
                if engine_pattern.search(subdir):
                    engine_path = os.path.join(root, subdir)
                    required_files = ['unet.engine', 'vae_encoder.engine', 'vae_decoder.engine']
                    missing_files = [file for file in required_files if file not in os.listdir(engine_path)]
                    
                    if not missing_files:
                        return True, f"Compatible TensorRT engine found: {subdir}"
                    else:
                        return False, f"Incomplete engine found: {subdir}. Missing files: {', '.join(missing_files)}"

        return False, "No compatible TensorRT engine found. A new engine may need to be built, which could take a long time."

    def get_current_engine_config(self):
        return {
            'model_id': self.normalize_model_id(self.ownerComp.par.Modelid.eval()),
            'lcm_lora': self.normalize_model_id(self.ownerComp.par.Customlcm.eval()) if self.ownerComp.par.Usecustomlcm.eval() else None,
            'max_batch': self.ownerComp.par.Tindexblock.sequence.numBlocks,
            'width': self.ownerComp.par.Width.eval(),
            'height': self.ownerComp.par.Height.eval()
        }

    def update_local_model_par(self, was_cloned, target_folder, parameter_name):
        """
        Updates the component parameters if the model is downloaded successfully.
        """
        if was_cloned:
            use_parameter_name = 'Use' + parameter_name.lower()
            setattr(self.ownerComp.par, use_parameter_name, True)
            setattr(self.ownerComp.par, parameter_name, target_folder)
            self.logger.log(f'Parameters updated for {parameter_name} with {target_folder}', level='INFO')

    def get_venv_python(self):
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        venv_path = os.path.join(base_folder, 'venv')
        
        if os.path.exists(venv_path):
            if platform.system() == 'Windows':
                return os.path.join(venv_path, 'Scripts', 'python.exe')
            elif platform.system() == 'Darwin':  # macOS
                return os.path.join(venv_path, 'bin', 'python')
            else:
                self.logger.log("Unsupported operating system.", level="ERROR")
                return None
        else:
            self.logger.log("Virtual environment not found.", level="ERROR")
            return None

    def get_daydream_venv_python(self):
        """Get Python executable from virtual environment that contains Daydream dependencies"""
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        
        # Check for venv in order of preference: main venv, .venv, daydream_venv
        venv_paths = [
            os.path.join(base_folder, 'venv'),
            os.path.join(base_folder, '.venv'),
            os.path.join(base_folder, 'daydream_venv')
        ]
        
        for venv_path in venv_paths:
            if os.path.exists(venv_path):
                if platform.system() == 'Windows':
                    python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
                    site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
                elif platform.system() == 'Darwin':  # macOS
                    python_exe = os.path.join(venv_path, 'bin', 'python')
                    # Find the correct python version folder
                    lib_path = os.path.join(venv_path, 'lib')
                    site_packages = None
                    if os.path.exists(lib_path):
                        for item in os.listdir(lib_path):
                            if item.startswith('python3.'):
                                site_packages = os.path.join(lib_path, item, 'site-packages')
                                break
                else:
                    self.logger.log("Unsupported operating system.", level="ERROR")
                    return None
                    
                if os.path.exists(python_exe) and site_packages and os.path.exists(site_packages):
                    # Check for Daydream dependencies by looking for package folders
                    required_packages = ['requests', 'pythonosc', 'numpy']
                    packages_found = []
                    
                    try:
                        for item in os.listdir(site_packages):
                            # Check for package folders or .dist-info folders
                            item_lower = item.lower()
                            if any(pkg in item_lower for pkg in required_packages):
                                if item_lower.startswith('requests'):
                                    packages_found.append('requests')
                                elif 'pythonosc' in item_lower or 'python_osc' in item_lower:
                                    packages_found.append('pythonosc')
                                elif item_lower.startswith('numpy'):
                                    packages_found.append('numpy')
                        
                        # Remove duplicates
                        packages_found = list(set(packages_found))
                        
                        if len(packages_found) >= 3:  # All required packages found
                            self.logger.log(f"Using Python from {venv_path} for Daydream (found: {', '.join(packages_found)})", level="DEBUG")
                            return python_exe
                        else:
                            missing = [pkg for pkg in required_packages if pkg not in packages_found]
                            self.logger.log(f"Venv {venv_path} missing packages: {', '.join(missing)}", level="DEBUG")
                    except Exception as e:
                        self.logger.log(f"Error checking packages in {site_packages}: {e}", level="DEBUG")
                        continue
        
        self.logger.log("No virtual environment with Daydream dependencies found.", level="ERROR")
        return None

    def run_script_in_venv(self, script):
        venv_python = self.get_venv_python()
        if not venv_python:
            return None
        try:
            if platform.system() == 'Windows':
                creation_flags = subprocess.CREATE_NO_WINDOW
            else:
                creation_flags = 0
            process = subprocess.Popen(
                [venv_python, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                creationflags=creation_flags
            )
            stdout, stderr = process.communicate()
            if stderr:
                self.logger.log(f"Error running script: {stderr}", level="ERROR")
                return None
            return stdout
        except Exception as e:
            self.logger.log(f"Error running script: {str(e)}", level="ERROR")
            return None

    def Dlhfmodel(self):
        self.Gethfmodelinfo(show_in_viewer=False)
        model_id = self.ownerComp.par.Dlhfmodelid.eval()
        model_type = self.ownerComp.par.Dlhfmodeltype.eval()
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        if model_type == 'SD':
            target_folder = os.path.join(base_folder, 'models', 'Model')
        elif model_type == 'LoRA':
            target_folder = tdu.expandPath(self.ownerComp.par.Lorafolder.eval())
        elif model_type == 'VAE':
            target_folder = os.path.join(base_folder, 'models', 'VAE')
        elif model_type == 'LCM':
            target_folder = os.path.join(base_folder, 'models', 'LCM')
        else:
            self.logger.log(f'Unknown model type: {model_type}.', level='ERROR')
            return

        if model_id.startswith('http') and any(model_id.endswith(ext) for ext in ['.safetensors', '.ckpt', '.pt', '.bin']):
            self.download_direct_file(model_id, model_type, target_folder)
            return

        if not self.is_git_lfs_installed():
            if ui.messageBox('Git LFS Required', 'Git LFS might be required. Install now?', buttons=['Install', 'Cancel']) == 0:
                self.install_git_lfs()
            else:
                self.logger.log("Git LFS installation canceled. Download aborted.", level="WARNING")
                return

        if os.path.exists(target_folder):
            choice = ui.messageBox('Folder Exists', f"The folder '{target_folder}' already exists. What would you like to do?", 
                                buttons=['Update', 'Delete and Download Fresh', 'Cancel'])
            if choice == 0:
                try:
                    subprocess.check_call(['git', '-C', target_folder, 'pull'])
                    self.logger.log(f'Successfully updated the repository at {target_folder}', level='INFO')
                    return
                except subprocess.CalledProcessError as e:
                    self.logger.log(f'Failed to update the repository at {target_folder}. Error: {e}', level='ERROR')
                    return
            elif choice == 1:
                try:
                    shutil.rmtree(target_folder)
                except Exception as e:
                    self.logger.log(f'Failed to delete the folder at {target_folder}. Error: {e}', level='ERROR')
                    ui.messageBox('Delete Failed', f"Failed to delete the folder at {target_folder}. Please manually remove the folder and try again.")
                    return
            else:
                return

        os.makedirs(target_folder, exist_ok=True)
        repo_url = model_id if "https://" in model_id else f'https://huggingface.co/{model_id}'
        try:
            subprocess.check_call(['git', 'clone', repo_url, target_folder])
            self.logger.log(f'Successfully cloned {model_id} into {target_folder}', level='INFO')
            git_lfs_objects_path = os.path.join(target_folder, '.git', 'lfs', 'objects')
            if os.path.exists(git_lfs_objects_path):
                shutil.rmtree(git_lfs_objects_path)
                self.logger.log(f'Removed Git LFS objects from {target_folder}', level='INFO')
            if model_type == 'SD':
                self.Savemodel(target_folder)
                self.sync_model_table()
        except subprocess.CalledProcessError as e:
            self.logger.log(f'Failed to clone {model_id}. Error: {e}', level='ERROR')

    def download_direct_file(self, file_url, model_type, target_folder):
        parsed_url = urlparse(file_url)
        filename = os.path.basename(parsed_url.path)
        target_path = os.path.join(target_folder, filename)
        try:
            if file_url.startswith("https://huggingface.co/"):
                # Use Hugging Face Hub download through venv
                venv_python = os.path.join(self.ownerComp.par.Basefolder.eval(), 'venv', 'Scripts', 'python.exe')
                if not os.path.exists(venv_python):
                    venv_python = os.path.join(self.ownerComp.par.Basefolder.eval(), '.venv', 'Scripts', 'python.exe')
                if not os.path.exists(venv_python):
                    raise FileNotFoundError(f"Python executable not found in virtual environment at {self.ownerComp.par.Basefolder.eval()}")

                script = f"""
import os
from huggingface_hub import hf_hub_download
url = "{file_url}"
target_folder = r"{target_folder}"
repo_id = url.split("/blob/")[0].split("https://huggingface.co/")[1]
filename = url.split("/")[-1]
file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=target_folder, cache_dir=target_folder)
print(file_path)
    """
                result = subprocess.run([venv_python, "-c", script], capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f"Error downloading file: {result.stderr}")
                target_path = result.stdout.strip()
            else:
                # Direct download for non-Hugging Face URLs
                response = requests.get(file_url, stream=True)
                response.raise_for_status()
                with open(target_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
            self.logger.log(f'Successfully downloaded {filename} to {target_path}', level='INFO')
            if model_type == 'SD':
                self.Savemodel(target_path)
                self.sync_model_table()
            return target_path
        except Exception as e:
            self.logger.log(f'Failed to download file from {file_url}. Error: {e}', level='ERROR')
            return None

    def Gethfmodelinfo(self, show_in_viewer=True):
        model_id = self.ownerComp.par.Dlhfmodelid.eval()
        base_folder = self.ownerComp.par.Basefolder.eval()
        # Path to the virtual environment's Python executable
        venv_path = os.path.join(base_folder, 'venv', 'Scripts', 'python.exe')
        if not os.path.exists(venv_path):
            # Try alternative location for virtual environments
            venv_path = os.path.join(base_folder, '.venv', 'Scripts', 'python.exe')
            if not os.path.exists(venv_path):
                self.logger.log("Python executable in virtual environment not found.", level="ERROR")
                return
        if model_id.startswith("https://huggingface.co/"):
            # Extract the model ID from the URL
            match = re.match(r"https://huggingface\.co/([^/]+/[^/]+)", model_id)
            if match:
                model_id = match.group(1)
            else:
                self.logger.log("Invalid Hugging Face URL format", level="ERROR")
                return
        python_script = f"""
import requests
import json
import subprocess

def get_repo_details(model_id):
    url = f"https://huggingface.co/api/models/{model_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {{'error': 'Failed to fetch details', 'status_code': response.status_code}}
model_details = get_repo_details("{model_id}")
print(json.dumps(model_details))
    """
        try:
            # Running the Python script with subprocess
            process = subprocess.Popen([venv_path, "-c", python_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, creationflags=subprocess.CREATE_NO_WINDOW)
            stdout, stderr = process.communicate()
            if stderr:
                self.logger.log(f"Error fetching model details: {stderr}", level="ERROR")
            else:
                model_details = json.loads(stdout)
                print(model_details)
                formatted_text = f"Model Information:\n"
                formatted_text += f"ID: {model_details['id']}\n"
                formatted_text += f"Author: {model_details['author']}\n"
                formatted_text += f"Last Modified: {model_details['lastModified']}\n"
                formatted_text += f"Private: {'Yes' if model_details['private'] else 'No'}\n"
                formatted_text += f"Disabled: {'Yes' if model_details['disabled'] else 'No'}\n"
                formatted_text += f"Pipeline Tag: {model_details['pipeline_tag']}\n"
                formatted_text += f"Tags: {', '.join(model_details['tags'])}\n"
                formatted_text += f"Likes: {model_details['likes']}\n"
                formatted_text += f"Downloads: {model_details['downloads']}\n"
                op('model_data').par.text = formatted_text    
                if show_in_viewer:
                    op('model_data').openViewer()
                if 'lora' in model_details['tags']:
                    self.ownerComp.par.Dlhfmodeltype = 'LoRA'
                return
        except Exception as e:
            self.logger.log(f"Error fetching model details", f"Error fetching model details: {str(e)}", level="WARNING")
            return

    def create_folder_if_not_exists(self, folder_path):
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                self.logger.log(f"Created folder: {folder_path}", level="INFO")
            except OSError as e:
                self.logger.log(f"Error creating folder: {folder_path}. Error: {e}", level="ERROR")

    def Uiviewlora(self):
        self.ownerComp.par.Uiviewlora = 0
        if time.time() - self.last_opened_time > 1:
            lora_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models', 'LoRA')
            self.create_folder_if_not_exists(lora_folder)
            ui.viewFile(lora_folder, showInFolder=False)
            self.last_opened_time = time.time()
        self.ownerComp.par.Uiviewlora = 0

    def Uiviewmodels(self):
        self.ownerComp.par.Uiviewmodels = 0
        if time.time() - self.last_opened_time > 1:
            model_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models')
            self.create_folder_if_not_exists(model_folder)
            ui.viewFile(model_folder, showInFolder=False)
            self.last_opened_time = time.time()
        self.ownerComp.par.Uiviewmodels = 0

    def Uiviewlcm(self):
        self.ownerComp.par.Uiviewlcm = 0
        if time.time() - self.last_opened_time > 1:
            lcm_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models', 'LCM')
            self.create_folder_if_not_exists(lcm_folder)
            ui.viewFile(lcm_folder, showInFolder=False)
            self.last_opened_time = time.time()
        self.ownerComp.par.Uiviewlcm = 0

    def Uiviewvae(self):
        self.ownerComp.par.Uiviewvae = 0
        if time.time() - self.last_opened_time > 1:
            vae_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models', 'VAE')
            self.create_folder_if_not_exists(vae_folder)
            ui.viewFile(vae_folder, showInFolder=False)
            self.last_opened_time = time.time()
        self.ownerComp.par.Uiviewvae = 0

    def find_python_exe(self):
        if platform.system() == 'Darwin':  # macOS specific handling
            # First try Homebrew Python 3.11 specifically
            homebrew_paths = [
                "/opt/homebrew/bin/python3.11",
                "/usr/local/opt/python@3.11/bin/python3.11",
                "/opt/homebrew/opt/python@3.11/bin/python3.11"
            ]
            
            # Try to find Homebrew Python 3.11 first
            for path in homebrew_paths:
                if os.path.exists(path):
                    try:
                        output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                        if version_match and version_match.group(1).startswith('3.11'):
                            return path, version_match.group(1)
                    except:
                        continue

            # If Homebrew Python 3.11 not found, try to find it using brew
            try:
                brew_path = subprocess.check_output(['which', 'brew']).decode().strip()
                if brew_path:
                    python_path = subprocess.check_output([brew_path, '--prefix', 'python@3.11']).decode().strip()
                    python_exe = os.path.join(python_path, 'bin', 'python3.11')
                    if os.path.exists(python_exe):
                        output = subprocess.check_output([python_exe, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                        if version_match:
                            return python_exe, version_match.group(1)
            except:
                pass

        else:  # Windows paths
            # Method 1: Try LOCALAPPDATA environment variable first (most reliable)
            localappdata = os.environ.get('LOCALAPPDATA')
            if localappdata:
                # First check Python.org installations (priority)
                python_paths = [
                    os.path.join(localappdata, 'Programs', 'Python', 'Python311', 'python.exe'),
                    os.path.join(localappdata, 'Programs', 'Python', 'Python310', 'python.exe')
                ]
                for path in python_paths:
                    if os.path.exists(path):
                        try:
                            output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                            version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                            if version_match:
                                version = version_match.group(1)
                                if version.startswith('3.11') or version.startswith('3.10'):
                                    return path, version
                        except:
                            continue
                
                # Then check Microsoft Store paths (fallback)
                ms_store_paths = [
                    os.path.join(localappdata, 'Microsoft', 'WindowsApps', 'python.exe'),
                    os.path.join(localappdata, 'Microsoft', 'WindowsApps', 'python3.exe'),
                    os.path.join(localappdata, 'Microsoft', 'WindowsApps', 'PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0', 'python.exe'),
                    os.path.join(localappdata, 'Microsoft', 'WindowsApps', 'PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0', 'python.exe')
                ]
                for path in ms_store_paths:
                    if os.path.exists(path):
                        try:
                            output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                            version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                            if version_match:
                                version = version_match.group(1)
                                if version.startswith('3.11') or version.startswith('3.10'):
                                    return path, version
                        except:
                            continue

            # Method 2: Try multiple username detection methods (backup approach)
            usernames = set()
            try:
                usernames.add(os.getlogin())
                usernames.add(os.environ.get('USERNAME'))
                usernames.add(os.path.expanduser('~').split('\\')[-1])
            except:
                pass

            # Add an alternative approach for paths with spaces
            user_home = os.path.expanduser('~')
            if user_home and os.path.exists(user_home):
                appdata_path = os.path.join(user_home, 'AppData', 'Local')
                if os.path.exists(appdata_path):
                    # First check Python.org installations (priority)
                    python_paths_alt = [
                        os.path.join(appdata_path, 'Programs', 'Python', 'Python311', 'python.exe'),
                        os.path.join(appdata_path, 'Programs', 'Python', 'Python310', 'python.exe')
                    ]
                    
                    for path in python_paths_alt:
                        if os.path.exists(path):
                            try:
                                output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                                version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                                if version_match:
                                    version = version_match.group(1)
                                    if version.startswith('3.11') or version.startswith('3.10'):
                                        return path, version
                            except:
                                continue
                    
                    # Then check Microsoft Store paths (fallback)
                    ms_store_paths_alt = [
                        os.path.join(appdata_path, 'Microsoft', 'WindowsApps', 'python.exe'),
                        os.path.join(appdata_path, 'Microsoft', 'WindowsApps', 'python3.exe'),
                        os.path.join(appdata_path, 'Microsoft', 'WindowsApps', 'PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0', 'python.exe'),
                        os.path.join(appdata_path, 'Microsoft', 'WindowsApps', 'PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0', 'python.exe')
                    ]
                    for path in ms_store_paths_alt:
                        if os.path.exists(path):
                            try:
                                output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                                version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                                if version_match:
                                    version = version_match.group(1)
                                    if version.startswith('3.11') or version.startswith('3.10'):
                                        return path, version
                            except:
                                continue

            for username in usernames:
                if username:
                    # First check Python.org installations (priority)
                    python_paths = [
                        f"C:/Users/{username}/AppData/Local/Programs/Python/Python311/python.exe",
                        f"C:/Users/{username}/AppData/Local/Programs/Python/Python310/python.exe"
                    ]
                    for path in python_paths:
                        if os.path.exists(path):
                            try:
                                output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                                version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                                if version_match:
                                    version = version_match.group(1)
                                    if version.startswith('3.11') or version.startswith('3.10'):
                                        return path, version
                            except:
                                continue
                    
                    # Then check Microsoft Store paths (fallback)
                    ms_store_paths = [
                        f"C:/Users/{username}/AppData/Local/Microsoft/WindowsApps/python.exe",
                        f"C:/Users/{username}/AppData/Local/Microsoft/WindowsApps/python3.exe",
                        f"C:/Users/{username}/AppData/Local/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/python.exe",
                        f"C:/Users/{username}/AppData/Local/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0/python.exe"
                    ]
                    for path in ms_store_paths:
                        if os.path.exists(path):
                            try:
                                output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                                version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                                if version_match:
                                    version = version_match.group(1)
                                    if version.startswith('3.11') or version.startswith('3.10'):
                                        return path, version
                            except:
                                continue

            # Method 3: Check standard installation locations (final fallback)
            standard_paths = [
                "C:/Python311/python.exe",
                "C:/Program Files/Python311/python.exe",
                "C:/Program Files (x86)/Python311/python.exe",
                "C:/Python310/python.exe",
                "C:/Program Files/Python310/python.exe",
                "C:/Program Files (x86)/Python310/python.exe"
            ]
            
            for path in standard_paths:
                if os.path.exists(path):
                    try:
                        output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                        if version_match:
                            version = version_match.group(1)
                            if version.startswith('3.11') or version.startswith('3.10'):
                                return path, version
                    except:
                        continue

            # Final attempt: Try to find python in PATH
            try:
                result = subprocess.run("where python", capture_output=True, text=True, shell=True)
                if result.returncode == 0 and result.stdout.strip():
                    python_path = result.stdout.strip().split("\n")[0]
                    if os.path.exists(python_path):
                        output = subprocess.check_output([python_path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                        if version_match:
                            version = version_match.group(1)
                            if version.startswith('3.11') or version.startswith('3.10'):
                                return python_path, version
            except:
                pass
                
            # Also try python3 in PATH
            try:
                result = subprocess.run("where python3", capture_output=True, text=True, shell=True)
                if result.returncode == 0 and result.stdout.strip():
                    python_path = result.stdout.strip().split("\n")[0]
                    if os.path.exists(python_path):
                        output = subprocess.check_output([python_path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                        version_match = re.search(r'Python (\d+\.\d+\.\d+)', output)
                        if version_match:
                            version = version_match.group(1)
                            if version.startswith('3.11') or version.startswith('3.10'):
                                return python_path, version
            except:
                pass

        return None, None  # Return None for both path and version if no matching executable is found

    def check_venv_python_version(self, venv_path):
        if platform.system() == 'Windows':
            python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
            creation_flags = subprocess.CREATE_NO_WINDOW
        else:
            python_exe = os.path.join(venv_path, 'bin', 'python')
            creation_flags = 0

        if not os.path.exists(python_exe):
            self.logger.log(f"Python executable not found in virtual environment at {venv_path}", level="ERROR")
            return

        try:
            if platform.system() == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                version_output = subprocess.check_output(
                    [python_exe, "--version"],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                    startupinfo=startupinfo,
                    creationflags=creation_flags
                ).strip()
            else:
                # Fixed version for macOS - removed stdout argument
                version_output = subprocess.check_output(
                    [python_exe, "--version"],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT
                ).strip()

            version_match = re.search(r'Python (\d+\.\d+\.\d+)', version_output) 
            if version_match:
                python_version = version_match.group(1)
                self.logger.log(f"Python version in venv: {python_version}", level="DEBUG")
                return python_version
            else:
                self.logger.log(f"Unable to parse Python version from output: {version_output}", level="ERROR")
                return None
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Error checking Python version: {str(e)}", level="ERROR")
            return None
        except Exception as e:
            self.logger.log(f"Unexpected error checking Python version: {str(e)}", level="ERROR")
            return None
        
    def check_venv_cuda_version(self, venv_path):
        if platform.system() == 'Darwin':  # macOS
            return None  # macOS doesn't use CUDA
            
        # Windows path handling
        try:
            site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
            if not os.path.exists(site_packages):
                return None
                
            for folder in os.listdir(site_packages):
                if folder.startswith('torch-') and '+cu' in folder:
                    cuda_version_match = re.search(r'\+cu(\d+)', folder)
                    if cuda_version_match:
                        cuda_version = f'cu{cuda_version_match.group(1)}'
                        return cuda_version        
        except (FileNotFoundError, OSError):
            return None
            
        return None


    def check_system_cuda_version(self):
        try:
            cuda_version_output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, universal_newlines=True)
            version_match = re.search(r'release (\d+\.\d+)', cuda_version_output)
            if version_match:
                detected_version = version_match.group(1)
                cuda_version = f'cu{detected_version.replace(".", "")}'

                version_info = {
                    'cu118': {
                        'status': '✓ Stable',
                        'config': 'PyTorch 2.1.0 + xFormers (default build)',
                        'notes': 'Fully supported'
                    },
                    'cu121': {
                        'status': '✓ Stable',
                        'config': 'PyTorch 2.1.0 + xFormers (default build)',
                        'notes': 'Fully supported'
                    },
                    'cu124': {
                        'status': '✓ Stable',
                        'config': 'PyTorch 2.1.1 + xFormers 0.0.23 (cu121 build)',
                        'notes': 'Fully supported - Using cu121 build'
                    },
                    'cu128': {
                        'status': '✓ Blackwell Ready',
                        'config': 'PyTorch 2.7.0 stable (cu128)',
                        'notes': 'Fully supported with Blackwell architecture (sm_120) for RTX 50-series GPUs.'
                    },
                    'cu129': {
                        'status': '✓ Blackwell Ready',
                        'config': 'PyTorch 2.7.0 stable (cu128)',
                        'notes': 'Fully supported with Blackwell architecture (sm_120) for RTX 50-series GPUs.'
                    },
                    'cu130': {
                        'status': '✓ Blackwell Ready',
                        'config': 'PyTorch 2.7.0 stable (cu128)',
                        'notes': 'Fully supported with Blackwell architecture (sm_120) for RTX 50-series GPUs. Using cu128 wheels (highest available).'
                    }
                }

                message = f"CUDA version: {detected_version}\n\n"

                if cuda_version in version_info:
                    info = version_info[cuda_version]
                    message += f"Status: {info['status']}\n"
                    message += f"Config: {info['config']}\n"
                    message += f"Notes: {info['notes']}\n\n"

                message += "Supported Versions:\n"
                message += "✓ CUDA 11.8\n"
                message += "✓ CUDA 12.1 (recommended for most GPUs)\n"
                message += "✓ CUDA 12.4 (recommended for most GPUs)\n"
                message += "✓ CUDA 12.8 (Blackwell/RTX 50-series)\n"
                message += "✓ CUDA 12.9 (Blackwell/RTX 50-series)\n"
                message += "✓ CUDA 13.0 (Blackwell/RTX 50-series)\n\n"

                if cuda_version in ['cu128', 'cu129', 'cu130']:
                    message += "ℹ️ Blackwell GPU detected. Using PyTorch 2.7.0 cu128 with sm_120 support.\n"
                    message += "For non-Blackwell GPUs, CUDA 12.1 or 12.4 is recommended.\n\n"

                message += "Continue with detected version?"

                choice = ui.messageBox('CUDA Version Check', message, buttons=['Continue', 'Cancel'])
                if choice != 0:
                    self.message_box_open = False
                    return None
                return cuda_version
            else:
                raise Exception("Unable to parse CUDA version from output.")
        except Exception as e:
            message = "CUDA not detected.\n\n"
            message += "Supported Versions:\n"
            message += "✓ CUDA 11.8 - PyTorch 2.4.0\n"
            message += "✓ CUDA 12.1 (recommended) - PyTorch 2.4.0\n"
            message += "✓ CUDA 12.4 (recommended) - PyTorch 2.4.0\n"
            message += "✓ CUDA 12.8 (Blackwell/RTX 50-series) - PyTorch 2.7.0 (cu128)\n"
            message += "✓ CUDA 12.9 (Blackwell/RTX 50-series) - PyTorch 2.7.0 (cu128)\n"
            message += "✓ CUDA 13.0 (Blackwell/RTX 50-series) - PyTorch 2.7.0 (cu128)\n\n"
            message += "Install a supported CUDA version to continue.\n\n"

            choice = ui.messageBox('CUDA Not Detected', message, buttons=['Download CUDA', 'Proceed Anyway', 'Cancel'])
            if choice == 0:
                import webbrowser
                webbrowser.open('https://developer.nvidia.com/cuda-12-1-0-download-archive')
                self.message_box_open = False
                return None
            elif choice == 1:
                return 'cpu'
            else:
                self.message_box_open = False
                return None

    def Installstreamdiffusion(self):
        
        backend = self.ownerComp.par.Backend.eval()
        if backend == 'Daydream':
            self.Installdaydream()
            return

        if self.message_box_open:
            return
        self.message_box_open = True
        self.copy_sdtd_code()
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        venv_path = os.path.join(base_folder, 'venv')

        if os.path.exists(venv_path):   
            cuda_version = self.check_venv_cuda_version(venv_path)
            self.check_venv_python_version(venv_path)
            if not cuda_version and platform.system() == 'Windows':
                cuda_version = self.check_system_cuda_version()       
            choice = ui.messageBox('Virtual Environment Detected',
                                'A virtual environment already exists. Do you want to update it?',
                                buttons=['Update', 'Cancel'])
            if choice != 0:  # Cancel
                self.message_box_open = False
                return False                    
            python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
        else:
            python_exe, pyversion = self.find_python_exe()
            if not python_exe:
                choice = ui.messageBox('Installation Error ! No Python detected.',
                                    'Python not found. Please ensure Python 3.11.9 or Python 3.10.9 is installed and accessible.\nDo you want to continue anyway?',
                                    buttons=['Continue Anyway', 'Cancel'])
                if choice == 0:
                    python_exe = 'python3'
                else: # Cancel
                    self.message_box_open = False
                    return False
            if platform.system() == 'Windows':
                # Show CUDA version selector popup
                cuda_message = (
                    "Select the PyTorch CUDA version for your GPU:\n\n"
                    "CUDA 12.4 - Recommended for RTX 40/30/20-series\n"
                    "CUDA 12.1 - Compatible with most modern GPUs\n"
                    "CUDA 11.8 - Older GPUs or fallback option\n"
                    "CUDA 12.8/12.9/13.0 - RTX 50-series Blackwell GPUs\n\n"
                    "Note: You do NOT need CUDA Toolkit installed.\n"
                    "PyTorch includes its own CUDA runtime."
                )

                cuda_choice = ui.messageBox('Select PyTorch CUDA Version',
                                          cuda_message,
                                          buttons=['CUDA 12.4 (Recommended)', 'CUDA 12.1', 'CUDA 11.8', 'CUDA 12.8 (RTX 50xx)', 'CUDA 13.0 (RTX 50xx)', 'Cancel'])

                if cuda_choice == 0:
                    cuda_version = 'cu124'
                elif cuda_choice == 1:
                    cuda_version = 'cu121'
                elif cuda_choice == 2:
                    cuda_version = 'cu118'
                elif cuda_choice == 3:
                    cuda_version = 'cu128'
                elif cuda_choice == 4:
                    cuda_version = 'cu130'
                else:  # Cancel
                    self.message_box_open = False
                    return False
            else:
                cuda_version = None
        if platform.system() == 'Windows':
            bat_file_path = os.path.join(base_folder, 'Install_StreamDiffusion.bat')
        else:
            bat_file_path = os.path.join(base_folder, 'Install_StreamDiffusion.sh')

        set_base_folder = False
        if base_folder is None or base_folder == '':
            previous_base_folder = base_folder
            base_folder = project.folder
            self.ownerComp.par.Basefolder = base_folder
            set_base_folder = True

        if not os.path.exists(venv_path):
            installation_details = f"Python version: {python_exe}\nCUDA version: {cuda_version}\nBase folder for venv: {base_folder}\nInstalling packages for: StreamDiffusionTD"
            choice = ui.messageBox('Installation Details',
                                f'{installation_details}\n\nDo you want to proceed with the installation?',
                                buttons=['Install', 'Cancel'])
            if choice != 0:
                if set_base_folder:
                    self.ownerComp.par.Basefolder = previous_base_folder
                self.message_box_open = False
                return False

        if platform.system() == 'Windows':
            requirements_file = 'requirements_pc.txt'
        else:
            requirements_file = 'requirements_mac.txt'

        no_cache = "--no-cache-dir" if self.ownerComp.par.Nocacheinstall else ""

        def get_cuda_install_command(cuda_version, no_cache=""):
            """Returns the appropriate pip install command for each CUDA version - LivePeer fork compatible"""
            cuda_configs = {
                'cu118': {
                    'torch': '2.4.0',
                    'torchvision': '0.19.0', 
                    'cuda_python': '11.8.7',
                    'xformers': ''
                },
                'cu121': {
                    'torch': '2.4.0',
                    'torchvision': '0.19.0',
                    'cuda_python': '12.9.0',  # LivePeer fork requirement
                    'xformers': ''
                },
                'cu124': {
                    'torch': '2.4.0',
                    'torchvision': '0.19.0', 
                    'cuda_python': '12.9.0',  # LivePeer fork requirement
                    'xformers': ''  # Skip xformers - causes version conflicts
                },

                'cu128': {
                    'torch_command': f"python -m pip install {no_cache} torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
                    'cuda_python_command': f"python -m pip install {no_cache} cuda-python==12.9.0",
                    'xformers_command': f""  # xformers not needed - PyTorch 2.7+ has native SDPA attention
                },
                'cu129': {
                    'torch_command': f"python -m pip install {no_cache} torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
                    'cuda_python_command': f"python -m pip install {no_cache} cuda-python==12.9.0",
                    'xformers_command': f""  # XFormers doesn't work on RTX 50-series, PyTorch 2.7+ has native optimizations
                },
                'cu130': {
                    'torch_command': f"python -m pip install {no_cache} torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
                    'cuda_python_command': f"python -m pip install {no_cache} cuda-python==12.9.0",
                    'xformers_command': f""  # XFormers doesn't work on RTX 50-series, PyTorch 2.7+ has native optimizations
                }
            }
            if cuda_version not in cuda_configs:
                return None

            config = cuda_configs[cuda_version]

            if cuda_version == 'cu128':
                # Install PyTorch 2.7.0 stable cu128 with Blackwell sm_120 support (no xformers - native SDPA used)
                return f"{config['torch_command']} && {config['cuda_python_command']}"

            # For CUDA 12.9 and 13.0, use PyTorch 2.7.0 cu128 builds (have Blackwell sm_120 support for RTX 50-series)
            # XFormers skipped - doesn't work on RTX 50-series
            if cuda_version in ['cu129', 'cu130']:
                return f"{config['torch_command']} && {config['cuda_python_command']}"

            # For CUDA 12.4, use cu121 index with LivePeer-compatible versions
            if cuda_version == 'cu124':
                return (
                    f"python -m pip install {no_cache} "
                    f"torch=={config['torch']} "
                    f"torchvision=={config['torchvision']} "
                    f"--index-url https://download.pytorch.org/whl/cu121 && "
                    f"python -m pip install {no_cache} "
                    f"cuda-python=={config['cuda_python']}"
                )

            # For cu118 and cu121 - LivePeer compatible versions
            return (
                f"python -m pip install {no_cache} "
                f"torch=={config['torch']} "
                f"torchvision=={config['torchvision']} "
                f"--index-url https://download.pytorch.org/whl/{cuda_version} && "
                f"python -m pip install {no_cache} "
                f"cuda-python=={config['cuda_python']}"
            )
        # In your main installation code:
        if platform.system() == 'Windows':
            torch_install_cmd = get_cuda_install_command(cuda_version, no_cache)
            if not torch_install_cmd:
                message = "Unsupported CUDA version detected.\n\n"
                message += "StreamDiffusion supports:\n"
                message += "- CUDA 11.8\n"
                message += "- CUDA 12.1\n"
                message += "- CUDA 12.4\n"
                message += "- CUDA 12.8 (Blackwell/RTX 50-series)\n"
                message += "- CUDA 12.9 (Blackwell/RTX 50-series)\n"
                message += "- CUDA 13.0 (Blackwell/RTX 50-series)\n"
                message += "Please install one of these CUDA versions."
                
                ui.messageBox('CUDA Version Error', message, buttons=['OK'])
                self.message_box_open = False
                return
        else:
            torch_install_cmd = ""
        batch_file_content_win = f"""
        @echo off
        echo ============================================================================
        echo StreamDiffusion Installation Script (FIXED for torch CPU issue)
        echo ============================================================================
        echo Current directory: %CD%
        cd /d "{base_folder}"
        echo Changed directory to: %CD%
        set "PIP_DISABLE_PIP_VERSION_CHECK=1"

        if exist "venv" (
            echo Clearing pip cache before update...
            python -m pip cache purge
        )

        if not exist "venv" (
            echo Creating Python venv at: "{base_folder}\\venv"
            "{python_exe}" -m venv venv
        ) else (
            echo Virtual environment already exists at: "{base_folder}\\venv"
        )

        echo Attempting to activate virtual environment...
        call "venv\\Scripts\\activate.bat"

        rem Check if the virtual environment was activated successfully
        if "%VIRTUAL_ENV%" == "" (
            echo Failed to activate virtual environment.
            pause
            exit /b 1
        ) else (
            echo Virtual environment activated.
        )

        echo.
        echo ============================================================================
        echo PHASE 1/7: Base System Setup
        echo ============================================================================
        echo Installing base requirements (pip, setuptools, wheel)...
        python -m pip install {no_cache} --upgrade pip setuptools wheel

        echo Installing compatible NumPy first (fixes NumPy 2.x conflicts)...
        python -m pip install {no_cache} "numpy<2.0.0"

        echo Installing nvidia-pyindex to ensure access to NVIDIA-specific packages...
        python -m pip install {no_cache} nvidia-pyindex --trusted-host pypi.org --trusted-host files.pythonhosted.org

        echo.
        echo ============================================================================
        echo PHASE 2/7: CUDA Stack Installation (CRITICAL - PREVENTS CPU TORCH)
        echo ============================================================================
        echo Installing PyTorch with CUDA support...
        echo This is the most critical step - DO NOT INTERRUPT!
        {torch_install_cmd}

        echo.
        echo Verifying CUDA PyTorch installation...
        python -c "import torch; print(f'PyTorch: {{torch.__version__}}'); print(f'CUDA Available: {{torch.cuda.is_available()}}'); print(f'CUDA Version: {{torch.version.cuda if torch.cuda.is_available() else \"N/A\"}}')" || echo "WARNING: PyTorch verification failed"

        echo.
        echo ============================================================================
        echo PHASE 3/7: Core Dependencies (careful order to prevent torch reinstall)
        echo ============================================================================
        echo Installing core dependencies without pulling torch...
        python -m pip install {no_cache} --no-deps diffusers transformers accelerate omegaconf protobuf

        echo Installing missing dependencies that don't conflict with torch...
        python -m pip install {no_cache} safetensors huggingface_hub regex requests tqdm filelock packaging pyyaml

        echo.
        echo ============================================================================
        echo PHASE 4/7: StreamDiffusion Installation
        echo ============================================================================
        echo Installing LivePeer StreamDiffusion fork (no deps to prevent conflicts)...
        python -m pip install {no_cache} --no-deps git+https://github.com/livepeer/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]

        echo Installing Diffusers IPAdapter (no deps)...
        python -m pip install {no_cache} --no-deps git+https://github.com/livepeer/Diffusers_IPAdapter.git@405f87da42932e30bd55ee8dca3ce502d7834a99

        echo.
        echo ============================================================================
        echo PHASE 5/7: Computer Vision Stack (opencv, controlnet_aux)
        echo ============================================================================
        echo Installing opencv and image processing libraries...
        python -m pip install {no_cache} opencv-python==4.8.1.78 Pillow scipy scikit-image

        echo Installing controlnet_aux WITHOUT dependencies (prevents torch conflicts)...
        python -m pip install {no_cache} --no-deps controlnet_aux

        echo Installing controlnet_aux missing dependencies manually...
        python -m pip install {no_cache} timm mediapipe

        echo.
        echo ============================================================================
        echo PHASE 6/7: TouchDesigner Integration and Utilities
        echo ============================================================================
        echo Installing TouchDesigner-specific packages...
        python -m pip install {no_cache} python-osc pywin32 fire mss einops peft>=0.17.0

        echo Installing matplotlib (large dependency tree, but safe after torch is locked)...
        python -m pip install {no_cache} matplotlib

        echo Installing insightface (optional, may have conflicts)...
        python -m pip install {no_cache} insightface || echo "WARNING: insightface installation failed - this is optional for FaceID"

        echo.
        echo ============================================================================
        echo PHASE 7/7: Final Verification and Fixes
        echo ============================================================================
        echo Fixing any version conflicts...
        python -m pip install {no_cache} "numpy<2.0.0" --force-reinstall

        echo Installing optional performance packages...
        python -m pip install {no_cache} triton || echo "INFO: triton not available - this is normal on Windows"

        echo.
        echo ============================================================================
        echo Final Verification
        echo ============================================================================
        echo Checking PyTorch CUDA availability...
        python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA not available!'; print(f'✅ PyTorch {{torch.__version__}} with CUDA {{torch.version.cuda}}')" || echo "❌ CUDA verification FAILED - check installation"

        echo.
        echo Checking StreamDiffusion installation...
        python -c "from streamdiffusion.config import load_config; print('✅ StreamDiffusion installed successfully')" || echo "❌ StreamDiffusion verification FAILED"

        echo.
        echo Checking diffusers installation...
        python -c "from diffusers import StableDiffusionPipeline; print('✅ Diffusers installed successfully')" || echo "❌ Diffusers verification FAILED"

        echo.
        echo ============================================================================
        echo Installation Summary
        echo ============================================================================
        python -m pip list | findstr /I "torch diffusers streamdiffusion xformers cuda-python"

        echo.
        echo ============================================================================
        echo Installation Finished!
        echo ============================================================================
        echo If you see "CUDA Available: True" above, installation was successful.
        echo If you see "CUDA Available: False", there was a problem - contact support.
        echo.
        pause
        """
        batch_file_content_mac = f'''
            #!/bin/bash
            
            # Unset PYTHONPATH to prevent conflicts
            unset PYTHONPATH
            
            echo "Current directory: $PWD"
            cd "{base_folder}"
            echo "Changed directory to: $PWD"
            export PIP_DISABLE_PIP_VERSION_CHECK=1
            if [ ! -d "venv" ]; then
                echo "Creating Python venv at: {base_folder}/venv"
                "{python_exe}" -m venv venv
            else:
                echo "Virtual environment already exists at: {base_folder}/venv"
            fi

            echo "Attempting to activate virtual environment..."
            source venv/bin/activate

            if [ -z "$VIRTUAL_ENV" ]; then
                echo "Failed to activate virtual environment. Please check the path and ensure the venv exists."
                echo "Path to venv: {base_folder}/venv"
                echo "VIRTUAL_ENV: $VIRTUAL_ENV"
                exit 1
            else:
                echo "Virtual environment activated."
            fi

            echo "Installing"
            
            pip install {no_cache} --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip 
            
            echo "Installing huggingface_hub==0.24.6 first to prevent version conflicts..."
            pip install {no_cache} --trusted-host pypi.org --trusted-host files.pythonhosted.org huggingface_hub==0.24.6
            
            echo "Installing 'wheel' to ensure successful building of packages..."
            pip install {no_cache} --trusted-host pypi.org --trusted-host files.pythonhosted.org wheel

            echo "Installing numpy==1.24.1 first to ensure correct version..."
            pip install {no_cache} --trusted-host pypi.org --trusted-host files.pythonhosted.org numpy==1.24.1

            echo "Installing dependencies with pip from the activated virtual environment..."
            pip install {no_cache} --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu --trusted-host download.pytorch.org
            pip {no_cache} --trusted-host pypi.org --trusted-host files.pythonhosted.org install .
            pip install {no_cache} --trusted-host pypi.org --trusted-host files.pythonhosted.org -r streamdiffusionTD/{requirements_file}

            echo "Ensuring huggingface_hub version is correct..."
            pip install {no_cache} --trusted-host pypi.org --trusted-host files.pythonhosted.org --force-reinstall huggingface_hub==0.24.6

            echo "Ensuring numpy version is still correct..."
            pip install {no_cache} --trusted-host pypi.org --trusted-host files.pythonhosted.org --force-reinstall numpy==1.24.1

            echo "Installation Finished"
            read -p "Press any key to continue..."
        '''

        if platform.system() == 'Windows':
            batch_file_content = batch_file_content_win
        else:
            batch_file_content = batch_file_content_mac
        print("Writing batch file for installation...")
        # Write to the batch file with UTF-8 encoding to handle Unicode characters like ✅
        with open(bat_file_path, 'w', encoding='utf-8') as bat_file:
            bat_file.write(batch_file_content)
        print(f"Executing batch file: {bat_file_path}")
        # Execute the batch file in a new command window
        if platform.system() == 'Windows':
            subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=base_folder)
        else:
            print("running sh file")
            os.system(f"chmod +x {bat_file_path}")
            subprocess.Popen(['open', '-a', 'Terminal', bat_file_path], cwd=base_folder)
        self.message_box_open = False   



    def print_installation_details(self, python_exe, cuda_version, base_folder):
        print("\n====================================")
        print("Installation Details:")
        print("------------------------------------")
        print(f"Python 3.10 executable: {python_exe}")
        print(f"CUDA version: {cuda_version}")
        print(f"Base folder for venv: {base_folder}")
        print("Installing packages for: StreamDiffusionTD")
        print("====================================\n")

    def Installtensorrt(self):
        if self.message_box_open:
            return
        self.message_box_open = True
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        venv_path = os.path.join(base_folder, 'venv')
        # Check if 'venv' directory exists
        if os.path.exists(venv_path):
            if self.ismac:
                ui.messageBox('Installing TensorRT',
                              'This installation is only supported on Windows. TensorRT installation does not work on macOS.',
                              buttons=['OK'])
                self.message_box_open = False
                return
            choice = ui.messageBox('Installing TensorRT',
                                    'Installing TensorRT...\n\n'
                                    'Note: This is for Windows only.\n'
                                    'TensorRT needs to be turned on by setting the\n'
                                    'model\'s acceleration mode to TensorRT in the\n'
                                    'Settings 1 parameter page.',
                                    buttons=['OK', 'Cancel'])
            if choice != 0:  # Cancel
                self.message_box_open = False
                return
        self.copy_sdtd_code()
        self.install_tensorrt()
        self.message_box_open = False

    def install_tensorrt(self):
        """
        Creates and executes a batch file to install TensorRT.
        It activates the Python virtual environment and runs the command for TensorRT installation.
        """
        bat_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'Install_TensorRT.bat')
        batch_file_content = f"""
@echo off
echo Current directory: %CD%
cd /d "{self.ownerComp.par.Basefolder.eval()}"

echo Attempting to activate virtual environment...
call "venv\\Scripts\\activate.bat"

rem Check if the virtual environment was activated successfully
if "%VIRTUAL_ENV%" == "" (
    echo Failed to activate virtual environment. Please check the path and ensure the venv exists.
    pause /b 1
) else (
    echo Virtual environment activated.
)

echo Installing TensorRT...
python "{self.ownerComp.par.Basefolder.eval()}/streamdiffusionTD/install_tensorrt.py"

echo TensorRT installation finished
pause
        """
        print("Writing batch file for TensorRT installation...")
        # Write the batch file content
        with open(bat_file_path, 'w') as bat_file:
            bat_file.write(batch_file_content)
        print(f"Executing batch file: {bat_file_path}")
        # Execute the batch file in a new command window
        subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=self.ownerComp.par.Basefolder.eval())
        print("TensorRT installation initiated.")

    def Installdaydream(self):
        """
        Installs minimal dependencies for Daydream cloud backend only.
        Creates a lightweight virtual environment with just the necessary packages.
        """
        if self.message_box_open:
            return
        self.message_box_open = True
        
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        
        # Check if base folder is set and valid
        if not base_folder or base_folder == '':
            choice = ui.messageBox('Select Installation Folder',
                                'Please choose where to install the Daydream backend.\n\n'
                                'This will create a lightweight virtual environment\n'
                                'with only the necessary dependencies for cloud processing.',
                                buttons=['Select Folder', 'Cancel'])
            if choice != 0:  # Cancel
                self.message_box_open = False
                return False
            
            # Open folder selection dialog
            folder_path = ui.chooseFolder(title='Select Daydream Installation Folder', 
                                        start=project.folder)
            if not folder_path:
                self.message_box_open = False
                return False
            
            # Set the base folder
            self.ownerComp.par.Basefolder = folder_path
            base_folder = folder_path
        
        # Check for existing virtual environments - prefer main venv over daydream_venv
        main_venv_path = os.path.join(base_folder, 'venv')
        alt_venv_path = os.path.join(base_folder, '.venv')
        daydream_venv_path = os.path.join(base_folder, 'daydream_venv')
        
        # Determine which venv to use
        if os.path.exists(main_venv_path):
            target_venv_path = main_venv_path
            venv_name = 'venv'
            venv_message = 'Main virtual environment detected. Will add Daydream dependencies to existing venv.'
        elif os.path.exists(alt_venv_path):
            target_venv_path = alt_venv_path
            venv_name = '.venv'
            venv_message = 'Main virtual environment detected. Will add Daydream dependencies to existing .venv.'
        elif os.path.exists(daydream_venv_path):
            target_venv_path = daydream_venv_path
            venv_name = 'daydream_venv'
            venv_message = 'Daydream virtual environment already exists. Do you want to update it?'
        else:
            target_venv_path = daydream_venv_path
            venv_name = 'daydream_venv'
            venv_message = None
        
        # Show appropriate message if venv exists
        if venv_message:
            if venv_name in ['venv', '.venv']:
                choice = ui.messageBox('Virtual Environment Detected',
                                    f'{venv_message}\n\nThis will install the minimal Daydream dependencies alongside your existing packages.',
                                    buttons=['Install to Existing', 'Cancel'])
            else:
                choice = ui.messageBox('Daydream Virtual Environment Detected',
                                    venv_message,
                                    buttons=['Update', 'Cancel'])
            if choice != 0:  # Cancel
                self.message_box_open = False
                return False
        
        # Check for Python installation
        python_exe, pyversion = self.find_python_exe()
        if not python_exe:
            choice = ui.messageBox('Python Required',
                                'Python not found on your system.\n\n'
                                'Daydream requires Python 3.10 or 3.11.\n'
                                'Please install Python from python.org\n\n'
                                'Do you want to continue anyway?',
                                buttons=['Continue Anyway', 'Cancel'])
            if choice != 0:  # Cancel
                self.message_box_open = False
                return False
            python_exe = 'python'  # Hope for the best
        
        # Final installation confirmation with all details
        installation_details = f"Ready to install Daydream Cloud Backend\n\n" \
                              f"Location: {base_folder}\n" \
                              f"Python: {python_exe} ({pyversion if python_exe != 'python' else 'system default'})\n\n" \
                              f"Will install:\n" \
                              f"• requests (HTTP API calls)\n" \
                              f"• python-osc (TouchDesigner communication)\n" \
                              f"• numpy (shared memory)\n\n" \
                              f"Size: ~50MB | Time: ~2 minutes | No GPU required"
        
        choice = ui.messageBox('Install Daydream Backend',
                            installation_details,
                            buttons=['Install', 'Cancel'])
        if choice != 0:
            self.message_box_open = False
            return False
        
        # Copy StreamDiffusionTD code including Daydream files
        self.copy_sdtd_code()
        
        # Create and execute installation batch file
        self.create_daydream_install_batch(base_folder, target_venv_path, venv_name)
        
        # Show final success message
        ui.messageBox('Installation Started',
                    'Installation has started!\n\n'
                    'A command window will open showing progress.\n'
                    'This typically takes 1-2 minutes.\n\n'
                    'After installation completes:\n'
                    '• Set your Daydream API key\n'
                    '• Set Backend to "Daydream"\n'
                    '• Click "Start Stream"\n\n'
                    'Check the command window or terminal for progress...',
                    buttons=['OK'])
        self.Apikey()
        self.message_box_open = False

    def create_daydream_install_batch(self, base_folder, target_venv_path, venv_name):
        """Creates batch file for Daydream installation"""
        
        python_exe, _ = self.find_python_exe()
        if not python_exe:
            python_exe = 'python'
        
        no_cache = "--no-cache-dir" if self.ownerComp.par.Nocacheinstall else ""
        
        if platform.system() == 'Windows':
            bat_file_path = os.path.join(base_folder, 'Install_Daydream.bat')
            
            # Determine activation script path
            if venv_name in ['venv', '.venv']:
                activate_script = f"{venv_name}\\Scripts\\activate.bat"
                create_check = f'if not exist "{venv_name}" ('
                create_command = f'    echo Creating {venv_name} virtual environment...\n    "{python_exe}" -m venv {venv_name}'
                exists_message = f'    echo {venv_name} virtual environment already exists.'
            else:
                activate_script = f"{venv_name}\\Scripts\\activate.bat"
                create_check = f'if not exist "{venv_name}" ('
                create_command = f'    echo Creating {venv_name} virtual environment...\n    "{python_exe}" -m venv {venv_name}'
                exists_message = f'    echo {venv_name} virtual environment already exists.'
                
            batch_content = f"""
@echo off
echo Installing Daydream Cloud Backend Dependencies...
cd /d "{base_folder}"

{create_check}
{create_command}
) else (
{exists_message}
)

echo Activating virtual environment...
call "{activate_script}"

rem Check if activation was successful
if "%VIRTUAL_ENV%" == "" (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Installing minimal dependencies...
python -m pip install {no_cache} --upgrade pip
python -m pip install {no_cache} -r streamdiffusionTD/requirements_daydream.txt

echo Daydream installation completed successfully!
echo You can now use the Daydream cloud backend.
pause
"""
        else:
            bat_file_path = os.path.join(base_folder, 'Install_Daydream.sh')
            
            # Determine activation script path
            activate_script = f"{venv_name}/bin/activate"
            create_check = f'if [ ! -d "{venv_name}" ]; then'
            create_command = f'    echo "Creating {venv_name} virtual environment..."\n    "{python_exe}" -m venv {venv_name}'
            exists_message = f'    echo "{venv_name} virtual environment already exists."'
                
            batch_content = f"""#!/bin/bash
echo "Installing Daydream Cloud Backend Dependencies..."
cd "{base_folder}"

{create_check}
{create_command}
else
{exists_message}
fi

# Unset PYTHONPATH to avoid TD Python interference
unset PYTHONPATH

echo "Activating virtual environment..."
source {activate_script}

echo "Installing minimal dependencies..."
pip install {no_cache} --upgrade pip
pip install {no_cache} -r streamdiffusionTD/requirements_daydream.txt

echo "Daydream installation completed successfully!"
echo "You can now use the Daydream cloud backend."
read -p "Press any key to continue..."
"""
        
        # Write and execute batch file
        with open(bat_file_path, 'w') as f:
            f.write(batch_content)
        
        print(f"Executing batch file: {bat_file_path}")
        if platform.system() == 'Windows':
            subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=base_folder)
        else:
            os.system(f"chmod +x {bat_file_path}")
            subprocess.Popen(['open', '-a', 'Terminal', bat_file_path], cwd=base_folder)

    def Apikey(self):
        """
        Manages secure storage and retrieval of DayDream API key.
        Updates parameter display based on key status.
        """
        current_value = self.ownerComp.par.Apikey.eval()
        invalid_key = False
        # Known status phrases that shouldn't be treated as actual keys
        status_phrases = ['Enter DayDream Key','✓ Loaded','DayDream Key Loaded', 'ENTER API KEY', 'API KEY LOADED']
        
        if current_value not in status_phrases and current_value.strip():
            key = current_value.strip()
            if len(key) < 30 or ' ' in key:
                self.logger.log("Invalid API key - must be 30+ chars with no spaces", level='WARNING')
                invalid_key = True
            else:
                self._save_daydream_key(key)
                self.ownerComp.par.Apikey = '✓ Loaded'
                self.ownerComp.par.Apikey.readOnly = False
                self.logger.log("DayDream Key Loaded", level='INFO')
                return
        
        base_folder = self.ownerComp.par.Basefolder.eval()
        if not base_folder:
            self.ownerComp.par.Apikey = 'Enter DayDream Key'
            return
        
        stored_key = self._load_daydream_key()
        if stored_key and len(stored_key) >= 30 and ' ' not in stored_key:
            self.ownerComp.par.Apikey = '✓ Loaded'
            self.ownerComp.par.Apikey.readOnly = False
            if invalid_key:
                self.logger.log("DayDream Key Loaded", level='INFO')
        else:
            self.ownerComp.par.Apikey = 'Enter DayDream Key'
            self.ownerComp.par.Apikey.readOnly = False
    
    def _get_daydream_config_path(self):
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        if not base_folder:
            return None
        daydream_folder = os.path.join(base_folder, 'daydream')
        config_file = os.path.join(daydream_folder, 'daydream_config.json')
        return daydream_folder, config_file
    
    def _save_daydream_key(self, api_key):
        try:
            daydream_folder, config_file = self._get_daydream_config_path()
            if not daydream_folder:
                return False
            os.makedirs(daydream_folder, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump({'api_key': api_key}, f, indent=2)
            return True
        except Exception as e:
            self.logger.log(f"Error saving API key: {e}", level='ERROR')
            return False
    
    def _load_daydream_key(self):
        try:
            daydream_folder, config_file = self._get_daydream_config_path()
            if not config_file or not os.path.exists(config_file):
                return None
            with open(config_file, 'r') as f:
                return json.load(f).get('api_key')
        except Exception as e:
            self.logger.log(f"Error loading API key: {e}", level='ERROR')
            return None
    
    def _get_actual_daydream_key(self):
        return self._load_daydream_key()

    def Updaterepo(self):
        base_folder = self.ownerComp.par.Basefolder.eval()
        if not os.path.exists(base_folder):
            self.logger.log("StreamDiffusion folder not found. Please make sure the repository is cloned.", level="ERROR")
            return
        # Path to the virtual environment's Python executable
        if platform.system() == 'Windows':
            venv_path = os.path.join(base_folder, 'venv', 'Scripts', 'python.exe')
            if not os.path.exists(venv_path):
                venv_path = os.path.join(base_folder, '.venv', 'Scripts', 'python.exe')
        elif platform.system() == 'Darwin':
            venv_path = os.path.join(base_folder, 'venv', 'bin', 'python')
            if not os.path.exists(venv_path):
                venv_path = os.path.join(base_folder, '.venv', 'bin', 'python')
        else:
            self.logger.log("Unsupported operating system.", level="ERROR")
            return
        if not os.path.exists(venv_path):
            self.logger.log("Python executable in virtual environment not found.", level="ERROR")
            return
        try:
            if platform.system() == 'Windows':
                # Create a subprocess to run the pip install command on Windows
                command = [venv_path, '-m', 'pip', 'install', '-e', base_folder]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            elif platform.system() == 'Darwin':
                # Create a subprocess to run the pip install command on macOS
                command = [venv_path, '-m', 'pip', 'install', '-e', base_folder]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            else:
                self.logger.log("Unsupported operating system.", level="ERROR")
                return
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                self.logger.log("StreamDiffusion repository updated successfully.", level="INFO")
            else:
                self.logger.log(f"Error updating StreamDiffusion repository: {stderr}", level="ERROR")
        except Exception as e:
            self.logger.log(f"Error updating StreamDiffusion repository: {str(e)}", level="ERROR")

    def copy_sdtd_code(self):
        """
        Copies the contents of Text DATs from the 'streamdiffusionTD' base component into corresponding files in the 'streamdiffusionTD' folder in the Basefolder.
        """
        base_folder = self.ownerComp.par.Basefolder.eval()
        # Ensure the 'streamdiffusionTD' folder exists
        streamdiffusionTD_folder = os.path.join(base_folder, 'streamdiffusionTD')
        if not os.path.exists(streamdiffusionTD_folder):
            os.makedirs(streamdiffusionTD_folder)
        # Paths to the Text DATs inside the 'streamdiffusionTD' base comp
        streamdiffusionTD_comp = self.ownerComp.op('streamdiffusionTD')
        text_dat_paths = {
            'td_main': 'td_main.py',
            'td_manager': 'td_manager.py',
            'td_osc_handler': 'td_osc_handler.py',
            'td_config': 'td_config.yaml',
            # 'requirements_pc': 'requirements_pc.txt',
            'requirements_mac': 'requirements_mac.txt',
            # 'pipeline_td': 'pipeline_td.py',
            # 'wrapper_td': 'wrapper_td.py',
            # 'fx_utils': 'fx_utils.py',
            # 'attention_processor': 'attention_processor.py',
            # 'ndi_utils': 'ndi_utils.py',
            'syphon_utils': 'syphon_utils.py',
            # 'daydream_client': 'daydream_client.py',
            # 'daydream_manager': 'daydream_manager.py',
            # 'webrtc_player': 'webrtc_player.html',
            'install_tensorrt': 'install_tensorrt.py'

        }
        # Copy each Text DAT's content into a file in the 'streamdiffusionTD' folder
        for dat_name, filename in text_dat_paths.items():
            file_path = os.path.join(streamdiffusionTD_folder, filename)
            
            # Special handling for td_config.yaml - generate dynamically
            if dat_name == "td_config":
                try:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        yaml_content = self.generate_td_config_yaml()
                        file.write(yaml_content)
                    self.logger.log(f'Generated dynamic td_config.yaml at {file_path}', level='Debug')
                except Exception as e:
                    self.logger.log(f'Error generating td_config.yaml: {str(e)}', level='ERROR')
                continue
            
            # Standard Text DAT copying for other files
            text_dat = streamdiffusionTD_comp.op(dat_name)
            if text_dat:
                try:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        text = text_dat.text    
                        if dat_name == "td_main":
                            text = text.replace("x.x.x123454321", self.get_version())
                        file.write(text)
                    # self.logger.log(f'Copied {dat_name} to {file_path}', level='Debug')
                except Exception as e:
                    self.logger.log(f'Error writing {dat_name} to {file_path}: {str(e)}', level='ERROR')
            else:
                self.logger.log(f'{dat_name} DAT not found in streamdiffusionTD', level='ERROR')
                
#         # Create the Daydream requirements file
#         daydream_req_dest = os.path.join(streamdiffusionTD_folder, 'requirements_daydream.txt')
#         daydream_requirements_content = """requests>=2.31.0
# python-osc>=1.8.0
# numpy>=1.24.0
# syphon-python>=0.1.0; sys_platform == "darwin" """
#         try:
#             with open(daydream_req_dest, 'w', encoding='utf-8') as file:
#                 file.write(daydream_requirements_content)
#             self.logger.log(f'Created requirements_daydream.txt at {daydream_req_dest}', level='Debug')
#         except Exception as e:
#             self.logger.log(f'Error creating requirements_daydream.txt: {str(e)}', level='ERROR')

    def check_install(self, force=True):
        backend = self.ownerComp.par.Backend.eval()
        
        if not force:
            clone_done = self.ownerComp.par.Clonestreamdiffusion.label.endswith(' ✓')
            install_done = self.ownerComp.par.Installstreamdiffusion.label.endswith(' ✓')
            tensorrt_done = self.ownerComp.par.Installtensorrt.label.endswith(' ✓')
            
            if clone_done and install_done and tensorrt_done:
                return
        
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        
        def update_pulse_label(pulse_par, is_completed):
            current_state = pulse_par.label.endswith(' ✓')
            if force or current_state != is_completed:
                current_label = pulse_par.label
                if current_label.endswith(' ✓') or current_label.endswith(' ✗'):
                    current_label = current_label[:-2]
                pulse_par.label = f"{current_label} {'✓' if is_completed else '✗'}"
        def check_if_installed(pulse_par):
            is_installed = pulse_par.label.endswith(' ✓')
            return is_installed
            
        # Update labels based on backend - only if they need to change
        if backend == 'Daydream':
            # For Daydream, we don't need to clone the repo or install TensorRT
            expected_labels = {
                'Clonestreamdiffusion': 'Download StreamDiffusion [ not needed ]',
                'Installstreamdiffusion': 'Install Daydream Backend',
                'Installtensorrt': 'Install TensorRT [ not needed ]'
            }
        else:
            # For Local backend, restore original labels
            expected_labels = {
                'Clonestreamdiffusion': '1. Download StreamDiffusion',
                'Installstreamdiffusion': '2. Install [ venv + all req ]',
                'Installtensorrt': '3. Install TensorRT [ optional ]'
            }
        
        # Only update labels if they don't match expected base labels
        for par_name, expected_label in expected_labels.items():
            current_par = getattr(self.ownerComp.par, par_name)
            current_base_label = current_par.label
            # Remove status indicators to get base label
            if current_base_label.endswith(' ✓') or current_base_label.endswith(' ✗'):
                current_base_label = current_base_label[:-2]
            
            if current_base_label != expected_label:
                # Preserve any existing status indicators
                status_suffix = ''
                if current_par.label.endswith(' ✓'):
                    status_suffix = ' ✓'
                elif current_par.label.endswith(' ✗'):
                    status_suffix = ' ✗'
                current_par.label = expected_label + status_suffix
            
        if base_folder == '':
            expected_message = 'Set Basefolder or hit "Install Daydream Backend" to install Daydream' if backend == 'Daydream' else 'Set Basefolder or pulse 1 to download'
            if self.ownerComp.par.Installstep != expected_message:
                self.ownerComp.par.Installstep = expected_message
            if self.ownerComp.par.Installstep.label != '':
                self.ownerComp.par.Installstep.label = ''
            
            if force or not self.ownerComp.par.Clonestreamdiffusion.label.endswith(' ✗'):
                update_pulse_label(self.ownerComp.par.Clonestreamdiffusion, False)
            if force or not self.ownerComp.par.Installstreamdiffusion.label.endswith(' ✗'):
                update_pulse_label(self.ownerComp.par.Installstreamdiffusion, False)
            if force or not self.ownerComp.par.Installtensorrt.label.endswith(' ✗'):
                update_pulse_label(self.ownerComp.par.Installtensorrt, False)
            return
        if backend == 'Daydream':
            # For Daydream backend, we only need to check for daydream_venv
            daydream_venv_path = os.path.join(base_folder, 'daydream_venv')
            daydream_installed = self.check_daydream_installation(base_folder)
            
            # Mark clone and tensorrt as not needed (always show as completed)
            update_pulse_label(self.ownerComp.par.Clonestreamdiffusion, True)
            update_pulse_label(self.ownerComp.par.Installtensorrt, True)
            
            # Check Daydream installation
            if force or not self.ownerComp.par.Installstreamdiffusion.label.endswith(' ✓'):
                if force or not check_if_installed(self.ownerComp.par.Installstreamdiffusion):
                    update_pulse_label(self.ownerComp.par.Installstreamdiffusion, daydream_installed)
            
            if daydream_installed:
                # Show Daydream installation info
                python_exe = self.get_daydream_venv_python()
                if python_exe:
                    expected_message = 'Daydream Backend Ready | Cloud Processing'
                    expected_label = 'Install Info:'
                    expected_enable = False
                else:
                    expected_message = 'Daydream venv found but Python missing'
                    expected_label = 'Warning'
                    expected_enable = True
                    
                if self.ownerComp.par.Installstep != expected_message:
                    self.ownerComp.par.Installstep = expected_message
                if self.ownerComp.par.Installstep.enable != expected_enable:
                    self.ownerComp.par.Installstep.enable = expected_enable
                if self.ownerComp.par.Installstep.label != expected_label:
                    self.ownerComp.par.Installstep.label = expected_label
            else:
                expected_message = 'Hit "Install Daydream Backend" to install Daydream'
                expected_label = ''
                if self.ownerComp.par.Installstep != expected_message:
                    self.ownerComp.par.Installstep = expected_message
                if self.ownerComp.par.Installstep.label != expected_label:
                    self.ownerComp.par.Installstep.label = expected_label
                
        else:
            # For Local backend, use original logic
            venv_path = os.path.join(base_folder, 'venv')
            git_path = os.path.join(base_folder, '.git')
            is_cloned = os.path.exists(git_path)
            is_installed = os.path.exists(venv_path)
            
            if force or not self.ownerComp.par.Clonestreamdiffusion.label.endswith(' ✓'):
                if force or not check_if_installed(self.ownerComp.par.Clonestreamdiffusion):
                    update_pulse_label(self.ownerComp.par.Clonestreamdiffusion, is_cloned)
                    
            if force or not self.ownerComp.par.Installstreamdiffusion.label.endswith(' ✓'):
                if force or not check_if_installed(self.ownerComp.par.Installstreamdiffusion):
                    update_pulse_label(self.ownerComp.par.Installstreamdiffusion, is_installed)

            if is_installed:
                if force or self.ownerComp.par.Installstep.enable:
                    python_version = self.check_venv_python_version(venv_path)   
                    
                    # Only check CUDA version on Windows
                    if platform.system() == 'Windows':
                        cuda_version = self.check_venv_cuda_version(venv_path)
                        version_text = f'python {python_version} | cuda {cuda_version.strip("cu")}' if cuda_version else f'python {python_version}'
                    else:
                        version_text = f'python {python_version}'
                    
                    # Only update if values have changed
                    if self.ownerComp.par.Installstep != version_text:
                        self.ownerComp.par.Installstep = version_text
                    if self.ownerComp.par.Installstep.enable != False:
                        self.ownerComp.par.Installstep.enable = False
                    if self.ownerComp.par.Installstep.label != 'Install Info:':
                        self.ownerComp.par.Installstep.label = 'Install Info:'
                    
                if force or not self.ownerComp.par.Installtensorrt.label.endswith(' ✓'):
                    if force or not check_if_installed(self.ownerComp.par.Installtensorrt):
                        tensorrt_installed = self.check_tensorrt_installation(venv_path)
                        update_pulse_label(self.ownerComp.par.Installtensorrt, tensorrt_installed)
            else:
                expected_message = 'Hit "2. Install" pulse' if is_cloned else 'Check Basefolder... it needs to be StreamDiffusion repo'
                expected_label = 'Help'
                
                if self.ownerComp.par.Installstep != expected_message:
                    self.ownerComp.par.Installstep = expected_message
                if self.ownerComp.par.Installstep.label != expected_label:
                    self.ownerComp.par.Installstep.label = expected_label
                update_pulse_label(self.ownerComp.par.Installtensorrt, False)
        self.logger.log(f"Basefolder updated", level='DEBUG')

    def Writeconfig(self):
        """
        Saves the current configuration to a JSON file if self.ownerComp.par.Writeconfig is True.
        The configuration is stored in the application config directory.
        """
        if not self.ownerComp.par.Writeconfig:
            self.ownerComp.par.Writeconfig.label = 'Save OP Config'
            return

        base_dir = self.get_config_dir()
        if not base_dir:
            return False
            
        config_file = os.path.join(base_dir, 'streamdiffusion_operator_config.json')
        config = {}

        # Parameters to skip
        skip_params = {'clone', 'Callbackdat', 'Log', 'Log2', 'Log3', 'Writeconfig', 'Status', 'Status2', 'Status3', 'Loadconfig'}

        # Handle sequence parameters first
        sequence_params = {
            'Promptdict': ['Concept', 'Weight'],
            'Seeddict': ['Seedval', 'Seedweight'],
            'Tindexblock': ['Step'],
            'Loradictblock': ['Lorapath', 'Weight'],
            'Textualinvblock': ['Embedpath', 'Token']
        }

        for seq_name, block_params in sequence_params.items():
            if hasattr(self.ownerComp.par, seq_name):
                seq_par = getattr(self.ownerComp.par, seq_name)
                blocks_data = []
                
                for block in seq_par.sequence:
                    block_data = {}
                    for param in block_params:
                        if hasattr(block.par, param):
                            par = getattr(block.par, param)
                            block_data[param] = {
                                'value': par.eval(),
                                'mode': str(par.mode),  # Store parameter mode
                                'expr': par.expr if par.mode == ParMode.EXPRESSION else None
                            }
                    blocks_data.append(block_data)
                    
                config[seq_name] = {
                    'numBlocks': seq_par.sequence.numBlocks,
                    'blocks': blocks_data
                }

        # Save regular parameters
        for page in self.ownerComp.customPages:
            for par in page.pars:
                if par.name in skip_params or isinstance(par, ParGroup) or par.name in sequence_params:
                    continue
                    
                try:
                    config[par.name] = {
                        'value': par.eval() if isinstance(par.eval(), (str, int, float, bool, type(None))) else str(par.eval()),
                        'mode': str(par.mode),
                        'expr': par.expr if par.mode == ParMode.EXPRESSION else None
                    }
                except:
                    continue

        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            self.logger.log(f'Operator configuration saved to {config_file}', level='INFO')
            self.ownerComp.par.Writeconfig.label = 'Save OP Config [ on Save ]'
            # run("me.par.Writeconfig = False",delayFrames = 50, fromOP = self.ownerComp)
            # run("me.par.Writeconfig.label = 'Save OP Config'",delayFrames = 50, fromOP = self.ownerComp)
            return True
        except Exception as e:
            self.logger.log(f'Error saving operator configuration: {str(e)}', level='ERROR')
            return False

    def Loadconfig(self):
        """
        Loads the operator configuration from the saved JSON file.
        """
        if not self.ownerComp.par.Loadconfig:
            return
        
        base_dir = self.get_config_dir()
        if not base_dir:
            return False
                
        config_file = os.path.join(base_dir, 'streamdiffusion_operator_config.json')
        
        if not os.path.exists(config_file):
            self.logger.log('No saved configuration found', level='INFO')
            return False

        skip_params = {'clone', 'Callbackdat', 'Log', 'Log2', 'Log3', 'Writeconfig', 'Status', 'Status2', 'Status3', 'Loadconfig'}
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            sequence_params = {
                'Promptdict': ['Concept', 'Weight'],
                'Seeddict': ['Seedval', 'Seedweight'],
                'Tindexblock': ['Step'],
                'Loradictblock': ['Lorapath', 'Weight'],
                'Textualinvblock': ['Embedpath', 'Token']
            }

            # Restore sequence parameters
            for seq_name, block_params in sequence_params.items():
                if seq_name in config and hasattr(self.ownerComp.par, seq_name):
                    seq_data = config[seq_name]
                    seq_par = getattr(self.ownerComp.par, seq_name)
                    
                    if 'numBlocks' in seq_data:
                        seq_par.sequence.numBlocks = seq_data['numBlocks']
                    
                    if 'blocks' in seq_data:
                        for i, block_data in enumerate(seq_data['blocks']):
                            if i < seq_par.sequence.numBlocks:
                                for param_name, param_info in block_data.items():
                                    if hasattr(seq_par.sequence[i].par, param_name):
                                        try:
                                            par = getattr(seq_par.sequence[i].par, param_name)
                                            if param_info['mode'] == str(ParMode.EXPRESSION) and param_info['expr']:
                                                par.expr = param_info['expr']
                                            else:
                                                par.val = param_info['value']
                                        except Exception as e:
                                            self.logger.log(f'Error setting {seq_name}[{i}].{param_name}: {str(e)}', level='WARNING')

            # Restore regular parameters
            for param_name, param_info in config.items():
                if param_name in sequence_params or param_name in skip_params:
                    continue
                    
                if hasattr(self.ownerComp.par, param_name):
                    try:
                        par = getattr(self.ownerComp.par, param_name)
                        if param_info['mode'] == str(ParMode.EXPRESSION) and param_info['expr']:
                            par.expr = param_info['expr']
                        else:
                            par.val = param_info['value']
                    except Exception as e:
                        self.logger.log(f'Could not restore parameter {param_name}: {str(e)}', level='WARNING')

            self.Basefolder(force=True)
            self.logger.log('OP Config Loaded from file', level='INFO')
            self.ownerComp.par.Writeconfig = False
            self.ownerComp.par.Loadconfig.label = 'OP Config Loaded ✓'
            run("me.par.Loadconfig = False",delayFrames = 50, fromOP = self.ownerComp)
            run("me.par.Loadconfig.label = 'Load OP Config'",delayFrames = 50, fromOP = self.ownerComp)

            return True
                
        except Exception as e:
            self.logger.log(f'Error loading operator configuration: {str(e)}', level='ERROR')
            return False

    def Resetop(self):
        """
        Resets the StreamDiffusionTD operator to default settings after user confirmation.
        """
        choice = ui.messageBox('Reset', 'How would you like to reset the StreamDiffusionTD operator?', 
                            buttons=['Reset Settings', 'Full Op Reset', 'No'])
        
        if choice == 0 or choice == 1:
            try:
                # Clear log
                op('Logger').par.Clearlog.pulse()
                
                # Reset basic parameters
                if choice == 1:  # Full Op Reset
                    self.ownerComp.par.Basefolder = ''
                
                self.ownerComp.par.Acceleration = 'none'
                self.ownerComp.par.Uselora = False
                
                # Set folder parameters to expression mode with empty expressions
                folder_params = {
                    'Sdmodelsfolder': 'me.par.Basefolder + "/models/Model"',
                    'Lorafolder': 'me.par.Basefolder + "\\\\models\\\\LoRA"',
                    # 'Cnfolder': 'me.par.Basefolder + "\\\\models\\\\ControlNet"',
                    'Hfcache': 'me.par.Basefolder + "/models"'
                }
                
                for param_name, expr in folder_params.items():
                    par = getattr(self.ownerComp.par, param_name)
                    par.mode = ParMode.EXPRESSION
                    par.expr = expr
                
                # Reset other parameters
                self.ownerComp.par.Loradictblock0lorapath = ''
                self.ownerComp.par.Customlcm = ''
                self.ownerComp.par.Usecustomlcm = False
                self.ownerComp.par.Lastpreset = ''
                self.update_preset_dropdown(reset=True)
                self.ownerComp.par.Modelid = 'stabilityai/sd-turbo'
                self.ownerComp.par.Promptdict0concept = 'default fancy banana'
                self.ownerComp.par.Seeddict0seedweight = 1
                self.ownerComp.par.Tindexblock.sequence.numBlocks = 1
                self.ownerComp.par.Tindexblock0step = 1
                # self.ownerComp.par.Cnmodelselect = ''
                self.ownerComp.par.Sethfcache = False
                self.ownerComp.par.Writeconfig = False
                
                # Final log clear
                op('Logger').par.Clearlog.pulse()
                
                reset_type = "Full Op Reset" if choice == 1 else "Settings Reset"
                self.logger.log(f'StreamDiffusionTD operator {reset_type} completed', level='INFO')
                
            except Exception as e:
                self.logger.log(f'Error during reset: {str(e)}', level='ERROR')


    def Basefolder(self, force=False):
        self.ownerComp.par.Installstep.enable = True
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        
        self.sync_model_table()
        self.update_preset_dropdown()
        self.update_engine_table(silent=True)
        self.check_install(force=True)
        if self.ownerComp.par.Basefolder.eval() == '' and force == False:
            return
        if base_folder:
            self.set_streamdiffusion_install_path(base_folder)
            # self.Loadconfig()
        elif saved_path := self.get_streamdiffusion_install_path():
            if os.path.exists(saved_path):
                self.ownerComp.par.Basefolder = saved_path
                self.Loadconfig()
        self.Apikey()
        self.Getpreprocessors()
                

    def get_config_dir(self):
        """Get the DotSimulate config directory path"""
        system = platform.system()
        if system == 'Windows':
            base_dir = os.path.join(os.getenv('APPDATA'), 'dotsimulate_TD_Tools')
        elif system == 'Darwin':  # macOS
            base_dir = os.path.expanduser('~/Library/Application Support/dotsimulate_TD_Tools')
        else:
            return None
        
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def get_streamdiffusion_install_path(self):
        """Get StreamDiffusionTD base folder path from config"""
        base_dir = self.get_config_dir()
        if not base_dir:
            return None
                
        config_file = os.path.join(base_dir, 'streamdiffusion_config.json')
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('base_folder')
            except json.JSONDecodeError:
                self.logger.log("Error reading config file", level='ERROR')
                return None
                    
        return None

    def set_streamdiffusion_install_path(self, base_folder):
        """Save StreamDiffusionTD base folder path to config"""
        base_dir = self.get_config_dir()
        if not base_dir:
            return False
                
        config_file = os.path.join(base_dir, 'streamdiffusion_config.json')
        
        config = {
            'base_folder': base_folder,
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            self.logger.log(f"Error saving config: {str(e)}", level='ERROR')
            return False



    def UnlinkBasefolder(self):
        """Remove the config file"""
        base_dir = self.get_config_dir()
        if not base_dir:
            return
        
        config_file = os.path.join(base_dir, 'streamdiffusion_config.json')
        if os.path.exists(config_file):
            try:
                os.remove(config_file)
                self.ownerComp.par.Basefolder = ''
                self.logger.log("StreamDiffusionTD config unlinked", level='INFO')
            except Exception as e:
                self.logger.log(f"Error removing config: {str(e)}", level='ERROR')





    def check_tensorrt_installation(self, venv_path):
        if platform.system() != 'Windows':
            expected_label = '3. Install TensorRT [ PC only ]'
            # Only update label if it's different
            current_label = self.ownerComp.par.Installtensorrt.label
            # Remove status indicators to get base label
            if current_label.endswith(' ✓') or current_label.endswith(' ✗'):
                current_base_label = current_label[:-2]
            else:
                current_base_label = current_label
                
            if current_base_label != expected_label:
                # Preserve any existing status indicators
                status_suffix = ''
                if current_label.endswith(' ✓'):
                    status_suffix = ' ✓'
                elif current_label.endswith(' ✗'):
                    status_suffix = ' ✗'
                self.ownerComp.par.Installtensorrt.label = expected_label + status_suffix
            return False
            
        expected_label = '3. Install TensorRT [ optional ]'
        # Only update label if it's different
        current_label = self.ownerComp.par.Installtensorrt.label
        # Remove status indicators to get base label
        if current_label.endswith(' ✓') or current_label.endswith(' ✗'):
            current_base_label = current_label[:-2]
        else:
            current_base_label = current_label
            
        if current_base_label != expected_label:
            # Preserve any existing status indicators
            status_suffix = ''
            if current_label.endswith(' ✓'):
                status_suffix = ' ✓'
            elif current_label.endswith(' ✗'):
                status_suffix = ' ✗'
            self.ownerComp.par.Installtensorrt.label = expected_label + status_suffix

        site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
        
        if not os.path.exists(site_packages):
            self.logger.log(f"Site-packages directory not found at {site_packages}", level="ERROR")
            return False

        tensorrt_pattern = re.compile(r'tensorrt-\d+\.\d+\.\d+.*\.dist-info')
        
        for item in os.listdir(site_packages):
            if tensorrt_pattern.match(item):
                version_match = re.search(r'tensorrt-(\d+\.\d+\.\d+)', item)
                if version_match:
                    tensorrt_version = version_match.group(1)
                    self.logger.log(f"TensorRT version {tensorrt_version} found in venv", level="DEBUG")
                    return True
        return False

    def check_daydream_installation(self, base_folder):
        """Check if Daydream virtual environment and dependencies are installed"""
        # Check for venv in order of preference: main venv, .venv, daydream_venv
        venv_paths = [
            os.path.join(base_folder, 'venv'),
            os.path.join(base_folder, '.venv'),
            os.path.join(base_folder, 'daydream_venv')
        ]
        
        for venv_path in venv_paths:
            if os.path.exists(venv_path):
                # Check if Python executable exists
                if platform.system() == 'Windows':
                    python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
                    site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
                elif platform.system() == 'Darwin':  # macOS
                    python_exe = os.path.join(venv_path, 'bin', 'python')
                    # Find the correct python version folder
                    lib_path = os.path.join(venv_path, 'lib')
                    site_packages = None
                    if os.path.exists(lib_path):
                        for item in os.listdir(lib_path):
                            if item.startswith('python3.'):
                                site_packages = os.path.join(lib_path, item, 'site-packages')
                                break
                else:
                    continue
                    
                if not os.path.exists(python_exe) or not site_packages or not os.path.exists(site_packages):
                    continue
                    
                # Check if required packages are installed by looking for package folders
                required_packages = ['requests', 'pythonosc', 'numpy']
                packages_found = []
                
                try:
                    for item in os.listdir(site_packages):
                        # Check for package folders or .dist-info folders
                        item_lower = item.lower()
                        if any(pkg in item_lower for pkg in required_packages):
                            if item_lower.startswith('requests'):
                                packages_found.append('requests')
                            elif 'pythonosc' in item_lower or 'python_osc' in item_lower:
                                packages_found.append('pythonosc')
                            elif item_lower.startswith('numpy'):
                                packages_found.append('numpy')
                    
                    # Remove duplicates
                    packages_found = list(set(packages_found))
                    
                    if len(packages_found) >= 3:  # All required packages found
                        self.logger.log(f"Daydream dependencies verified in {venv_path} (found: {', '.join(packages_found)})", level="DEBUG")
                        return True
                    else:
                        missing = [pkg for pkg in required_packages if pkg not in packages_found]
                        self.logger.log(f"Venv {venv_path} missing packages: {', '.join(missing)}", level="DEBUG")
                        
                except Exception as e:
                    self.logger.log(f"Error checking packages in {site_packages}: {e}", level="DEBUG")
                    continue
            
        return False

    def generate_td_config_yaml(self):
        """
        Generate td_config.yaml content dynamically from TouchDesigner parameters.
        Returns YAML string that matches current parameter settings.
        """
        # Get current parameter values
        model_id = self.ownerComp.par.Modelid.eval()
        width = round(self.ownerComp.par.Width.eval() / 8) * 8
        height = round(self.ownerComp.par.Height.eval() / 8) * 8
        
        # Determine model architecture type for ControlNet/IPAdapter selection
        config_type = self.get_config_type(model_id.lower())
        
        # Build YAML content as string
        yaml_content = f"""# StreamDiffusionTD Configuration
# Based on LivePeer StreamDiffusion fork config system

# Base model configuration
model_id: "{model_id}"

# Core StreamDiffusion parameters  
t_index_list: {list(block.par.Step.eval() for block in self.ownerComp.par.Tindexblock.sequence)}
width: {width}
height: {height}
device: "cuda"
dtype: "float16"

# Generation parameters (defaults, can be updated via OSC)
guidance_scale: {self.ownerComp.par.Guidancescale.eval()}
num_inference_steps: 50
seed: {self.ownerComp.par.Seeddict0seedval.eval()}
delta: {self.ownerComp.par.Delta.eval()}

# Prompt configuration (supports both single and blending)
prompt: "{self.ownerComp.par.Promptdict0concept.eval()}"
negative_prompt: "{self.ownerComp.par.Negprompt.eval()}"

# Optimization settings
mode: "img2img"  # Always use img2img engines (mode switching handled at runtime)
frame_buffer_size: {self.ownerComp.par.Framesize.eval()}
use_denoising_batch: {str(self.ownerComp.par.Denoisebatch.eval()).lower()}
use_lcm_lora: true
use_tiny_vae: true
acceleration: "{self.ownerComp.par.Acceleration.eval()}"
cfg_type: "{self.ownerComp.par.Cfgtype.eval()}"

# TensorRT engine directory
engine_dir: "./engines/td"

"""

        # Add ControlNet configuration
        if self.ownerComp.par.Usecontrolnet.eval() and hasattr(self.ownerComp.par, 'Cn'):
            yaml_content += "# ControlNet configuration\n"
            yaml_content += "use_controlnet: true\n"
            yaml_content += "controlnets:\n"

            for block in self.ownerComp.par.Cn.sequence:
                if block.par.Enable.eval():
                    model_id = block.par.Id.eval()
                    weight = block.par.Weight.eval()
                    preprocessor = block.par.Preprocessor.eval() if hasattr(block.par, 'Preprocessor') else "canny"

                    if model_id and model_id != "none":
                        # Build preprocessor params based on preprocessor type
                        preprocessor_params = {}
                        if "tensorrt" in preprocessor.lower():
                            preprocessor_params = {"engine_path": "./engines/preprocessors"}

                        yaml_content += f'  - model_id: "{model_id}"\n'
                        yaml_content += f'    conditioning_scale: {weight}\n'
                        yaml_content += f'    preprocessor: "{preprocessor}"\n'
                        if preprocessor_params:
                            yaml_content += f'    preprocessor_params:\n'
                            for param_key, param_value in preprocessor_params.items():
                                yaml_content += f'      {param_key}: "{param_value}"\n'
                        yaml_content += f'    enabled: true\n'
        else:
            yaml_content += "# ControlNet configuration (disabled)\n"
            yaml_content += "use_controlnet: false\n"

        yaml_content += "\n"

        # Add IPAdapter configuration
        if self.ownerComp.par.Ipadapterenable.eval():
            ipadapter_scale = self.ownerComp.par.Ipadapterscale.eval()

            # Check if FaceID mode is enabled
            is_faceid = (hasattr(self.ownerComp.par, 'Ipfaceid') and
                        self.ownerComp.par.Ipfaceid.eval())

            if config_type in ["sdxl", "sdxl-turbo"]:
                if is_faceid:
                    # SDXL FaceID model
                    ipadapter_model_path = "h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin"
                    image_encoder_path = "h94/IP-Adapter/sdxl_models/image_encoder"
                else:
                    # Regular SDXL model
                    ipadapter_model_path = "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
                    image_encoder_path = "h94/IP-Adapter/sdxl_models/image_encoder"
            else:
                # SD1.5 and other models
                if is_faceid:
                    # SD1.5 FaceID model (using .bin format as per h94 repo)
                    ipadapter_model_path = "h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin"
                    image_encoder_path = "h94/IP-Adapter/models/image_encoder"
                else:
                    # Regular SD1.5 model
                    ipadapter_model_path = "h94/IP-Adapter/models/ip-adapter_sd15.bin"
                    image_encoder_path = "h94/IP-Adapter/models/image_encoder"

            yaml_content += "# IPAdapter configuration\n"
            yaml_content += "use_ipadapter: true\n"
            yaml_content += "ipadapters:\n"
            yaml_content += f'  - ipadapter_model_path: "{ipadapter_model_path}"\n'
            yaml_content += f'    image_encoder_path: "{image_encoder_path}"\n'
            yaml_content += f'    scale: {ipadapter_scale}\n'
            yaml_content += '    enabled: true\n'

            # Add FaceID-specific configuration if enabled
            if is_faceid:
                yaml_content += '    type: faceid\n'  # CRITICAL: Must be 'type: faceid', not 'is_faceid: true'
                yaml_content += '    insightface_model_name: "buffalo_l"\n'
            else:
                yaml_content += '    type: regular\n'  # Explicit type for regular IP-Adapter
        else:
            yaml_content += "# IPAdapter configuration (disabled)\n"
            yaml_content += "use_ipadapter: false\n"

        yaml_content += "\n"

        # Add Image Preprocessing (Fx processors for image domain)
        # Pre-VAE feedback loop
        use_image_feedback = (hasattr(self.ownerComp.par, 'Useprefxfeedback') and
                              self.ownerComp.par.Useprefxfeedback.eval())
        use_color_correction_feedback = (hasattr(self.ownerComp.par, 'Usecolorcorrectionfeedback') and
                                        self.ownerComp.par.Usecolorcorrectionfeedback.eval())

        if use_image_feedback or use_color_correction_feedback:
            yaml_content += """# Multi-stage Image Preprocessing (Pre-Fx)
image_preprocessing:
  enabled: true
  processors:
"""
            processor_order = 1

            if use_image_feedback:
                params = self.gather_fx_parameters_for_processor('feedback_transform')
                yaml_content += f'    - type: "feedback_transform"\n'
                yaml_content += f'      order: {processor_order}\n'
                yaml_content += f'      enabled: true\n'
                yaml_content += f'      params:\n'
                # CRITICAL: Force sync processing to avoid 1-frame delay from pipelined orchestrator
                yaml_content += f'        requires_sync_processing: true\n'
                for param_name, param_value in params.items():
                    yaml_content += f'        {param_name}: {param_value}\n'
                processor_order += 1

            if use_color_correction_feedback:
                params = self.gather_fx_parameters_for_processor('color_correction_feedback')
                yaml_content += f'    - type: "color_correction_feedback"\n'
                yaml_content += f'      order: {processor_order}\n'
                yaml_content += f'      enabled: true\n'
                yaml_content += f'      params:\n'
                # CRITICAL: Force sync processing to avoid 1-frame delay from pipelined orchestrator
                yaml_content += f'        requires_sync_processing: true\n'
                for param_name, param_value in params.items():
                    yaml_content += f'        {param_name}: {param_value}\n'
        else:
            yaml_content += """# Multi-stage Image Preprocessing (disabled)
# image_preprocessing:
#   enabled: false
"""

        yaml_content += "\n"

        # Add Latent Preprocessing (Multi-stage preprocessing)
        # Fx processors: latent_feedback and latent_transform
        use_latent_feedback = (hasattr(self.ownerComp.par, 'Uselatentfeedback') and
                               self.ownerComp.par.Uselatentfeedback.eval())
        use_latent_transform = (hasattr(self.ownerComp.par, 'Uselatenttransform') and
                                self.ownerComp.par.Uselatenttransform.eval())

        if use_latent_feedback or use_latent_transform:
            yaml_content += """# Multi-stage Latent Preprocessing (Fx)
latent_preprocessing:
  enabled: true
  processors:
"""
            processor_order = 1

            if use_latent_feedback:
                params = self.gather_fx_parameters_for_processor('latent_feedback')
                yaml_content += f'    - type: "latent_feedback"\n'
                yaml_content += f'      order: {processor_order}\n'
                yaml_content += f'      enabled: true\n'
                yaml_content += f'      params:\n'
                for param_name, param_value in params.items():
                    yaml_content += f'        {param_name}: {param_value}\n'
                processor_order += 1

            if use_latent_transform:
                params = self.gather_fx_parameters_for_processor('latent_transform')
                yaml_content += f'    - type: "latent_transform"\n'
                yaml_content += f'      order: {processor_order}\n'
                yaml_content += f'      enabled: true\n'
                yaml_content += f'      params:\n'
                for param_name, param_value in params.items():
                    yaml_content += f'        {param_name}: {param_value}\n'
        else:
            yaml_content += """# Multi-stage Latent Preprocessing (disabled)
# latent_preprocessing:
#   enabled: false
"""

        yaml_content += "\n"

        # Add TouchDesigner specific settings
        osc_in_port = self.ownerComp.par.Oscinport.eval()
        osc_out_port = self.ownerComp.par.Oscoutport.eval()
        stream_name = self.ownerComp.par.Streamoutname.eval()
        
        yaml_content += """# TouchDesigner specific settings
td_settings:
  # OSC communication
"""
        yaml_content += f'  osc_receive_port: {osc_in_port}\n'
        yaml_content += f'  osc_transmit_port: {osc_out_port}\n'
        yaml_content += '  \n'
        yaml_content += '  # Memory interface\n'
        yaml_content += f'  input_mem_name: "{stream_name}"\n'
        yaml_content += f'  output_mem_name: "{stream_name}_out"\n'
        yaml_content += '  \n'
        yaml_content += '  # Performance settings\n'
        yaml_content += '  enable_timing_stats: false\n'
        yaml_content += '  fps_target: 30\n'

        return yaml_content
    
    def Printinversions(self):
        """
        Debug helper function to print current textual inversion settings.
        """
        print("\n=== Textual Inversions Status ===")
        # print(f"{'Enabled' if self.ownerComp.par.Usetextualinv.eval() else 'Disabled'}")
        
        if not self.ownerComp.par.Usetextualinv.eval():
            print("❌ Textual inversions are currently disabled")
            print("--------------------------------------------------")
            return
            
        if not hasattr(self.ownerComp.par, 'Textualinvblock'):
            print("❌ No textual inversion blocks found in configuration")
            print("--------------------------------------------------")
            return
            
        print("Current Embeddings:")
        
        embeddings_found = False
        for block in self.ownerComp.par.Textualinvblock.sequence:
            path = block.par.Embedpath.eval()
            token = block.par.Token.eval()
            
            if path == 'select_embedding_from_dropdown':
                continue
                
            if path:
                embeddings_found = True
                print("--------------------------------------------------")
                print(f"Embedding Path: {path}")
                if token:
                    print(f"Token: {token}")
        
        if not embeddings_found:
            print("❌ No embeddings currently configured")
        print("--------------------------------------------------")


    def Sethfcache(self):
        try:
            start = "HF model cache: "
            if self.ownerComp.par.Sethfcache.eval():
                # print(self.ownerComp.par.Hfcache.mode)
                if self.ownerComp.par.Hfcache.mode == ParMode.EXPRESSION:
                    if self.ownerComp.par.Hfcache == self.ownerComp.par.Basefolder + '/models' or self.ownerComp.par.Hfcache.expr == "me.par.Basefolder + '/models'":
                        self.ownerComp.par.Hfheader.label = start + "StreamDiffusion/models"
                    else:
                        self.ownerComp.par.Hfheader.label = start + self.ownerComp.par.Hfcache.eval()
                else:
                    self.ownerComp.par.Hfheader.label = start + self.get_huggingface_cache_path()
            else:
                self.ownerComp.par.Hfheader.label = start + self.get_huggingface_cache_path()
        except Exception as e:
            self.logger.log(f"Error determining HF cache path: {str(e)}", level="WARNING")

    def get_huggingface_cache_path(self):
        user_home = os.path.expanduser("~")
        default_cache_path = os.path.join(user_home, ".cache", "huggingface", "hub")
        default_cache_path = default_cache_path.replace("\\", "/")
        if os.path.exists(default_cache_path):
            return default_cache_path
        else:
            return "Hugging Face cache path does not exist."

    def get_config_type(self, model_id):
        force_model_type = self.ownerComp.par.Forcemodeltype.eval()
        if force_model_type != 'auto':
            return force_model_type
        model_id_lower = model_id.lower()
        # Check SDXL variants first (including sdxl-turbo)
        if "xl" in model_id_lower or "sdxl" in model_id_lower:
            if "turbo" in model_id_lower:
                return "sdxl-turbo" 
            return "sdxl"
        # Non-SDXL turbo models (like SD-turbo)
        if "turbo" in model_id_lower or "sdxs" in model_id_lower:
            return "sd21"
        return "sd15"

    def normalize_model_id(self, model_id):
        """
        Normalizes a model ID to extract the file name or the last folder name from a path.
        If the model ID is a simple identifier with a single '/', it returns the last segment.
        """
        if not model_id or model_id == "":
            return None
        normalized_id = os.path.basename(model_id)
        if len(normalized_id) > 15:
            normalized_id = normalized_id[:15]
        return normalized_id
    
    def Updateengines(self):
        self.update_engine_table()

    def update_engine_table(self, silent = None):
        """
        Updates the 'my_engines' table with the list of engines in the current Basefolder installation.
        Checks for the presence of required files and logs the results.
        """
        engines_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'engines')
        required_files = ['unet.engine', 'vae_encoder.engine', 'vae_decoder.engine']

        my_engines_table = self.setup_table('my_engines', headers=['Engine Configuration', 'Label', 'Status', 'Missing Files'])
        my_engines_table.clear(keepFirstRow=True)

        if not os.path.exists(engines_folder):
            if not silent:
                self.logger.log(f"No engines folder found in {engines_folder}", level='WARNING')
            return

        engine_pattern = re.compile(r'--lcm_lora-.*--tiny_vae-.*--max_batch-.*--min_batch-.*--mode-.*')
        complete_engines = []
        incomplete_engines = {}

        for root, dirs, files in os.walk(engines_folder):
            for subdir in dirs:
                engine_path = os.path.join(root, subdir)
                if not engine_pattern.search(subdir):
                    continue
                engine_files = self.get_engine_files(engine_path)
                missing_files = [file for file in required_files if file not in engine_files]
                # Check for shared VAE files if they are missing
                if 'vae_encoder.engine' in missing_files or 'vae_decoder.engine' in missing_files:
                    vae_encoder_path, vae_decoder_path = self.find_shared_vae_files(engines_folder, engine_path)
                    if vae_encoder_path and 'vae_encoder.engine' in missing_files:
                        missing_files.remove('vae_encoder.engine')
                    if vae_decoder_path and 'vae_decoder.engine' in missing_files:
                        missing_files.remove('vae_decoder.engine')

                engine_name = os.path.relpath(engine_path, engines_folder)
                label = self.generate_engine_label(engine_name)
                if not missing_files:
                    complete_engines.append(engine_name)
                    my_engines_table.appendRow([engine_name, label, 'Complete', ''])
                else:
                    incomplete_engines[engine_name] = missing_files
                    my_engines_table.appendRow([engine_name, label, 'Incomplete', ', '.join(missing_files)])
        
        self.logger.log(f"Engine table updated with {len(complete_engines)} complete engines and {len(incomplete_engines)} incomplete engines.", level='DEBUG' if silent else 'INFO')

    def get_engine_files(self, engine_path):
        """
        Gets the list of files in the given engine path.
        """
        engine_files = []
        for file in os.listdir(engine_path):
            if os.path.isfile(os.path.join(engine_path, file)):
                engine_files.append(file)
        return engine_files

    def find_shared_vae_files(self, engines_folder, current_engine_path):
        """
        Searches for shared VAE encoder and decoder files in other engine folders.
        """
        vae_encoder_path = None
        vae_decoder_path = None
        for root, dirs, files in os.walk(engines_folder):
            for subdir in dirs:
                engine_path = os.path.join(root, subdir)
                if engine_path == current_engine_path:
                    continue
                engine_files = self.get_engine_files(engine_path)
                if 'vae_encoder.engine' in engine_files and not vae_encoder_path:
                    vae_encoder_path = os.path.join(engine_path, 'vae_encoder.engine')
                if 'vae_decoder.engine' in engine_files and not vae_decoder_path:
                    vae_decoder_path = os.path.join(engine_path, 'vae_decoder.engine')
                if vae_encoder_path and vae_decoder_path:
                    break

        return vae_encoder_path, vae_decoder_path

    def generate_engine_label(self, engine_name):
        """
        Generates a condensed label for the engine.
        """
        parts = engine_name.split('--')
        base_name = parts[0]
        max_batch = [part.split('-')[1] for part in parts if part.startswith('max_batch')][0]
        steps_label = f'{max_batch} step'
        width = [part.split('-')[1] for part in parts if part.startswith('width')]
        height = [part.split('-')[1] for part in parts if part.startswith('height')]
        if width and height:
            resolution = f'{width[0]}x{height[0]}'
            label = f'{base_name}, {steps_label}, {resolution}'
        else:
            label = f'{base_name}, {steps_label}'
        return label

    def Loadengine(self):
        engine = self.ownerComp.par.Engine.eval()
        if not engine:
            return
        if self.set_engine_parameters(engine):
            self.logger.log(f"Engine set to {engine}", level='INFO')

    def Log(self):
        log = self.ownerComp.par.Log.eval()
        if log == 'Server stopped...' and self.ownerComp.par.Streamactive.eval():
            self.logger.log(f"Server is streaming...", level='INFO')

    def set_engine_parameters(self, engine):
        """
        Sets the parameters based on the engine details.
        """
        engines_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'engines')
        engine_path = os.path.join(engines_folder, engine)

        if not os.path.exists(engine_path):
            self.logger.log(f"Engine folder {engine_path} does not exist.", level='ERROR')
            return False

        engine_details = self.parse_engine_details(engine)
        if not engine_details:
            return False
        model_base_name = engine_details['base_name']
        self.ownerComp.par.Mymodels = model_base_name
        model_id = self.get_model_id_from_json(model_base_name)
        if model_id:
            self.ownerComp.par.Modelid = model_id
        else:
            self.logger.log(f"Model ID for base name {model_base_name} not found in working_models.json.", level='ERROR')
            return False
        self.ownerComp.par.Width = engine_details['width']
        self.ownerComp.par.Height = engine_details['height']
        self.ownerComp.par.Tindexblock.sequence.numBlocks = engine_details['max_batch']
        self.ownerComp.par.Acceleration = 'tensorrt'
        self.ownerComp.par.Usecontrolnet = False
        # self.ownerComp.par.Enablefx = False
        return True

    def parse_engine_details(self, engine):
        """
        Parses the engine details from the engine string.
        """
        parts = engine.split('--')
        details = {
            'base_name': parts[0],
            'max_batch': int([part.split('-')[1] for part in parts if part.startswith('max_batch')][0]),
            'width': int([part.split('-')[1] for part in parts if part.startswith('width')][0] if 'width' in engine else 512),
            'height': int([part.split('-')[1] for part in parts if part.startswith('height')][0] if 'height' in engine else 512)
        }
        return details

    def get_model_id_from_json(self, model_base_name):
        models_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'working_models.json')
        if not os.path.exists(models_file_path):
            self.logger.log(f'working_models.json not found at {models_file_path}.', level='WARNING')
            return None
        with open(models_file_path, 'r') as file:
            try:
                working_models = json.load(file)
            except json.JSONDecodeError:
                self.logger.log(f'Error reading working_models.json at {models_file_path}.', level='ERROR')
                return None
        for model in working_models:
            if isinstance(model, str):
                normalized_model = os.path.normpath(model).replace("\\", "/").lower()
                normalized_base_name = os.path.normpath(model_base_name).replace("\\", "/").lower()
                if normalized_base_name in normalized_model:
                    return model
        self.logger.log(f"Model ID for base name {model_base_name} not found in working_models.json.", level='WARNING')
        return None

    def Savepreset(self, preset_name=None):
        if preset_name is None:
            simple_name = self.generate_simple_preset_name()
            op.TDResources.PopDialog.OpenDefault(
                text=f'Enter the name for the preset or use the suggested name:\n\n{simple_name}',
                title='Name Preset',
                buttons=['OK', 'Cancel'],
                callback=self._presetNameDialogChoice,
                textEntry=True,
                escButton=1,
                enterButton=0,
                escOnClickAway=True
            )
        else:
            self._saveAndHandlePreset(preset_name)

    def _presetNameDialogChoice(self, info):
        if info['button'] == "OK":
            preset_name = info['enteredText']
        else:
            return
        if preset_name == "":
            preset_name = self.generate_simple_preset_name() 
        self._saveAndHandlePreset(preset_name)

    def _saveAndHandlePreset(self, preset_name):
        # Save the preset
        preset_name = self.save_preset(preset_name)
        self.update_preset_dropdown()
        return preset_name

    def generate_simple_preset_name(self):
        model_id = self.normalize_model_id(self.ownerComp.par.Modelid.eval())
        acceleration = self.ownerComp.par.Acceleration.eval()
        numblocks = self.ownerComp.par.Tindexblock.sequence.numBlocks
        if self.ownerComp.par.Usecustomlcm.eval():
            customlcm = self.normalize_model_id(self.ownerComp.par.Customlcm.eval())
        else:
            customlcm = False
        simple_name = f"{model_id}"
        if acceleration != 'none':
            simple_name += f"_{acceleration}"
        simple_name += f"_{numblocks}steps"
        if customlcm:
            simple_name += f"_{customlcm}"
        if len(simple_name) > 65:
            simple_name = simple_name[:65]
        return simple_name

    def save_preset(self, preset_name=None, preset_file_path=None):
        if preset_name is None:
            preset_name = "Default"
        if preset_file_path is None:
            preset_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'model_presets.json')
        file_exists = os.path.exists(preset_file_path)
        os.makedirs(os.path.dirname(preset_file_path), exist_ok=True)
        presets_list = self.load_presets_list(preset_file_path)
        current_preset = {"preset_name": preset_name}
        for param_name in model_preset_params:
            if param_name == "Loradictblock":
                if self.ownerComp.par.Uselora.eval():
                    current_preset[param_name] = {block.par.Lorapath.eval(): block.par.Weight.eval() for block in self.ownerComp.par.Loradictblock.sequence}
            elif param_name == "Customlcm":
                if self.ownerComp.par.Usecustomlcm.eval():
                    current_preset[param_name] = getattr(self.ownerComp.par, param_name).eval()
            elif param_name == "Customvae":
                if self.ownerComp.par.Usecustomvae.eval():
                    current_preset[param_name] = getattr(self.ownerComp.par, param_name).eval()
            elif param_name == "Tindexblock":
                current_preset[param_name] = self.ownerComp.par.Tindexblock.sequence.numBlocks
            else:
                current_preset[param_name] = getattr(self.ownerComp.par, param_name).eval()

        if not any(preset['preset_name'] == preset_name for preset in presets_list):
            presets_list.append(current_preset)
            with open(preset_file_path, 'w') as file:
                json.dump(presets_list, file, indent=4)
            self.logger.log(f'Preset saved: {preset_name}', level='INFO')
        else:
            choice = ui.messageBox('Preset Exists',
                                f'A preset named "{preset_name}" already exists. Do you want to overwrite it?',
                                buttons=['Overwrite', 'Cancel'])
            if choice == 0:
                presets_list = [preset for preset in presets_list if preset['preset_name'] != preset_name]
                presets_list.append(current_preset)
                with open(preset_file_path, 'w') as file:
                    json.dump(presets_list, file, indent=4)
                self.logger.log(f'Preset overwritten: {preset_name}', level='INFO')
        if not file_exists:
            # Use 'model_preset_params' to generate the parameter list
            param_list = '\n'.join(model_preset_params)
            ui.messageBox('First Time Setup',
                        f'The following parameters are stored in the presets:\n{param_list}',
                        buttons=['OK'])
        return preset_name

    def load_preset(self, preset_name=None, preset_file_path=None):
        if preset_name is None:
            preset_name = "Default"
        if preset_file_path is None:
            preset_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'model_presets.json')
        presets_list = self.load_presets_list(preset_file_path)
        preset = next((p for p in presets_list if p['preset_name'] == preset_name), None)
        if preset is not None:
            for param_name in model_preset_params:
                if param_name == "Loradictblock":
                    loradict = preset.get(param_name, {})
                    lora_sequence = self.ownerComp.par.Loradictblock.sequence
                    if loradict:
                        lora_sequence.numBlocks = len(loradict)
                        for i, (block_path, weight) in enumerate(loradict.items()):
                            lora_sequence[i].par.Lorapath = block_path
                            lora_sequence[i].par.Weight = weight
                elif param_name == "Tindexblock":
                    self.ownerComp.par.Tindexblock.sequence.numBlocks = preset.get(param_name, self.ownerComp.par.Tindexblock.sequence.numBlocks)
                elif param_name == "Modelid":
                    try:
                        model_name = preset.get(param_name, getattr(self.ownerComp.par, param_name).default)
                        self.ownerComp.par.Mymodels = model_name
                        # print(model_name)
                    except:
                        continue
                    setattr(self.ownerComp.par, param_name, preset.get(param_name, getattr(self.ownerComp.par, param_name).default))
                elif param_name.startswith("Cn"):
                    #if Cnfolder not in preset, skip setting    
                    if param_name in preset:
                        setattr(self.ownerComp.par, param_name, preset.get(param_name, getattr(self.ownerComp.par, param_name).default))
                else:
                    setattr(self.ownerComp.par, param_name, preset.get(param_name, getattr(self.ownerComp.par, param_name).default))
            self.logger.log(f'Preset loaded: {preset_name}', level='INFO')
            self.ownerComp.par.Lastpreset = preset_name
        else:
            self.logger.log(f'Preset with name "{preset_name}" not found.', level='WARNING')
            self.update_preset_dropdown()

    def load_presets_list(self, preset_file_path):
        if os.path.exists(preset_file_path):
            with open(preset_file_path, 'r') as file:
                return json.load(file)
        else:
            return []

    def Loadpreset(self, preset_name=None):
        if preset_name is None:
            preset_name = self.ownerComp.par.Presetname.eval()
        self.load_preset(preset_name)

    def update_preset_dropdown(self, reset=False):
        if reset:
            # Reset the Presetname menu to a default message
            self.ownerComp.par.Presetname.menuNames = ['Saved Model Configs Will Show Here']
            self.ownerComp.par.Presetname.menuLabels = ['Saved Model Configs Will Show Here']
        else:
            preset_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'model_presets.json')
            presets_list = self.load_presets_list(preset_file_path)
            preset_names = [preset['preset_name'] for preset in presets_list]
            if preset_names:
                self.ownerComp.par.Presetname.menuNames = preset_names
                self.ownerComp.par.Presetname.menuLabels = preset_names
            else:
                self.ownerComp.par.Presetname.menuNames = ['Saved Model Configs Will Show Here']
                self.ownerComp.par.Presetname.menuLabels = ['Saved Model Configs Will Show Here']

    def Savemodel(self, model=None):
        if self.ownerComp.par.Basefolder.eval() == '':
            return
        models_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'working_models.json')
        os.makedirs(os.path.dirname(models_file_path), exist_ok=True)
        models_list = self.load_models_list(models_file_path)
        # Function to check if a string is a Windows absolute path
        def is_windows_abs_path(s):
            return len(s) > 2 and s[1] == ':' and (s[0].isalpha() and s[2] == '\\' or s[2] == '/')
        # Get the current model ID/PATH and normalize it if it's a full Windows path
        if not model:
            current_model_id = self.ownerComp.par.Modelid.eval()
            if is_windows_abs_path(current_model_id):
                current_model_id = os.path.normpath(current_model_id)
        else:
            current_model_id = model
        # Normalize all paths in the models_list for comparison, only if they are full Windows paths
        normalized_models_list = [os.path.normpath(model_id) if is_windows_abs_path(model_id) else model_id for model_id in models_list]
        # Check if the current model ID/PATH is not in the normalized list
        if current_model_id not in normalized_models_list:
            models_list.append(current_model_id)  # Append the original ID/path
            with open(models_file_path, 'w') as file:
                json.dump(models_list, file, indent=4)
            self.logger.log(f'Model ID {current_model_id} added to the working models list.', level='DEBUG')
        else:
            self.logger.log(f'Model ID {current_model_id} is already in the working models list.', level='DEBUG')
        self.sync_model_table()

    def sync_model_table(self):
        """
        Synchronizes the model IDs from the working_models.json file, models/Model folder, models/checkpoints folder,
        and the custom SD models folder with the 'model_table' DAT. Adds a second column 'Label' to the table, 
        which contains only the file name if the model path is a full file path, or the base model path/id if it's 
        a relative path or not an actual file. Ensures only unique models are added to the table.
        """
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        if os.path.exists(base_folder):
            models_file_path = os.path.join(base_folder, 'StreamDiffusionTD', 'working_models.json')
            models_list = self.load_models_list(models_file_path)

            # Ensure Model and checkpoints folders exist
            model_folder = os.path.join(base_folder, 'models', 'Model')
            checkpoints_folder = os.path.join(base_folder, 'models', 'checkpoints')
            os.makedirs(model_folder, exist_ok=True)
            os.makedirs(checkpoints_folder, exist_ok=True)

            # Get models from models/Model folder
            models_list.extend(self.get_models_from_folder(model_folder))

            # Get models from models/checkpoints folder
            models_list.extend(self.get_models_from_folder(checkpoints_folder))

            # Get models from custom SD models folder
            sd_models_folder = tdu.expandPath(self.ownerComp.par.Sdmodelsfolder.eval())
            if os.path.exists(sd_models_folder) and sd_models_folder != model_folder and sd_models_folder != checkpoints_folder:
                models_list.extend(self.get_models_from_folder(sd_models_folder))

            # Remove duplicates while preserving order
            unique_models = []
            seen = set()
            for model in models_list:
                model_key = os.path.basename(model)  # Use filename as key for uniqueness
                if model_key not in seen:
                    unique_models.append(model)
                    seen.add(model_key)

            # Get the 'model_table' DAT operator
            model_table = op('model_table')
            # Clear the table before adding new data
            model_table.clear()
            # Add headers
            model_table.appendRow(['Name', 'Label'])
            model_table.appendRow(['select_working_model_from_dropdown', 'Select Working Model from Dropdown'])

            # Add each unique model ID as a new row in the table
            for model_id in unique_models:
                # Determine the label based on whether the model_id is a full file path
                if os.path.isabs(model_id) and os.path.isfile(model_id):
                    # Extract the file name from the full file path
                    label = os.path.basename(model_id)
                else:
                    # Use the base model path/id as the label
                    label = os.path.basename(model_id)
                model_table.appendRow([model_id, label])
            
            self.logger.log('Model table synchronized with unique models from working models list and all local model folders.', level='DEBUG')
    def get_models_from_folder(self, folder_path):
        """
        Helper function to get model files from a specified folder.
        """
        model_extensions = ['.ckpt', '.safetensors', '.pt', '.pth']
        models = []
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if any(file.endswith(ext) for ext in model_extensions):
                    models.append(os.path.join(folder_path, file))
        return models

    def load_models_list(self, models_file_path):
        """
        Helper function that loads the list of models from the specified JSON file.
        If the file doesn't exist, it creates it with default presets.

        Parameters:
        models_file_path (str): The file path to the JSON file containing the models list.

        Returns:
        list: The list of models loaded from the file, or default presets if the file doesn't exist.
        """
        default_presets = [
            "stabilityai/sd-turbo",
            "IDKiro/sdxs-512-dreamshaper",
            # "stabilityai/sdxl-turbo"
        ]
        # Check if the file exists
        if not os.path.exists(models_file_path):
            # If the file doesn't exist, create it with default presets
            os.makedirs(os.path.dirname(models_file_path), exist_ok=True)
            with open(models_file_path, 'w') as file:
                json.dump(default_presets, file, indent=4)
            self.logger.log(f'Created working_models.json with default presets at {models_file_path}', level='INFO')
            return default_presets
        # If the file exists, read the current list of models
        with open(models_file_path, 'r') as file:
            try:
                models_list = json.load(file)
            except json.JSONDecodeError:
                # If there's a JSON decode error, use default presets
                self.logger.log(f'Error decoding JSON in {models_file_path}. Using default presets.', level='WARNING')
                models_list = default_presets
        return models_list

    def Mymodels(self):
        modelpreset = self.ownerComp.par.Mymodels.eval()
        if modelpreset != 'select_working_model_from_dropdown':
            self.ownerComp.par.Modelid = modelpreset

    def Modelid(self):
        self.update_cn_id_menus()

        
    def Updateloramodelpath(self, block, path):
        setattr(self.ownerComp.par, f'Loradictblock{block}lorapath', path)

    def Openvenv(self):
        base_folder = self.ownerComp.par.Basefolder.eval()
        # Check if 'venv' or '.venv' directory exists
        if os.path.exists(os.path.join(base_folder, 'venv')):
            venv_path = os.path.join(base_folder, 'venv')
        elif os.path.exists(os.path.join(base_folder, '.venv')):
            venv_path = os.path.join(base_folder, '.venv')
        else:
            self.logger.log("Virtual environment not found in the specified base folder.", level="ERROR")
            return

        if platform.system() == 'Windows':
            # Windows-specific activation
            activate_script_path = os.path.join(venv_path, 'Scripts', 'activate.bat')
            if not os.path.exists(activate_script_path):
                self.logger.log("activate.bat script not found in the virtual environment.", level="ERROR")
                return
            cmd_command = f'start cmd.exe /K "cd /d {base_folder} & {activate_script_path}"'
            try:
                subprocess.Popen(cmd_command, shell=True)
                self.logger.log("Virtual environment activated in a new CMD window.", level="INFO")
            except Exception as e:
                self.logger.log(f"Error opening CMD window and activating virtual environment: {e}", level="ERROR")
        else:
            # macOS-specific activation
            activate_script_path = os.path.join(venv_path, 'bin', 'activate')
            if not os.path.exists(activate_script_path):
                self.logger.log("activate script not found in the virtual environment.", level="ERROR")
                return
            terminal_command = f'tell application "Terminal" to do script "unset PYTHONPATH && cd {base_folder} && source {activate_script_path}"'
            try:
                subprocess.run(['osascript', '-e', terminal_command])
                self.logger.log("Virtual environment activated in a new Terminal window.", level="INFO")
            except Exception as e:
                self.logger.log(f"Error opening Terminal and activating virtual environment: {e}", level="ERROR")

    def Showbuiltin(self):  
        self.ownerComp.showCustomOnly = 1- self.ownerComp.par.Showbuiltin.eval()

    def Createpars(self):
        # Configuration as per stream_config.json
        config = {
            "Modelid": {"type": "str", "default": "stabilityai/sd-turbo", "help": "The name of the model to use for image generation."},
            "Tindexlist": {"type": "str", "default": "[18, 25, 33, 41]", "help": "List of T-indices for the model."},
            "Loradict": {"type": "str", "default": "{\"F:\\ComfyUI\\ComfyUI_windows_portable\\ComfyUI\\models\\loras\\node-network-000013.safetensors\": 3.0}", "help": "Dictionary of LoRA names and scales."},
            "Prompt": {"type": "str", "default": "abstract artistic portal to surreal smoky void, colorful bright light god rays, a complex smoky billowing smoke tunnel, complex detailed ribbons and sparks, high quality stunning stark contrast, collage 3d and 2d digital art, ripped paper and folded collage paper cut", "help": "The prompt to generate images from."},
            "Negprompt": {"type": "str", "default": "text, words, black and white, shit, shit, shit, low quality, bad quality, blurry, low resolution", "help": "The negative prompt to avoid."},
            "Framesize": {"type": "int", "default": 1, "help": "The frame buffer size for denoising batch."},
            "Width": {"type": "int", "default": 512, "help": "The width of the image."},
            "Height": {"type": "int", "default": 512, "help": "The height of the image."},
            "Acceleration": {"type": "menu", "default": ["none", "xformers", "tensorrt"], "help": "The acceleration method."},
            "Denoisebatch": {"type": "bool", "default": True, "help": "Whether to use denoising batch or not."},
            "Seed": {"type": "int", "default": 3242346, "help": "The seed for the generation."},
            "Cfgtype": {"type": "menu", "default": ["none", "full", "self", "initialize"], "help": "The cfg_type for img2img mode."},
            "Guidancescale": {"type": "float", "default": 0.5, "help": "The CFG scale."},
            "Delta": {"type": "float", "default": 0.5, "help": "The delta multiplier of virtual residual noise."},
            "Addnoise": {"type": "bool", "default": True, "help": "Whether to add noise for following denoising steps or not."},
            "Imagefilter": {"type": "bool", "default": False, "help": "Whether to enable similar image filter or not."},
            "Filterthresh": {"type": "float", "default": 0.99, "help": "The threshold for similar image filter."},
            "Maxskipframe": {"type": "int", "default": 2, "help": "The max skip frame for similar image filter."}
        }

        for par_name, details in config.items():
            par_type = details["type"]
            default = details["default"]
            help_text = details["help"]

            # Use different approach for menu type to set menu names and labels
            if par_type == "menu":
                self.create_parameter(par_name, par_type, default=default, label=par_name, menu_items=default)
            else:
                self.create_parameter(par_name, par_type, default=default, label=par_name)

            # Set help text for the parameter
            getattr(self.ownerComp.par, par_name).help = help_text

    def setup_table(self, table_name, headers=None):
        # Check if the table exists, if not, create it
        table = self.ownerComp.op(table_name)
        if table is None:
            table = self.ownerComp.create(tableDAT, table_name)
            table.clear()
            if headers:
                table.appendRow(headers)
        return table

    def _ensure_daydream_status_table(self):
        """Ensure daydream_web_status table exists with all required columns"""
        try:
            table = op('daydream_web_status')
        except:
            table = None

        if not table:
            self.logger.log("daydream_web_status table not found!", level='ERROR')
            return

        # Define all required columns
        required_columns = [
            'parameter', 'value',
            'active_client', 'stream_id', 'status', 'whip_url', 'whep_url',
            'playback_id', 'output_stream_url', 'frames_received', 'frames_sent',
            'input_fps', 'output_fps', 'last_update', 'stream_state', 'stream_key',
            'is_active', 'last_poll', 'capacity_idle', 'capacity_total', 'capacity_available',
            'api_response', 'api_timestamp', 'last_params',
            'connection_quality', 'packet_loss_pct', 'jitter_ms', 'rtt_ms',
            'daydream_input_fps', 'last_error'
        ]

        # Get existing columns
        existing_cols = [table[i, 0].val for i in range(table.numRows)]

        # Add missing columns
        for col in required_columns:
            if col not in existing_cols:
                table.appendRow([col, ''])
                self.logger.log(f"Added missing column to status table: {col}", level='INFO')

    def _update_osc_table(self, table, column_name, value):
        """Update stream_osc_data table - ROW-BASED: each row is [param_name, value]"""
        if not table:
            return

        # Find row with matching parameter name in column 0
        found_row = None
        for i in range(table.numRows):
            if table[i, 0].val == column_name:
                found_row = i
                break

        # If row doesn't exist, add it
        if found_row is None:
            table.appendRow([column_name, value])
        else:
            # Update existing row's value in column 1
            table[found_row, 1] = value

    def setup_project_info_table(self):
        """
        Sets up a table with project information.
        """
        # Define the headers for the project info table
        headers = ['Info', 'Value']
        project_info_table = self.setup_table('projectInfoTable', headers)
        project_info_table.clear(keepFirstRow = True)
        # Populate the table with project information
        project_data = [
            ('Project Folder', project.folder),
            ('Project Name', project.name),
            ('Last Save Time', project.saveTime),
            ('TDversion', str(app.build)),
            ('OS Name at Last Save', project.saveOSName),
            ('OS Version at Last Save', project.saveOSVersion),
            ('Real-Time State', str(project.realTime)),
            ('Cook Rate', str(project.cookRate)),
            ('Licenses', ', '.join([l.type for l in licenses])),
        ]

        for info, value in project_data:
            project_info_table.appendRow([info, value])


    def create_parameter(self, par_name, par_type, page='Custom', default=None, norm_min=None, norm_max=None, size=1, menu_items=None, label=None, order=None, replace=False, section=None, clamp=True):
        '''
        Creates a custom parameter on a specified page in a TouchDesigner component, supporting a wide 
        range of parameter types. This function allows for creating basic types like 'float', 'int', 
        'str', 'bool', and 'menu', as well as node reference types such as 'op', 'comp', etc. Advanced 
        options like 'label', 'order', and 'replace' enable further customization. For 'float' and 'int' 
        types, 'size' determines the number of values associated with the parameter. The 'default' 
        parameter sets the initial value, with special handling for 'menu' types where it sets the menu 
        options. If 'replace' is set to False, the function errors if the parameter already exists.

        Parameters:
        par_name (str): The name of the parameter.
        par_type (str): The type of the parameter. Supported types are:
                        'float', 'int', 'str', 'bool', 'menu', 'op', 'comp', 'object', 'panelcomp', 
                        'top', 'chop', 'sop', 'mat', 'dat', 'xy', 'xyz', 'xyzw', 'wh', 'uv', 'uvw', 
                        'rgb', 'rgba', 'file', 'folder', 'pulse', 'momentary', 'python', 'par', 'header'.
        page (str): The page name where the parameter will be added. Defaults to 'Custom'.
        default: The default value for the parameter. For 'menu', it should be a list of strings.
        norm_min, norm_max (optional): Normalized minimum and maximum values for 'float' and 'int' types.
        size (int, optional): Number of values for 'float' and 'int' types. Defaults to 1.
        menu_items (list of str, optional): List of menu items for 'menu' type parameters.
        label (str, optional): Display label of the parameter. Defaults to par_name.
        order (int, optional): Display order of the parameter.
        replace (bool, optional): Determines whether to replace an existing parameter. Defaults to True.
                                If set to False, the function errors if the parameter already exists.
        section (bool, optional): If set to True, adds a visual separator above this parameter. Useful for organizing parameters into distinct sections on the UI.

        Returns:
        The created parameter object.
        '''
        # print(f"Creating parameter: Name: {par_name}, Type: {par_type}, Page: {page}")
        # Check if the parameter already exists
        if hasattr(self.ownerComp.par, par_name):
            if not replace:
                # If the parameter exists and 'replace' is False, skip creating the parameter
                return getattr(self.ownerComp.par, par_name)
        # Check if the page exists
        custom_page = next((p for p in self.ownerComp.customPages if p.name == page), None)
        if not custom_page:
            # If the page doesn't exist, create it
            custom_page = self.ownerComp.appendCustomPage(page)

        # Mapping for parameter creation based on type
        create_method = {
            'float': lambda: custom_page.appendFloat(par_name, label=label, size=size, order=order, replace=replace),
            'int': lambda: custom_page.appendInt(par_name, label=label, size=size, order=order, replace=replace),
            'str': lambda: custom_page.appendStr(par_name, label=label, order=order, replace=replace),
            'bool': lambda: custom_page.appendToggle(par_name, label=label, order=order, replace=replace),
            'menu': lambda: custom_page.appendMenu(par_name, label=label, order=order, replace=replace),
            'op': lambda: custom_page.appendOP(par_name, label=label, order=order, replace=replace),
            'comp': lambda: custom_page.appendCOMP(par_name, label=label, order=order, replace=replace),
            'object': lambda: custom_page.appendObject(par_name, label=label, order=order, replace=replace),
            'panelcomp': lambda: custom_page.appendPanelCOMP(par_name, label=label, order=order, replace=replace),
            'top': lambda: custom_page.appendTOP(par_name, label=label, order=order, replace=replace),
            'chop': lambda: custom_page.appendCHOP(par_name, label=label, order=order, replace=replace),
            'sop': lambda: custom_page.appendSOP(par_name, label=label, order=order, replace=replace),
            'mat': lambda: custom_page.appendMAT(par_name, label=label, order=order, replace=replace),
            'dat': lambda: custom_page.appendDAT(par_name, label=label, order=order, replace=replace),
            'xy': lambda: custom_page.appendXY(par_name, label=label, order=order, replace=replace),
            'xyz': lambda: custom_page.appendXYZ(par_name, label=label, order=order, replace=replace),
            'xyzw': lambda: custom_page.appendXYZW(par_name, label=label, order=order, replace=replace),
            'wh': lambda: custom_page.appendWH(par_name, label=label, order=order, replace=replace),
            'uv': lambda: custom_page.appendUV(par_name, label=label, order=order, replace=replace),
            'uvw': lambda: custom_page.appendUVW(par_name, label=label, order=order, replace=replace),
            'rgb': lambda: custom_page.appendRGB(par_name, label=label, order=order, replace=replace),
            'rgba': lambda: custom_page.appendRGBA(par_name, label=label, order=order, replace=replace),
            'file': lambda: custom_page.appendFile(par_name, label=label, order=order, replace=replace),
            'folder': lambda: custom_page.appendFolder(par_name, label=label, order=order, replace=replace),
            'pulse': lambda: custom_page.appendPulse(par_name, label=label, order=order, replace=replace),
            'momentary': lambda: custom_page.appendMomentary(par_name, label=label, order=order, replace=replace),
            'python': lambda: custom_page.appendPython(par_name, label=label, order=order, replace=replace),
            'par': lambda: custom_page.appendPar(par_name, label=label, order=order, replace=replace),
            'header': lambda: custom_page.appendHeader(par_name, label=label, order=order, replace=replace)
        }.get(par_type.lower())

        if create_method is None:
            raise ValueError(f"Unsupported parameter type: {par_type}")

        new_param_group = create_method()

        if new_param_group is None:
            raise Exception("Parameter group creation failed")

        if not hasattr(new_param_group, 'pars'):
            raise Exception(f"Expected ParGroup, got {type(new_param_group)}")

        new_param = new_param_group[0] if new_param_group.pars else None

        if new_param is None:
            raise Exception("Parameter creation failed")
        # Set default, norm_min, norm_max, and menu_items based on the parameter type
        if default is not None:
            if par_type == 'menu' and isinstance(default, list) and all(isinstance(item, str) for item in default):
                new_param.menuNames = default
                new_param.menuLabels = default
            else:
                setattr(self.ownerComp.par, par_name, default)

        if par_type in ['float', 'int']:
            if norm_min is not None:
                new_param.normMin = norm_min
                new_param.min = norm_min
                new_param.clampMin = clamp
            if norm_max is not None:
                new_param.normMax = norm_max
                new_param.max = norm_max
                new_param.clampMax = clamp

        if section:
            new_param.startSection = True
        return new_param 
    
    def Printpardetails(self):
        """
        This function prints the details of the parameters of the owner component.
        It prints the parameter name, label, value, default value, and the page name.
        """
        print(f"{'Parameter Name':<18}|{'Label':<18}|{'Value':<18}|{'Default':<18}|{'Page':<18}")
        print("-" * 90)
        for par in self.ownerComp.customPars:
            print(f"{par.name:<18}|{par.label:<18}|{str(par.eval()):<18}|{str(par.default):<18}|{par.page.name:<18}")


    def setup_par_details_table(self):
        """
        This function sets up a table with the details of the parameters of the owner component.
        It defines the headers for the table and populates it with the parameter details.
        """
        # Define the headers for the parameter details table
        headers = ['Parameter Name', 'Label', 'Value', 'Default', 'Page', 'Norm Min', 'Norm Max', 'Min', 'Max', 'Clamp Min', 'Clamp Max', 'Enabled', 'Menu Names', 'Menu Labels']
        
        par_details_table = self.setup_table('parDetailsTable', headers)
        par_details_table.clear(keepFirstRow = True)

        # Populate the table with parameter details
        for par in self.ownerComp.customPars:
            par_details = [
                par.name,
                par.label,
                str(par.eval()),
                str(par.default),
                par.page.name,
                str(par.normMin),
                str(par.normMax),
                str(par.min),
                str(par.max),
                str(par.clampMin),
                str(par.clampMax),
                str(par.enable),
                str(par.menuNames),
                str(par.menuLabels)
            ]
            par_details_table.appendRow(par_details)


    def Editcallbacksscript(self):
        viewop = self.ownerComp.par.Callbackdat.eval()
        viewop.openViewer(unique=False, borders=True)

    def Viewinstallguide(self):
        import webbrowser
        webbrowser.open('https://dotsimulate.com/docs/streamdiffusiontd')

    def setup_shared_memory_change_detection(self):
        """
        Sets up the shared memory change detection system for optimal frame change callbacks.
        """
        try:
            # Find shared memory input operators
            shmem_ops = ['numpy_share_in', 'numpy_share_in_cn']  # Add other shmem op names as needed
            
            for op_name in shmem_ops:
                shmem_op = self.ownerComp.op(op_name)
                if shmem_op and hasattr(shmem_op, 'ext') and hasattr(shmem_op.ext, 'shMemExt'):
                    # Enable change detection with this component as the callback target
                    shmem_op.ext.shMemExt.enable_image_change_detection(
                        callback_target_path=str(self.ownerComp.path),
                        hash_method='md5'  # You can make this configurable via parameters
                    )
                    # self.logger.log(f"Enabled change detection for {op_name}", level='DEBUG')
                    
        except Exception as e:
            self.logger.log(f"Error setting up shared memory change detection: {str(e)}", level='WARNING')

    def send_frame_acknowledgment(self):
        """
        Sends frame acknowledgment to StreamDiffusion server for loopback synchronization.
        This ensures the server waits for TD to process each frame before continuing.
        """
        try:
            if not self.ownerComp.par.Serveractive.eval():
                return False
            
            # Check if OSC is properly set up
            osc_out = op('oscout1')
            if not osc_out:
                self.logger.log("ERROR: oscout1 operator not found!", level='ERROR')
                return False
                
            # Send frame acknowledgment
            self.send_parameter_update('frame_ack', 1, 'command')
            # self.logger.log("✓ SENT frame acknowledgment (/frame_ack) to server", level='INFO')
            return True
            
        except Exception as e:
            self.logger.log(f"✗ ERROR sending frame acknowledgment: {str(e)}", level='ERROR')
            return False

    def enable_shmem_change_detection(self, enable=True, hash_method='md5'):
        """
        Enable or disable shared memory change detection.
        
        Args:
            enable (bool): Whether to enable change detection
            hash_method (str): Hash method to use ('md5', 'python_hash', 'checksum')
        """
        try:
            shmem_ops = ['numpy_share_in', 'numpy_share_in_cn']
            
            for op_name in shmem_ops:
                shmem_op = self.ownerComp.op(op_name)
                if shmem_op and hasattr(shmem_op, 'ext') and hasattr(shmem_op.ext, 'shMemExt'):
                    if enable:
                        shmem_op.ext.shMemExt.enable_image_change_detection(
                            callback_target_path=str(self.ownerComp.path),
                            hash_method=hash_method
                        )
                    else:
                        shmem_op.ext.shMemExt.disable_image_change_detection()
                        
            status = "enabled" if enable else "disabled"
            self.logger.log(f"Shared memory change detection {status}", level='INFO')
            
        except Exception as e:
            self.logger.log(f"Error configuring shared memory change detection: {str(e)}", level='ERROR')

    def update_webrender_top(self, url):
        """
        Updates the URL of the 'webrender_daydream' TOP and sets source to URL mode.
        """
        webrender_op = self.ownerComp.op('webrender_daydream')
        if webrender_op:
            # Set source to "URL or File" mode (not DAT mode)
            webrender_op.par.source = 'url'
            webrender_op.par.url = url
            webrender_op.par.active = True

            # CRITICAL: Must pulse reloadsrc to actually load the URL
            webrender_op.par.reloadsrc.pulse()

            self.logger.log(f"Updated Daydream Web Render TOP to URL mode: {url}", level='INFO')
        else:
            self.logger.log("Could not find a 'webrender_daydream' TOP to update.", level='WARNING')

    def update_videostreamout_top(self, url):
        """
        Updates the Destination URL of the 'videostreamout1' TOP and restarts the stream.
        """
        streamout_op = self.ownerComp.op('videostreamout1')
        if streamout_op:
            # Deactivate to ensure a clean connection to the new URL
            streamout_op.par.active = False
            # Set the new destination URL
            streamout_op.par.url = url
            # Activate the stream
            streamout_op.par.active = True
            self.logger.log(f"Updated and reactivated Video Stream Out TOP ('videostreamout1') with URL: {url}", level='INFO')
        else:
            self.logger.log("Could not find a 'videostreamout1' TOP to update.", level='WARNING')

    def Cnblock(self):
        """
        Sends ControlNet configurations from each block in the Cn sequence to the OSC Out DAT.
        Filters out duplicate model_ids and keeps only the highest weight for each unique model.
        Backend-aware: sends appropriate format based on par.Backend setting.
        """
        import json
        if not self.ownerComp.par.Streamactive:
            return
            
        backend = self.ownerComp.par.Backend.eval()
        osc_out = op('oscout1')
        
        # if backend == 'Daydream':
        # For Daydream backend, build full ControlNet array
        model_weights = {}  # Track highest weight for each model_id

        # Iterate over each block in the Cn sequence
        for block in self.ownerComp.par.Cn.sequence:
            model_id = block.par.Id.eval()
            weight = block.par.Weight.eval()
            enabled = block.par.Enable.eval()

            self.logger.log(f"Cnblock: Processing block with Id='{model_id}', enabled={enabled}", level='DEBUG')

            # Skip if not enabled or empty ID
            if not enabled or not model_id:
                continue

            # Track highest weight for each model
            if model_id not in model_weights or weight > model_weights[model_id]['weight']:
                # Get preprocessor using new unified logic
                auto_preprocess = self.ownerComp.par.Autopreprocess.eval() if hasattr(self.ownerComp.par, 'Autopreprocess') else True
                manual_preprocessor = block.par.Preprocessor.eval() if hasattr(block.par, 'Preprocessor') else None

                preprocessor = self.get_preprocessor_for_controlnet(
                    model_id,
                    backend,
                    auto=auto_preprocess,
                    manual_value=manual_preprocessor
                )

                # Build preprocessor params - use dynamic Dyn* parameters if available
                preprocessor_params = self.gather_dynamic_cn_parameters_for_preprocessor(preprocessor)

                # Fallback to hardcoded parameters if no dynamic params found
                if not preprocessor_params:
                    if "pose" in preprocessor or "openpose" in preprocessor:
                        preprocessor_params = {
                            "confidence_threshold": self.ownerComp.par.Confidencethreshold.eval() if hasattr(self.ownerComp.par, 'Confidencethreshold') else 0.5
                        }
                    elif "canny" in preprocessor:
                        preprocessor_params = {
                            "low_threshold": self.ownerComp.par.Lowthreshold.eval() if hasattr(self.ownerComp.par, 'Lowthreshold') else 100,
                            "high_threshold": self.ownerComp.par.Highthreshold.eval() if hasattr(self.ownerComp.par, 'Highthreshold') else 200
                        }

                controlnet_config = {
                    "model_id": model_id,
                    "conditioning_scale": weight,
                    "enabled": enabled,
                    "preprocessor": preprocessor,
                    "preprocessor_params": preprocessor_params,
                    "control_guidance_start": 0.0,
                    "control_guidance_end": 1.0
                }

                model_weights[model_id] = {
                    'weight': weight,
                    'config': controlnet_config
                }

        # Extract final configs (only highest weight for each model)
        final_configs = [data['config'] for data in model_weights.values()]
        
        # Send to Daydream API format
        if final_configs and self.ownerComp.par.Streamactive.eval():
            # DEBUG: Log what's being sent
            self.logger.log(f"Cnblock: Sending final_configs to API: {final_configs}", level='DEBUG')
            self.send_parameter_update('controlnets', final_configs, 'json')
                
        # else:
        #     # For local backend, send individual ControlNet parameters
        #     # Use the first enabled ControlNet from the sequence
        #     for block in self.ownerComp.par.Cn.sequence:
        #         if block.par.Enable.eval():
        #             model_id = block.par.Id.eval()
        #             weight = block.par.Weight.eval()
                    
        #             # Add "thibaud/" prefix if not already present
        #             if not model_id.startswith("thibaud/"):
        #                 model_id = f"thibaud/{model_id}"
                    
        #             osc_out.sendOSC('/controlnet_weight', [weight])
        #             osc_out.sendOSC('/use_controlnet', [True])
        #             osc_out.sendOSC('/controlnet_model', [model_id])
        #             break  # Only send the first enabled one for local backend

    def Ipadapterenable(self):
        """
        Sends IPAdapter enable/disable status via OSC.
        Called when Ipadapterenable parameter changes.
        """
        if not self.ownerComp.par.Streamactive:
            return
            
        enable_value = self.ownerComp.par.Ipadapterenable.eval()
        self.send_parameter_update('ipadapter_enable', enable_value)
        
        self.logger.log(f'IPAdapter enabled: {enable_value}', level='INFO')

    def Ipadapterupdate(self):
        """
        Sends IPAdapter style image update request via OSC.
        Called when Ipadapterupdate parameter is pulsed.
        """
        if not self.ownerComp.par.Streamactive:
            return

        osc_out = op('oscout1')
        self.send_parameter_update('ipadapter_update', 1, 'command')

        self.logger.log('IPAdapter style image update requested', level='INFO')

        # Only upload image if IP adapter is actually enabled AND backend is Daydream
        if self.ownerComp.par.Backend.eval() == 'Daydream':
            if hasattr(self.ownerComp.par, 'Ipadapterenable') and self.ownerComp.par.Ipadapterenable.eval():
                self.upload_ipadapter_image()
            else:
                self.logger.log('IP adapter image upload skipped - IP adapter is disabled', level='WARNING')

    def Ipadapterscale(self):
        """
        Updates IPAdapter scale value.
        Called when Ipadapterscale parameter changes.
        NOTE: IP adapter IS dynamic (web prototype confirms this)
        """
        if not self.ownerComp.par.Streamactive:
            return

        scale_value = self.ownerComp.par.Ipadapterscale.eval()
        self.send_parameter_update('ipadapter_scale', scale_value)
        self.logger.log(f'IPAdapter scale: {scale_value}', level='INFO')

    def Uselatentfeedback(self):
        self.logger.log('Uselatentfeedback changed', level='INFO')
        """Called when Uselatentfeedback toggle changes - updates Fx dynamic parameters"""
        self.update_fx_dynamic_parameters()

    def Uselatenttransform(self):
        self.logger.log('Uselatenttransform changed', level='INFO')
        """Called when Uselatenttransform toggle changes - updates Fx dynamic parameters"""
        self.update_fx_dynamic_parameters()

    def Useprefxfeedback(self):
        self.logger.log('Useprefxfeedback changed', level='INFO')
        """Called when Useprefxfeedback toggle changes - updates Fx dynamic parameters"""
        self.update_fx_dynamic_parameters()

    def Usecolorcorrectionfeedback(self):
        self.logger.log('Usecolorcorrectionfeedback changed', level='INFO')
        """Called when Usecolorcorrectionfeedback toggle changes - updates Fx dynamic parameters"""
        self.update_fx_dynamic_parameters()

    def Fxparameterupdate(self, par):
        """
        Generic callback for ALL Fx* dynamic parameters.
        Automatically called when any Fx* parameter changes via onValueChange.
        Sends OSC to update Fx processor parameters live.
        """
        self.logger.log(f"ENTER Fxparameterupdate: {par.name}", level='ERROR')

        if not self.ownerComp.par.Streamactive:
            self.logger.log("EXIT: Streamactive False", level='ERROR')
            return

        import re
        import json

        par_name = par.name
        if not par_name.startswith('Fx'):
            self.logger.log(f"EXIT: Not Fx param: {par_name}", level='ERROR')
            return

        clean_name = par_name[2:].lower()  # Remove 'Fx' prefix (2 chars)
        self.logger.log(f"Processing Fx param: {par_name}, clean: {clean_name}", level='ERROR')

        table_dat = self.ownerComp.op('table_preprocessors')
        if not table_dat:
            self.logger.log("EXIT: table_preprocessors not found", level='ERROR')
            return

        self.logger.log(f"table_preprocessors found, rows: {table_dat.numRows}", level='ERROR')

        # Match parameter to processor
        for row in range(1, table_dat.numRows):
            processor_type = table_dat[row, 'name'].val
            params_json = table_dat[row, 'parameters_json'].val

            if not params_json or params_json == '{}':
                continue

            try:
                params_metadata = json.loads(params_json)
                for param_name in params_metadata.keys():
                    expected_name = re.sub(r'[^a-zA-Z0-9]', '', f'{processor_type}_{param_name}').lower()
                    if clean_name == expected_name:
                        osc_address = f'/fx/{processor_type}/{param_name}'
                        param_value = par.eval()
                        osc_out = self.ownerComp.op('oscout1')
                        if osc_out:
                            osc_out.sendOSC(osc_address, [param_value])
                            self.logger.log(f"Fx OSC: {osc_address} = {param_value}", level='DEBUG')
                        return
            except Exception as e:
                self.logger.log(f"Fxparameterupdate error: {e}", level='ERROR')

    def get_preprocessor_for_controlnet(self, model_id, backend, auto=True, manual_value=None):
        """
        Get preprocessor for ControlNet model with auto-detection or manual override.

        Args:
            model_id: ControlNet model ID (e.g., "xinsir/controlnet-depth-sdxl-1.0")
            backend: 'Daydream' or 'Local'
            auto: If True, auto-detect from model_id. If False, use manual_value
            manual_value: Manual preprocessor selection (used if auto=False)

        Returns:
            preprocessor name string
        """
        if not auto and manual_value:
            return manual_value

        model_id_lower = model_id.lower()

        if "openpose" in model_id_lower or "pose" in model_id_lower:
            return "pose_tensorrt"
        elif "canny" in model_id_lower:
            return "canny"
        elif "depth" in model_id_lower:
            return "depth_tensorrt"
        elif "hed" in model_id_lower:
            return "hed"
        elif "scribble" in model_id_lower:
            return "scribble"
        elif "soft" in model_id_lower or "softedge" in model_id_lower:
            return "soft_edge"
        elif "lineart" in model_id_lower:
            return "standard_lineart"
        elif "tile" in model_id_lower:
            return "feedback"
        elif "color" in model_id_lower:
            return "passthrough"
        elif "normal" in model_id_lower:
            return "normal_bae"
        else:
            return "depth_tensorrt"

    def Getpreprocessors(self):
        """
        Get all available preprocessors by parsing __init__.py.
        Outputs to table_preprocessors DAT with columns:
        name, is_tensorrt, display_name, description, parameters_json, use_cases
        """
        try:
            import os
            import re
            import json

            base_folder = self.ownerComp.par.Basefolder.eval()
            if not base_folder:
                self.logger.log('ERROR: Basefolder parameter is not set', level='ERROR')
                return

            init_file = os.path.join(base_folder, 'src', 'streamdiffusion', 'preprocessing', 'processors', '__init__.py')

            if not os.path.exists(init_file):
                self.logger.log(f'ERROR: __init__.py not found at {init_file}', level='ERROR')
                return

            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()

            registry_pattern = r'_preprocessor_registry\s*=\s*\{([^}]+)\}'
            match = re.search(registry_pattern, content, re.DOTALL)

            if not match:
                self.logger.log('ERROR: Could not find _preprocessor_registry', level='ERROR')
                return

            registry_content = match.group(1)
            name_pattern = r'"([^"]+)":\s*\w+'
            processors = re.findall(name_pattern, registry_content)

            if 'DEPTH_TENSORRT_AVAILABLE' in content:
                processors.append('depth_tensorrt')
            if 'POSE_TENSORRT_AVAILABLE' in content:
                processors.append('pose_tensorrt')
            if 'TEMPORAL_NET_TENSORRT_AVAILABLE' in content:
                processors.append('temporal_net_tensorrt')
            if 'MEDIAPIPE_POSE_AVAILABLE' in content:
                processors.append('mediapipe_pose')
            if 'MEDIAPIPE_SEGMENTATION_AVAILABLE' in content:
                processors.append('mediapipe_segmentation')

            processors_dir = os.path.join(base_folder, 'src', 'streamdiffusion', 'preprocessing', 'processors')

            table_dat = self.ownerComp.op('table_preprocessors')
            if not table_dat:
                self.logger.log('ERROR: table_preprocessors DAT not found', level='ERROR')
                return

            table_dat.clear()
            table_dat.appendRow(['name', 'is_tensorrt', 'display_name', 'description', 'parameters_json', 'use_cases'])

            for name in sorted(set(processors)):
                is_trt = 'tensorrt' in name.lower() or 'trt' in name.lower()

                # Try to extract metadata by parsing the Python file directly
                display_name = name
                description = ''
                parameters_json = '{}'
                use_cases = ''

                # Try to find and parse the preprocessor file
                preprocessor_file = os.path.join(processors_dir, f'{name}.py')
                if os.path.exists(preprocessor_file):
                    try:
                        metadata = self._extract_metadata_from_file(preprocessor_file)
                        if metadata:
                            display_name = metadata.get('display_name', name)
                            description = metadata.get('description', '')

                            # Extract parameters with full metadata
                            params_dict = metadata.get('parameters', {})
                            if params_dict:
                                parameters_json = json.dumps(params_dict, indent=2)

                            # Extract use cases as comma-separated string
                            use_cases_list = metadata.get('use_cases', [])
                            if use_cases_list:
                                use_cases = ', '.join(use_cases_list)

                    except Exception as meta_error:
                        self.logger.log(f'WARNING: Could not extract metadata for {name}: {meta_error}', level='WARNING')

                table_dat.appendRow([
                    name,
                    '1' if is_trt else '0',
                    display_name,
                    description,
                    parameters_json,
                    use_cases
                ])

            self.logger.log(f'Loaded {len(processors)} preprocessors with metadata into table_preprocessors', level='INFO')

        except Exception as e:
            self.logger.log(f'ERROR getting preprocessors: {e}', level='ERROR')
            import traceback
            traceback.print_exc()

    def _extract_metadata_from_file(self, filepath):
        """
        Extract get_preprocessor_metadata() return value from a Python file
        without importing it (avoids dependency issues).
        """
        import ast

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()

            # Parse the file as AST
            tree = ast.parse(file_content)

            # Find the class definition
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Look for get_preprocessor_metadata method
                    for item in node.body:
                        if (isinstance(item, ast.FunctionDef) and
                            item.name == 'get_preprocessor_metadata'):
                            # Find the return statement
                            for stmt in ast.walk(item):
                                if isinstance(stmt, ast.Return) and stmt.value:
                                    # Extract the return value (should be a dict)
                                    metadata_str = ast.get_source_segment(file_content, stmt.value)
                                    if metadata_str:
                                        # Safely evaluate the dictionary literal
                                        try:
                                            metadata = ast.literal_eval(metadata_str)
                                            return metadata
                                        except:
                                            # If literal_eval fails, try regex extraction
                                            return self._extract_metadata_with_regex(file_content)

            return None

        except Exception as e:
            self.logger.log(f'WARNING: Failed to parse {filepath}: {e}', level='WARNING')
            return None

    def _extract_metadata_with_regex(self, file_content):
        """
        Fallback method to extract metadata using regex pattern matching.
        """
        import re
        import json

        try:
            # Find get_preprocessor_metadata method
            pattern = r'def get_preprocessor_metadata\(.*?\):\s*return\s*(\{.*?\n\s*\})'
            match = re.search(pattern, file_content, re.DOTALL)

            if not match:
                return None

            metadata_str = match.group(1)

            # Try to convert Python dict syntax to JSON
            # Replace single quotes with double quotes
            metadata_str = re.sub(r"'([^']*)'", r'"\1"', metadata_str)
            # Handle Python True/False/None
            metadata_str = metadata_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')

            # Parse as JSON
            metadata = json.loads(metadata_str)
            return metadata

        except Exception as e:
            self.logger.log(f'WARNING: Regex metadata extraction failed: {e}', level='WARNING')
            return None

    # ============================================================================
    # DYNAMIC PARAMETER SYSTEM - Dyn* parameters for preprocessors
    # ============================================================================

    def update_cn_dynamic_parameters(self):
        """
        Updates dynamic Dyn* parameters for all enabled ControlNet blocks.
        Called when Autoupdatecnpars is True and CN selection changes.
        Follows Hydra/Strudel dynamic parameter pattern.
        """
        if not hasattr(self.ownerComp.par, 'Autoupdatecnpars') or not self.ownerComp.par.Autoupdatecnpars.eval():
            return

        # Collect all unique preprocessors from ALL CN blocks (enabled or disabled)
        # The Preprocessor parameter is the source of truth
        active_preprocessors = set()

        # Filter out latent-domain processors (they belong on Fx page, not ControlNet page)
        latent_only_processors = ['latent_feedback', 'latent_transform']

        for block in self.ownerComp.par.Cn.sequence:
            # Get preprocessor value directly from the parameter (source of truth)
            if hasattr(block.par, 'Preprocessor'):
                preprocessor_name = block.par.Preprocessor.eval()
                if preprocessor_name and preprocessor_name.strip():
                    # Skip latent-domain processors - they're for Fx page only
                    if preprocessor_name not in latent_only_processors:
                        active_preprocessors.add(preprocessor_name)

        # Remove old Dyn* parameters first
        self._remove_all_dynamic_cn_parameters()

        # Create new Dyn* parameters for all active preprocessors
        is_very_first = True
        for preprocessor_name in active_preprocessors:
            self._create_dynamic_parameters_for_preprocessor(preprocessor_name, is_first_preprocessor=is_very_first)
            is_very_first = False

    def _remove_all_dynamic_cn_parameters(self):
        """
        Remove all Dyn* parameters (ControlNet preprocessor dynamic parameters).
        IMPORTANT: Saves state of all Dyn* parameters before removal!
        """
        params_to_remove = []

        # Find all Dyn* parameters
        for par_tuple in self.ownerComp.customPars:
            # customPars returns tuples of parameters
            par = par_tuple[0] if isinstance(par_tuple, tuple) else par_tuple
            if par.name.startswith('Dyn'):
                params_to_remove.append(par)

        # Save state of all parameters BEFORE removing them
        for par in params_to_remove:
            self.save_dyn_param_state(par.name)

        # Remove them
        for par in params_to_remove:
            try:
                par.destroy()
            except Exception as e:
                self.logger.log(f'WARNING: Failed to remove parameter {par.name}: {e}', level='WARNING')

    def _create_dynamic_parameters_for_preprocessor(self, preprocessor_name, is_first_preprocessor=False):
        """
        Create Dyn* parameters for a specific preprocessor based on its metadata.
        """
        import json

        # Get metadata from table_preprocessors
        table_dat = self.ownerComp.op('table_preprocessors')
        if not table_dat:
            return

        # Find the preprocessor row
        preprocessor_row = None
        for row in range(1, table_dat.numRows):  # Skip header
            if table_dat[row, 'name'].val == preprocessor_name:
                preprocessor_row = row
                break

        if preprocessor_row is None:
            self.logger.log(f'WARNING: Preprocessor {preprocessor_name} not found in table_preprocessors', level='WARNING')
            return

        # Get parameters_json
        parameters_json = table_dat[preprocessor_row, 'parameters_json'].val
        if not parameters_json or parameters_json == '{}':
            # No parameters for this preprocessor
            return

        try:
            params_dict = json.loads(parameters_json)
        except Exception as e:
            self.logger.log(f'ERROR: Failed to parse parameters_json for {preprocessor_name}: {e}', level='ERROR')
            return

        # Create Dyn* parameters for each parameter in the metadata
        # Only add section separator if this is NOT the very first parameter overall
        is_first_param = True
        for param_name, param_metadata in params_dict.items():
            add_section = is_first_param and not is_first_preprocessor
            self._create_single_dynamic_parameter(preprocessor_name, param_name, param_metadata, section=add_section)
            is_first_param = False

    def _create_single_dynamic_parameter(self, preprocessor_name, param_name, param_metadata, section=False, page='ControlNet', prefix='Dyn', order=None):
        """
        Create a single Dyn* parameter using the existing create_parameter() method.
        Format: Dyn{preprocessor}_{paramname} (e.g., Dyncannylowthreshold)
        """
        param_type = param_metadata.get('type', 'float')
        default = param_metadata.get('default', 0)
        param_range = param_metadata.get('range', None)
        description = param_metadata.get('description', '')
        options = param_metadata.get('options', None)

        # Handle string type: if it has options, use menu; otherwise use str
        if param_type == 'string':
            if options and len(options) > 0:
                param_type = 'menu'
                default = options  # For menu, default is the list of options
            else:
                param_type = 'str'

        # Create normalized parameter name (TD convention: Firstlowercase)
        # TouchDesigner STRICT rule: First letter uppercase, REST ALL LOWERCASE (no mixed case!)
        # Format: dyn_canny_low_threshold -> Dyncannylowthreshold
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', f'{preprocessor_name}_{param_name}')
        # First letter uppercase, rest all lowercase
        normalized_name = prefix + clean_name.lower()

        # Use specified page
        page_name = page

        # Get min/max from range if available
        norm_min = param_range[0] if param_range and len(param_range) > 0 else None
        norm_max = param_range[1] if param_range and len(param_range) > 1 else None

        # For Fx transform parameters, disable clamping so users can type values outside slider range
        # Transform params: zoom, pan_x, pan_y, rotation (not noise_strength, noise_seed_mix, feedback_blend)
        transform_params = ['zoom', 'pan_x', 'pan_y', 'rotation', 'noise_strength']
        should_clamp = param_name not in transform_params

        # Create parameter using the existing create_parameter method
        try:
            new_par = self.create_parameter(
                par_name=normalized_name,
                par_type=param_type,
                page=page_name,
                default=default,
                norm_min=norm_min,
                norm_max=norm_max,
                label=param_name,
                replace=True,
                section=section,
                clamp=should_clamp,
                order=order
            )

            # Set help text if available
            if description and new_par:
                new_par.help = description

            # Try to restore saved state if it exists
            saved_state = self.restore_dyn_param_state(normalized_name)
            if saved_state:
                self.apply_dyn_param_state(normalized_name, saved_state)
                # self.logger.log(f'Restored saved state for {normalized_name}', level='INFO')

        except Exception as e:
            self.logger.log(f'ERROR: Failed to create parameter {normalized_name}: {e}', level='ERROR')

    def gather_dynamic_cn_parameters_for_preprocessor(self, preprocessor_name):
        """
        Gather current values of Dyn* parameters for a specific preprocessor.
        Returns dict of {original_param_name: current_value}
        """
        import re

        params_dict = {}

        # Get metadata to map parameter names correctly
        table_dat = self.ownerComp.op('table_preprocessors')
        if not table_dat:
            return params_dict

        # Find preprocessor metadata
        import json
        preprocessor_row = None
        for row in range(1, table_dat.numRows):
            if table_dat[row, 'name'].val == preprocessor_name:
                preprocessor_row = row
                break

        if not preprocessor_row:
            return params_dict

        # Get parameter names from metadata
        parameters_json = table_dat[preprocessor_row, 'parameters_json'].val
        if not parameters_json or parameters_json == '{}':
            return params_dict

        try:
            params_metadata = json.loads(parameters_json)
        except:
            return params_dict

        # For each parameter in metadata, find its Dyn* parameter and get value
        for param_name in params_metadata.keys():
            # Reconstruct the normalized name the same way it was created
            clean_name = re.sub(r'[^a-zA-Z0-9]', '', f'{preprocessor_name}_{param_name}')
            normalized_name = 'Dyn' + clean_name.lower()

            # Get the parameter value
            if hasattr(self.ownerComp.par, normalized_name):
                par = getattr(self.ownerComp.par, normalized_name)
                params_dict[param_name] = par.eval()

        return params_dict

    # ============================================================================
    # Fx DYNAMIC PARAMETERS - Latent-domain processors on Fx page
    # ============================================================================

    def gather_fx_parameters_for_processor(self, preprocessor_name):
        """
        Gather current values of Fx* parameters for a specific Fx processor.
        Returns dict of {original_param_name: current_value}
        """
        import re
        import json

        params_dict = {}

        table_dat = self.ownerComp.op('table_preprocessors')
        if not table_dat:
            return params_dict

        # Find preprocessor metadata
        for row in range(1, table_dat.numRows):
            if table_dat[row, 'name'].val == preprocessor_name:
                parameters_json = table_dat[row, 'parameters_json'].val
                if not parameters_json or parameters_json == '{}':
                    return params_dict

                try:
                    params_metadata = json.loads(parameters_json)
                except:
                    return params_dict

                # For each parameter in metadata, find its Fx* parameter and get value
                for param_name in params_metadata.keys():
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '', f'{preprocessor_name}_{param_name}')
                    normalized_name = 'Fx' + clean_name.lower()

                    if hasattr(self.ownerComp.par, normalized_name):
                        par = getattr(self.ownerComp.par, normalized_name)
                        params_dict[param_name] = par.eval()

                return params_dict

        return params_dict

    def update_fx_dynamic_parameters(self):
        if not self.ownerComp.par.Updatefxpars.eval():
            return
        """Updates Fx* parameters for feedback_transform (image), latent_feedback, latent_transform, and color_correction_feedback on Fx page"""
        active_fx = []
        if hasattr(self.ownerComp.par, 'Useprefxfeedback') and self.ownerComp.par.Useprefxfeedback.eval():
            active_fx.append('feedback_transform')
        if hasattr(self.ownerComp.par, 'Uselatentfeedback') and self.ownerComp.par.Uselatentfeedback.eval():
            active_fx.append('latent_feedback')
        if hasattr(self.ownerComp.par, 'Uselatenttransform') and self.ownerComp.par.Uselatenttransform.eval():
            active_fx.append('latent_transform')
        if hasattr(self.ownerComp.par, 'Usecolorcorrectionfeedback') and self.ownerComp.par.Usecolorcorrectionfeedback.eval():
            active_fx.append('color_correction_feedback')

        # Remove old Fx* params
        for par_tuple in self.ownerComp.customPars:
            par = par_tuple[0] if isinstance(par_tuple, tuple) else par_tuple
            if par.name.startswith('Fx') and not par.name.startswith('Fxuse'):
                self.save_dyn_param_state(par.name)
                try:
                    par.destroy()
                except:
                    pass

        # Create Fx* params on Fx page with Fx prefix
        table_dat = self.ownerComp.op('table_preprocessors')
        if not table_dat:
            return

        # Calculate base order to ensure Fx params appear at END of Fx page
        # Get the highest order from existing Fx page parameters (non-Fx* params)
        fx_page = next((p for p in self.ownerComp.customPages if p.name == 'Fx'), None)
        if fx_page:
            existing_orders = []
            for par_tuple in fx_page.pars:
                par = par_tuple[0] if isinstance(par_tuple, tuple) else par_tuple
                # Skip Fx* params (we're about to recreate those)
                if not par.name.startswith('Fx') or par.name.startswith('Fxuse'):
                    if hasattr(par, 'order') and par.order is not None:
                        existing_orders.append(par.order)

            # Start Fx params after all existing non-Fx params
            base_order = max(existing_orders) + 1000 if existing_orders else 10000
        else:
            base_order = 10000

        is_first_processor = True
        current_order = base_order

        for proc_name in active_fx:
            for row in range(1, table_dat.numRows):
                if table_dat[row, 'name'].val == proc_name:
                    import json
                    params_json = table_dat[row, 'parameters_json'].val
                    if params_json and params_json != '{}':
                        try:
                            params_dict = json.loads(params_json)
                            is_first_param = True
                            for param_name, param_metadata in params_dict.items():
                                add_section = is_first_param
                                self._create_single_dynamic_parameter(
                                    proc_name, param_name, param_metadata,
                                    section=add_section, page='Fx', prefix='Fx',
                                    order=current_order
                                )
                                is_first_param = False
                                current_order += 1  # Increment order for each param
                            is_first_processor = False
                        except:
                            pass
                    break

    # ============================================================================
    # DYNAMIC PARAMETER STATE MANAGEMENT - Save/Restore last_val and last_mode
    # ============================================================================

    def _ensure_dyn_param_state_table(self):
        """
        Ensure table_dyn_param_states exists with proper columns.
        Columns: param_name, last_val, last_mode, last_expr
        """
        state_table = self.ownerComp.op('table_dyn_param_states')

        if not state_table:
            self.logger.log('WARNING: table_dyn_param_states not found, cannot save parameter states', level='WARNING')
            return None

        # Check if table has correct headers, if not initialize
        if state_table.numRows == 0 or state_table[0, 0].val != 'param_name':
            state_table.clear()
            state_table.appendRow(['param_name', 'last_val', 'last_mode', 'last_expr'])

        return state_table

    def save_dyn_param_state(self, param_name):
        """
        Save the current value, mode, and expression of a Dyn* parameter.
        Called before switching parameter modes to preserve the previous state.

        Args:
            param_name: Name of the parameter (e.g., 'Dyncannylowthreshold')
        """
        state_table = self._ensure_dyn_param_state_table()
        if not state_table:
            return

        # Get the parameter
        if not hasattr(self.ownerComp.par, param_name):
            return

        par = getattr(self.ownerComp.par, param_name)

        # Get current state
        current_val = par.eval()
        current_mode = str(par.mode)  # 'ParMode.CONSTANT', 'ParMode.EXPRESSION', 'ParMode.BIND'
        current_expr = par.expr if par.mode == ParMode.EXPRESSION else ''

        # Find existing row or create new one
        row_index = None
        for row in range(1, state_table.numRows):
            if state_table[row, 'param_name'].val == param_name:
                row_index = row
                break

        if row_index is None:
            # Create new row
            state_table.appendRow([param_name, str(current_val), current_mode, current_expr])
        else:
            # Update existing row
            state_table[row_index, 'last_val'] = str(current_val)
            state_table[row_index, 'last_mode'] = current_mode
            state_table[row_index, 'last_expr'] = current_expr

        # self.logger.log(f'Saved state for {param_name}: val={current_val}, mode={current_mode}', level='DEBUG')

    def restore_dyn_param_state(self, param_name):
        """
        Restore the last saved value, mode, and expression of a Dyn* parameter.
        Call this when switching back to a previous mode.

        Args:
            param_name: Name of the parameter (e.g., 'Dyncannylowthreshold')

        Returns:
            dict with 'val', 'mode', 'expr' or None if no saved state exists
        """
        state_table = self._ensure_dyn_param_state_table()
        if not state_table:
            return None

        # Find saved state
        for row in range(1, state_table.numRows):
            if state_table[row, 'param_name'].val == param_name:
                saved_state = {
                    'val': state_table[row, 'last_val'].val,
                    'mode': state_table[row, 'last_mode'].val,
                    'expr': state_table[row, 'last_expr'].val
                }
                # self.logger.log(f'Restored state for {param_name}: {saved_state}', level='DEBUG')
                return saved_state

        return None

    def apply_dyn_param_state(self, param_name, state):
        """
        Apply a saved state to a Dyn* parameter.

        Args:
            param_name: Name of the parameter
            state: dict with 'val', 'mode', 'expr' from restore_dyn_param_state()
        """
        if not state or not hasattr(self.ownerComp.par, param_name):
            return

        par = getattr(self.ownerComp.par, param_name)

        # Apply based on mode
        mode_str = state['mode']

        if 'EXPRESSION' in mode_str and state['expr']:
            par.mode = ParMode.EXPRESSION
            par.expr = state['expr']
        elif 'BIND' in mode_str:
            par.mode = ParMode.BIND
            # Note: bind expressions need to be set appropriately
            if state['expr']:
                par.bindExpr = state['expr']
        else:  # CONSTANT or default
            par.mode = ParMode.CONSTANT
            # Try to set value with proper type conversion
            try:
                if par.isFloat:
                    par.val = float(state['val'])
                elif par.isInt:
                    par.val = int(float(state['val']))
                elif par.isToggle:
                    par.val = bool(int(float(state['val'])))
                else:
                    par.val = state['val']
            except:
                self.logger.log(f'WARNING: Could not restore value for {param_name}', level='WARNING')

        self.logger.log(f'Applied state to {param_name}', level='DEBUG')

    def save_all_dyn_param_states(self):
        """
        Save state for ALL Dyn* parameters currently on the component.
        Useful to call before regenerating parameters.
        """
        for par_tuple in self.ownerComp.customPars:
            par = par_tuple[0] if isinstance(par_tuple, tuple) else par_tuple
            if par.name.startswith('Dyn'):
                self.save_dyn_param_state(par.name)

    # ============================================================================
    # ASYNC HTTP HELPERS - Methods using TDAsyncIO for non-blocking API calls
    # ============================================================================

    async def async_http_patch(self, url, payload, headers, timeout=10):
        """Async PATCH request using requests library"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.patch(url, json=payload, headers=headers, timeout=timeout)
        )
        return response

    async def async_http_post(self, url, payload, headers, timeout=30):
        """Async POST request using requests library"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, json=payload, headers=headers, timeout=timeout)
        )
        return response

    async def async_http_get(self, url, headers, timeout=10):
        """Async GET request using requests library"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(url, headers=headers, timeout=timeout)
        )
        return response

    async def async_upload_ipadapter_image(self, numpy_array, stream_id, api_key, timeout=30):
        """
        Async upload of IP adapter image to Daydream API.
        Converts numpy array to base64 JPEG and uploads.

        IMPORTANT: numpy_array must be extracted on main thread before calling this!

        Args:
            numpy_array: numpy array from TOP (extracted on main thread)
            stream_id: Daydream stream ID to update
            api_key: Daydream API key for authorization
            timeout: Request timeout in seconds

        Returns:
            response object from requests
        """
        loop = asyncio.get_event_loop()

        def _process_and_upload():
            """Synchronous function to run in executor - NO TD OBJECTS"""
            import cv2

            # numpy_array is already extracted on main thread - safe to use
            image = numpy_array

            # Convert float [0-1] to uint8 [0-255] if needed
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Flip and convert color space (TD uses BGR, need RGB)
            image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0)

            # Encode as JPEG
            success, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                raise ValueError("Failed to encode image as JPEG")

            # Convert to base64 data URI
            data_uri = f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('utf-8')}"

            # Upload to Daydream API
            url = f"https://api.daydream.live/v1/streams/{stream_id}"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                "params": {
                    "ip_adapter_style_image_url": data_uri
                }
            }

            response = requests.patch(url, json=payload, headers=headers, timeout=timeout)
            return response

        # Run the blocking operations in executor
        response = await loop.run_in_executor(None, _process_and_upload)
        return response

    def upload_ipadapter_image(self):
        """
        Uploads IP adapter image as base64 to Daydream API.
        Triggered when user pulses the Ipadapterupdate parameter.
        Calls async_upload_ipadapter_image() with extracted numpy array.
        """
        backend = self.ownerComp.par.Backend.eval()
        if backend != 'Daydream':
            self.logger.log('IP Adapter upload only supported for Daydream backend', level='WARNING')
            return

        # Check if stream is active
        if not self.ownerComp.par.Streamactive:
            self.logger.log('Cannot upload IP adapter image: stream not active', level='WARNING')
            return

        # Get stream_id from status table
        status_table = self.ownerComp.op('daydream_web_status')
        if not status_table:
            self.logger.log('Cannot find daydream_web_status table', level='ERROR')
            return

        stream_id = status_table['stream_id', 1].val
        if not stream_id:
            self.logger.log('No stream_id available', level='ERROR')
            return

        # Get API key
        api_key = self._load_daydream_key()
        if not api_key:
            return

        # Get image TOP
        image_top = self.ownerComp.par.Ipadapterimage.eval()
        if not image_top:
            self.logger.log('Ipadapterimage parameter is empty', level='WARNING')
            return

        # Extract numpy array on main thread (required before async call)
        numpy_array = image_top.numpyArray(delayed=False)

        # Call async upload function via TDAsyncIO
        asyncio_op = self.ownerComp.op('TDAsyncIO')
        if asyncio_op:
            asyncio_op.ext.AsyncIOManager.Run(
                self.async_upload_ipadapter_image(numpy_array, stream_id, api_key),
                description="Upload IPAdapter image"
            )
        else:
            self.logger.log('TDAsyncIO not found', level='ERROR')

    def handle_whip_proxy(self, whip_url, sdp_offer, api_key):
        """
        Handle WHIP SDP forwarding (synchronous - required for WebRTC).
        Called from web callbacks to forward browser SDP to Daydream WHIP endpoint.

        MUST be synchronous because WebRTC negotiation requires immediate SDP answer.

        Args:
            whip_url: Daydream WHIP endpoint URL
            sdp_offer: SDP offer string from browser
            api_key: Daydream API key

        Returns:
            dict with status_code, reason, text (SDP answer)
        """
        try:
            import requests
            response = requests.post(
                whip_url,
                headers={
                    'Content-Type': 'application/sdp',
                    'Authorization': f'Bearer {api_key}'
                },
                data=sdp_offer,
                timeout=10
            )

            return {
                'status_code': response.status_code,
                'reason': response.reason,
                'text': response.text
            }

        except Exception as e:
            self.logger.log(f'WHIP proxy error: {e}', level='ERROR')
            return {
                'status_code': 500,
                'reason': 'Internal Error',
                'text': f'WHIP proxy error: {str(e)}'
            }

    def _on_ipadapter_upload_complete(self, task):
        """
        Callback when IP adapter image upload completes.

        Args:
            task: AsyncIOTask object from TDAsyncIO
        """
        try:
            # Check for error (not exception!)
            if task.error:
                self.logger.log(f'IP adapter image upload failed: {str(task.error)}', level='ERROR')
                return

            # Get response from result
            response = task.result
            if response and hasattr(response, 'status_code'):
                if response.status_code == 200:
                    self.logger.log(f'✓ IP adapter image uploaded successfully (HTTP {response.status_code})', level='INFO')
                else:
                    self.logger.log(f'✗ IP adapter image upload returned HTTP {response.status_code}', level='WARNING')
                    if hasattr(response, 'text'):
                        self.logger.log(f'Response: {response.text}', level='DEBUG')
            else:
                self.logger.log(f'✓ IP adapter upload completed (no response info)', level='INFO')

        except Exception as e:
            self.logger.log(f'Error in IP adapter upload callback: {str(e)}', level='ERROR')
            import traceback
            self.logger.log(traceback.format_exc(), level='ERROR')

    def _on_patch_complete(self, task):
        """
        Callback when parameter PATCH request completes.

        Args:
            task: AsyncIOTask object from TDAsyncIO
        """
        try:
            # Check for error (not exception!)
            if task.error:
                self.logger.log(f'✗ PATCH params failed: {str(task.error)}', level='ERROR')
                return

            # Get response from result
            response = task.result

            # DEBUG: Log what we actually got
            if response is None:
                self.logger.log(f'⚠ PATCH params: response is None!', level='ERROR')
            elif not hasattr(response, 'status_code'):
                self.logger.log(f'⚠ PATCH params: response has no status_code! Type: {type(response)}', level='ERROR')
            elif response.status_code == 200:
                self.logger.log(f'✓ PATCH params HTTP {response.status_code}', level='DEBUG')
            else:
                self.logger.log(f'✗ PATCH params HTTP {response.status_code}', level='WARNING')
                if hasattr(response, 'text'):
                    self.logger.log(f'Response: {response.text}', level='DEBUG')

        except Exception as e:
            self.logger.log(f'Error in PATCH callback: {str(e)}', level='ERROR')
            import traceback
            self.logger.log(traceback.format_exc(), level='ERROR')






    # def update_stream_config_dat_LEGACY(self):
    #     """LEGACY VERSION - keeping for reference but not used"""
    #     stream_config_dat = self.ownerComp.op('streamdiffusionTD/stream_config')
    #     try:
    #         config = json.loads(stream_config_dat.text)
    #     except json.JSONDecodeError as e:
    #         self.logger.log(f"Error decoding JSON: {e}", level='ERROR')
    #         return
    #     keys_to_remove = []
    #     for param_name, json_key in param_json_map.items():
    #         if param_name == 'Modelid':
    #             if self.ownerComp.par.Modelid.eval() == '':
    #                 self.logger.log(f"Modelid (Settings 1) is empty", level='ERROR')
    #                 return
    #             model_id = self.ownerComp.par.Modelid.eval()
    #             if '/' in model_id and model_id.count('/') == 1:
    #                 config[json_key] = model_id
    #             else:
    #                 config[json_key] = tdu.expandPath(model_id)
    #             continue

    #         if param_name == "Promptdict0concept":
    #             max_weight = 0
    #             max_weight_concept = ""
    #             for block in self.ownerComp.par.Promptdict.sequence:
    #                 concept = block.par.Concept.eval()
    #                 weight = block.par.Weight.eval()
    #                 if weight > max_weight:
    #                     max_weight = weight
    #                     max_weight_concept = concept
    #             config[json_key] = max_weight_concept
    #             continue
    #         if param_name == "Customlcm":
    #             model_id = self.ownerComp.par.Modelid.eval().lower()
    #             config_type = self.get_config_type(model_id)
    #             if self.ownerComp.par.Skiplcm or config_type == "sd21":
    #                 #mark as skip
    #                 config[json_key] = "skip"
    #                 continue
    #             if not self.ownerComp.par.Usecustomlcm:
    #                 keys_to_remove.append(json_key)
    #                 continue
    #         if param_name == "Customvae":
    #         # if param_name == "Customvae" and not self.ownerComp.par.Usecustomvae:
    #             keys_to_remove.append(json_key)
    #             continue
    #         if json_key == "t_index_list":
    #             t_index_list = []
    #             # Iterate over each block in the Tindexblock sequence
    #             for block in self.ownerComp.par.Tindexblock.sequence:
    #                 step_value = block.par.Step.eval()  # Extract the Step value
    #                 t_index_list.append(step_value)     # Append the step value to the list
    #             config[json_key] = t_index_list
    #         else:
    #             param_value = getattr(self.ownerComp.par, param_name).eval()
    #             config[json_key] = param_value
    #         #round widht and height to 8 multiplier    
    #         # Correct the key names in the print statements
    #         if param_name == "Width":
    #             config[json_key] = round(param_value / 8) * 8
    #             # print(f"Width: {config['width']}")  # Use 'width' instead of 'Width'
    #             if self.ownerComp.par.Width.mode == ParMode.CONSTANT:
    #                 self.ownerComp.par.Width = config['width']
    #         if param_name == "Height":
    #             config[json_key] = round(param_value / 8) * 8
    #             # print(f"Height: {config['height']}")  # Use 'height' instead of 'Height'
    #             if self.ownerComp.par.Height.mode == ParMode.CONSTANT:
    #                 self.ownerComp.par.Height = config['height']
    #         if param_name == "Usecontrolnet":
    #             config[json_key] = self.ownerComp.par.Usecontrolnet.eval()
    #             self.ownerComp.op('streamdiffusionTD').par.Cnactive = self.ownerComp.par.Usecontrolnet.eval()

    #         if param_name == "Forcemodeltype":
    #             if self.ownerComp.par.Forcemodeltype.eval() == 'auto':
    #                 config[json_key] = None
    #                 keys_to_remove.append(json_key)
    #             else: 
    #                 config[json_key] = self.ownerComp.par.Forcemodeltype.eval()

    #         if param_name == "Hfcache":
    #             if not self.ownerComp.par.Sethfcache.eval() or self.ownerComp.par.Hfcache.eval() == '':
    #                 # print(f"Removing {json_key}")
    #                 if self.ownerComp.par.Hfcache.eval() == '':
    #                     print(f"Hfcache is empty")
    #                 keys_to_remove.append(json_key)
    #                 continue
    #             config[json_key] = tdu.expandPath(self.ownerComp.par.Hfcache.eval())
    #     # REMOVED: Old TensorRT blocking for ControlNet/IPAdapter
    #     # Modern StreamDiffusion fork supports TensorRT with both ControlNet and IPAdapter
    #     if self.ownerComp.par.Acceleration == 'tensorrt':
    #         # Only disable Fx for TensorRT (still incompatible)
    #         if self.ownerComp.par.Enablefx.mode == ParMode.CONSTANT:
    #             self.ownerComp.par.Enablefx = False
    #     for key in keys_to_remove:
    #         if key in config:
    #             del config[key]
    #     if self.ownerComp.par.Uselora:
    #         lora_dict = {}
    #         for block in self.ownerComp.par.Loradictblock.sequence:
    #             lora_path = block.par.Lorapath.eval()
    #             weight = block.par.Weight.eval()
    #             if lora_path == 'select_lora_from_dropdown':
    #                 pass
    #             if lora_path and weight != 0:  # Check if lora_path is not None or an empty string and weight is not 0
    #                 lora_dict[lora_path] = weight
    #     else:
    #         lora_dict = None
    #     config["lora_dict"] = lora_dict

    #     # Add this before the final stream_config_dat.text = json.dumps(config, indent=4)
    #     if self.ownerComp.par.Usetextualinv:
    #         ti_dict = {}
    #         for block in self.ownerComp.par.Textualinvblock.sequence:
    #             embed_path = block.par.Embedpath.eval()
    #             if embed_path and embed_path != 'select_embedding_from_dropdown':
    #                 token = block.par.Token.eval()
    #                 # Expand the path if it's local
    #                 if not embed_path.startswith('http'):
    #                     embed_path = tdu.expandPath(embed_path)
    #                 ti_dict[embed_path] = token if token else None
            
    #         if ti_dict:  # Only set if we have valid entries
    #             config["textual_inversion_dict"] = ti_dict
    #             print(f"\nTextual Inversion Config:")
    #             print(json.dumps(ti_dict, indent=2))
    #     else:
    #         config["textual_inversion_dict"] = None

    #     # ControlNet array configuration (modern format)
    #     if config.get('use_controlnet', False) and hasattr(self.ownerComp.par, 'Cn'):
    #         controlnets = []
    #         for block in self.ownerComp.par.Cn.sequence:
    #             if block.par.Enable.eval():
    #                 model_id = block.par.Id.eval()
    #                 weight = block.par.Weight.eval()
                    
    #                 # Add "thibaud/" prefix if not already present and not HuggingFace format
    #                 if not model_id.startswith(("thibaud/", "lllyasviel/", "xinsir/")):
    #                     model_id = f"thibaud/{model_id}"
                    
    #                 # Determine preprocessor based on model type
    #                 if "depth" in model_id.lower():
    #                     preprocessor = "depth"
    #                 elif "canny" in model_id.lower():
    #                     preprocessor = "canny_tensorrt"
    #                 elif "openpose" in model_id.lower():
    #                     preprocessor = "pose_tensorrt"
    #                 else:
    #                     preprocessor = "depth"  # Default
                    
    #                 controlnet_config = {
    #                     "model_id": model_id,
    #                     "conditioning_scale": weight,
    #                     "enabled": True,
    #                     "preprocessor": preprocessor,
    #                     "preprocessor_params": {},
    #                     "control_guidance_start": 0.0,
    #                     "control_guidance_end": 1.0
    #                 }
    #                 controlnets.append(controlnet_config)
            
    #         if controlnets:
    #             config["controlnets"] = controlnets
        
    #     # IPAdapter array configuration (modern format)
    #     if config.get('use_ipadapter', False):
    #         ipadapter_scale = config.get('ipadapter_scale', 0.7)
    #         ipadapters = [{
    #             "ipadapter_model_path": "h94/IP-Adapter/models/ip-adapter_sd15.safetensors",
    #             "image_encoder_path": "h94/IP-Adapter/models/image_encoder",
    #             "scale": ipadapter_scale,
    #             "enabled": True
    #         }]
    #         config["ipadapters"] = ipadapters

    #     stream_config_dat.text = json.dumps(config, indent=4)
    #     return True



    # def Cnweight(self):
    #     if self.ownerComp.par.Streamactive:
    #         backend = self.ownerComp.par.Backend.eval()
    #         osc_out = op('oscout1')
    #         controlnet_weight_value = self.ownerComp.par.Cnweight.eval()
            
    #         if backend == 'Daydream':
    #             # For Daydream, trigger the sequence block function instead
    #             self.Cnblock()
    #         else:
    #             # For local backend, send individual parameter
    #             osc_out.sendOSC('/controlnet_weight', [controlnet_weight_value])



    # def Caactive(self):
    #     if not self.ownerComp.par.Enablefx.eval():
    #         self.Enablefx()   
    #         return
    #     if self.ownerComp.par.Caactive:
    #         osc_out = op('oscout1')
    #         self.send_parameter_update('use_cached_attn', self.ownerComp.par.Caactive.eval(), 'single')
    #         if self.ownerComp.par.Streamactive: 
    #             use_cached_attn_settings = {
    #                 "use_feature_injection": self.ownerComp.par.Causefeatinj.eval(),
    #                 "feature_injection_strength": self.ownerComp.par.Cafeatinjstr.eval(),
    #                 "feature_similarity_threshold": self.ownerComp.par.Cafeatsimthr.eval(),
    #                 "cache_interval": int(self.ownerComp.par.Cacacheintvl.eval()),
    #                 "cache_maxframes": int(self.ownerComp.par.Cacachemaxfr.eval()),
    #                 "use_tome_cache": self.ownerComp.par.Causetomecache.eval(),
    #                 "tome_metric": "keys",
    #                 "tome_ratio": self.ownerComp.par.Catomeratio.eval(),
    #                 "use_grid": self.ownerComp.par.Causegrid.eval(),
    #             }
    #             self.send_parameter_update('use_cached_attn_settings', use_cached_attn_settings, 'json')
