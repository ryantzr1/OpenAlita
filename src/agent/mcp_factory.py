class MCPFactory:
    """Responsible for taking a script string (from LLM) and loading it."""
    
    def _parse_mcp_metadata_and_code(self, script_string: str):
        metadata = {
            "name": None,
            "description": "No description provided by LLM.",
            "args": "N/A",
            "returns": "N/A",
            "requires": None,
            "code_body": script_string
        }
        
        lines = script_string.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines.pop(0)
            if lines and lines[-1].strip() == "```":
                lines.pop()
        
        script_string = "\n".join(lines)
        
        header_lines = []
        code_body_lines = []
        in_header = True

        for line_num, line in enumerate(script_string.splitlines()):
            stripped_line = line.strip()
            if in_header:
                if stripped_line.startswith("#"):
                    header_lines.append(stripped_line)
                elif stripped_line and not stripped_line.startswith("#"):
                    in_header = False
                    code_body_lines.append(line)
                elif not stripped_line:
                    header_lines.append(line) 
            else:
                code_body_lines.append(line)
        
        metadata["code_body"] = "\n".join(code_body_lines).strip()

        if "from typing import str" in metadata["code_body"]:
            code_lines = metadata["code_body"].splitlines()
            cleaned_code_lines = [line for line in code_lines if line.strip() != "from typing import str"]
            metadata["code_body"] = "\n".join(cleaned_code_lines)

        for line in header_lines:
            if line.startswith("# MCP Name:"):
                metadata["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Description:"):
                metadata["description"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Arguments:"):
                metadata["args"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Returns:"):
                metadata["returns"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Requires:"):
                if metadata["requires"] is None: 
                    metadata["requires"] = []
                metadata["requires"].append(line.split(":", 1)[1].strip())
        
        if isinstance(metadata["requires"], list):
            metadata["requires"] = ", ".join(metadata["requires"])

        return metadata

    def create_mcp_from_script(self, mcp_name_expected: str, script_string: str):
        if not script_string or not script_string.strip():
            return None, {}

        metadata = self._parse_mcp_metadata_and_code(script_string)
        actual_mcp_name = metadata.get("name") or mcp_name_expected
        
        if not metadata["code_body"]:
            return None, metadata

        try:
            restricted_globals = {
                "__builtins__": {
                    "print": print, "len": len, "str": str, "int": int, "float": float, "bool": bool,
                    "list": list, "dict": dict, "tuple": tuple, "set": set,
                    "True": True, "False": False, "None": None,
                    "abs": abs, "round": round, "range": range, "zip": zip, "map": map, "filter": filter,
                    "sum": sum, "min": min, "max": max, "pow": pow, "sorted": sorted,
                    "isinstance": isinstance, "type": type, "hasattr": hasattr, "getattr": getattr,
                    "setattr": setattr, "delattr": delattr,
                    "format": format, "repr": repr, "ascii": ascii, "ord": ord, "chr": chr,
                    "enumerate": enumerate, "iter": iter, "next": next, "reversed": reversed,
                    "ValueError": ValueError, "TypeError": TypeError, "Exception": Exception,
                    "AttributeError": AttributeError, "KeyError": KeyError, "IndexError": IndexError,
                    "ZeroDivisionError": ZeroDivisionError, "ImportError": ImportError,
                    "callable": callable, "vars": vars, "dir": dir,
                    "all": all, "any": any,
                    "__import__": __import__
                }
            }
            
            requires_str = metadata.get("requires", "").lower() if metadata.get("requires") else ""
            
            if "math" in requires_str:
                import math
                restricted_globals["math"] = math
                
            if "random" in requires_str:
                import random
                restricted_globals["random"] = random
                
            if "datetime" in requires_str:
                import datetime
                restricted_globals["datetime"] = datetime
                
            if "json" in requires_str:
                import json
                restricted_globals["json"] = json
                
            if "re" in requires_str:
                import re
                restricted_globals["re"] = re
                
            if "socket" in requires_str:
                import socket
                restricted_globals["socket"] = socket
                
            if "requests" in requires_str:
                try:
                    import requests
                    restricted_globals["requests"] = requests
                except ImportError:
                    pass
            
            if "time" in requires_str:
                import time
                restricted_globals["time"] = time
                    
            if "os" in requires_str:
                import os
                class SafeOS:
                    def __init__(self):
                        self.path = os.path
                        self.environ = dict(os.environ)
                    
                    def getenv(self, key, default=None):
                        return self.environ.get(key, default)
                        
                restricted_globals["os"] = SafeOS()
            
            if "typing" in requires_str or "Optional" in script_string:
                from typing import Optional, List, Dict, Any
                restricted_globals["Optional"] = Optional
                restricted_globals["List"] = List
                restricted_globals["Dict"] = Dict
                restricted_globals["Any"] = Any

            mcp_namespace = dict(restricted_globals) 
            exec(metadata['code_body'], mcp_namespace)

            if actual_mcp_name in mcp_namespace and callable(mcp_namespace[actual_mcp_name]):
                return mcp_namespace[actual_mcp_name], metadata
            else:
                return None, metadata
        except Exception as e:
            return None, metadata