class MCPBox:
    def __init__(self):
        self.mcps = {}

    def add_mcp(self, name: str, function, description: str, args_info: str = "N/A", returns_info: str = "N/A", source: str = "pre-loaded", requires: str = None, script_content: str = None, original_command: str = None):
        if name not in self.mcps:
            self.mcps[name] = {
                "function": function,
                "description": description,
                "args_info": args_info,
                "returns_info": returns_info,
                "source": source,
                "requires": requires,
                "script_content": script_content if source == "dynamically-generated" else None,
                "original_command": original_command  # Store the original user command
            }
            print(f"   Agent Log: New MCP '{name}' ({source}) added. Description: {description}")
        else:
            print(f"   Agent Log: MCP '{name}' already exists. Not overriding.")

    def get_mcp(self, name: str):
        return self.mcps.get(name)

    def list_mcps(self):
        if not self.mcps:
            return "MCP Box is empty."
        mcp_list = "Available MCPs:\n"
        for name, data in self.mcps.items():
            mcp_list += f"- {name} (Source: {data['source']})\n"
            mcp_list += f"    Description: {data['description']}\n"
            mcp_list += f"    Arguments: {data['args_info']}\n"
            mcp_list += f"    Returns: {data['returns_info']}\n"
            if data['requires']:
                mcp_list += f"    Requires: {data['requires']}\n"
            if data.get('script_content'):
                mcp_list += f"    --- Script Start ---\n{data['script_content'][:200]}...\n    --- Script End ---\n"
        return mcp_list