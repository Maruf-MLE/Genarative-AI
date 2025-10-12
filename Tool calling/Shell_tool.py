from langchain_community.tools import ShellTool

Shell_Tool = ShellTool()

result = Shell_Tool.invoke('dir')
print(result)
