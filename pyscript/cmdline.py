import os

def exe(master_setting_file_path):
    SolutionDir = "C:/HHWork/ImWarping/Development/"
    BuildDir = "build/"
    Platform = "Win32"
    Configuration = "Debug"
    ProjectName = "core_lib"
    ExecutableExt = ".exe"
    execute_path = SolutionDir + BuildDir + Platform + "/" + Configuration + "/" + ProjectName + ExecutableExt
    command_line = execute_path + " " + master_setting_file_path
    os.system(command_line)

