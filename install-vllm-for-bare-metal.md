# Install-vllm-for-bare-metal
A shell script for vllm installatoin on bare-metal


###  Description
The script is used to build and install vllm and its dependencies from source code. Build and install from source code helps to achieve best performance. 

Not all dependencies are required for vllm installation, it depends on use cases. The script comments out all optional dependencies, and just leaves pytorch and vllm installation. If you want to install dependencies from this script, you can uncomment specific script block to install dependencies.

### Using the script
Go to your working directory and 
```
bash install-vllm-for-bare-metal.sh
```
Please note, for optional dependencies, the script will clone its source code into the working directory. If you do not want to keep source code for optional dependencies, you can delete them after installation.