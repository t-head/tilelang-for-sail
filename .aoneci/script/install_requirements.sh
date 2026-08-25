
. ./PPU_SDK/envsetup.sh

pip install pytest==9.0.0 pyyaml==6.0.3 junit-xml 
pip install einops==0.8.1 ninja==1.13.0 mkl==2021.4.0 cuda-bindings==13.0
pip install 'z3-solver>=4.13.0,<4.15.5' tabulate
yes | pip uninstall nvidia-cuda-nvrtc

# torch 2.9
mkdir torch_2_9 && cd torch_2_9
wget --quiet https://art-pub.eng.t-head.cn/artifactory/ptgai-pypi_ppu_generic/torch/2.9.0+v0.1.0.ppu2.1.0.oe/torch-2.9.0+cu130ubuntu2404oe-cp312-cp312-linux_x86_64.whl
pip install *.whl --force-reinstall --no-deps

cd - && mkdir triton_3_5 && cd triton_3_5
wget --quiet https://art-pub.eng.t-head.cn/artifactory/ptgai-pypi_ppu_generic/triton/3.5.0+v0.1.0.ppu2.1.0.oe/triton-3.5.0+cu130ubuntu2404oe-cp312-cp312-linux_x86_64.whl
pip install *.whl --force-reinstall --no-deps

cd - && mkdir flash_attn_2_8_2 && cd flash_attn_2_8_2
wget --quiet https://art-pub.eng.t-head.cn/artifactory/ptgai-pypi_ppu_generic/flash_attn/2.8.2+v0.1.0.ppu2.1.0.oe/flash_attn-2.8.2+cu130torch2.9.0ubuntu2404oe-cp312-cp312-linux_x86_64.whl
pip install *.whl --force-reinstall --no-deps
cd -

sed -i '10,13 s/^/# /' tilelang/testing/conftest.py

pip install $(find tilelang/dist -name *.whl)

apt-get update && apt-get upgrade -y
apt-get install libopenmpi-dev -y
