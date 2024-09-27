#!/bin/bash

cd ./mujoco || exit
mkdir -p .build

cd ./build || exit
cmake ..
cmake --build .

cmake .. -DCMAKE_INSTALL_PREFIX="/move/u/hshi74/projects/toddlerbot/mujoco/build/install"
cmake --install .

cd ../python || exit
python3 -m venv /tmp/mujoco
source /tmp/mujoco/bin/activate
bash make_sdist.sh
deactivate

cd dist || exit
export MUJOCO_PATH=/move/u/hshi74/projects/toddlerbot/mujoco/build
export MUJOCO_PLUGIN_PATH=/move/u/hshi74/projects/toddlerbot/mujoco/build/plugin
pip install mujoco-3.2.3.tar.gz

cd ../../mjx || exit
pip install -e .