#!/usr/bin/env bash
# Bakes 3DSceneGraph annotations onto raw Gibson meshes into the .scn/_semantic.ply pair
# habitat-sim's Gibson loader expects, via habitat-sim's `Datatool` (removed upstream after
# v0.3.2, see CLAUDE.md). Run from repo root.
#
# Get the data:
#   wget --no-check-certificate https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat_trainval.zip && unzip gibson_habitat_trainval.zip -d gibson_data   # native .glb/.navmesh -> --gibson-glb (only 492/572 scenes)
#   wget https://storage.googleapis.com/gibson_scenes/gibson_tiny.tar.gz && tar -xzf gibson_tiny.tar.gz -C gibson_data   # raw .obj -> --gibson-raw
#   # 3DSceneGraph .npz -> --scene-graph: no direct URL, sign in at https://redivis.com/datasets/1kf9-cfjvtqc7q/files and grab the "verified" split into gibson_data/verified_graph/
#
# Build Datatool:
#   git clone --branch v0.3.2 --recurse-submodules --shallow-submodules https://github.com/facebookresearch/habitat-sim.git third_party/habitat-sim
#   # bug: Datatool.cpp's main() never sets up a logging context that create_gibson_semantic_mesh needs
#   sed -i 's/int main(int argc, char\*\* argv) {/int main(int argc, char** argv) {\n  esp::logging::LoggingContext loggingContext;/' third_party/habitat-sim/src/utils/datatool/Datatool.cpp
#   # conda's mesalib/libglvnd ship no GLVND opengl/glx split -> force legacy libGL + explicit paths
#   cmake -S third_party/habitat-sim/src -B third_party/habitat-sim/build -GNinja \
#     -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_DATATOOL=ON \
#     -DBUILD_GUI_VIEWERS=OFF -DBUILD_TEST=OFF -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_AUDIO=OFF \
#     -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DOpenGL_GL_PREFERENCE=LEGACY \
#     -DOPENGL_INCLUDE_DIR=$CONDA_PREFIX/include -DOPENGL_gl_LIBRARY=$CONDA_PREFIX/lib/libGL.so.1 \
#     -DEGL_LIBRARY=$CONDA_PREFIX/lib/libEGL.so.1 -DEGL_INCLUDE_DIR=$CONDA_PREFIX/include \
#     -DCMAKE_CXX_FLAGS="-I$CONDA_PREFIX/include" -DCMAKE_C_FLAGS="-I$CONDA_PREFIX/include" \
#     -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib" \
#     -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
#   cmake --build third_party/habitat-sim/build --target Datatool -j"$(nproc)"   # -> build/Release/bin/Datatool, this script's default
#
# Run: scripts/gen_gibson_semantics.sh --out $HABITAT_DATA/scene_datasets/gibson_semantic

set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: $0 [--datatool PATH] [--habitat-sim-tools PATH] [--gibson-raw PATH] [--scene-graph PATH] [--gibson-glb PATH] --out PATH

  --datatool           habitat-sim Datatool binary (default: third_party/habitat-sim/build/Release/bin/Datatool)
  --habitat-sim-tools  habitat-sim's tools/ dir, for npz2ids.py/npz2scn.py (default: third_party/habitat-sim/tools)
  --gibson-raw         raw Gibson meshes, one <scene>/mesh.obj subdir per scene (default: gibson_data/gibson_tiny)
  --scene-graph        3DSceneGraph_<scene>.npz files (default: gibson_data/verified_graph)
  --gibson-glb         native <scene>.glb/.navmesh mirror, flat dir (default: gibson_data/gibson)
  --out                output dir -- ends up a full ready-to-use scene_dataset (.scn,
                       _semantic.ply, .glb, .navmesh, scene_dataset_config.json per scene)
EOF
    exit 1
}

datatool="third_party/habitat-sim/build/Release/bin/Datatool"
habitat_sim_tools="third_party/habitat-sim/tools"
gibson_raw="gibson_data/gibson_tiny"
scene_graph_path="gibson_data/verified_graph"
gibson_glb="gibson_data/gibson"
out_path=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datatool) datatool="$2"; shift 2 ;;
        --habitat-sim-tools) habitat_sim_tools="$2"; shift 2 ;;
        --gibson-raw) gibson_raw="$2"; shift 2 ;;
        --scene-graph) scene_graph_path="$2"; shift 2 ;;
        --gibson-glb) gibson_glb="$2"; shift 2 ;;
        --out) out_path="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -z "$out_path" ]] && usage

mkdir -p "$out_path"

# matches dl.fbaipublicfiles.com/habitat/gibson/config_v1/gibson_semantic.scene_dataset_config.json
cat > "${out_path}/gibson_semantic.scene_dataset_config.json" <<'EOF'
{
  "stages": {
    "paths": {
      ".glb": [
        "*.glb"
      ]
    },
    "default_attributes": {
      "shader_type": "flat",
      "nav_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.navmesh",
      "semantic_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%_semantic.ply",
      "semantic_descriptor_filename": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.scn",
      "up": [0, 0, 1],
      "front": [0, 1, 0],
      "semantic_up": [0, 1, 0],
      "semantic_front": [0, 0, -1],
      "origin": [0, 0, 0]
    }
  },
  "objects": {},
  "light_setups": {},
  "scene_instances": {
    "default_attributes": {
      "default_lighting": "no_lights"
    }
  }
}
EOF

for npz in "${scene_graph_path}"/3DSceneGraph_*.npz; do
    filename=$(basename "${npz}")
    scene=${filename#3DSceneGraph_}
    scene=${scene%.npz}
    echo "${scene}"

    tmp_scene_dir=$(mktemp -d)
    "${habitat_sim_tools}/npz2ids.py" "${npz}" "${tmp_scene_dir}/${scene}.ids"
    "${habitat_sim_tools}/npz2scn.py" "${npz}" "${out_path}/${scene}.scn"

    "${datatool}" create_gibson_semantic_mesh \
        "${gibson_raw}/${scene}/mesh.obj" \
        "${tmp_scene_dir}/${scene}.ids" \
        "${out_path}/${scene}_semantic.ply"

    rm -rf "${tmp_scene_dir}"

    if [[ -f "${gibson_glb}/${scene}.glb" ]]; then
        cp "${gibson_glb}/${scene}.glb" "${gibson_glb}/${scene}.navmesh" "${out_path}/"
    else
        echo "  no ${scene}.glb under ${gibson_glb} -- skipping .glb/.navmesh for this scene" >&2
    fi
done
