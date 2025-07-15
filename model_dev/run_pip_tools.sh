set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
PIPELINE_PATH=`realpath "${SCRIPT_PATH}"/../`

pip install pip-tools
pip-compile "${PIPELINE_PATH}"/model_dev/requirements_tune.in
