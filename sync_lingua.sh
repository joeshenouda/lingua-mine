set -euxo pipefail

USER="jsheno"

HOSTNAME="${USER}-sleeper-job-gy6j-worker-0"

RSYNC_HELPER="/Users/jsheno/experiments/param_norms/rsync_helper.sh"
LINGUA="/Users/jsheno/experiments/lingua"

rsync -av --exclude '.venv/*' --progress --stats -e ${RSYNC_HELPER} ${LINGUA} ${HOSTNAME}:/scratch/scratch/${USER}/pnorms_lingua --exclude=brazil/env --exclude=.idea --exclude="*.pyc" --exclude=.git --exclude=__pycache__ --exclude=.gradle --exclude=buildSrc --exclude=gradle --exclude=out --exclude=brazil-documentation --exclude=brazil_unit_tests --exclude=generated_make --exclude=venv --exclude=.pytest_cache --exclude=build_tools --exclude=brazil/build --exclude=build/private --exclude=release-info --exclude=.venv --exclude=local_data
