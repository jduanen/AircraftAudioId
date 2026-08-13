#!/usr/bin/env bash
# Sync the necessary parts of this repo to the Standalone Data Collection
# unit (CM4). Run this on the dev/recording machine, not on the CM4 itself.
#
# Usage:
#   bash scripts/syncToStandalone.sh <standalone-hostname-or-ip> [--skipFaaDb]
#
# What is synced:
#   src/aircraftAudio/       — the whole package (recorder, adsb, audioStream,
#                               dataset, standalone, capture). Synced whole
#                               rather than cherry-picked per submodule so
#                               internal imports (e.g. standaloneRecorder.py's
#                               ..dataset.faaDatabase) never break.
#   scripts/standaloneRecord.py — the main recording entry point.
#   scripts/shutdownButtonWatch.py — the shutdown-button entry point.
#   standaloneUnit/           — requirements.txt, systemd services,
#                               cm4-led-gpio.dts, cm4-sdspi.dts,
#                               cm4-shutdown-button.dts
#   data/ReleasableAircraft/  — offline FAA registry; --faaDatabaseDir is
#                               required for StandaloneRecorder. ~500 MB on
#                               first sync, incremental after. Skip with
#                               --skipFaaDb once it's already on the device.
#
# What is NOT synced:
#   src/aircraftClassifier/  — model/training code; not needed for data
#                              collection and pulls in torch/torchaudio,
#                              which aren't in standaloneUnit/requirements.txt
#   recordings/, dataset/, checkpoints/ — collected/derived data; this unit
#                              produces recordings/ locally, it doesn't
#                              receive it
#   .git/, docs/, cad/, assets/, hold/, tests/, tools/
#
# The destination path /home/jdn/Code/AircraftAudioId/ is preserved to match
# standaloneRecorder.service's ExecStart paths and sys.path.insert(0, "src").
#
# This script only pushes files — it does not create the venv (see
# audioCapture/StandaloneDataCollection.md "Installation") or restart
# standaloneRecorder.service.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/syncToStandalone.sh <standalone-hostname-or-ip> [--skipFaaDb]"
    exit 1
fi

STANDALONE_HOST="$1"
SKIP_FAA_DB=false
if [[ "${2:-}" == "--skipFaaDb" ]]; then
    SKIP_FAA_DB=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_ROOT="jdn@${STANDALONE_HOST}:/home/jdn/Code/AircraftAudioId"

OPTS="-avz --progress"

echo "Syncing to ${STANDALONE_HOST}:/home/jdn/Code/AircraftAudioId/ ..."

# Remote rsync only creates one missing destination directory level, not the
# full parent chain — on a fresh host (nothing under ~/Code/ yet) that fails
# with "mkdir ... No such file or directory". Create the full tree upfront.
ssh "jdn@${STANDALONE_HOST}" "mkdir -p \
    /home/jdn/Code/AircraftAudioId/src/aircraftAudio \
    /home/jdn/Code/AircraftAudioId/scripts \
    /home/jdn/Code/AircraftAudioId/standaloneUnit \
    /home/jdn/Code/AircraftAudioId/data/ReleasableAircraft"

rsync ${OPTS} \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "${PROJECT_ROOT}/src/aircraftAudio/" \
    "${REMOTE_ROOT}/src/aircraftAudio/"

rsync ${OPTS} \
    "${PROJECT_ROOT}/scripts/standaloneRecord.py" \
    "${REMOTE_ROOT}/scripts/standaloneRecord.py"

rsync ${OPTS} \
    "${PROJECT_ROOT}/scripts/shutdownButtonWatch.py" \
    "${REMOTE_ROOT}/scripts/shutdownButtonWatch.py"

rsync ${OPTS} \
    "${PROJECT_ROOT}/standaloneUnit/" \
    "${REMOTE_ROOT}/standaloneUnit/"

if [[ "${SKIP_FAA_DB}" == false ]]; then
    rsync ${OPTS} \
        "${PROJECT_ROOT}/data/ReleasableAircraft/" \
        "${REMOTE_ROOT}/data/ReleasableAircraft/"
else
    echo "Skipping data/ReleasableAircraft/ (--skipFaaDb)"
fi

echo "Sync complete."
echo "If standaloneRecorder/shutdownButton services are running on ${STANDALONE_HOST}, restart them to pick up the new code:"
echo "  ssh jdn@${STANDALONE_HOST} sudo systemctl restart standaloneRecorder shutdownButton"
