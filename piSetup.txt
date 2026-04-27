sudo timedatectl set-ntp true
python3 -m venv ./aaiVenv
source ./aaiVenv/bin/activate
python3 -m pip install -r requirements.txt.pi0
sudo apt update
sudo apt upgrade
sudo apt install git
git config --global user.email "duane.northcutt@gmail.com"
git config --global user.name "J. Duane Northcutt"
git clone git@github.com:jduanen/AircraftAudioId.git
