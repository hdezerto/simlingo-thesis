# mkdir ~/software
# mkdir ~/software/carla0915
# cd ~/software/carla0915

source env.sh

CARLA_BASE="${BASE_DIR}/carla"
mkdir -p "$CARLA_ROOT"
cd "$CARLA_BASE"

wget -c "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_${CARLA_VERSION}.tar.gz"
tar -xvf "CARLA_${CARLA_VERSION}.tar.gz"
rm -f "CARLA_${CARLA_VERSION}.tar.gz"
mkdir -p "$CARLA_ROOT"
shopt -s dotglob extglob
mv -- "$CARLA_BASE"/!("CARLA_${CARLA_VERSION}"|"CARLA_${CARLA_VERSION}.tar.gz") "$CARLA_ROOT"/
cd "$CARLA_ROOT/Import" && wget -c "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_${CARLA_VERSION}.tar.gz"
cd "$CARLA_ROOT" && bash ImportAssets.sh