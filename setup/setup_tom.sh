
export ROS_DISTRO=kinetic
export CTP_FILE=setup_kinetic.sh\?token\=ADImZ3UI07E04DzgwetRl93sHErSv9yyks5aL5uBwA%3D%3D
wget https://raw.githubusercontent.com/cpaxton/costar_plan/master/setup/setup_kinetic.sh?token=ADImZ3UI07E04DzgwetRl93sHErSv9yyks5aL5uBwA%3D%3D I
chmod +x $CTP_FILE
source ./$CTP_FILE

wget https://raw.githubusercontent.com/cpaxton/costar_stack/master/install_indigo.sh
chmod +x install_indigo.sh
source ./install_indigo.sh

git clone git@github.com:cpaxton/costar_plan.git


