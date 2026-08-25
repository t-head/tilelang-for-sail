source $AONE_CI_WORKSPACE/opt/PPU_SDK/envsetup.sh
# pip install $AONE_CI_WORKSPACE/opt/torch/*.whl --force-reinstall
python -m pip install wheel build setuptools
python -m pip install cython packaging setuptools==68.1.2

set -e

# build tilelang
cd $AONE_CI_SOURCE/tilelang
python -m build

export build_output=$AONE_CI_SOURCE/output
mkdir -p ${build_output}
cp $AONE_CI_SOURCE/tilelang/.aoneci/script/* ${build_output}/
cp -rf $AONE_CI_WORKSPACE/opt/* ${build_output}/
cp -rf $AONE_CI_SOURCE/tilelang ${build_output}/

cd ${build_output}
tar -czf  $AONE_CI_SOURCE/output.tar.gz *
ls $AONE_CI_WORKSPACE/source
