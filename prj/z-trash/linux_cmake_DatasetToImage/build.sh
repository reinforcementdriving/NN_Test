#! /bin/bash

real_path=$(realpath $0)
dir_name=`dirname "${real_path}"`
echo "real_path: ${real_path}, dir_name: ${dir_name}"

data_dir="data"
if [ -d ${dir_name}/${data_dir} ]; then
	rm -rf ${dir_name}/${data_dir}
fi

cifar100_path=${dir_name}/./../../${data_dir}/database/CIFAR/CIFAR-100
echo "cifar100_path: ${cifar100_path}"
cd ${cifar100_path}
7za -y x "*.7z.*"

cd -

ln -s ${dir_name}/./../../${data_dir} ${dir_name}

mkdir -p ${dir_name}/${data_dir}/tmp/MNIST/test_images
mkdir -p ${dir_name}/${data_dir}/tmp/MNIST/train_images
mkdir -p ${dir_name}/${data_dir}/tmp/cifar-10_train
mkdir -p ${dir_name}/${data_dir}/tmp/cifar-10_test
mkdir -p ${dir_name}/${data_dir}/tmp/cifar-100_train
mkdir -p ${dir_name}/${data_dir}/tmp/cifar-100_test

rc=$?
if [[ ${rc} != 0 ]]; then
	echo "##### Error: some of thess commands have errors above, please check"
	exit ${rc}
fi

new_dir_name=${dir_name}/build
mkdir -p ${new_dir_name}
cd ${new_dir_name}
cmake ..
make

cd -

