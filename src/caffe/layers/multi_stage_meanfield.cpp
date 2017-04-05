/*!
 *  \brief     The Caffe layer that implements the CRF-RNN described in the paper:
 *             Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *  \authors   Sadeep Jayasumana, Bernardino Romera-Paredes, Shuai Zheng, Zhizhong Su.
 *  \version   1.0
 *  \date      2015
 *  \copyright Torr Vision Group, University of Oxford.
 *  \details   If you use this code, please consider citing the paper:
 *             Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du,
 *             Chang Huang, Philip H. S. Torr. Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             For more information about CRF-RNN, please visit the project website http://crfasrnn.torr.vision.
 */
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/tvg_util.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <cmath>

namespace caffe {

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const caffe::MultiStageMeanfieldParameter meanfield_param = this->layer_param_.multi_stage_meanfield_param();

  num_iterations_ = meanfield_param.num_iterations();

  CHECK_GT(num_iterations_, 1) << "Number of iterations must be greater than 1.";

  theta_alpha_ = meanfield_param.theta_alpha();
  theta_beta_ = meanfield_param.theta_beta();
  theta_gamma_ = meanfield_param.theta_gamma();

  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_pixels_ = height_ * width_;

  //POTTS==1, CUSTOM==11
  compat_mode_ = meanfield_param.compatibility_mode();//Implementing CUSTOM matrix
  //LOG(INFO) << "Implementing compatibility matrix, selected mode:" << compat_mode_;

  LOG(INFO) << "This implementation has not been tested batch size > 1.";

  top[0]->Reshape(num_, channels_, height_, width_);

  // Initialize the parameters that will updated by backpropagation.
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Multimeanfield layer skipping parameter initialization.";
  } else {

    this->blobs_.resize(3);// blobs_[0] - spatial kernel weights, blobs_[1] - bilateral kernel weights, blobs_[2] - compatability matrix

    // Allocate space for kernel weights.
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_));
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_));

    caffe_set(channels_ * channels_, Dtype(0.), this->blobs_[0]->mutable_cpu_data());
    caffe_set(channels_ * channels_, Dtype(0.), this->blobs_[1]->mutable_cpu_data());

    // Initialize the kernels weights. The two files spatial.par and bilateral.par should be available.
    FILE * pFile;
    pFile = fopen("spatial.par", "r");
    CHECK(pFile) << "The file 'spatial.par' is not found. Please create it with initial spatial kernel weights.";
    for (int i = 0; i < channels_; i++) {
      fscanf(pFile, "%lf", &this->blobs_[0]->mutable_cpu_data()[i * channels_ + i]);
    }
    fclose(pFile);

    pFile = fopen("bilateral.par", "r");
    CHECK(pFile) << "The file 'bilateral.par' is not found. Please create it with initial bilateral kernel weights.";
    for (int i = 0; i < channels_; i++) {
      fscanf(pFile, "%lf", &this->blobs_[1]->mutable_cpu_data()[i * channels_ + i]);
    }
    fclose(pFile);

    if (compat_mode_ == 1){//POTTS
      // Initialize the compatibility matrix.
      this->blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_));
      caffe_set(channels_ * channels_, Dtype(0.), this->blobs_[2]->mutable_cpu_data());

      // Initialize it to have the Potts model.
      for (int c = 0; c < channels_; ++c) {
        (this->blobs_[2]->mutable_cpu_data())[c * channels_ + c] = Dtype(-1.);
      }
    }else if (compat_mode_ == 11){//CUSTOM
	//Init Custom Matrix
	LOG(INFO)<<"Initializing custom matrix";

	// Initialize the compatibility matrix.
      this->blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_));
      caffe_set(channels_ * channels_, Dtype(-0.012345), this->blobs_[2]->mutable_cpu_data());

      // Initialize it to have the Potts model.
      for (int c = 0; c < channels_; ++c) {
        (this->blobs_[2]->mutable_cpu_data())[c * channels_ + c] = Dtype(-0.899);
	
	//set each channel (label)
	//case c represent the row of the matrix. For each row set the values of all columns
	switch(c)
	{
		case 0://background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.789);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(-0.5);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(-0.5);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.5);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(-0.5);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(-0.5);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.5);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.5);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.5);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(-0.5);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(-0.5);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.5);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.5);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(-0.5);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(-0.5);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.5);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(-0.5);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(-0.5);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(-0.5);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.5);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(-0.5);//tvmonitor
		break;
		case 1://aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(-0.987);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.02);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(-0.0005);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.001);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.001);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(0);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(0);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.00005);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.03);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.0002);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 2://bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(-0.987);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.005);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(-0.0002);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(-0.0025);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.05);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.15);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.001);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(0);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.005);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(-0.003);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.3);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.0002);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 3://bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(-0.002);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.987);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.0001);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.0005);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.015);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(-0.002);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(-0.002);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.003);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.005);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(-0.00085);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.025);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(-0.003);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(-0.0007);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.00005);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(-0.0004);//tvmonitor
		break;
		case 4://boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.0025);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(-0.987);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.0001);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.005);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(0);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(0);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(0);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.05);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.00005);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 5://bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(-0.0023);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(0);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(-0.987);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(0);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(0);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(0);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(-0.0015);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.2);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.0015);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(-0.0001);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.05);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(-0.0015);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(0);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 6://bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(-0.015);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(-0.0001);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.0025);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(-0.001);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.987);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.15);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(0);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(0);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(0);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(-0.0001);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.15);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.0005);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 7://car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(-0.0015);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(-0.075);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.00035);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(-0.0085);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.007);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.987);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(0);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(0);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.00015);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(-0.025);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.15);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.0003);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 8://cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(-0.002);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.05);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.0001);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.0005);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.987);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(-0.085);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.025);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.085);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(-0.00085);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.1);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(-0.0025);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(-0.0005);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(-0.04);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(0);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(-0.0004);//tvmonitor
		break;
		case 9://chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(0);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(-0.0015);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(0);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(0);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.075);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(-0.987);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.35);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.05);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.35);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(-0.0015);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(0);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(-0.0005);//tvmonitor
		break;
		case 10://cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(-0.0005);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.0005);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.0001);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.0005);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.015);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(-0.987);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(0);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.005);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(-0.05);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.06);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(-0.0008);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(-0.04);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(0);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 11://diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.0025);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(-0.075);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(0);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(0);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.075);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(-0.45);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.987);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.005);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.35);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(-0.0015);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(0);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(-0.008);//tvmonitor
		break;
		case 12://dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(-0.002);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.05);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.0001);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.0005);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.025);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(-0.01);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(-0.005);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.025);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.987);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(-0.0085);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.1);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(-0.0025);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(-0.005);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(-0.04);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(0);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 13://horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.0085);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(0);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(0);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.025);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(-0.05);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(0);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.085);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(-0.987);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.1);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(-0.0025);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(-0.005);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.003);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 14://motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(-0.0015);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(-0.075);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.00035);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.007);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.15);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(0);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(0);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(0);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(-0.987);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.15);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.0003);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 15://person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(-0.02);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(-0.25);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.05);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(-0.05);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(-0.05);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.1);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.15);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.05);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(-0.2);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(-0.005);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.15);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.05);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(-0.085);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(-0.085);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.987);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(-0.025);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(-0.005);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(-0.15);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.035);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(-0.0025);//tvmonitor
		break;
		case 16://pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.0025);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(0);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(0);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.075);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.005);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.005);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.02);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(-0.987);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(-0.0015);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(0);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 17://sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.0085);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(0);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(0);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.025);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(-0.05);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(0);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.085);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(-0.09);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.05);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(-0.0025);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(-0.987);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(0);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 18://sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(0);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(-0.0015);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(0);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(0);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.075);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(-0.05);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.005);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.05);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.35);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(-0.987);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(0);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(-0.0005);//tvmonitor
		break;
		case 19://train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(-0.0015);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(-0.00075);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.00035);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(-0.0085);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(-0.05);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(-0.085);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(0);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(0);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(0);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(0);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.15);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(0);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(0);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(-0.987);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(0);//tvmonitor
		break;
		case 20://tvmonitor
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 0] = Dtype(-0.5);//background
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 1] = Dtype(0);//aeroplane
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 2] = Dtype(0);//bicycle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 3] = Dtype(-0.0025);//bird
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 4] = Dtype(0);//boat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 5] = Dtype(0);//bottle
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 6] = Dtype(0);//bus
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 7] = Dtype(0);//car
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 8] = Dtype(-0.075);//cat
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 9] = Dtype(-0.006);//chair
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 10] = Dtype(0);//cow
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 11] = Dtype(-0.005);//diningtable
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 12] = Dtype(-0.005);//dog
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 13] = Dtype(0);//horse
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 14] = Dtype(0);//motorbike
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 15] = Dtype(-0.02);//person
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 16] = Dtype(-0.0004);//pottedplant
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 17] = Dtype(0);//sheep
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 18] = Dtype(-0.0015);//sofa
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 19] = Dtype(0);//train
		(this->blobs_[2]->mutable_cpu_data())[c * channels_ + 20] = Dtype(-0.987);//tvmonitor
		break;
	}
      }
    }else{
        LOG(INFO)<<"Compatibility mode not defined!!";
    }
  }

  // Initialize the spatial lattice. This does not need to be computed for every image because we use a fixed size.
  float spatial_kernel[2 * num_pixels_];
  compute_spatial_kernel(spatial_kernel);
  spatial_lattice_.reset(new ModifiedPermutohedral());
  spatial_lattice_->init(spatial_kernel, 2, num_pixels_);

  // Calculate spatial filter normalization factors.
  norm_feed_.reset(new Dtype[num_pixels_]);
  caffe_set(num_pixels_, Dtype(1.0), norm_feed_.get());
  spatial_norm_.Reshape(1, 1, height_, width_);
  Dtype* norm_data = spatial_norm_.mutable_cpu_data();
  spatial_lattice_->compute(norm_data, norm_feed_.get(), 1);
  for (int i = 0; i < num_pixels_; ++i) {
    norm_data[i] = 1.0f / (norm_data[i] + 1e-20f);
  }

  // Allocate space for bilateral kernels. This is a temporary buffer used to compute bilateral lattices later.
  // Also allocate space for holding bilateral filter normalization values.
  bilateral_kernel_buffer_.reset(new float[5 * num_pixels_]);
  bilateral_norms_.Reshape(num_, 1, height_, width_);

  // Configure the split layer that is used to make copies of the unary term. One copy for each iteration.
  // It may be possible to optimize this calculation later.
  split_layer_bottom_vec_.clear();
  split_layer_bottom_vec_.push_back(bottom[0]);

  split_layer_top_vec_.clear();

  split_layer_out_blobs_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; i++) {
    split_layer_out_blobs_[i].reset(new Blob<Dtype>());
    split_layer_top_vec_.push_back(split_layer_out_blobs_[i].get());
  }

  LayerParameter split_layer_param;
  split_layer_.reset(new SplitLayer<Dtype>(split_layer_param));
  split_layer_->SetUp(split_layer_bottom_vec_, split_layer_top_vec_);

  // Make blobs to store outputs of each meanfield iteration. Output of the last iteration is stored in top[0].
  // So we need only (num_iterations_ - 1) blobs.
  iteration_output_blobs_.resize(num_iterations_ - 1);
  for (int i = 0; i < num_iterations_ - 1; ++i) {
    iteration_output_blobs_[i].reset(new Blob<Dtype>(num_, channels_, height_, width_));
  }

  // Make instances of MeanfieldIteration and initialize them.
  meanfield_iterations_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    meanfield_iterations_[i].reset(new MeanfieldIteration<Dtype>());
    meanfield_iterations_[i]->OneTimeSetUp(
        split_layer_out_blobs_[i].get(), // unary terms
        (i == 0) ? bottom[1] : iteration_output_blobs_[i - 1].get(), // softmax input
        (i == num_iterations_ - 1) ? top[0] : iteration_output_blobs_[i].get(), // output blob
        spatial_lattice_, // spatial lattice
        &spatial_norm_); // spatial normalization factors.
  }

  this->param_propagate_down_.resize(this->blobs_.size(), true);

  LOG(INFO) << ("MultiStageMeanfieldLayer initialized.");
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// Do nothing.
}


/**
 * Performs filter-based mean field inference given the image and unaries.
 *
 * bottom[0] - Unary terms
 * bottom[1] - Softmax input/Output from the previous iteration (a copy of the unary terms if this is the first stage).
 * bottom[2] - RGB images
 *
 * top[0] - Output of the mean field inference (not normalized).
 */
template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  split_layer_bottom_vec_[0] = bottom[0];
  split_layer_->Forward(split_layer_bottom_vec_, split_layer_top_vec_);

  // Initialize the bilateral lattices.
  bilateral_lattices_.resize(num_);
  for (int n = 0; n < num_; ++n) {

    compute_bilateral_kernel(bottom[2], n, bilateral_kernel_buffer_.get());
    bilateral_lattices_[n].reset(new ModifiedPermutohedral());
    bilateral_lattices_[n]->init(bilateral_kernel_buffer_.get(), 5, num_pixels_);

    // Calculate bilateral filter normalization factors.
    Dtype* norm_output_data = bilateral_norms_.mutable_cpu_data() + bilateral_norms_.offset(n);
    bilateral_lattices_[n]->compute(norm_output_data, norm_feed_.get(), 1);
    for (int i = 0; i < num_pixels_; ++i) {
      norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
    }
  }

  for (int i = 0; i < num_iterations_; ++i) {

    meanfield_iterations_[i]->PrePass(this->blobs_, &bilateral_lattices_, &bilateral_norms_);

    meanfield_iterations_[i]->Forward_cpu();
  }
}

/**
 * Backprop through filter-based mean field inference.
 */
template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  for (int i = (num_iterations_ - 1); i >= 0; --i) {
    meanfield_iterations_[i]->Backward_cpu();
  }

  vector<bool> split_layer_propagate_down(1, true);
  split_layer_->Backward(split_layer_top_vec_, split_layer_propagate_down, split_layer_bottom_vec_);

  // Accumulate diffs from mean field iterations.
  for (int blob_id = 0; blob_id < this->blobs_.size(); ++blob_id) {

    Blob<Dtype>* cur_blob = this->blobs_[blob_id].get();

    if (this->param_propagate_down_[blob_id]) {

      caffe_set(cur_blob->count(), Dtype(0), cur_blob->mutable_cpu_diff());

      for (int i = 0; i < num_iterations_; ++i) {
        const Dtype* diffs_to_add = meanfield_iterations_[i]->blobs()[blob_id]->cpu_diff();
        caffe_axpy(cur_blob->count(), Dtype(1.), diffs_to_add, cur_blob->mutable_cpu_diff());
      }
    }
  }
}

template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n,
                                                               float* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[5 * p] = static_cast<float>(p % width_) / theta_alpha_;
    output_kernel[5 * p + 1] = static_cast<float>(p / width_) / theta_alpha_;

    const Dtype * const rgb_data_start = rgb_blob->cpu_data() + rgb_blob->offset(n);
    output_kernel[5 * p + 2] = static_cast<float>(rgb_data_start[p] / theta_beta_);
    output_kernel[5 * p + 3] = static_cast<float>((rgb_data_start + num_pixels_)[p] / theta_beta_);
    output_kernel[5 * p + 4] = static_cast<float>((rgb_data_start + num_pixels_ * 2)[p] / theta_beta_);
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::compute_spatial_kernel(float* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[2*p] = static_cast<float>(p % width_) / theta_gamma_;
    output_kernel[2*p + 1] = static_cast<float>(p / width_) / theta_gamma_;
  }
}

INSTANTIATE_CLASS(MultiStageMeanfieldLayer);
REGISTER_LAYER_CLASS(MultiStageMeanfield);
}  // namespace caffe
