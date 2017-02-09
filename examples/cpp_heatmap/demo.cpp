/* This file uses a FLIC trained model and applies it to a video sequence from Poses in the Wild
 *
 * Download the model:
 *    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/
 */

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include<iostream>
#include <ctime>

//#include "demo.hpp"

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;


//Options
class Options{
public:
  bool visualise;
  bool useGPU;
  int dims[2];
  int numJoints;
  char layerName[20];
  char modelDefFile[80];
  char modelFile[80];
  char inputDir[40];
  int numFiles;
  
  Options();
};

class applyNet{

public:
  applyNet(Options opt);
  
  void printOptions(Options opt);
  void initCaffe(Options opt);
  void applyNetImages(char files[][10], Options opt, vector< vector<vector<int> > > &joints);
  vector<vector<int> > applyNetImage(char file[10], Options opt);
  Mat prepareImagePose(Mat img, Options opt);
  Mat processHeatmap(Mat features, Options opt);
  vector<vector<int> > heatmapToJoints(vector<Mat> heatmaps, int numJoints);

private:
  shared_ptr<Net<float> > net;
  Size input_geometry;
  int num_channels;
};


Options::Options(){
  visualise = true; // Visualise predictions
  useGPU = true; // Run on GPU
  dims[0] = 256; dims[1] = 256; // Input dimensions
  numJoints = 7; // Number of joints
  strcpy(layerName, "conv5_fusion"); // Output layer name
  strcpy(modelDefFile, "models/heatmap-flic-fusion/matlab.prototxt"); // Model definition
  strcpy(modelFile,  "models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel");// Model weights    
  strcpy(inputDir, "examples/cpp_heatmap/my_sample/"); // Image input directory
  numFiles = 29; // Number of input images  
}


void applyNet::printOptions(Options opt){
  printf("config:\n");
  printf("%s\n", opt.visualise ? "true" : "false");
  printf("%s\n", opt.useGPU ? "true" : "false");
  printf("[%d, %d]\n", opt.dims[0], opt.dims[1]);
  printf("%d\n", opt.numJoints);
  printf("%s\n", opt.layerName);
  printf("%s\n", opt.modelDefFile);
  printf("%s\n", opt.modelFile);
  printf("%s\n", opt.inputDir);
  printf("%d\n", opt.numFiles);
  printf("\n");

  return;
}

applyNet::applyNet(Options opt){
  // Initialise caffe

#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
  int gpu_id = 0;
  Caffe::SetDevice(gpu_id);
#endif

  /* Load the network. */
  this->net.reset(new Net<float>(opt.modelDefFile, TEST));
  this->net->CopyTrainedLayersFrom(opt.modelFile);

  Blob<float>* input_layer = net->input_blobs()[0];
  num_channels = input_layer->channels();
  input_geometry = Size(input_layer->width(), input_layer->height());
  Blob<float>* output_layer = net->output_blobs()[0];
}

void applyNet::applyNetImages(char files[][10], Options opt, vector< vector<vector<int> > > &joints){
  // Apply network separately to each image
  for (int ind = 0; ind < opt.numFiles; ind++){
    joints[ind] = applyNetImage(files[ind], opt);
    if (opt.visualise)
      cin.get();
  }

  return;
}

vector<vector<int> > applyNet::applyNetImage(char file[10], Options opt){
  // Apply network to a single image

  // Read & reformat input image
  Mat img;
  //printf("%s\n",strcat(opt.inputDir,file));
  img = imread(strcat(opt.inputDir,file), IMREAD_COLOR);   // Read the file into BGR channels 
  //namedWindow( "Display window", WINDOW_AUTOSIZE );
  //imshow( "Display window", img );
  //waitKey(0);

  Mat input_data;
  input_data = prepareImagePose(img, opt);

  // Forward pass
  clock_t start = clock();

  Blob<float>* input_layer = net->input_blobs()[0];
  input_layer->Reshape(1, num_channels, input_geometry.height, input_geometry.width);
  //cout << input_layer->shape(0) <<"x"<< input_layer->shape(1) <<"x"<< input_layer->shape(2) <<"x"<< input_layer->shape(3) <<"\n"; //1x3x256x256
  /* Forward dimension change to all layers. */
  net->Reshape();
  net->Forward();
  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net->output_blobs()[0];
  //cout << output_layer->shape(0) <<"x"<< output_layer->shape(1) <<"x"<< output_layer->shape(2) <<"x"<< output_layer->shape(3) <<"\n"; //1x7x64x64
  
  float* output = output_layer->mutable_cpu_data();
  //vector<vector<vector<int> > > output_mat (7, vector<vector<int> >(64, vector<int>(64, 0)));
  //int x, y, z ,w;
  //for(w=0; w<64*64*7; w++){
  //  x = w/(64*64);
  //  y = (w - x*64*64)/64;
  //  z = w-x*64*64-y*64;
  //  output_mat[x][y][z] = output[w];
  //}

  Size size(64, 64);
  vector<Mat> output_mat;
  for(int i=0; i<7; i++){
    Mat mat = Mat(size, CV_32F, output+i*64*64);
    output_mat.push_back(mat);
  }

  //vector<vector<vector<int> > > heatmaps (7, vector<vector<int> >(256, vector<int>(256, 0)));
  Size size2(256, 256);
  vector<Mat> heatmaps;
  for(int i=0; i<7; i++){
    heatmaps.push_back(processHeatmap(output_mat[i], opt)); //7x256x256
  }

  vector<vector<int> > j (2, vector<int>(opt.numJoints, 0));
  j = heatmapToJoints(heatmaps, opt.numJoints);
  
  clock_t end = clock();
  clock_t duration = end - start / (double) CLOCKS_PER_SEC;
  cout << duration << "\n";

  // Visualisation
  if (opt.visualise){
    
  }

  return j;
}

vector<vector<int> > applyNet::heatmapToJoints(vector<Mat> heatmaps, int numJoints){
  vector<vector<int> > j (2, vector<int>(numJoints, 0));
  double  minVal; 
  double maxVal; 
  Point minLoc; 
  Point maxLoc;
  for (int i=0; i<numJoints; i++){
    Mat sub_img = heatmaps[i];
    minMaxLoc( sub_img, &minVal, &maxVal, &minLoc, &maxLoc );
    //cout << maxLoc << "\n";
    j[0][i] = maxLoc.x;
    j[1][i] = maxLoc.y;
  }

  return j;
}

Mat applyNet::prepareImagePose(Mat img, Options opt){
  // Prepare input image for caffe: change to single & permute color channels

  Mat imgOut = img;
  Mat channels[3], tmp1, tmp2;
  split(imgOut, channels);
  tmp1 = channels[0];
  tmp2 = channels[1];
  channels[0] = channels[2];
  channels[1] = tmp1;
  channels[2] = tmp2;
  merge(channels, 3, imgOut);
  imgOut.convertTo(imgOut, CV_32FC1);
  return imgOut;
}

Mat applyNet::processHeatmap(Mat features, Options opt){
  // Reformat output heatmap: rotate & permute color channels
  //vector<vector<vector<int> > > out (7, vector<vector<int> >(256, vector<int>(256, 0)));
  Size size(256, 256);
  Mat out(size, CV_32F);
  resize(features, out, size, 0, 0, INTER_CUBIC);
  //cout << features.at<float>(0,0) <<"\n";
  //cout << features.at<float>(40,40) <<"\n";
  return out;
}

int main(){

  Options opt;
  applyNet app(opt);
  
  char files[opt.numFiles][10];
  //Create image file list
  for (int ind = 0; ind < opt.numFiles; ind++){
    sprintf(files[ind], "%d.png", ind+1);
    //printf("%s\n", files[ind]);
  }

  // Apply network
  vector<vector<vector<int> > > joints (opt.numFiles, vector<vector<int> >(2, vector <int>(opt.numJoints, 0)));
  // Run network on multiple images
  app.printOptions(opt);
  // Run network on multiple images
  app.applyNetImages(files, opt, joints);
  
  return 0;
}

#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
