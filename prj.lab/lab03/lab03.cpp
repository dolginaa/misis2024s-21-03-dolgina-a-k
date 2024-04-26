#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <iostream>

const int HISTOGR_SIZE = 256;
const int HISTOGR_WIDTH = 256;
const int HISTOGR_HIGHT = 256;
const uchar B_BORDER = 0;
const uchar W_BORDER = 255;

uchar contrast(const uchar col, const uchar col_min_old, const uchar col_max_old,
               const uchar col_min_new, const uchar col_max_new) {
  float res = (col_min_new
               + (col - col_min_old)
               * (col_max_new - col_min_new)
               / (col_max_old - col_min_old));
  return cvRound(res);
}

std::pair<uchar, uchar> find_min_max(cv::Mat hist, double qb, double qw) {
  uchar min = 0, max = 0;
  int hist_sum = cv::sum(hist)[0];
  long int acc_sum = 0;
  for (int i = 0; i < hist.size().height; i++) {
    if (hist.at<float>(i) > 0) {
      acc_sum += hist.at<float>(i);
    }
    if (acc_sum >= qb * hist_sum) {
      min = i;
      break;
    }
  }
  acc_sum = 0;
  for (int i = hist.size().height - 1; i >= 0; i--) {
    if (hist.at<float>(i) > 0) {
      acc_sum += hist.at<float>(i);
    }
    if (acc_sum >= qw * hist_sum) {
      max = i;
      break;
    }
  }
  return std::pair<uchar, uchar>(min, max);
}

cv::Mat generate_histgram(const cv::Mat img) {
    cv::Mat histogram;
    float range[] = {0, 256};
    const float* hist_range[] = {range};
    cv::calcHist(&img, 1, 0, cv::Mat(), histogram, 1, &HISTOGR_SIZE, hist_range, true, false);

    int bin_w = cvRound(double(HISTOGR_WIDTH / HISTOGR_SIZE));

    cv::Mat hist_image(HISTOGR_WIDTH, HISTOGR_HIGHT, 0, 230);

    cv::normalize(histogram, histogram, 0, 230, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 0; i < HISTOGR_SIZE; i++) {
        cv::rectangle(hist_image, cv::Point(bin_w * i, HISTOGR_HIGHT), cv::Point(bin_w * (i + 1), HISTOGR_HIGHT - cvRound(histogram.at<float>(i))), 0, cv::FILLED);
    }

    cv::Mat flipped_histogr;
    cv::flip(hist_image, flipped_histogr, 0);
    return flipped_histogr;
}


void channel_contrast(cv::Mat& channel, const uchar black_cutoff, const uchar white_cutoff) {
  for (int x = 0; x < channel.rows; x++) {
    for(int y = 0; y < channel.cols; y++) {
      if (channel.at<uchar>(x, y) <= black_cutoff){
        channel.at<uchar>(x, y) = B_BORDER;
      }
      else if(channel.at<uchar>(x,y) >= white_cutoff){
        channel.at<uchar>(x, y) = W_BORDER;
      }
      else {
        channel.at<uchar>(x, y) = contrast(channel.at<uchar>(x, y), black_cutoff,
                                           white_cutoff, B_BORDER, W_BORDER);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  const cv::String parser_keys = 
  "{qb     |0.3|}"
  "{qw     |0.3|}"
  "{mode m |1  |}"
  ;
  cv::CommandLineParser parser(argc, argv, parser_keys);
  double q1 = parser.get<float>("qb");
  double q2 = parser.get<float>("qw");
  int mode = parser.get<int>("mode");

  cv::Mat pict = cv::imread("C:/misis2024s-21-03-dolgina-a-k/prj.lab/lab03/fish_homes.jpg");

  std::vector<cv::Mat> channels;
  cv::split(pict, channels);
  cv::Mat hist;
  cv::Mat hist2;
  float range[] = {0, 256};
  hist = generate_histgram(channels[0]);
  cv::imshow("hist3", hist);
  uchar black_cutoff = 255, white_cutoff = 0;

  for (cv::Mat& channel: channels) {
    if (mode ==1) {
      const float* hist_range[] = {range};
      cv::calcHist(&channel, 1, 0, cv::Mat(), hist2, 1, &HISTOGR_SIZE, hist_range, true, false);
      std::pair<uchar, uchar> cutoffs = find_min_max(hist2, q1, q2);
      if (cutoffs.first <= black_cutoff) {
        black_cutoff = cutoffs.first;
      } 
      if (cutoffs.second >= white_cutoff) {
        white_cutoff = cutoffs.second;
      }
    } else {
      const float* hist_range[] = {range};
      cv::calcHist(&channel, 1, 0, cv::Mat(), hist2, 1, &HISTOGR_SIZE, hist_range, true, false);
      std::pair<uchar, uchar> cutoffs = find_min_max(hist2, q1, q2);
      black_cutoff = cutoffs.first; white_cutoff = cutoffs.second;
    }
    channel_contrast(channel, black_cutoff, white_cutoff);
  }

  cv::Mat res_img = cv::Mat(pict.size(), pict.type());
  cv::merge(channels, res_img);

  hist = generate_histgram(channels[0]);
  cv::imshow("hist2", hist);

  cv::hconcat(pict, res_img, res_img);
  cv::imwrite("C:/misis2024s-21-03-dolgina-a-k/prj.lab/lab03/fish_homes_3.png", res_img);
  cv::waitKey(0);
}