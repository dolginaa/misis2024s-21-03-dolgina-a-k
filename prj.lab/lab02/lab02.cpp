#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/fast_math.hpp>

#include <iostream>

const uchar RECTANGLE_SIDE = 209;
const uchar CIRCLE_RADIUS = 83;
const int HISTOGR_SIZE = 256;
const int HISTOGR_WIDTH = 256;
const int HISTOGR_HIGHT = 256;

cv::Mat generate_sample(const cv::Scalar colors) {
    cv::Mat img = cv::Mat(256, 256, 0, colors[0]);
    cv::Size img_dim = img.size();

    uchar point_coord = (img_dim.width - RECTANGLE_SIDE) / 2;

    cv::Point top_left = cv::Point(point_coord, point_coord);
    cv::Point bottom_right = cv::Point(img_dim.width - point_coord, img_dim.width - point_coord);
    cv::rectangle(img, top_left, bottom_right, colors[1], cv::FILLED);

    cv::Point circle_center = cv::Point(img_dim.width / 2, img_dim.width / 2);
    cv::Size thicc = cv::Size(CIRCLE_RADIUS, CIRCLE_RADIUS);
    cv::ellipse(img, circle_center, thicc, 0, 0, 360, colors[2], cv::FILLED);
    return img;
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

void calc_mean_stddev(const cv::Mat img, const cv::Mat mask) {
    cv::Mat histogram;
    float range[] = {0, 256};
    const float* hist_range[] = {range};
    cv::calcHist(&img, 1, 0, mask, histogram, 1, &HISTOGR_SIZE, hist_range, true, false);

    double mean = 0;
    int cnt = cv::sum(histogram)[0];
    for (int i = 0; i < histogram.size().height; i++) {
        mean += i * histogram.at<float>(i);
    }
    mean /= cnt;

    double stddev = 0;
    for (int i = 0; i < histogram.size().height; i++) {
        stddev += histogram.at<float>(i) * (i - mean) * (i - mean);
    }
    stddev /= cnt;

    printf("%.2f %.2f\n", mean, cv::sqrt(stddev));
}

int main(int argc, char* argv[]) {
    cv::Scalar color_arr[] = {cv::Scalar(0, 127, 255), cv::Scalar(20, 127, 235),
                              cv::Scalar(55, 127, 200), cv::Scalar(90, 127, 165)};
    int stddevs_arr[] = {3, 7, 15};

    const cv::Mat kMaskSquareOuter = generate_sample(cv::Scalar(255, 0, 0)),
                  kMaskSquareInner = generate_sample(cv::Scalar(0, 255, 0)),
                  kMaskCircle = generate_sample(cv::Scalar(0, 0, 255));

    std::vector<std::vector<float>> stats_table(4 * 3, std::vector<float>(2 * 4, 0.0));

    std::vector<cv::Mat> res_images, sample_images;
    for (cv::Scalar colors: color_arr) {
        std::vector<cv::Mat> noised_images;
        cv::Mat img_hist;
        cv::Mat img = generate_sample(colors);
        printf("For colors %.0f %.0f %.0f\n", colors[0], colors[1], colors[2]);
        for (int stddev: stddevs_arr){
            cv::Mat_<int> noise(img.size());
            cv::randn(noise, 0, stddev);
            cv::Mat noised_img = img.clone();
            noised_img += noise;
            printf("Real deviation %d\n", stddev);
            calc_mean_stddev(noised_img, kMaskSquareOuter);
            calc_mean_stddev(noised_img, kMaskSquareInner);
            calc_mean_stddev(noised_img, kMaskCircle);
            printf("\n");

            cv::Mat hist_image = generate_histgram(noised_img);
            cv::vconcat(noised_img, hist_image, noised_img);
            noised_images.push_back(noised_img);
        }
        printf("\n");
        cv::vconcat(noised_images, img_hist);
        res_images.push_back(img_hist);
        sample_images.push_back(img);
    }

    cv::Mat samples;
    cv::hconcat(sample_images, samples);

    cv::Mat res_img;
    cv::hconcat(res_images,res_img);
    cv::vconcat(samples, res_img, res_img);

    cv::imshow("lab02", res_img);
    // cv::imwrite("../prj.lab/lab02/res.png", res_img);
    cv::waitKey(0);
    return 0;
}