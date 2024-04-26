#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

const int RECTANGLE_SIDE = 99;
const int RECTANGLE_CNT = 3;
const int ROWS_COUNT = 2;
const int CIRCLE_RADIUS = 25;
const int GREY_L_1 = 0;
const int GREY_L_2 = 127;
const int GREY_L_3 = 255;

std::tuple<cv::Mat1b, cv::Mat1b, cv::Mat1b> process(cv::Mat1f sample) {
    cv::Mat1f sample_k1 = sample.clone(), sample_k2 = sample.clone();
    cv::Mat kernel1 = (cv::Mat1f(2, 2) << 1.0, 0.0, 0.0, -1.0);
    cv::filter2D(sample, sample_k1, -1, kernel1, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::Mat kernel2 = (cv::Mat1f(2, 2) << 0.0, 1.0, -1.0, 0.0);
    cv::filter2D(sample, sample_k2, -1, kernel2, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::Mat1f s1_tmp = sample_k1.clone(), s2_tmp = sample_k2.clone(), s3_tmp = s2_tmp.clone();

    s1_tmp = s1_tmp * 0.5 + 127.5;
    s2_tmp = s2_tmp * 0.5 + 127.5;
    cv::sqrt(s1_tmp.mul(s1_tmp) + s2_tmp.mul(s2_tmp), s3_tmp);
    cv::Mat1b s1 = s1_tmp.clone(), s2 = s2_tmp.clone(), s3 = s3_tmp.clone();
    
    return std::make_tuple(s1, s2, s3);
}

int main() {
    cv::Mat1f sample(RECTANGLE_SIDE * ROWS_COUNT, RECTANGLE_SIDE * RECTANGLE_CNT);
    std::vector<cv::Vec2b> colors{
        cv::Vec2b{GREY_L_1, GREY_L_2},
        cv::Vec2b{GREY_L_2, GREY_L_1},
        cv::Vec2b{GREY_L_3, GREY_L_1},
        cv::Vec2b{GREY_L_3, GREY_L_2},
        cv::Vec2b{GREY_L_1, GREY_L_3},
        cv::Vec2b{GREY_L_2, GREY_L_3}
    };
    for (int i = 0; i < RECTANGLE_CNT; i++) {
        for (int j = 0; j < ROWS_COUNT; j++) {
            cv::Vec2b color = colors[j * RECTANGLE_CNT + i];

            cv::Point top_left = cv::Point(i * RECTANGLE_SIDE, j * RECTANGLE_SIDE);
            cv::Point bottom_right = cv::Point((i + 1) * RECTANGLE_SIDE, (j + 1) * RECTANGLE_SIDE);
            cv::rectangle(sample, top_left, bottom_right, color[0], cv::FILLED);

            cv::Point center = cv::Point(i * RECTANGLE_SIDE + RECTANGLE_SIDE / 2., j * RECTANGLE_SIDE + RECTANGLE_SIDE / 2.);
            cv::ellipse(sample, center, cv::Size(CIRCLE_RADIUS, CIRCLE_RADIUS), 0, 0, 360, color[1], cv::FILLED);
        }
    }

    cv::Mat1b s1, s2, s3;
    std::tie(s1, s2, s3) = process(sample);
    std::vector<cv::Mat1b> channels{s3, s2, s1};
    cv::Mat3b res;
    cv::merge(channels, res);
    cv::imwrite("C:/misis2024s-21-03-dolgina-a-k/prj.lab/lab05/l_1.png", s1);
    cv::imwrite("C:/misis2024s-21-03-dolgina-a-k/prj.lab/lab05/l_2.png", s2);
    cv::imwrite("C:/misis2024s-21-03-dolgina-a-k/prj.lab/lab05/l_3.png", s3);
    cv::imwrite("C:/misis2024s-21-03-dolgina-a-k/prj.lab/lab05/res.png", res);
    cv::imshow("res", res);
    cv::waitKey(0);
    return 0;
}