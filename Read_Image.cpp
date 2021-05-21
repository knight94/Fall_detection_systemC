#include <systemc.h>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#undef int64
#include<iostream>

//#define TEMP_FILE_NAME "C:\\Users\\nsahu\\Documents\\Semester-I\\COL812\\Dataset\\fall-01-cam0-d\\fall-01-cam0-d-%03d.png"
//#define TEMP_FILE_NAME "C:\\Users\\nsahu\\Documents\\Semester-I\\COL812\\Dataset\\fall-02-cam-0-d\\fall-02-cam0-d-%03d.png"
//#define TEMP_FILE_NAME "C:\\Users\\nsahu\\Documents\\Semester-I\\COL812\\Dataset\\fall-03-cam0-d\\fall-03-cam0-d-%03d.png"
#define TEMP_FILE_NAME "C:\\Users\\nsahu\\Documents\\Semester-I\\COL812\\Dataset\\fall-05-cam0-d\\fall-05-cam0-d-%03d.png"
//#define TEMP_FILE_NAME "C:\\Users\\nsahu\\Documents\\Semester-I\\COL812\\Dataset\\fall-29-cam0-d\\fall-29-cam0-d-%03d.png"
#define ROWS 480
#define COLS 640
#define DEPTH_SIZE 13
#define TAKE_FRAME 8
#define SIZE ROWS*COLS

#define BACK_THRES 20

#define B 0.075
#define F 580
//fall
#define C0 6000
//ADL
#define C1 7000
#define SCALE 65535

#define HT_THRES 100

#define FALL_T 100
#define FALL_T_RATIO 0.6
#define FALL_T_Y 90

using namespace cv;
using namespace std;

/*For centroid and bounding box */
typedef struct cen_bound {
    double data[5];
    cen_bound(void) {
        for (int i = 0; i < 5; i++) {
            data[i] = 0;
        }
    }
    cen_bound& operator=(const cen_bound& copy) {
        for (int i = 0; i < 5; i++) {
            data[i] = copy.data[i];
        }
        return *this;
    }
    operator double* () {
        return data;
    }
    double& operator[](const int index) {
        return data[index];
    }
}CEN_BOUND;

/*For centroid and bounding box */
typedef struct plane_p {
    double data[3];
    plane_p(void) {
        for (int i = 0; i < 3; i++) {
            data[i] = 0;
        }
    }
    plane_p& operator=(const plane_p& copy) {
        for (int i = 0; i < 3; i++) {
            data[i] = copy.data[i];
        }
        return *this;
    }
    operator double* () {
        return data;
    }
    double& operator[](const int index) {
        return data[index];
    }
}PLANE_PARA;

SC_MODULE(Depth_image) {
    sc_fifo_out<Mat> ImgOutPort;
    sc_fifo_out<Mat> ImgOutPort1;
    sc_fifo_out<Mat> ImgOutPort2;
    sc_fifo_out<Mat> ImgOutPort3;

    //sc_in<bool> clock;

    SC_CTOR(Depth_image)
    {
        SC_THREAD(Read_input_image);
    }

    void Read_input_image()
    {
        //fall-01-cam0-d-001
        char inputfilename[256];
        for (int i = 0; i < 160; i++) {
            sprintf_s(inputfilename, 256, TEMP_FILE_NAME, i + 1);
            Mat image = imread(inputfilename, IMREAD_GRAYSCALE);
            Mat image2 = imread(inputfilename, IMREAD_ANYDEPTH);
            if (image.empty()) // Check for invalid input
            {
                cout << "Could not open or find the image" << endl;
                return;
            }
            /*for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                    img[i * image.rows + j] = image.at<double>(i, j);
                }
            }*/
            /*wait for posedge of clock*/
            //wait();
            ImgOutPort.write(image);
            ImgOutPort1.write(image);
            ImgOutPort2.write(image2);
            ImgOutPort3.write(image2);
        }
    }

};


SC_MODULE(Reference_image) {
    sc_fifo_in<Mat> ImgInPort;
    sc_fifo_out<Mat> ImgOutPort; 
    unsigned int index;
    unsigned int frame_counter;
    const unsigned int use_frame = TAKE_FRAME;
    bool buffer_ready;

    Mat img;
    Mat background_img = Mat(ROWS,COLS,CV_8U);
    //vector<int> elements_pixels[ROWS * COLS];
    uchar elements_pixels[ROWS*COLS][DEPTH_SIZE];

    //sc_in<bool> clock;
    //sc_event backimg_ready;
    //sc_out<bool> back_ready;

    SC_CTOR(Reference_image)
    {
        SC_THREAD(median_depth_images);
        index = 0;
        frame_counter = 0;
        buffer_ready = false;
    }

    //int computeMedian(vector<int> elements)
    uchar computeMedian(uchar A[], int len)
    {
        //nth_element(elements.begin(), elements.begin() + elements.size() / 2, elements.end());

        sort(A, A+len);
        if (len % 2 != 0)
            return A[len / 2];
        return (A[len/2] + A[(len-1)/2])/ 2;
    }

    void median_depth_images() {
        while (true) {
            ImgInPort.read(img);
            cout << "back thread" <<frame_counter << endl;
            wait(0, SC_MS);
            //back_ready.write(0);
            if (buffer_ready == false && index == DEPTH_SIZE - 1)
                buffer_ready = true;
            //cout << index << ":" << frame_counter << buffer_ready << endl;
            if (frame_counter % use_frame == 0) {
                for (int i = 0; i < ROWS; i++) {
                    for (int j = 0; j < COLS; j++) {
                        elements_pixels[(i *COLS) + j][index % DEPTH_SIZE] = img.at<uchar>(i, j);
                        //if (buffer_ready) {
                            background_img.at<uchar>(i, j) = computeMedian(elements_pixels[(i * COLS) + j], DEPTH_SIZE);
                            //cout << background_img.at<uchar>(i, j) << endl;
                        //}
                    }
                }
                index++;
            }
            //if (buffer_ready) {
                char outputfilename[256];
                sprintf_s(outputfilename, 256, "Back%d.png", frame_counter);
                //cout << outputfilename;
                imwrite(outputfilename, background_img);
                /*wait for pos edge of clock*/
                //wait();
                wait(1000, SC_MS);
                ImgOutPort.write(background_img);
                //backimg_ready.notify();
                //back_ready.write(1);
                //extract_person();
            //}
            frame_counter++;
        }
    }
};

SC_MODULE(Person_extraction) {
    sc_fifo_in<Mat> ImgInPort;
    sc_fifo_in<Mat> BackImgInPort;
    sc_fifo_out<CEN_BOUND> outPort;
    //sc_in<bool> back_ready;
    //sc_out<bool> per_ready;
    //sc_fifo_out<Mat> PerImgOutPort;


    Mat img, backimg;
    Mat subtract_img = Mat(ROWS, COLS, CV_8U);
    Mat thres_img = Mat(ROWS, COLS, CV_8U);
    Mat erode_img = Mat(ROWS, COLS, CV_8U);
    Mat label_img;
    Mat stats_img;
    Mat centroids_img;
    Mat person_img = Mat(ROWS, COLS, CV_8U, Scalar(0));

    //sc_in<bool> clock;
    //sc_event backimg_ready;

    int erosion_size = 1;
    unsigned int frame_counter;
    SC_CTOR(Person_extraction)
    {
        SC_THREAD(extract_person);
        frame_counter = 0;
    }

    void extract_person() {
        while (true) {
            //per_ready.write(0);
            ImgInPort.read(img);
            wait(0, SC_MS);
            //if (back_ready.read()) {
                BackImgInPort.read(backimg);
                wait(0, SC_MS);
                char outputfilename[256];
                //sprintf_s(outputfilename, 256, "Back%d.png", frame_counter);
                //imwrite(outputfilename, backimg);
                //TODO: add signal to know if backgroundimg is ready for processing or fifo read
                //Subtract current image from background image and threadhold
                absdiff(img, backimg, subtract_img);
                //imwrite("sub.jpg", subtract_img);
                threshold(subtract_img, thres_img, BACK_THRES, 255, THRESH_BINARY);

                erode(thres_img, erode_img, Mat(), Point(-1, -1), 10);
                dilate(erode_img, thres_img, Mat(), Point(-1, -1), 10);
                //sprintf_s(outputfilename, 256, "THRES%d.png", frame_counter);
                //imwrite(outputfilename, thres_img);
                //Find the connected components in the image
                int nLabels = connectedComponentsWithStats(thres_img, label_img, stats_img, centroids_img, 8, CV_16U);
                cout << "per_thread:" << frame_counter << ":" << nLabels << endl;
                int max_area = -1, label = 0;
                //sprintf_s(outputfilename, 256, "label%d.jpg", frame_counter);
                //imwrite(outputfilename, label_img);
                for (int i = 1; i < nLabels; i++) {
                    int val = stats_img.at<int>(i, 4);
                    if (val > max_area) {
                        max_area = val;
                        label = i;
                    }
                }
                //cout << "area" << stats_img.at<int>(label, 4) << ":" << stats_img.at<int>(label, 0) << ":" << stats_img.at<int>(label, 1) << ":" << stats_img.at<int>(label, 2) << ":" << stats_img.at<int>(label, 3) << endl;
                cout << "Centroid" << centroids_img.at<double>(label, 0) << ":" << centroids_img.at<double>(label, 1) << " height/width of box" << stats_img.at<int>(label, 3) << ":" << stats_img.at<int>(label, 2) << endl;

                for (int i = 0; i < ROWS; i++) {
                    for (int j = 0; j < COLS; j++) {
                        if (label_img.at<ushort>(i, j) == label)
                            person_img.at<uchar>(i, j) = 255;
                        else
                            person_img.at<uchar>(i, j) = 0;
                    }
                }
                //PerImgOutPort.write(person_img);
                //wait();
                CEN_BOUND R;
                R[0] = centroids_img.at<double>(label, 0);
                R[1] = centroids_img.at<double>(label, 1);
                R[2] = img.at<uchar>(ceil(R[1]), ceil(R[0]));
                R[3] = stats_img.at<int>(label, 3);
                R[4]= stats_img.at<int>(label, 2);
                outPort.write(R);
                //per_ready.write(1);
                sprintf_s(outputfilename, 256, "output_images\\person%d.png", frame_counter);
                imwrite(outputfilename, person_img);
                frame_counter++;
            //}
        }
    }
};

SC_MODULE(Floor_plane) {
    sc_fifo_in<Mat> ImgInPort;
    sc_fifo_out<PLANE_PARA> outPort;
    //sc_out<bool> floor_ready;


    Mat img;
    Mat d_img = Mat(ROWS, COLS, CV_32S, Scalar(0));
    Mat v_img, v_img_t;
    Mat edges, cedges;
    vector<Vec3f> lines;
    vector<Vec3f> pointset;
    vector<float> rightside;
    Mat result;

    //sc_in<bool> clock;

    unsigned int frame_counter;
    SC_CTOR(Floor_plane)
    {
        SC_THREAD(get_floor_plane);
        frame_counter = 0;
    }

    void get_floor_plane() {
        while (true) {
            //floor_ready.write(0);
            cout << "floor thread" << endl;
            ImgInPort.read(img);
            wait(0, SC_MS);
            
            float max_v_val, minp = 250, d, disp;
            //cout << minp << ":" << maxp << endl;
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < COLS; j++) {
                    if (img.at<ushort>(i, j) < minp && img.at<ushort>(i, j) > 0) {
                        minp = img.at<ushort>(i, j);
                    }
                }
            }
            d = C0 * minp / (SCALE * 1000);
            //cout << d << ":" << minp << endl;
            max_v_val = B * F / d;
            
            v_img = Mat(ROWS, (int)max_v_val + 1, CV_8U, Scalar(0));
            Rect crop_mat(0, ROWS / 2, 50, ROWS / 2);
            //cout << (int)max_v_val << ":" << v_img.size() << endl;
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < COLS; j++) {
                    if (img.at<ushort>(i, j) > 0) {
                        d = C0 * (float)img.at<ushort>(i, j) / (SCALE * 1000);
                        //cout << (float)img.at<uchar>(i, j) << ":" << d << endl;
                        disp = ((B * F) / d);
                        d_img.at<int>(i, j) = (int)disp;
                        //cout << disp << ":" << d_img.at<int>(i, j) << endl;
                        v_img.at<uchar>(i, d_img.at<int>(i, j)) += 1;
                    }
                }
            }

            /*Truncated the image to take bottom half only*/
            v_img_t = v_img(crop_mat);

            /*Canny Edge*/
            Canny(v_img_t, edges, 50, 200, 3);
            /*Hough Transform*/
            HoughLines(edges, lines, 1, CV_PI / 180, 100, 0, 0);
            if(lines.size() == 0)
                HoughLines(edges, lines, 1, CV_PI / 180, 80, 0, 0);

            cvtColor(edges, cedges, COLOR_GRAY2BGR);

            cout << "lines:" << lines.size() << endl;
            for (int i = 0; i < lines.size(); i++) {
                //cout << lines[i][0] << "rho:theta" << lines[i][1]  << ":" << lines[i][2] << endl;
                float rho = lines[i][0], theta = lines[i][1];
                Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a * rho, y0 = b * rho;
                pt1.x = cvRound(x0 + 1000 * (-b));
                pt1.y = cvRound(y0 + 1000 * (a));
                pt2.x = cvRound(x0 - 1000 * (-b));
                pt2.y = cvRound(y0 - 1000 * (a));
                line(cedges, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
            }

            float rho = lines[0][0], theta = lines[0][1];
            double a = cos(theta), b = sin(theta);
            /*Collect all points corresponding to detemined line*/
            for (int i = 0; i < v_img_t.rows; i++) {
                for (int j = 0; j < v_img_t.cols; j++) {
                    float pix = v_img_t.at<uchar>(i, j);
                    if (abs((i * a + b * j) - rho) < 0.1) {
                        int actual_r = i + ROWS / 2;
                        for (int k = 0; k < COLS; k++) {
                            if (abs(d_img.at<int>(actual_r, k) - j) < 3) {
                                if (img.at<ushort>(actual_r, k) > 0) {
                                    Vec3f v0 = Vec3f(actual_r, k, img.at<ushort>(actual_r, k));
                                    pointset.push_back(v0);
                                    //cout << "points-1" << actual_r << ":" << k << ":" << img.at<ushort>(actual_r, k) << endl;
                                    rightside.push_back(1);
                                }
                            }
                        }
                    }
                }
            }
            Mat lefts = Mat(pointset.size(), 3, CV_32F);
            Mat rights = Mat(pointset.size(), 1, CV_32F);

            for (int i = 0; i < pointset.size(); i++) {
                lefts.at<float>(i, 0) = pointset[i][0];
                lefts.at<float>(i, 1) = pointset[i][1];
                lefts.at<float>(i, 2) = pointset[i][2];
                rights.at<float>(i, 0) = 1;
                //cout << "pointcovert" << lefts.at<float>(i, 0) << ":" << lefts.at<float>(i, 1) << ":" << lefts.at<float>(i, 2) << endl;
                //cout << "pointcovert" << rights.at<float>(i, 0)  << endl;
            }

            //cout << "Point Cloud size" << pointset.size() << endl;
            int ret = solve(lefts, rights, result, DECOMP_SVD);
            //cout << "Solve return value" << ret << endl;
            //cout << "done" << result.depth() << result.size() << endl;
            cout << "Result: " << result.at<float>(0, 0) << ":" << result.at<float>(1, 0) << ":" << result.at<float>(2, 0) << endl;
            PLANE_PARA R;
            R[0] = result.at<float>(0, 0);
            R[1] = result.at<float>(1, 0);
            R[2] = result.at<float>(2, 0);

            char outputfilename[256];
            //sprintf_s(outputfilename, 256, "dmap%d.png", frame_counter);
            //imwrite(outputfilename, d_img);
            sprintf_s(outputfilename, 256, "output_images\\vmap%d.png", frame_counter);
            imwrite(outputfilename, v_img_t);

            sprintf_s(outputfilename, 256, "output_images\\edgesHTmap%d.png", frame_counter);
            imwrite(outputfilename, cedges);
            //wait();
            //floor_ready.write(1);
            outPort.write(R);
            frame_counter++;
            
        }
    }
};

SC_MODULE(Distance_centroid) {
    sc_fifo_in<CEN_BOUND> ImgInPort1;
    sc_fifo_in<Mat> ImgInPort;
    sc_fifo_in<PLANE_PARA> ImgInPort2;
    sc_out<bool> fall;

    //sc_in<bool>per_ready;
    //sc_in<bool>floor_ready;

    //sc_in<bool> clock;
    CEN_BOUND c1;
    PLANE_PARA p1;
    Mat img;

    unsigned int frame_counter;

    SC_CTOR(Distance_centroid)
    {
        SC_THREAD(compute_result);
        frame_counter = 0;
    }

    void compute_result()
    {
        ofstream Result("Result_distance.txt");
        while (true) {
            //if(per_ready.read() && floor_ready.read()){
                ImgInPort.read(img);
                ImgInPort1.read(c1);
                ImgInPort2.read(p1);
                wait(0, SC_NS);

                double distance, d;
                d = C0 * (double)img.at<ushort>(ceil(c1[1]), ceil(c1[0])) / (SCALE * 1000);
                distance = abs(p1[0] * c1[1] + p1[1] * c1[0] + p1[2] * d + 1) / sqrt(p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2]);

                cout << "Distance:" << frame_counter  << ":" << distance  << ":" << (c1[3] / c1[4]) << ":" << (ROWS - c1[1]) << endl;
                Result << frame_counter << ":" << distance << ":" << c1[0] << ":" << c1[1] << ":" << c1[3] << ":" << c1[4] << endl;
                //wait();
                if (((c1[3] / c1[4]) < FALL_T_RATIO) || ((ROWS - c1[1]) < FALL_T_Y)) {
                    fall.write(1);
                    cout << "FALL detected" << endl;
                }
                else {
                    fall.write(0);
                    cout << "Normal conditions" << endl;
                }
                frame_counter++;
            //}
        }
        Result.close();
    }

};

int sc_main(int argc, char* argv[])
{
    //For Linux based system: set_stack_size(256*ROWS*COLS);
    //sc_clock clock("System_clock", 1, 0.5);
    sc_fifo<Mat> q1;
    sc_fifo<Mat> q2;
    sc_fifo<Mat> q3;
    sc_fifo<Mat> q4;
    sc_fifo<CEN_BOUND> q5;
    sc_fifo<PLANE_PARA> q6;
    sc_fifo<Mat> q7;
    //sc_signal<bool> back_ready_1;
    //sc_signal<bool> per_ready_1;
    //sc_signal<bool> floor_ready_1;

    sc_signal<bool> output_fall;

    Depth_image input_read_img("module_1");
    //input_read_img.clock(clock);
    input_read_img.ImgOutPort.bind(q1);
    input_read_img.ImgOutPort1.bind(q2);
    input_read_img.ImgOutPort2.bind(q4);
    input_read_img.ImgOutPort3.bind(q7);

    Reference_image background_image("module_2");
    //background_image.clock(clock);
    //background_image.back_ready(back_ready_1);
    background_image.ImgInPort.bind(q1);
    background_image.ImgOutPort.bind(q3);

    Person_extraction per_image("module_3");
    //per_image.clock(clock);
    //per_image.back_ready(back_ready_1);
    //per_image.per_ready(per_ready_1);
    per_image.ImgInPort.bind(q2);
    per_image.BackImgInPort.bind(q3);
    per_image.outPort.bind(q5);

    Floor_plane floor_plane("module_4");
    //floor_plane.clock(clock);
    //floor_plane.floor_ready(floor_ready_1);
    floor_plane.ImgInPort.bind(q4);
    floor_plane.outPort.bind(q6);

    Distance_centroid result("module_5"); 
    //result.clock(clock);
    //result.per_ready(per_ready_1);
    //result.floor_ready(floor_ready_1);
    result.fall(output_fall);
    result.ImgInPort1.bind(q5);
    result.ImgInPort2.bind(q6);
    result.ImgInPort.bind(q7);

    sc_trace_file* wf = sc_create_vcd_trace_file("Signals_waveform");

    //sc_trace(wf, clock, "clock");
    //sc_trace(wf, back_ready_1, "back_ready");
    //sc_trace(wf, per_ready_1, "per_ready");
    //sc_trace(wf, floor_ready_1, "floor_ready");
    sc_trace(wf, output_fall, "output_fall");

    sc_start();
    sc_close_vcd_trace_file(wf);
    //sc_start();
    return (0);
}



