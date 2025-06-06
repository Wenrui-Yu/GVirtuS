// ----------------------------- OpenPose CUDA-Compatible Example -----------------------------
// This reads an image, processes it with OpenPose, and displays the pose keypoints.

#include <opencv2/opencv.hpp> 
//#define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
#include <openpose/headers.hpp>
#include <iostream>

// Custom flags
DEFINE_string(image_path, "examples/media/COCO_val2014_000000000192.jpg",
              "Process an image. Read all standard formats (jpg, png, bmp, etc.).");

DEFINE_bool(no_display, false,
            "Enable to disable the visual display.");

//DEFINE_bool(disable_multi_thread, false,
//            "Disable multi-threading for debugging or latency reduction.");

void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        std::cout << "Inside display" << std::endl;
        if (datumsPtr && !datumsPtr->empty())
        {
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            if (!cvMat.empty())
            {
                cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
                cv::waitKey(0);
            }
            else
                op::opLog("Empty cv::Mat as output.", op::Priority::High);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        if (datumsPtr && !datumsPtr->empty())
        {
            op::opLog("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int tutorialApiCuda()
{
    try
    {
        op::opLog("Starting OpenPose with CUDA (.cu file)...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // // Configure OpenPose
        // op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        // if (FLAGS_disable_multi_thread)
        //     opWrapper.disableMultiThreading();

        // opWrapper.start();


        // Configure OpenPose
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();

        // CPU-only configuration for older OpenPose
        op::WrapperStructPose poseConfig;
        poseConfig.poseMode = op::PoseMode::Enabled;
        poseConfig.poseModel = op::PoseModel::BODY_25;
        poseConfig.netInputSize = op::Point<int>{656, 368};
        poseConfig.outputSize = op::Point<int>{-1, -1};
        poseConfig.renderMode = op::RenderMode::None; // No GPU rendering
        poseConfig.alphaKeypoint = 0.6;
        poseConfig.alphaHeatMap = 0.7;
        poseConfig.defaultPartToRender = 0; // BODY_25
        poseConfig.enableGoogleLogging = true;

        // 👇 THIS IS THE KEY TO CPU MODE
        poseConfig.gpuNumber = 1;     // Use 0 GPUs
        poseConfig.gpuNumberStart = 0;

        // Apply configuration
        opWrapper.configure(poseConfig);
        opWrapper.start();


        const cv::Mat cvImageToProcess = cv::imread(FLAGS_image_path);
        if (cvImageToProcess.empty())
        {
            std::cerr << "Failed to load image: " << FLAGS_image_path << std::endl;
            return -1;
        }

        const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
        auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);

        if (datumProcessed != nullptr)
        {
            printKeypoints(datumProcessed);
            if (!FLAGS_no_display)
                display(datumProcessed);
        }
        else
            op::opLog("Image could not be processed.", op::Priority::High);

        op::printTime(opTimer, "OpenPose finished. Total time: ", " seconds.", op::Priority::High);
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    return tutorialApiCuda();
}
