#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

class MNISTClassifier : public torch::nn::Module
{
public:
    MNISTClassifier()
        : conv1(torch::nn::Conv2dOptions(1, 10, 5)),
          pool1(torch::nn::AvgPool2dOptions(2).stride(2)),
          conv2(torch::nn::Conv2dOptions(10, 20, 5)),
          pool2(torch::nn::AvgPool2dOptions(2).stride(2)),
          conv3(torch::nn::Conv2dOptions(20, 50, 3))
    {
        register_module("conv1", conv1);
        register_module("pool1", pool1);
        register_module("conv2", conv2);
        register_module("pool2", pool2);
        register_module("conv3", conv3);

        // Use a dummy input to determine the size of the fully connected layer
        torch::Tensor dummy_input = torch::randn({1, 1, 28, 28});
        auto output = forward_conv(dummy_input);
        auto num_features = output.numel() / output.size(0);

        fc1 = register_module("fc1", torch::nn::Linear(num_features, 500));
        fc2 = register_module("fc2", torch::nn::Linear(500, 10));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = forward_conv(x);
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = torch::log_softmax(fc2->forward(x), 1);
        return x;
    }

private:
    torch::Tensor forward_conv(torch::Tensor x)
    {
        x = torch::relu(pool1->forward(conv1->forward(x)));
        x = torch::relu(pool2->forward(conv2->forward(x)));
        x = torch::relu(conv3->forward(x));
        return x;
    }

    torch::nn::Conv2d conv1, conv2, conv3;
    torch::nn::AvgPool2d pool1, pool2;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main(int argc, char **argv)
{

    if (!torch::mps::is_available())
    {
        std::cerr << "MPS is not available. Exiting..." << std::endl;
        return 1;
    }

    if (argc < 2)
    {
        std::cout << "Usage: ./mnist_classifier <command>" << std::endl;
        std::cout << "Commands:" << std::endl;
        std::cout << "train - Train the model" << std::endl;
        std::cout << "test - Test the model" << std::endl;
        return 1;
    }

    std::string command = argv[1];

    torch::Device device(torch::kMPS);

    // Load the model
    auto model = std::make_shared<MNISTClassifier>();

    if (command == "train")
    {
        auto train_dataset = torch::data::datasets::MNIST("../resources/train", torch::data::datasets::MNIST::Mode::kTrain)
                                 .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                                 .map(torch::data::transforms::Stack<>());

        auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(64));

        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

        model->to(device);

        for (size_t epoch = 1; epoch <= 10; ++epoch)
        {
            model->train();
            size_t batch_index = 0;

            for (auto &batch : *train_data_loader)
            {
                auto data = batch.data.to(device);
                auto targets = batch.target.to(device);

                optimizer.zero_grad();
                auto output = model->forward(data);
                auto loss = torch::nll_loss(output, targets);
                loss.backward();
                optimizer.step();

                if (++batch_index % 100 == 0)
                {
                    std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << std::endl;
                }
            }
        }

        torch::save(model, "mnist_model.pt");
    }
    else if (command == "test")
    {
        torch::load(model, "mnist_model.pt");

        // Load the test dataset
        auto test_dataset = torch::data::datasets::MNIST("../resources/test", torch::data::datasets::MNIST::Mode::kTest)
                                .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                                .map(torch::data::transforms::Stack<>());

        // Create a data loader
        auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(64));

        model->eval();
        torch::NoGradGuard no_grad;
        size_t correct = 0;
        size_t total = 0;

        bool close = false;

        for (const auto &batch : *test_data_loader)
        {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            auto output = model->forward(data);
            auto predictions = output.argmax(1);

            total += targets.size(0);
            correct += predictions.eq(targets).sum().item<int>();

            // Visualize the images and predictions
            for (size_t i = 0; i < data.size(0); ++i)
            {

                if (close)
                    break;

                if (predictions[i].item<int>() == targets[i].item<int>())
                    continue;

                auto img_tensor = data[i].cpu();
                img_tensor = img_tensor.mul(0.5).add(0.5); // De-normalize the image
                auto img = img_tensor.squeeze().clone().detach();
                cv::Mat img_mat(28, 28, CV_32F, img.data_ptr());
                img_mat.convertTo(img_mat, CV_8U, 255);
                cv::resize(img_mat, img_mat, cv::Size(280, 280), 0, 0, cv::INTER_NEAREST);

                int pred = predictions[i].item<int>();
                std::string label = "Pred: " + std::to_string(pred);
                std::string actual = "Actual: " + std::to_string(targets[i].item<int>());
                cv::putText(img_mat, label, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(120, 120, 255), 1);
                cv::putText(img_mat, actual, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(120, 120, 255), 1);
                cv::imshow("MNIST Test Image", img_mat);

                if (cv::waitKey(0) == 27)
                {
                    close = true;
                    break;
                }
            }
        }

        std::cout << "Accuracy: " << (100.0 * correct / total) << "%" << std::endl;
    }

    return 0;
}
