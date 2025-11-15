#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include "base64.h"
#include <cxxopts.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;


std::string get_api_key(const std::string& api_key_env) {

    const char* api_key = std::getenv(api_key_env.c_str());
    if (api_key == nullptr) 
        throw std::runtime_error(std::string(api_key_env) + " environment variable not set");

    return std::string(api_key);
}

// Function to check if the string is a URL
bool is_url(const std::string& image_path) {
    return image_path.rfind("http://", 0) == 0 || image_path.rfind("https://", 0) == 0;
}


// Function to encode the image

std::string encode_image(const std::string& image_path, int target_size) {
    // Read the image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Unable to open image file: " + image_path);
    }

    // Calculate the aspect ratio and resize the image while maintaining the aspect ratio
    int original_width = image.cols;
    int original_height = image.rows;
    float aspect_ratio = (float)original_width / (float)original_height;

    int new_width, new_height;
    if (aspect_ratio >= 1.0f) {
        // Width is greater than or equal to height, so scale based on width
        new_width = target_size;
        new_height = static_cast<int>(target_size / aspect_ratio);
    } else {
        // Height is greater than width, so scale based on height
        new_height = target_size;
        new_width = static_cast<int>(target_size * aspect_ratio);
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));

    // Create a square canvas (target_size x target_size) and place the resized image in the center
    int top = (target_size - new_height) / 2;
    int bottom = target_size - new_height - top;
    int left = (target_size - new_width) / 2;
    int right = target_size - new_width - left;

    cv::Mat squared_image;
    cv::copyMakeBorder(resized_image, squared_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));  // Padding with black color

    // Convert the square image to a vector of unsigned chars (binary data)
    std::vector<unsigned char> buf;
    cv::imencode(".jpg", squared_image, buf);  // Encode as JPG (or PNG if needed)

    // Encode the image data as base64
    return base64_encode(buf.data(), buf.size());
}

// Callback function for CURL
size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *s) {
    size_t newLength = size * nmemb;
    try {
        s->append((char*)contents, newLength);
    } catch(std::bad_alloc &e) {
        return 0;
    }
    return newLength;
}

int main(int argc, char* argv[]) {
    try {
        // Set up command line options
        cxxopts::Options options("VLM Inference Client", "Vision Language Model inference using OpenAI-compatible APIs");
        options.add_options()
            ("p,prompt", "Text prompt for image analysis", cxxopts::value<std::string>())
            ("a,api_key_env", "API key environment variable", cxxopts::value<std::string>())
            ("i,images", "Image file paths", cxxopts::value<std::vector<std::string>>())
            ("m,model", "Model name", cxxopts::value<std::string>()->default_value(""))
            ("e,api_endpoint", "API endpoint URL", cxxopts::value<std::string>())
            ("r,provider", "API provider", cxxopts::value<std::string>()->default_value(""))  
            ("d,detail", "Image detail level (auto, low, high)", cxxopts::value<std::string>()->default_value("low"))
            ("t,tokens", "Max tokens for response", cxxopts::value<int>()->default_value("300"))
            ("s,size", "Image size for encoding", cxxopts::value<int>()->default_value("512"))
            ("h,help", "Print usage")
        ;
        options.parse_positional({"images"});

        auto result = options.parse(argc, argv);

        if (result.count("help") || !result.count("images")) {
            std::cout << options.help() << std::endl;
            return 0;
        }
        std::string API_KEY_ENV = result["api_key_env"].as<std::string>();
        std::string prompt = result["prompt"].as<std::string>();
        std::vector<std::string> image_paths = result["images"].as<std::vector<std::string>>();
        std::string model = result["model"].as<std::string>();
        std::string api_endpoint = result["api_endpoint"].as<std::string>();
        std::string detail = result["detail"].as<std::string>();
        int max_tokens = result["tokens"].as<int>();
        std::string provider = result["provider"].as<std::string>();
        int target_size = result["size"].as<int>();

        // Get the API key/endpoint from environment variable
        std::string api_key = get_api_key(API_KEY_ENV);

        // Prepare JSON payload
        json payload = {
            {"model", model},
            {"messages", json::array({
                {
                    {"role", "user"},
                    {"content", json::array()}
                }
            })},
            {"max_tokens", max_tokens}
        };

        // Add text prompt
        payload["messages"][0]["content"].push_back({
            {"type", "text"},
            {"text", prompt}
        });

        // Add images (distinguish between URLs and local files)
        for (const auto& image_path : image_paths) {
            json image_content;
            if (is_url(image_path)) {
                // If it's a URL, use it directly
                image_content = {
                    {"type", "image_url"},
                    {"image_url", {
                        {"url", image_path},
                        {"detail", detail}
                    }}
                };
            } else {
                // If it's a local file, encode it as base64
                std::string base64_image = encode_image(image_path, target_size);
                image_content = {
                    {"type", "image_url"},
                    {"image_url", {
                        {"url", "data:image/jpeg;base64," + base64_image},
                        {"detail", detail}
                    }}
                };
            }
            payload["messages"][0]["content"].push_back(image_content);
        }

        // Initialize CURL
        CURL *curl = curl_easy_init();
        if(curl) {
            struct curl_slist *headers = NULL;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key).c_str());

            std::string response_string;
            std::string payload_string = payload.dump();

            curl_easy_setopt(curl, CURLOPT_URL, api_endpoint.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_string.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

            CURLcode res = curl_easy_perform(curl);

            if(res != CURLE_OK) {
                std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            } else {
                std::cout << response_string << std::endl;
            }

            curl_easy_cleanup(curl);
            curl_slist_free_all(headers);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}